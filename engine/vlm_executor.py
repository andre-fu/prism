"""VLM (Vision-Language Model) executor.

Extends the text-only executor to handle image inputs:
1. Image preprocessing (resize, normalize, patch extraction)
2. Vision encoder forward pass (ViT)
3. Cross-modal projection (vision → LLM embedding space)
4. Standard LLM decode (text generation conditioned on image features)

Supports Qwen2-VL architecture. The vision encoder runs once during prefill;
decode is identical to text-only (same CUDA graphs).
"""

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoConfig
from PIL import Image
import requests as http_requests
import base64
import io

from .fa_kv_cache import FlashAttnKVCache
from .weight_pool import StaticWeightPool
from .fused_kernels import fused_rms_norm


class VLMWeightPool:
    """Static weight pool for VLMs: includes both vision encoder and LLM weights."""

    def __init__(self, hf_config, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.config = hf_config

        # We store the full model and extract components
        # Vision encoder weights are relatively small (~630M params = ~1.3GB bf16)
        # LLM weights are handled like text-only models
        self._model: Qwen2VLForConditionalGeneration | None = None

    def load_model(self, model_id: str):
        """Load the full VLM model to GPU."""
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, dtype=self.dtype, device_map=self.device,
        )
        self._model.eval()
        return self._model

    def load_from_pinned(self, pinned_weights: dict[str, torch.Tensor], hf_config):
        """Load VLM weights from pinned RAM into the model."""
        if self._model is None:
            with torch.device("meta"):
                self._model = Qwen2VLForConditionalGeneration._from_config(hf_config)

            # Load weights
            gpu_weights = {n: t.to(self.dtype).to(self.device) for n, t in pinned_weights.items()}

            # Handle tied weights
            if getattr(hf_config, "tie_word_embeddings", False):
                if "model.embed_tokens.weight" in gpu_weights:
                    gpu_weights["lm_head.weight"] = gpu_weights["model.embed_tokens.weight"]

            self._model.load_state_dict(gpu_weights, strict=False, assign=True)

            # Reinit rotary buffers
            for name, buf in list(self._model.named_buffers()):
                if buf.device == torch.device("meta") and "inv_freq" in name:
                    dim = buf.shape[0] * 2
                    rope_theta = getattr(hf_config, "rope_theta", 10000.0)
                    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                    parent = self._model
                    for part in name.rsplit(".", 1)[0].split("."):
                        parent = getattr(parent, part)
                    parent.inv_freq = inv_freq.to(self.device)

            self._model.to(self.device)
            self._model.eval()
            del gpu_weights

        return self._model

    @property
    def model(self):
        return self._model

    @property
    def total_gb(self):
        if self._model:
            return sum(p.nbytes for p in self._model.parameters()) / 1e9
        return 0


class VLMExecutor:
    """Executes VLM inference: image processing + text generation.

    The vision encoder runs during prefill (processes image patches).
    Decode is standard autoregressive generation (same as text-only).
    """

    def __init__(self, model: Qwen2VLForConditionalGeneration, kv_cache: FlashAttnKVCache,
                 processor, device: str):
        self.model = model
        self.kv = kv_cache
        self.processor = processor
        self.device = device

        config = model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_layers = config.num_hidden_layers

    def process_image(self, image_input) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image_input, Image.Image):
            return image_input
        if isinstance(image_input, str):
            if image_input.startswith("http"):
                response = http_requests.get(image_input)
                return Image.open(io.BytesIO(response.content))
            elif image_input.startswith("data:image"):
                # base64 data URL
                header, data = image_input.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(data)))
            else:
                return Image.open(image_input)
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def prefill_with_image(self, text: str, images: list, seq_id: int) -> torch.Tensor:
        """Run prefill with image(s) + text.

        Uses the model's processor to handle image-text interleaving,
        then runs the full model forward pass (vision encoder + LLM).
        """
        # Process inputs using the model's processor
        pil_images = [self.process_image(img) for img in images]

        # Build conversation format for Qwen2-VL
        messages = [{"role": "user", "content": []}]
        for img in pil_images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": text})

        # Process with the model's processor
        inputs = self.processor(
            text=self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=pil_images if pil_images else None,
            return_tensors="pt",
        ).to(self.device)

        # Get input length for KV cache
        input_len = inputs["input_ids"].shape[1]

        # Allocate KV cache
        self.kv.new_sequence(seq_id)
        self.kv._ensure_capacity(seq_id, input_len)

        # Run the full model forward (vision encoder + LLM)
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=False)

        # Update KV cache length
        self.kv.seq_len[seq_id] = input_len

        # Return logits for last token
        return outputs.logits[:, -1:, :]

    def decode_step(self, token_id: torch.Tensor, seq_id: int) -> torch.Tensor:
        """Standard autoregressive decode step (no vision, just LLM).

        After prefill processes the image, decode is text-only.
        Uses the model's LLM layers directly.
        """
        kv = self.kv
        pos = kv.seq_len[seq_id]
        kv._ensure_capacity(seq_id, pos + 1)
        kv.seq_len[seq_id] = pos + 1

        with torch.no_grad():
            # Run just the LLM part (no vision encoder)
            hidden = self.model.model.embed_tokens(token_id)

            # Standard transformer decode
            for layer in self.model.model.layers:
                hidden = layer(hidden, position_ids=torch.tensor([[pos]], device=self.device))[0]

            hidden = self.model.model.norm(hidden)
            logits = self.model.lm_head(hidden)

        return logits

    @torch.no_grad()
    def generate(self, text: str, images: list | None = None, seq_id: int = 0,
                 max_new_tokens: int = 100, eos_token_id: int | None = None) -> list[int]:
        """Full VLM generation: image + text → text output.

        Uses the model's built-in generate() for proper vision-text handling.
        The vision encoder processes images during prefill; decode is autoregressive.
        """
        images = images or []
        pil_images = [self.process_image(img) for img in images]

        # Build conversation
        messages = [{"role": "user", "content": []}]
        for img in pil_images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": text})

        # Process with the model's processor
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text_input,
            images=pil_images if pil_images else None,
            return_tensors="pt",
        ).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]

        # Generate using model's native generate (handles vision+text KV cache properly)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Extract generated token IDs (excluding prompt)
        generated = output_ids[0][prompt_len:].tolist()
        return generated
