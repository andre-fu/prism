"""Model upload: accept safetensors files, validate, store, and register.

Workflow:
1. Customer uploads safetensors files + config.json via multipart POST
2. We validate: correct format, shapes match claimed architecture, no corruption
3. Store files on disk (configurable upload directory)
4. Register in engine for serving
"""

import os
import json
import shutil
import hashlib
import time
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoConfig
import torch


class ModelUploadManager:
    """Handles model file upload, validation, and storage."""

    def __init__(self, upload_dir: str = "./model_uploads", max_upload_gb: float = 100.0):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = int(max_upload_gb * 1e9)

    def validate_and_store(
        self,
        name: str,
        config_json: dict,
        safetensor_files: list[tuple[str, bytes]],  # [(filename, content), ...]
    ) -> dict:
        """Validate uploaded model files and store to disk.

        Returns: {"status": "ok", "path": ..., "num_params": ..., "architecture": ...}
        Raises: ValueError on validation failure
        """
        model_dir = self.upload_dir / name
        if model_dir.exists():
            raise ValueError(f"Model '{name}' already exists. Delete first or use a different name.")

        # Step 1: Validate config
        self._validate_config(config_json)
        architecture = config_json.get("architectures", ["unknown"])[0]

        # Step 2: Check total size
        total_bytes = sum(len(content) for _, content in safetensor_files)
        if total_bytes > self.max_upload_bytes:
            raise ValueError(f"Upload too large: {total_bytes/1e9:.1f}GB > {self.max_upload_bytes/1e9:.0f}GB limit")

        # Step 3: Write files to temporary directory
        tmp_dir = self.upload_dir / f".tmp_{name}_{int(time.time())}"
        tmp_dir.mkdir(parents=True)

        try:
            # Write config
            with open(tmp_dir / "config.json", "w") as f:
                json.dump(config_json, f, indent=2)

            # Write safetensors files
            for filename, content in safetensor_files:
                if not filename.endswith(".safetensors"):
                    raise ValueError(f"Invalid file: {filename} (must be .safetensors)")
                filepath = tmp_dir / filename
                with open(filepath, "wb") as f:
                    f.write(content)

            # Step 4: Validate safetensors integrity + shapes
            validation = self._validate_weights(tmp_dir, config_json)

            # Step 5: Move to final location
            tmp_dir.rename(model_dir)

            return {
                "status": "ok",
                "name": name,
                "path": str(model_dir),
                "architecture": architecture,
                "num_params": validation["num_params"],
                "size_gb": total_bytes / 1e9,
                "num_shards": len(safetensor_files),
                "checksum": validation["checksum"],
            }

        except Exception:
            # Cleanup on failure
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise

    def _validate_config(self, config_json: dict):
        """Validate the model config has required fields."""
        required = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
        for field in required:
            if field not in config_json:
                raise ValueError(f"config.json missing required field: {field}")

        nh = config_json["num_attention_heads"]
        nkv = config_json.get("num_key_value_heads", nh)
        hs = config_json["hidden_size"]

        if hs % nh != 0:
            raise ValueError(f"hidden_size ({hs}) not divisible by num_attention_heads ({nh})")
        if nh % nkv != 0:
            raise ValueError(f"num_attention_heads ({nh}) not divisible by num_key_value_heads ({nkv})")

    def _validate_weights(self, model_dir: Path, config_json: dict) -> dict:
        """Validate safetensors files: loadable, correct shapes, complete."""
        st_files = sorted(model_dir.glob("*.safetensors"))
        if not st_files:
            raise ValueError("No .safetensors files found in upload")

        hs = config_json["hidden_size"]
        nl = config_json["num_hidden_layers"]
        nh = config_json["num_attention_heads"]
        nkv = config_json.get("num_key_value_heads", nh)
        hd = hs // nh
        mlp = config_json.get("intermediate_size", hs * 4)
        vocab = config_json.get("vocab_size", 32000)

        all_params = {}
        total_params = 0
        checksum = hashlib.sha256()

        for filepath in st_files:
            try:
                shard = load_file(str(filepath), device="cpu")
            except Exception as e:
                raise ValueError(f"Failed to load {filepath.name}: {e}")

            for name, tensor in shard.items():
                all_params[name] = tensor.shape
                total_params += tensor.numel()
                checksum.update(name.encode())
                checksum.update(str(tensor.shape).encode())

        # Validate critical weight shapes
        errors = []

        # Check embedding
        if "model.embed_tokens.weight" in all_params:
            shape = all_params["model.embed_tokens.weight"]
            if shape[1] != hs:
                errors.append(f"embed_tokens.weight dim 1 = {shape[1]}, expected {hs}")

        # Check first layer
        q_key = "model.layers.0.self_attn.q_proj.weight"
        if q_key in all_params:
            q_shape = all_params[q_key]
            expected_q = nh * hd
            if q_shape[0] != expected_q:
                errors.append(f"q_proj shape {q_shape}, expected [{expected_q}, {hs}]")

        # Check all layers exist
        for i in range(nl):
            prefix = f"model.layers.{i}"
            required_suffixes = [
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".self_attn.o_proj.weight",
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
                ".mlp.down_proj.weight",
            ]
            for suffix in required_suffixes:
                full_key = prefix + suffix
                if full_key not in all_params:
                    errors.append(f"Missing weight: {full_key}")

        if errors:
            raise ValueError(f"Weight validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return {
            "num_params": total_params,
            "checksum": checksum.hexdigest()[:16],
        }

    def delete_model(self, name: str):
        """Delete an uploaded model from disk."""
        model_dir = self.upload_dir / name
        if model_dir.exists():
            shutil.rmtree(model_dir)

    def list_uploaded(self) -> list[dict]:
        """List all uploaded models."""
        result = []
        for d in self.upload_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                config_path = d / "config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    st_files = list(d.glob("*.safetensors"))
                    size_gb = sum(f.stat().st_size for f in st_files) / 1e9
                    result.append({
                        "name": d.name,
                        "architecture": config.get("architectures", ["unknown"])[0],
                        "hidden_size": config.get("hidden_size"),
                        "num_layers": config.get("num_hidden_layers"),
                        "size_gb": round(size_gb, 2),
                        "path": str(d),
                    })
        return result
