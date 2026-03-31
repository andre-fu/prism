FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git wget nginx \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch with CUDA 12.8
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# flash-attn (compile from source for CUDA compatibility)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Python deps
RUN pip install --no-cache-dir \
    transformers>=4.57 \
    safetensors \
    huggingface-hub \
    accelerate \
    flashinfer \
    triton \
    fastapi \
    uvicorn \
    sse-starlette \
    pyyaml

WORKDIR /app
COPY engine/ /app/engine/
COPY gpu_swap/ /app/gpu_swap/

# Nginx config for TLS termination (optional)
COPY deploy/nginx.conf /etc/nginx/sites-available/prism

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TRANSFORMERS_NO_TF=1
ENV USE_TF=0

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "engine.serve"]
CMD ["--config", "/app/config.yaml"]
