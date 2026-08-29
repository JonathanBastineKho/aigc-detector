FROM python:3.12-slim

# libgl/libglib are needed by opencv, which comes in via the imaging stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first so the layer caches across code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY app.py .
COPY checkpoints/lora_r32_film_slim.pt checkpoints/
COPY data/cache/probe.joblib data/cache/

# Bake DINOv3 into the image. Without this every cold start pulls 1.2 GB from
# HuggingFace before it can answer a single request.
ENV HF_HOME=/app/.hf_cache
RUN uv run python -c "import timm; timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=True)"

# Cloud Run sends traffic to $PORT; Gradio must bind 0.0.0.0 to receive it.
ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=8080
EXPOSE 8080
CMD ["uv", "run", "python", "app.py", "--checkpoint", "checkpoints/lora_r32_film_slim.pt"]
