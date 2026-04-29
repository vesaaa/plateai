# PlateAI training image: PyTorch CPU + plateai trainer + bundled pretrained
# weights so that `docker run plateai train --csv ...` works out of the box.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Native libs needed by opencv-python-headless and PyTorch CPU.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch CPU first (large wheel) so the layer is cached cleanly.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu

# Install the rest of the runtime dependencies. Pinning narrowly to keep the
# layer reproducible across builds.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    requests==2.32.3 \
    onnx==1.16.2 \
    openpyxl==3.1.5

# Copy package source and the bundled pretrained weights.
COPY pyproject.toml /workspace/pyproject.toml
COPY plateai/ /workspace/plateai/
COPY weights/ /workspace/weights/

# Install plateai itself (no extras; deps were installed above).
RUN pip install --no-cache-dir --no-deps .

# Default mounted directories for outputs and download cache.
RUN mkdir -p /workspace/cache /workspace/checkpoints /workspace/output

VOLUME ["/workspace/cache", "/workspace/checkpoints", "/workspace/output"]

ENTRYPOINT ["plateai"]
CMD ["info"]
