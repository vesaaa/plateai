#!/usr/bin/env bash
# One-shot `plateai train` with persistent host cache (reuses downloaded images).
# Default paths match 192.168.1.66 layout; override with env.
set -euo pipefail
DATA="${DATA:-/opt/vscc/plateai/data}"
CACHE="${CACHE:-/opt/vscc/plateai/cache}"
CKPT="${CKPT:-/opt/vscc/plateai/checkpoints}"
OUT="${OUT:-/opt/vscc/plateai/output}"
IMAGE="${PLATEAI_IMAGE:-ghcr.io/vesaaa/plateai:v1.0.3-cpu}"
mkdir -p "$CACHE" "$CKPT" "$OUT"
echo "cache: $DATA -> /data:ro  |  $CACHE -> /workspace/cache (reuse)  |  $CKPT -> /workspace/checkpoints  |  $OUT -> /workspace/output"
exec docker run --rm \
  -e PLATEAI_DEVICE="${PLATEAI_DEVICE:-cpu}" \
  -e PLATEAI_CACHE_DIR=/workspace/cache \
  -v "$DATA:/data:ro" \
  -v "$CACHE:/workspace/cache" \
  -v "$CKPT:/workspace/checkpoints" \
  -v "$OUT:/workspace/output" \
  "$IMAGE" train "$@"
