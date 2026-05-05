#!/usr/bin/env bash
# One-shot: merge 20万_err_nev.csv (or any fixed err CSV) + positive pool -> docker train.
# Defaults match /opt/vscc/plateai layout.
set -euo pipefail

DATA="${DATA:-/opt/vscc/plateai/data}"
CACHE="${CACHE:-/opt/vscc/plateai/cache}"
CKPT="${CKPT:-/opt/vscc/plateai/checkpoints}"
OUT="${OUT:-/opt/vscc/plateai/output}"
TOOLS="${TOOLS:-/opt/vscc/plateai/tools}"
PREFIX="${URL_PREFIX:-https://huizhoupark.obs.cn-south-1.myhuaweicloud.com}"

ERR_CSV="${ERR_CSV:-$DATA/20万_err_nev.csv}"
POOL_CSV="${POOL_CSV:-$DATA/2000原图.csv}"
MIX_OUT="${MIX_OUT:-$DATA/train_mix_nev_err.csv}"
MIX_MAX_ROWS="${MIX_MAX_ROWS:-12000}"
POS_RATIO="${POS_RATIO:-0.55}"
SEED="${SEED:-42}"

IMAGE="${PLATEAI_IMAGE:-ghcr.io/vesaaa/plateai:cpu}"

mkdir -p "$CACHE" "$CKPT" "$OUT"
[[ -f "$ERR_CSV" ]] || { echo "missing ERR_CSV=$ERR_CSV"; exit 1; }
[[ -f "$POOL_CSV" ]] || { echo "missing POOL_CSV=$POOL_CSV"; exit 1; }

echo "[mix] err=$ERR_CSV pool=$POOL_CSV -> $MIX_OUT max_rows=$MIX_MAX_ROWS pos_ratio=$POS_RATIO"
python3 "$TOOLS/build_train_mix.py" \
  --err "$ERR_CSV" \
  --pool "$POOL_CSV" \
  --output "$MIX_OUT" \
  --max-rows "$MIX_MAX_ROWS" \
  --pos-ratio "$POS_RATIO" \
  --seed "$SEED"

PT_MOUNT=()
if [[ -f "$CKPT/best.pth" ]]; then
  PT_MOUNT=(-v "$CKPT/best.pth:/workspace/init.pth:ro")
  PT_ARG=(--pretrained /workspace/init.pth)
  echo "[train] pretrained=$CKPT/best.pth"
else
  echo "[train] no $CKPT/best.pth; use image default pretrained (/workspace/weights/plate_rec_color.pth)"
  PT_ARG=()
fi

extra_max=()
if [[ -n "${TRAIN_MAX_ROWS:-}" && "${TRAIN_MAX_ROWS}" != "0" ]]; then
  extra_max=(--max-rows "$TRAIN_MAX_ROWS")
fi

docker run --rm \
  -e PLATEAI_DEVICE="${PLATEAI_DEVICE:-cpu}" \
  -e PLATEAI_WORKERS="${PLATEAI_WORKERS:-4}" \
  -e PLATEAI_BATCH_SIZE="${PLATEAI_BATCH_SIZE:-14}" \
  -e PLATEAI_CACHE_DIR=/workspace/cache \
  -v "$DATA:/data:ro" \
  -v "$CACHE:/workspace/cache" \
  -v "$CKPT:/workspace/checkpoints" \
  -v "$OUT:/workspace/output" \
  "${PT_MOUNT[@]}" \
  "$IMAGE" train \
  --csv "/data/$(basename "$MIX_OUT")" \
  "${PT_ARG[@]}" \
  --url-prefix "$PREFIX" \
  --output /workspace/output/plate_rec_color.onnx \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${PLATEAI_BATCH_SIZE:-14}" \
  --workers "${PLATEAI_WORKERS:-4}" \
  --lr "${LR:-3.5e-4}" \
  --hard-case-repeat "${HARD_REPEAT:-2}" \
  --val-ratio "${VAL_RATIO:-0.1}" \
  --seed "${TRAIN_SEED:-1234}" \
  ${TRAIN_EXTRA:-} \
  "${extra_max[@]}"

echo "[done] ONNX -> $OUT/plate_rec_color.onnx  checkpoint -> $CKPT/best.pth"
echo "Deploy: cp -a $OUT/plate_rec_color.onnx /opt/vscc/platex/models/ && docker restart <platex>"
