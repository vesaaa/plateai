#!/usr/bin/env bash
# Verify golden WE baseline files exist before autotune (avoids silent AUTO_RESTORE_BASELINE no-op).
#
# Defaults match iter_train_platex_loop.sh / autotune.env.example.
# Exit 1 if required files missing (unless ONLY_WARN=1 prints messages and exits 0).
#
set -euo pipefail

BACK="${BACK:-/opt/vscc/plateai/backups}"
BASELINE_ACC="${BASELINE_ACC:-0.926}"
BASELINE_ONNX="${BASELINE_ONNX:-$BACK/OPTIMAL_plate_rec_color_acc_0_926000.onnx}"
BASELINE_PTH="${BASELINE_PTH:-$BACK/OPTIMAL_best_we_acc_0_926000.pth}"
ONLY_WARN="${ONLY_WARN:-0}"

log() { echo "[check_baseline] $*" >&2; }

missing=0
if [[ ! -f "$BASELINE_ONNX" ]] || [[ ! -r "$BASELINE_ONNX" ]]; then
  log "MISSING or unreadable ONNX: $BASELINE_ONNX"
  missing=1
fi
if [[ ! -f "$BASELINE_PTH" ]] || [[ ! -r "$BASELINE_PTH" ]]; then
  log "MISSING or unreadable PTH:  $BASELINE_PTH"
  missing=1
fi

if (( missing )); then
  log "BASELINE_ACC=${BASELINE_ACC} — copy OPTIMAL backups here or set BASELINE_ONNX / BASELINE_PTH."
  if [[ "$ONLY_WARN" == "1" ]]; then
    exit 0
  fi
  exit 1
fi

echo "OK baseline acc=${BASELINE_ACC}"
echo "  ONNX $BASELINE_ONNX ($(wc -c <"$BASELINE_ONNX") bytes)"
echo "  PTH  $BASELINE_PTH ($(wc -c <"$BASELINE_PTH") bytes)"
exit 0
