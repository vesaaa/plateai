#!/usr/bin/env bash
# Restore WE weights (plate_rec_color.onnx + checkpoints/best.pth) and restart platex.
#
# Default policy (no args): do NOT trust BEST_EVAL.txt — it may record a weak tuning run.
# Priority:
#   1) RESTORE_ONNX or first positional argument
#   2) BASELINE_ONNX when file exists (default: backups OPTIMAL_*0_926000*)
#   3) Highest acc among BACK/OPTIMAL_plate_rec_color_acc_*.onnx (parsed from filename)
#   4) Only if RESTORE_FROM_BEST_EVAL=1: use OUT/BEST_EVAL.txt paths
#
# Usage:
#   ./restore_optimal_we.sh
#   ./restore_optimal_we.sh /opt/vscc/plateai/backups/OPTIMAL_plate_rec_color_acc_0_926000.onnx
#   RESTORE_ONNX=/path/a.onnx RESTORE_PTH=/path/b.pth ./restore_optimal_we.sh
#   RESTORE_FROM_BEST_EVAL=1 ./restore_optimal_we.sh   # legacy / debugging
#
set -euo pipefail

BACK="${BACK:-/opt/vscc/plateai/backups}"
OUT="${OUT:-/opt/vscc/plateai/output}"
CKPT="${CKPT:-/opt/vscc/plateai/checkpoints}"
PLATEX_MODELS="${PLATEX_MODELS:-/opt/vscc/platex/models}"
PLATEX_CID="${PLATEX_CID:-$(docker ps --filter publish=8080 --format '{{.ID}}' | head -1)}"

BASELINE_ACC="${BASELINE_ACC:-0.926}"
BASELINE_ONNX="${BASELINE_ONNX:-$BACK/OPTIMAL_plate_rec_color_acc_0_926000.onnx}"
BASELINE_PTH="${BASELINE_PTH:-$BACK/OPTIMAL_best_we_acc_0_926000.pth}"

log() { echo "[$(date -Iseconds)] $*"; }

pick_max_acc_optimal_onnx() {
  local d="$1"
  python3 - "$d" <<'PY'
import glob
import os
import re
import sys

d = sys.argv[1]
files = glob.glob(os.path.join(d, "OPTIMAL_plate_rec_color_acc_*.onnx"))
best_v = -1.0
best_f = ""
for f in files:
    bn = os.path.basename(f)
    m = re.search(r"acc_(.+)\.onnx$", bn)
    if not m:
        continue
    try:
        v = float(m.group(1).replace("_", "."))
    except ValueError:
        continue
    if v > best_v:
        best_v, best_f = v, f
print(best_f)
PY
}

pick_pth_for_onnx() {
  local onnx="$1"
  local base
  base=$(basename "$onnx" .onnx)
  base=${base#OPTIMAL_plate_rec_color_acc_}
  echo "${BACK}/OPTIMAL_best_we_acc_${base}.pth"
}

resolve_sources() {
  RESTORE_ONNX="${RESTORE_ONNX:-}"
  RESTORE_PTH="${RESTORE_PTH:-}"

  if [[ -n "$RESTORE_ONNX" ]]; then
    [[ -f "$RESTORE_ONNX" ]] || { log "FATAL: RESTORE_ONNX not found: $RESTORE_ONNX"; exit 1; }
    if [[ -z "$RESTORE_PTH" ]]; then
      RESTORE_PTH="$(pick_pth_for_onnx "$RESTORE_ONNX")"
    fi
    log "from RESTORE_ONNX"
    return 0
  fi

  if [[ -n "${1:-}" ]]; then
    RESTORE_ONNX="$1"
    [[ -f "$RESTORE_ONNX" ]] || { log "FATAL: file not found: $RESTORE_ONNX"; exit 1; }
    RESTORE_PTH="$(pick_pth_for_onnx "$RESTORE_ONNX")"
    log "from positional argument"
    return 0
  fi

  if [[ "${RESTORE_FROM_BEST_EVAL:-0}" == "1" ]] && [[ -f "$OUT/BEST_EVAL.txt" ]]; then
    RESTORE_ONNX=$(grep '^we_onnx=' "$OUT/BEST_EVAL.txt" | head -1 | sed 's/^we_onnx=//')
    RESTORE_PTH=$(grep '^we_pth=' "$OUT/BEST_EVAL.txt" | head -1 | sed 's/^we_pth=//')
    if [[ -n "$RESTORE_ONNX" ]] && [[ -f "$RESTORE_ONNX" ]]; then
      log "from BEST_EVAL (RESTORE_FROM_BEST_EVAL=1): $OUT/BEST_EVAL.txt"
      return 0
    fi
    log "WARN: RESTORE_FROM_BEST_EVAL=1 but BEST_EVAL paths missing; falling through"
  fi

  if [[ -f "$BASELINE_ONNX" ]]; then
    RESTORE_ONNX="$BASELINE_ONNX"
    if [[ -f "$BASELINE_PTH" ]]; then
      RESTORE_PTH="$BASELINE_PTH"
    else
      RESTORE_PTH="$(pick_pth_for_onnx "$RESTORE_ONNX")"
    fi
    log "from BASELINE_ONNX (target acc=${BASELINE_ACC}): $RESTORE_ONNX"
    return 0
  fi

  RESTORE_ONNX="$(pick_max_acc_optimal_onnx "$BACK")"
  if [[ -n "$RESTORE_ONNX" ]] && [[ -f "$RESTORE_ONNX" ]]; then
    RESTORE_PTH="$(pick_pth_for_onnx "$RESTORE_ONNX")"
    log "picked max-acc backup under $BACK: $RESTORE_ONNX"
    return 0
  fi

  log "FATAL: no OPTIMAL onnx under $BACK (and no BASELINE_ONNX at $BASELINE_ONNX)"
  exit 1
}

resolve_sources "${1:-}"

[[ -f "$RESTORE_PTH" ]] || log "WARN: pth missing (continue onnx-only): $RESTORE_PTH"

log "restore ONNX: $RESTORE_ONNX -> $PLATEX_MODELS/plate_rec_color.onnx"
cp -a "$RESTORE_ONNX" "$PLATEX_MODELS/plate_rec_color.onnx"

if [[ -f "$RESTORE_PTH" ]]; then
  mkdir -p "$CKPT"
  log "restore CKPT: $RESTORE_PTH -> $CKPT/best.pth"
  cp -a "$RESTORE_PTH" "$CKPT/best.pth"
fi

if [[ -n "$PLATEX_CID" ]]; then
  log "docker restart $PLATEX_CID"
  docker restart "$PLATEX_CID"
else
  log "WARN: PLATEX_CID empty; restart platex manually"
fi

log "done"
