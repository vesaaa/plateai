#!/usr/bin/env bash
# Restore WE weights (plate_rec_color.onnx + checkpoints/best.pth) from backups or BEST_EVAL.txt, then restart platex.
#
# Usage:
#   ./restore_optimal_we.sh                    # use OUT/BEST_EVAL.txt or latest OPTIMAL_* in BACK
#   RESTORE_ONNX=/path/a.onnx RESTORE_PTH=/path/b.pth ./restore_optimal_we.sh
#   ./restore_optimal_we.sh /opt/vscc/plateai/backups/OPTIMAL_plate_rec_color_acc_0_926000.onnx
#
set -euo pipefail

BACK="${BACK:-/opt/vscc/plateai/backups}"
OUT="${OUT:-/opt/vscc/plateai/output}"
CKPT="${CKPT:-/opt/vscc/plateai/checkpoints}"
PLATEX_MODELS="${PLATEX_MODELS:-/opt/vscc/platex/models}"
PLATEX_CID="${PLATEX_CID:-$(docker ps --filter publish=8080 --format '{{.ID}}' | head -1)}"

log() { echo "[$(date -Iseconds)] $*"; }

pick_latest_optimal_onnx() {
  local d="$1"
  shopt -s nullglob
  local -a files=("$d"/OPTIMAL_plate_rec_color_acc_*.onnx)
  shopt -u nullglob
  if ((${#files[@]} == 0)); then
    echo ""
    return 0
  fi
  ls -t "${files[@]}" | head -1
}

pick_pth_for_onnx() {
  # OPTIMAL_plate_rec_color_acc_X.onnx -> OPTIMAL_best_we_acc_X.pth (same suffix pattern)
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
    return 0
  fi

  if [[ -n "${1:-}" ]]; then
    RESTORE_ONNX="$1"
    [[ -f "$RESTORE_ONNX" ]] || { log "FATAL: file not found: $RESTORE_ONNX"; exit 1; }
    RESTORE_PTH="$(pick_pth_for_onnx "$RESTORE_ONNX")"
    return 0
  fi

  if [[ -f "$OUT/BEST_EVAL.txt" ]]; then
    RESTORE_ONNX=$(grep '^we_onnx=' "$OUT/BEST_EVAL.txt" | head -1 | sed 's/^we_onnx=//')
    RESTORE_PTH=$(grep '^we_pth=' "$OUT/BEST_EVAL.txt" | head -1 | sed 's/^we_pth=//')
    if [[ -n "$RESTORE_ONNX" ]] && [[ -f "$RESTORE_ONNX" ]]; then
      log "from BEST_EVAL: $OUT/BEST_EVAL.txt"
      return 0
    fi
  fi

  RESTORE_ONNX="$(pick_latest_optimal_onnx "$BACK")"
  if [[ -z "$RESTORE_ONNX" ]] || [[ ! -f "$RESTORE_ONNX" ]]; then
    log "FATAL: no OPTIMAL onnx in $BACK and no valid BEST_EVAL"
    exit 1
  fi
  RESTORE_PTH="$(pick_pth_for_onnx "$RESTORE_ONNX")"
  log "picked latest backup: $RESTORE_ONNX"
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
