#!/usr/bin/env bash
# Iterative WE tuning: benchmark platex (default mode=full) -> mixed train CSV -> docker plateai train -> deploy ONNX.
# No platex heuristic edits — weights only.

set -euo pipefail

DATA="${DATA:-/opt/vscc/plateai/data}"
OUT="${OUT:-/opt/vscc/plateai/output}"
CKPT="${CKPT:-/opt/vscc/plateai/checkpoints}"
CACHE="${CACHE:-/opt/vscc/plateai/cache}"
BACK="${BACK:-/opt/vscc/plateai/backups}"
TOOLS="${TOOLS:-/opt/vscc/plateai/tools}"
PLATEX_MODELS="${PLATEX_MODELS:-/opt/vscc/platex/models}"

PREFIX="${URL_PREFIX:-https://huizhoupark.obs.cn-south-1.myhuaweicloud.com}"
API="${PLATEX_API:-http://127.0.0.1:8080/api/v1/recognize}"
IMAGE="${PLATEAI_IMAGE:-ghcr.io/vesaaa/plateai:v1.0.3-cpu}"

BENCH_MODE="${BENCH_MODE:-full}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-120}"

BENCH_CSV="${BENCH_CSV:-$DATA/2000原图.csv}"
POOL_CSV="${POOL_CSV:-$DATA/2000原图.csv}"
INIT_PTH="${INIT_PTH:-$DATA/best_10w_01.pth}"

MAX_ROUNDS="${MAX_ROUNDS:-10}"
TARGET_ACC="${TARGET_ACC:-0.97}"
PLATEAU_ROUNDS="${PLATEAU_ROUNDS:-3}"
MIN_GAIN="${MIN_GAIN:-0.002}"

PLATEX_CID="${PLATEX_CID:-$(docker ps --filter publish=8080 --format '{{.ID}}' | head -1)}"

mkdir -p "$OUT" "$CKPT" "$CACHE" "$BACK" "$TOOLS"
printf '%s\n' "$BENCH_MODE" >"$OUT/.bench_mode_env.txt"

log() { echo "[$(date -Iseconds)] $*"; }

notify_async() {
  local body="$1" title="${2:-platex 自动训练}"
  [[ -n "${NOTIFY_WEBHOOK_URL:-}" ]] || return 0
  [[ -f "$TOOLS/notify_webhook.py" ]] || return 0
  nohup env NOTIFY_WEBHOOK_URL="$NOTIFY_WEBHOOK_URL" python3 "$TOOLS/notify_webhook.py" "$body" "$title" \
    </dev/null >/dev/null 2>&1 &
}

parse_result_acc() {
  grep '^RESULT acc=' | tail -1 | sed -n 's/^RESULT acc=\([0-9.]*\).*/\1/p'
}

save_optimal() {
  local acc="$1" rnd="$2"
  python3 - "$acc" "$rnd" "$BACK" "$OUT" "$PLATEX_MODELS" "$CKPT" <<'PY'
import pathlib, shutil, sys
acc, rnd, back, out, models, ckpt = sys.argv[1:7]
suffix = acc.replace(".", "_")
b = pathlib.Path(back)
onnx_dst = b / f"OPTIMAL_plate_rec_color_acc_{suffix}.onnx"
pth_dst = b / f"OPTIMAL_best_we_acc_{suffix}.pth"
shutil.copy2(pathlib.Path(models) / "plate_rec_color.onnx", onnx_dst)
p = pathlib.Path(ckpt) / "best.pth"
if p.exists():
    shutil.copy2(p, pth_dst)
bench_mode = (pathlib.Path(out) / ".bench_mode_env.txt")
bm = bench_mode.read_text(encoding="utf-8").strip() if bench_mode.exists() else "unknown"
(pathlib.Path(out) / "BEST_EVAL.txt").write_text(
    f"round={rnd}\nacc={acc}\nbench_mode={bm}\nwe_onnx={onnx_dst}\nwe_pth={pth_dst}\n",
    encoding="utf-8",
)
print("OPTIMAL saved", onnx_dst, pth_dst)
PY
}

best_acc=""
stall=0
last_acc=""

for round in $(seq 1 "$MAX_ROUNDS"); do
  log "==== ROUND $round: benchmark ===="
  ERR_CSV="$OUT/bench_err_r${round}.csv"
  REP="$OUT/bench_report_r${round}.jsonl"
  BENCH_LOG="$OUT/bench_stdout_r${round}.log"

  log "bench mode=$BENCH_MODE timeout=${BENCH_TIMEOUT}s"
  python3 "$TOOLS/bench_platex_csv.py" \
    --csv "$BENCH_CSV" --url-prefix "$PREFIX" --api "$API" \
    --mode "$BENCH_MODE" --workers 10 --timeout "$BENCH_TIMEOUT" \
    --out-err "$ERR_CSV" --out-report "$REP" \
    | tee "$BENCH_LOG"

  acc="$(parse_result_acc <"$BENCH_LOG")"
  if [[ -z "$acc" ]]; then
    log "FATAL: no RESULT acc in log"
    exit 1
  fi
  log "eval acc=$acc (target=$TARGET_ACC)"

  if python3 - <<PY
import sys
sys.exit(0 if float("$acc") >= float("$TARGET_ACC") else 1)
PY
  then
    log "target reached"
    save_optimal "$acc" "$round"
    notify_async "platex(${BENCH_MODE}) 已达 ${TARGET_ACC}: acc=$acc round=$round。见 backups/OPTIMAL_* 与 BEST_EVAL.txt。" "达到目标识别率"
    exit 0
  fi

  if [[ -z "$best_acc" ]] || python3 - <<PY
import sys
sys.exit(0 if float("$acc") > float("$best_acc") + 1e-9 else 1)
PY
  then
    best_acc="$acc"
    save_optimal "$acc" "$round"
    log "new best acc=$best_acc"
  fi

  if [[ -n "$last_acc" ]]; then
    if python3 - <<PY
import sys
gain = float("$acc") - float("$last_acc")
sys.exit(0 if gain < float("$MIN_GAIN") else 1)
PY
    then
      stall=$((stall + 1))
      log "low gain vs previous bench stall=$stall"
    else
      stall=0
    fi
  fi
  last_acc="$acc"

  if (( stall >= PLATEAU_ROUNDS )); then
    log "plateau: no gain >= ${MIN_GAIN} for ${PLATEAU_ROUNDS} rounds; best_acc=$best_acc"
    notify_async "多轮微调提升低于 ${MIN_GAIN}，已停止。history best_eval=$best_acc 当前=$acc。见 $OUT/BEST_EVAL.txt 与 backups/OPTIMAL_*。" "训练进入平台期"
    exit 0
  fi

  n_lines=$(wc -l <"$ERR_CSV")
  if (( n_lines > 1 )); then n_err=$((n_lines - 1)); else n_err=0; fi
  if (( n_err < 4 )); then
    log "too few errors ($n_err); stop (best_acc=$best_acc)"
    notify_async "错例过少(n_err=$n_err)已结束。best_eval=$best_acc。" "自动训练结束"
    exit 0
  fi

  TRAIN_HOST="$DATA/train_mix_current.csv"
  log "build mix -> $TRAIN_HOST"
  python3 "$TOOLS/build_train_mix.py" \
    --err "$ERR_CSV" --pool "$POOL_CSV" \
    --output "$TRAIN_HOST" --pos-ratio 0.55 --max-rows 9000 --seed "$((42 + round))"

  PT_IN="$INIT_PTH"
  if [[ -f "$CKPT/best.pth" ]]; then
    PT_IN="$CKPT/best.pth"
  fi

  log "docker train (pretrained=$PT_IN); cache host=$CACHE -> /workspace/cache (reuse downloads)"
  docker rm -f "plateai_autotune_${round}" 2>/dev/null || true
  docker run --name "plateai_autotune_${round}" \
    -e PLATEAI_DEVICE=cpu -e PLATEAI_WORKERS=4 -e PLATEAI_BATCH_SIZE=14 \
    -e PLATEAI_CACHE_DIR=/workspace/cache \
    -v "$DATA:/data:ro" -v "$CACHE:/workspace/cache" -v "$CKPT:/workspace/checkpoints" \
    -v "$OUT:/workspace/output" -v "$PT_IN:/workspace/init.pth:ro" \
    "$IMAGE" train \
      --csv /data/train_mix_current.csv \
      --pretrained /workspace/init.pth \
      --url-prefix "$PREFIX" \
      --output /workspace/output/plate_rec_color.onnx \
      --epochs 5 --batch-size 14 --workers 4 --lr 3.5e-4 --hard-case-repeat 2 \
      --val-ratio 0.1 --seed "$((1234 + round))"

  log "deploy WE onnx"
  cp -a "$PLATEX_MODELS/plate_rec_color.onnx" "$BACK/plate_rec_color.before_round_${round}.onnx"
  cp -a "$OUT/plate_rec_color.onnx" "$PLATEX_MODELS/plate_rec_color.onnx"

  if [[ -n "$PLATEX_CID" ]]; then
    log "restart platex $PLATEX_CID"
    docker restart "$PLATEX_CID"
    sleep 6
  else
    log "WARN: no platex container id (publish 8080); skip restart"
  fi
done

log "max rounds done best_acc=${best_acc:-unknown}"
notify_async "已达最大轮次 $MAX_ROUNDS。best_eval=${best_acc:-?} 见 backups/OPTIMAL_*。" "自动训练结束"
exit 0
