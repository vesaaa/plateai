#!/usr/bin/env bash
# One-shot targeted loop:
# 1) diagnose err CSV -> JSONL
# 2) build confusion-focused hard subset CSV
# 3) mix with positive pool and train
# 4) optional gate benchmarks (2000 + 20w), fail if any drops
set -euo pipefail

DATA="${DATA:-/opt/vscc/plateai/data}"
CACHE="${CACHE:-/opt/vscc/plateai/cache}"
CKPT="${CKPT:-/opt/vscc/plateai/checkpoints}"
OUT="${OUT:-/opt/vscc/plateai/output}"
TOOLS="${TOOLS:-/opt/vscc/plateai/tools}"
PREFIX="${URL_PREFIX:-https://huizhoupark.obs.cn-south-1.myhuaweicloud.com}"

ERR_CSV="${ERR_CSV:-$DATA/20w_err_03.csv}"
POOL_CSV="${POOL_CSV:-$DATA/2000原图.csv}"

DIAG_JSONL="${DIAG_JSONL:-$OUT/diag_confusion.jsonl}"
HARD_CSV="${HARD_CSV:-$DATA/train_hard_confusion.csv}"
MIX_OUT="${MIX_OUT:-$DATA/train_mix_confusion.csv}"

DIAG_SAMPLE="${DIAG_SAMPLE:-0}" # 0 = all
DIAG_WORKERS="${DIAG_WORKERS:-10}"
DIAG_TIMEOUT="${DIAG_TIMEOUT:-120}"
API="${API:-http://127.0.0.1:8080/api/v1/recognize}"
MODE="${MODE:-full}"

PAIRS="${PAIRS:-D:0,0:D,F:E,E:F,B:8,8:B,S:5,5:S}"
PAIR_CATEGORIES="${PAIR_CATEGORIES:-char_1_2,len_mismatch}"
MAX_PER_PAIR="${MAX_PER_PAIR:-1200}"
HARD_MAX_TOTAL="${HARD_MAX_TOTAL:-8000}"
SEED="${SEED:-42}"

MIX_MAX_ROWS="${MIX_MAX_ROWS:-12000}"
POS_RATIO="${POS_RATIO:-0.55}"

GATE_2000_CSV="${GATE_2000_CSV:-$DATA/2000原图.csv}"
GATE_20W_CSV="${GATE_20W_CSV:-$DATA/20w.csv}"
RUN_GATE="${RUN_GATE:-1}"
GATE_WORKERS="${GATE_WORKERS:-8}"
GATE_TIMEOUT="${GATE_TIMEOUT:-45}"
BASE_2000="${BASE_2000:-0.9565}"
BASE_20W="${BASE_20W:-0.9120}"
EPS="${EPS:-0.0001}"

mkdir -p "$CACHE" "$CKPT" "$OUT"
[[ -f "$ERR_CSV" ]] || { echo "missing ERR_CSV=$ERR_CSV"; exit 1; }
[[ -f "$POOL_CSV" ]] || { echo "missing POOL_CSV=$POOL_CSV"; exit 1; }

diag_sample_args=()
if [[ "$DIAG_SAMPLE" != "0" ]]; then
  diag_sample_args=(--sample "$DIAG_SAMPLE")
fi

echo "[1/4] diagnose: $ERR_CSV -> $DIAG_JSONL"
python3 "$TOOLS/diagnose_errors.py" \
  --err-csv "$ERR_CSV" \
  --api "$API" \
  --mode "$MODE" \
  --workers "$DIAG_WORKERS" \
  --timeout "$DIAG_TIMEOUT" \
  --seed "$SEED" \
  "${diag_sample_args[@]}" \
  --out-jsonl "$DIAG_JSONL"

echo "[2/4] build hard confusion subset -> $HARD_CSV"
python3 "$TOOLS/build_confusion_subset.py" \
  --diag-jsonl "$DIAG_JSONL" \
  --output "$HARD_CSV" \
  --pairs "$PAIRS" \
  --categories "$PAIR_CATEGORIES" \
  --max-per-pair "$MAX_PER_PAIR" \
  --max-total "$HARD_MAX_TOTAL" \
  --seed "$SEED"

echo "[3/4] train mix (hard subset + positives)"
ERR_CSV="$HARD_CSV" \
POOL_CSV="$POOL_CSV" \
MIX_OUT="$MIX_OUT" \
MIX_MAX_ROWS="$MIX_MAX_ROWS" \
POS_RATIO="$POS_RATIO" \
SEED="$SEED" \
DATA="$DATA" CACHE="$CACHE" CKPT="$CKPT" OUT="$OUT" TOOLS="$TOOLS" URL_PREFIX="$PREFIX" \
bash "$TOOLS/train_nev_err_mix.sh"

if [[ "$RUN_GATE" != "1" ]]; then
  echo "[4/4] gate skipped (RUN_GATE=$RUN_GATE)"
  exit 0
fi

echo "[4/4] gate benchmark on 2000 + 20w"
[[ -f "$GATE_2000_CSV" ]] || { echo "missing GATE_2000_CSV=$GATE_2000_CSV"; exit 1; }
[[ -f "$GATE_20W_CSV" ]] || { echo "missing GATE_20W_CSV=$GATE_20W_CSV"; exit 1; }

run_bench() {
  local csv="$1"
  python3 "$TOOLS/bench_platex_csv.py" \
    --csv "$csv" \
    --url-prefix "$PREFIX" \
    --api "$API" \
    --mode full \
    --workers "$GATE_WORKERS" \
    --timeout "$GATE_TIMEOUT"
}

acc2000="$(run_bench "$GATE_2000_CSV" | awk -F' ' '/RESULT/{for(i=1;i<=NF;i++) if($i ~ /^acc=/){split($i,a,"="); print a[2]}}' | tail -n1)"
acc20w="$(run_bench "$GATE_20W_CSV" | awk -F' ' '/RESULT/{for(i=1;i<=NF;i++) if($i ~ /^acc=/){split($i,a,"="); print a[2]}}' | tail -n1)"

echo "gate_result acc2000=$acc2000 baseline2000=$BASE_2000 acc20w=$acc20w baseline20w=$BASE_20W eps=$EPS"

python3 - "$acc2000" "$BASE_2000" "$acc20w" "$BASE_20W" "$EPS" <<'PY'
import sys
acc2000, base2000, acc20w, base20w, eps = map(float, sys.argv[1:])
ok2000 = acc2000 + eps >= base2000
ok20w = acc20w + eps >= base20w
if ok2000 and ok20w:
    print("GATE PASS")
    raise SystemExit(0)
print("GATE FAIL", {"ok2000": ok2000, "ok20w": ok20w})
raise SystemExit(2)
PY
