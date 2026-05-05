#!/usr/bin/env bash
# Start iterative platex benchmark -> mixed train -> docker train -> deploy WE (see iter_train_platex_loop.sh).
# Optional: copy autotune.env.example -> autotune.env next to this script and edit.
#
# Usage:
#   ./start_autotune.sh                    # foreground (Ctrl+C stops)
#   AUTO_BACKGROUND=1 ./start_autotune.sh   # nohup to $AUTO_LOG
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATEAI_HOME="${PLATEAI_HOME:-/opt/vscc/plateai}"
TOOLS="${TOOLS:-$SCRIPT_DIR}"
export TOOLS
export PLATEAI_HOME

ENV_FILE="${AUTO_ENV:-$TOOLS/autotune.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source <(sed 's/\r$//' "$ENV_FILE")
  set +a
fi

# Defaults aligned with previous remote_start_autotune_env.sh
export BENCH_MODE="${BENCH_MODE:-full}"
export BENCH_TIMEOUT="${BENCH_TIMEOUT:-120}"
export PLATEAU_ROUNDS="${PLATEAU_ROUNDS:-5}"
export MIN_GAIN="${MIN_GAIN:-0.001}"
export MAX_ROUNDS="${MAX_ROUNDS:-14}"
export PLATEAI_IMAGE="${PLATEAI_IMAGE:-ghcr.io/vesaaa/plateai:v1.0.3-cpu}"

# Large-pool defaults when file exists
export DATA="${DATA:-$PLATEAI_HOME/data}"
if [[ -z "${POOL_CSV:-}" ]] && [[ -f "$DATA/pool_sampled_50k.csv" ]]; then
  export POOL_CSV="$DATA/pool_sampled_50k.csv"
fi
export MIX_MAX_ROWS="${MIX_MAX_ROWS:-25000}"
export TRAIN_MAX_ROWS="${TRAIN_MAX_ROWS:-20000}"

export OUT="${OUT:-$PLATEAI_HOME/output}"
export CKPT="${CKPT:-$PLATEAI_HOME/checkpoints}"
export CACHE="${CACHE:-$PLATEAI_HOME/cache}"
export BACK="${BACK:-$PLATEAI_HOME/backups}"

AUTO_LOG="${AUTO_LOG:-$OUT/autotune_run.log}"
mkdir -p "$OUT" "$CKPT" "$CACHE" "$BACK"

LOOP="$TOOLS/iter_train_platex_loop.sh"
if [[ ! -f "$LOOP" ]]; then
  echo "FATAL: missing $LOOP (run sync_tools.sh first?)" >&2
  exit 1
fi

BLCHK="$TOOLS/check_baseline_files.sh"
if [[ "${AUTO_CHECK_BASELINE:-1}" == "1" ]] && [[ -f "$BLCHK" ]]; then
  bash "$BLCHK"
fi

appended=false
if [[ "${AUTO_APPEND_HEADER:-1}" == "1" ]]; then
  {
    echo ""
    echo "=== autotune $(date -Iseconds) BENCH_MODE=$BENCH_MODE POOL_CSV=${POOL_CSV:-default} MIX_MAX_ROWS=${MIX_MAX_ROWS} TRAIN_MAX_ROWS=${TRAIN_MAX_ROWS} MAX_ROUNDS=$MAX_ROUNDS ==="
  } >>"$AUTO_LOG"
  appended=true
fi

if [[ "${AUTO_BACKGROUND:-0}" == "1" ]]; then
  nohup bash "$LOOP" >>"$AUTO_LOG" 2>&1 &
  echo "started autotune pid=$! log=$AUTO_LOG"
  exit 0
fi

bash "$LOOP" 2>&1 | tee -a "$AUTO_LOG"
