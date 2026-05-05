#!/usr/bin/env bash
# Backwards-compatible thin wrapper: same env as before, then start_autotune.sh in background.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${AUTO_LOG:-/opt/vscc/plateai/output/autotune_run.log}"
{
  echo ""
  echo "=== remote_start_autotune_env $(date -Iseconds) (delegates to start_autotune.sh) ==="
} >>"$LOG"
export BENCH_MODE="${BENCH_MODE:-full}"
export BENCH_TIMEOUT="${BENCH_TIMEOUT:-120}"
export PLATEAU_ROUNDS="${PLATEAU_ROUNDS:-5}"
export MIN_GAIN="${MIN_GAIN:-0.001}"
export MAX_ROUNDS="${MAX_ROUNDS:-14}"
export PLATEAI_IMAGE="${PLATEAI_IMAGE:-ghcr.io/vesaaa/plateai:v1.0.3-cpu}"
export AUTO_LOG="$LOG"
export AUTO_BACKGROUND=1
export AUTO_APPEND_HEADER=0
bash "$HERE/start_autotune.sh"
