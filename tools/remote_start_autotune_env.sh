#!/usr/bin/env bash
# Server-side: append log header and start iterative training (no secrets in repo).
set -euo pipefail
LOG=/opt/vscc/plateai/output/autotune_run.log
{
  echo ""
  echo "=== autotune resume platex v1.0.4 $(date -Iseconds) PLATEAU_ROUNDS=5 MIN_GAIN=0.001 MAX_ROUNDS=14 ==="
} >>"$LOG"
export PLATEAU_ROUNDS=5
export MIN_GAIN=0.001
export MAX_ROUNDS=14
export PLATEAI_IMAGE=ghcr.io/vesaaa/plateai:v1.0.3-cpu
# Caller must: export NOTIFY_WEBHOOK_URL='https://...' (optional)
nohup bash /opt/vscc/plateai/tools/iter_train_platex_loop.sh >>"$LOG" 2>&1 &
echo "started pid=$!"
