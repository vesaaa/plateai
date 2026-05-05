#!/usr/bin/env bash
# Download latest plateai tools/*.sh and tools/*.py from GitHub main (no git required).
# Usage: PLATEAI_HOME=/opt/vscc/plateai bash sync_tools.sh
set -euo pipefail

PLATEAI_HOME="${PLATEAI_HOME:-/opt/vscc/plateai}"
ARCHIVE_URL="${PLATEAI_ARCHIVE_URL:-https://github.com/vesaaa/plateai/archive/refs/heads/main.tar.gz}"
DEST="$PLATEAI_HOME/tools"

command -v curl >/dev/null 2>&1 || {
  echo "FATAL: curl required" >&2
  exit 1
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "fetch $ARCHIVE_URL"
curl -fsSL "$ARCHIVE_URL" -o "$TMP/t.tar.gz"
tar -xzf "$TMP/t.tar.gz" -C "$TMP"
SRC=$(find "$TMP" -maxdepth 1 -type d -name "plateai-*" | head -1)
if [[ -z "$SRC" ]] || [[ ! -d "$SRC/tools" ]]; then
  echo "FATAL: unexpected archive layout under $TMP" >&2
  exit 1
fi

install -d -m 0755 "$DEST"
shopt -s nullglob
for f in "$SRC/tools"/*.sh "$SRC/tools"/*.py; do
  [[ -e "$f" ]] || continue
  install -m 0755 "$f" "$DEST/"
  echo "installed $(basename "$f")"
done
shopt -u nullglob

echo "OK: tools synced to $DEST"
