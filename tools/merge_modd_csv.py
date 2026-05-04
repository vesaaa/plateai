#!/usr/bin/env python3
"""Merge plate label CSVs (label, path) with dedupe by image path; optional max rows."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_rows(path: Path) -> list[tuple[str, str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    rows: list[list[str]] = []
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                rows = [r for r in csv.reader(f) if r and len(r) >= 2]
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Cannot decode {path}")

    if rows and rows[0][0].strip().lower() in {"test1", "plate", "车牌", "label"}:
        rows = rows[1:]

    out: list[tuple[str, str]] = []
    for r in rows:
        label, src = r[0].strip(), r[1].strip()
        if not label or not src:
            continue
        out.append((label, src))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_path: dict[str, str] = {}
    for p in args.inputs:
        for label, src in read_rows(p):
            by_path[src] = label

    items: list[tuple[str, str]] = list(by_path.items())  # (path, label)
    rng = random.Random(args.seed)
    rng.shuffle(items)
    if args.max_rows and len(items) > args.max_rows:
        items = items[: args.max_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plate", "path"])
        for path, label in sorted(items, key=lambda x: x[0]):
            w.writerow([label, path])

    print(f"Wrote {len(items)} rows to {args.output}")


if __name__ == "__main__":
    main()
