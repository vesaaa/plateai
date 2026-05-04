#!/usr/bin/env python3
"""Build a training CSV: hard negatives (err) + random correct samples from a pool (no path overlap)."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_pairs(path: Path) -> list[tuple[str, str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    rows: list[list[str]] = []
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                rows = [r for r in csv.reader(f) if r and len(r) >= 2]
            break
        except UnicodeDecodeError:
            continue
    if rows and rows[0][0].strip().lower() in {"test1", "plate", "车牌", "label"}:
        rows = rows[1:]
    out: list[tuple[str, str]] = []
    for r in rows:
        a, b = r[0].strip(), r[1].strip()
        if a and b:
            out.append((a, b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--err", type=Path, required=True, help="bench miss CSV (label, path)")
    ap.add_argument("--pool", type=Path, required=True, help="positive pool e.g. 2000原图")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pos-ratio", type=float, default=0.55, help="target fraction of positives in output")
    ap.add_argument("--max-rows", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    err_pairs = read_pairs(args.err)
    pool_pairs = read_pairs(args.pool)
    err_paths = {p for _, p in err_pairs}
    pos_candidates = [(l, p) for l, p in pool_pairs if p not in err_paths]

    rng = random.Random(args.seed)
    n_err = len(err_pairs)
    # Solve for n_pos / (n_err + n_pos) ≈ pos_ratio  -> n_pos ≈ pos_ratio * n_err / (1-pos_ratio)
    target_pos = int(round(args.pos_ratio * max(1, n_err) / max(1e-6, 1.0 - args.pos_ratio)))
    target_pos = min(target_pos, len(pos_candidates), max(0, args.max_rows - n_err))
    rng.shuffle(pos_candidates)
    chosen_pos = pos_candidates[:target_pos]

    merged: dict[str, str] = {}
    for lab, path in chosen_pos:
        merged[path] = lab
    for lab, path in err_pairs:
        merged[path] = lab

    items = list(merged.items())
    rng.shuffle(items)
    if len(items) > args.max_rows:
        items = items[: args.max_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plate", "path"])
        for path, lab in sorted(items, key=lambda x: x[0]):
            w.writerow([lab, path])

    print(
        f"wrote {len(items)} rows (errs={n_err}, positives_included={len(chosen_pos)}) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
