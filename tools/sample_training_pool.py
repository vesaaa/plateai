#!/usr/bin/env python3
"""Build a smaller high-quality pool CSV from 10万/20万-scale lists for mixed training.

Filters: mainland plate length 7–8, valid charset, optional province-prefix stratification
(round-robin across first-character buckets so one province does not dominate).

Output columns match build_train_mix.py / plateai train: plate, path
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plateai.alphabets import PLATE_CHR  # noqa: E402

# Province / special leading chars (same order prefix region as PLATE_CHR[1:35])
_PROV_CHARS = set(PLATE_CHR[1:35])


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


def valid_plate_label(lab: str) -> bool:
    s = lab.strip()
    if len(s) not in (7, 8):
        return False
    if s[0] not in _PROV_CHARS:
        return False
    return all(c in PLATE_CHR for c in s)


def dedupe_paths(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for lab, p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append((lab, p))
    return out


def stratified_round_robin(
    buckets: dict[str, list[tuple[str, str]]], max_total: int, rng: random.Random
) -> list[tuple[str, str]]:
    keys = sorted(buckets.keys())
    queues: dict[str, list[tuple[str, str]]] = {
        k: rng.sample(v, len(v)) for k, v in buckets.items() if v
    }
    out: list[tuple[str, str]] = []
    while len(out) < max_total:
        progressed = False
        for k in keys:
            if len(out) >= max_total:
                break
            if queues.get(k):
                out.append(queues[k].pop())
                progressed = True
        if not progressed:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Large CSV (label, path).")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-total", type=int, default=50_000, help="Cap rows after filter/sample.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stratify-prefix",
        action="store_true",
        help="Round-robin sample across first label char (省份 bucket).",
    )
    ap.add_argument(
        "--no-dedupe-path",
        action="store_true",
        help="Keep duplicate paths (not recommended).",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = read_pairs(args.input)
    valid = [(a, b) for a, b in raw if valid_plate_label(a)]
    if not args.no_dedupe_path:
        valid = dedupe_paths(valid)

    if len(valid) <= args.max_total:
        chosen = list(valid)
        rng.shuffle(chosen)
    elif args.stratify_prefix:
        buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for lab, p in valid:
            buckets[lab.strip()[0]].append((lab, p))
        chosen = stratified_round_robin(dict(buckets), args.max_total, rng)
        chosen_paths = {p for _, p in chosen}
        if len(chosen) < args.max_total:
            rest = [(a, b) for a, b in valid if b not in chosen_paths]
            rng.shuffle(rest)
            for item in rest:
                if len(chosen) >= args.max_total:
                    break
                chosen.append(item)
        rng.shuffle(chosen)
    else:
        rng.shuffle(valid)
        chosen = valid[: args.max_total]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plate", "path"])
        for lab, path in chosen:
            w.writerow([lab, path])

    print(
        f"pooled {len(chosen)} rows from {len(raw)} raw ({len(valid)} after filter) -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
