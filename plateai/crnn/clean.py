"""Build a label-cleaned copy of a prep'd array dir.

Usage:
    python -m plateai.crnn.clean --data arr32x160/ --scored train_scored.jsonl \
        --trusted plate_mini_10w.csv --out arr32x160_clean/

Rules for training rows (the test split is linked unchanged):
- rows listed in ``--trusted`` (a human-cleaned CSV) are always kept;
- untrusted rows from a date folder (``/YYYYMMDD...``) that the trusted CSV
  mostly covers (``--reviewed-share``, default 80% of that day's rows) are
  dropped: the human pass looked at that day and left them out. Days only
  partly covered (e.g. the trusted export ended mid-day) are not treated
  as reviewed;
- other untrusted rows are dropped when the model's prediction differs from
  the label by an edit distance >= 2 (``--max-edit``). On the cleaned 10w set,
  88% of genuine model errors are a single-char edit, while a label that
  belongs to another car usually differs in several chars; single-char
  mismatches are kept as hard examples.

Reference (2026-10-22): 179,805 -> 172,557 train rows (5,201 dropped by the
date rule, 2,047 by the edit rule).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np

from plateai.crnn.prep import read_rows

_DATE = re.compile(r"/(20\d{6})")  # /YYYYMMDD/ or /YYYYMMDDhhmmss/


def edit_distance(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def _date(path: str) -> str | None:
    m = _DATE.search(path)
    return m.group(1) if m else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="prep output dir")
    ap.add_argument("--scored", required=True, help="plateai.crnn.score output for the train split")
    ap.add_argument("--trusted", default=None, help="human-cleaned CSV (label,path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-edit", type=int, default=1, help="keep untrusted mismatches up to this edit distance")
    ap.add_argument("--reviewed-share", type=float, default=0.8,
                    help="a date counts as human-reviewed when the trusted CSV covers this share of its rows")
    args = ap.parse_args(argv)

    trusted = {p for _, p in read_rows(args.trusted)} if args.trusted else set()
    with open(args.scored, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    per_date = Counter(_date(r["path"]) for r in rows)
    trusted_per_date = Counter(_date(r["path"]) for r in rows if r["path"] in trusted)
    reviewed = {d for d, n in trusted_per_date.items() if d and n >= args.reviewed_share * per_date[d]}
    keep, dropped = [], Counter()
    for i, r in enumerate(rows):
        if r["path"] in trusted:
            keep.append(i)
        elif _date(r["path"]) in reviewed:
            dropped["reviewed_date_not_trusted"] += 1
        elif r["pred"] != r["label"] and edit_distance(r["pred"], r["label"]) > args.max_edit:
            dropped["edit_distance"] += 1
        else:
            keep.append(i)
    print(f"train kept {len(keep)} of {len(rows)}; dropped {dict(dropped)}; reviewed dates {sorted(reviewed)}")

    os.makedirs(args.out, exist_ok=True)
    x = np.load(f"{args.data}/train_x.npy", mmap_mode="r")
    np.save(f"{args.out}/train_x.npy", x[np.array(keep)])
    for kind in ("labels", "paths"):
        with open(f"{args.data}/train_{kind}.txt", encoding="utf-8") as f:
            lines = f.read().split("\n")
        with open(f"{args.out}/train_{kind}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines[i] for i in keep) + "\n")
    for name in ("test_x.npy", "test_labels.txt", "test_paths.txt"):
        dst = f"{args.out}/{name}"
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f"{args.data}/{name}"), dst)


if __name__ == "__main__":
    main()
