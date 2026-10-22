"""Decode + resize labeled plate crops into uint8 arrays for CRNN training.

Usage:
    python -m plateai.crnn.prep --csv plate_20w.csv --img-dir img/ --out arr32x160/

The CSV is ``label,path`` (same layout as the rest of plateai; a single quoted
``"label,url"`` cell is also accepted). Paths may be absolute URLs or
``/relative/paths``; either way the URL host is stripped and the file is read
from ``<img-dir>/<path>`` (see ``tools/fetch_images.py`` to mirror them).

The split is by plate number: plates whose ``md5(label) % 10 == 0`` go to the
test set, so the same car never appears in both train and test.
Outputs ``<out>/{train,test}_{x.npy,labels.txt,paths.txt}``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from multiprocessing import Pool

import cv2
import numpy as np

from plateai.crnn.model import CHARS, INPUT_H, INPUT_W


def read_rows(path: str):
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if len(r) == 1 and "," in r[0]:
                r = r[0].split(",", 1)
            if len(r) >= 2:
                yield r[0].strip(), re.sub(r"^https?://[^/]+", "", r[1].strip())


def _load(args):
    img_dir, path = args
    full = os.path.join(img_dir, path.lstrip("/"))
    if not os.path.exists(full):
        return None
    arr = np.fromfile(full, dtype=np.uint8)
    if arr.size < 32:
        return None
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)


def is_test_plate(label: str, test_mod: int = 10) -> bool:
    return int(hashlib.md5(label.encode()).hexdigest(), 16) % test_mod == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    data = [(l, p) for l, p in read_rows(args.csv) if 5 <= len(l) <= 10 and all(c in CHARS for c in l)]
    print("usable rows", len(data))
    with Pool(args.workers) as pool:
        imgs = pool.map(_load, [(args.img_dir, p) for _, p in data], chunksize=256)

    split = {"train": [], "test": []}
    for (label, path), img in zip(data, imgs):
        if img is not None:
            split["test" if is_test_plate(label) else "train"].append((label, path, img))
    for name, items in split.items():
        np.save(f"{args.out}/{name}_x.npy", np.stack([i[2] for i in items]))
        with open(f"{args.out}/{name}_labels.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(i[0] for i in items) + "\n")
        with open(f"{args.out}/{name}_paths.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(i[1] for i in items) + "\n")
        print(name, len(items))


if __name__ == "__main__":
    main()
