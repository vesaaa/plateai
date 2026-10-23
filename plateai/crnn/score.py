"""Score a split with a trained PlateCRNN (for label cleaning / error analysis).

Usage:
    python -m plateai.crnn.score --run runs/crnn_s --data arr32x160/ --split train --out train_scored.jsonl

Each output row: ``{"label", "pred", "conf", "path"}`` where conf is the
minimum per-character probability.
"""
from __future__ import annotations

import argparse
import json

import torch

from plateai.crnn.model import DEFAULT_CHANNELS, PlateCRNN, decode, parse_channels
from plateai.crnn.train import load_split, to_tensor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", default=",".join(map(str, DEFAULT_CHANNELS)))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    model = PlateCRNN(c=parse_channels(args.channels))
    model.load_state_dict(torch.load(f"{args.run}/best.pt", map_location="cpu"))
    model.eval()
    x, y = load_split(args.data, args.split)
    with open(f"{args.data}/{args.split}_paths.txt", encoding="utf-8") as f:
        paths = f.read().split("\n")[: len(x)]
    with open(args.out, "w", encoding="utf-8") as f, torch.no_grad():
        for i in range(0, len(x), 1024):
            preds = decode(model(to_tensor(x[i:i + 1024])))
            for (pred, conf), label, path in zip(preds, y[i:i + 1024], paths[i:i + 1024]):
                f.write(json.dumps({"label": label, "pred": pred, "conf": round(conf, 4), "path": path}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
