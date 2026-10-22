"""Export a trained PlateCRNN to ONNX and evaluate it on the test split.

Usage:
    python -m plateai.crnn.evaluate --run runs/crnn_s --data arr32x160/ \
        [--platex-bench test_full.jsonl]

Writes ``<run>/plate_rec_crnn.onnx`` and ``<run>/test_preds.jsonl``. When a
platex bench JSONL for the same test images is given (``tools/bench_platex_local.py``),
prints a side-by-side comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch

from plateai.crnn.model import CHARS, DEFAULT_CHANNELS, INPUT_H, INPUT_W, MEAN, STD, PlateCRNN, parse_channels


def export_onnx(model: PlateCRNN, path: str) -> None:
    model.eval()
    torch.onnx.export(
        model, torch.zeros(1, 3, INPUT_H, INPUT_W), path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def _prep(b: np.ndarray) -> np.ndarray:
    return ((b.transpose(0, 3, 1, 2).astype(np.float32) / 255.0) - MEAN) / STD


def _decode(logits: np.ndarray):
    e = np.exp(logits - logits.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    out = []
    for b in range(logits.shape[0]):
        s, ps, prev = [], [], 0
        for t, i in enumerate(p[b].argmax(-1)):
            if i and i != prev:
                s.append(CHARS[i - 1])
                ps.append(p[b, t, i])
            prev = i
        out.append(("".join(s), float(min(ps)) if ps else 0.0))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--channels", default=",".join(map(str, DEFAULT_CHANNELS)))
    ap.add_argument("--platex-bench", default=None)
    args = ap.parse_args(argv)

    model = PlateCRNN(c=parse_channels(args.channels))
    model.load_state_dict(torch.load(f"{args.run}/best.pt", map_location="cpu"))
    onnx_path = f"{args.run}/plate_rec_crnn.onnx"
    export_onnx(model, onnx_path)

    x = np.load(f"{args.data}/test_x.npy")
    with open(f"{args.data}/test_labels.txt", encoding="utf-8") as f:
        labels = f.read().split("\n")[: len(x)]
    with open(f"{args.data}/test_paths.txt", encoding="utf-8") as f:
        paths = f.read().split("\n")[: len(x)]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    preds = []
    for i in range(0, len(x), 256):
        preds += _decode(sess.run(None, {"input": _prep(x[i:i + 256])})[0])

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s1 = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    one = _prep(x[:1])
    for _ in range(20):
        s1.run(None, {"input": one})
    t = time.time()
    for _ in range(500):
        s1.run(None, {"input": one})
    latency = (time.time() - t) / 500 * 1000

    n = len(x)
    new_ok = [p == g for (p, _), g in zip(preds, labels)]
    print(f"test n={n}")
    print(f"crnn (onnx): {sum(new_ok) / n:.4f}  latency 1-thread={latency:.2f}ms  size={os.path.getsize(onnx_path) / 1e6:.2f}MB")
    for L in (7, 8):
        ids = [i for i in range(n) if len(labels[i]) == L]
        if ids:
            print(f"  len{L}: n={len(ids)} acc={sum(new_ok[i] for i in ids) / len(ids):.4f}")
    confs = np.array([c for _, c in preds])
    for th in (0.5, 0.7, 0.9):
        keep = confs >= th
        print(f"  conf>={th}: coverage={keep.mean():.4f} acc={np.mean(np.array(new_ok)[keep]):.4f}")

    base = {}
    if args.platex_bench:
        with open(args.platex_bench, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                base[r["path"]] = r
        base_ok = [base[p]["ok"] for p in paths]
        print(f"platex bench on same split: {sum(base_ok) / n:.4f}")
        print(f"  both right {sum(a and b for a, b in zip(new_ok, base_ok))}"
              f"  only crnn {sum(a and not b for a, b in zip(new_ok, base_ok))}"
              f"  only platex {sum(b and not a for a, b in zip(new_ok, base_ok))}"
              f"  both wrong {sum(not a and not b for a, b in zip(new_ok, base_ok))}")

    with open(f"{args.run}/test_preds.jsonl", "w", encoding="utf-8") as f:
        for path, label, (pred, conf) in zip(paths, labels, preds):
            row = {"label": label, "path": path, "pred": pred, "conf": conf, "ok": pred == label}
            if path in base:
                row["platex"] = base[path]["pred"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
