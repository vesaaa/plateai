"""Train PlateCRNN from scratch on arrays produced by ``plateai.crnn.prep``.

Usage:
    python -m plateai.crnn.train --data arr32x160/ --out runs/crnn_s --epochs 8 [--device cuda]

Reference run (2026-10-22, 12-core CPU, 8 threads, 8 epochs, ~2.5h):
test_acc 0.9351 on the 20,082-image plate-held-out split of the 20万 set.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from plateai.crnn.model import CHARS, DEFAULT_CHANNELS, MEAN, STD, PlateCRNN, decode, parse_channels


def to_tensor(xb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(xb)).permute(0, 3, 1, 2).float()
    return (t / 255.0 - MEAN) / STD


def augment(t: torch.Tensor) -> torch.Tensor:
    """Batch-level geometric + photometric jitter (runs on the training device)."""
    B, dev = t.size(0), t.device
    rnd = lambda *s: torch.rand(*s, device=dev)
    ang = (rnd(B) * 2 - 1) * math.radians(4)
    sc = 0.88 + rnd(B) * 0.14          # only zoom out, so no character gets cropped
    shx = (rnd(B) * 2 - 1) * 0.15
    tx = (rnd(B) * 2 - 1) * 0.04
    ty = (rnd(B) * 2 - 1) * 0.08
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    aspect = t.size(3) / t.size(2)      # rotation in normalized coords of a 5:1 image
    theta = torch.stack([torch.stack([cos, -sin / aspect + shx, tx], 1),
                         torch.stack([sin * aspect, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, t.shape, align_corners=False)
    t = F.grid_sample(t, grid, padding_mode="border", align_corners=False)
    t = t * (0.6 + rnd(B, 1, 1, 1) * 0.8) + (rnd(B, 1, 1, 1) * 2 - 1) * 0.3
    blur = rnd(B) < 0.25
    if blur.any():
        t[blur] = F.avg_pool2d(t[blur], 3, 1, 1)
    t = t + torch.randn_like(t) * (rnd(B, 1, 1, 1) * 0.08)
    gray = rnd(B) < 0.1
    if gray.any():
        t[gray] = t[gray].mean(1, keepdim=True).expand(-1, 3, -1, -1)
    return t.clamp(-1.5, 1.5)


def load_split(data: str, split: str):
    x = np.load(f"{data}/{split}_x.npy", mmap_mode="r")
    with open(f"{data}/{split}_labels.txt", encoding="utf-8") as f:
        y = f.read().split("\n")[: len(x)]
    return x, y


def evaluate(model, x, y, device) -> float:
    model.eval()
    ok = 0
    with torch.no_grad():
        for i in range(0, len(x), 512):
            preds = decode(model(to_tensor(x[i:i + 512]).to(device)).cpu())
            ok += sum(p == g for (p, _), g in zip(preds, y[i:i + 512]))
    model.train()
    return ok / len(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--channels", default=",".join(map(str, DEFAULT_CHANNELS)))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (ignored on cuda)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train.log"), "a", encoding="utf-8")

    def P(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    device = torch.device(args.device)
    xtr, ytr = load_split(args.data, "train")
    xte, yte = load_split(args.data, "test")
    idx = {c: i + 1 for i, c in enumerate(CHARS)}
    P(f"train={len(xtr)} test={len(xte)} classes={len(CHARS) + 1} device={device}")

    model = PlateCRNN(c=parse_channels(args.channels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(xtr) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    best, step, t0 = 0.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(len(xtr))
        total = 0.0
        for bi in range(steps_per_epoch):
            ids = np.sort(perm[bi * args.batch_size:(bi + 1) * args.batch_size])
            x = augment(to_tensor(xtr[ids]).to(device))
            labels = [ytr[i] for i in ids]
            targets = torch.tensor([idx[c] for l in labels for c in l])
            target_lens = torch.tensor([len(l) for l in labels])
            lp = model(x).log_softmax(-1).transpose(0, 1)
            loss = ctc(lp, targets, torch.full((len(ids),), lp.size(0), dtype=torch.long), target_lens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            total += loss.item()
            if step % 200 == 0:
                P(f"ep{ep} step{step}/{args.epochs * steps_per_epoch} loss={total / (bi + 1):.4f} {step * args.batch_size / (time.time() - t0):.0f}img/s")
        acc = evaluate(model, xte, yte, device)
        P(f"== epoch {ep} test_acc={acc:.4f} elapsed={(time.time() - t0) / 60:.1f}min")
        torch.save(model.state_dict(), f"{args.out}/last.pt")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), f"{args.out}/best.pt")
    P(f"done best={best:.4f}")


if __name__ == "__main__":
    main()
