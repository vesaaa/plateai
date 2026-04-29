"""Fine-tuning entry point: take a CSV of (label, image-source) and produce
an updated plate_rec_color recognizer that platex can deploy as-is.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from plateai.alphabets import NUM_CLASSES, NUM_COLORS, PLATE_CHR
from plateai.dataset import (
    PlateDataset,
    load_csv_samples,
    repeat_hard_cases,
    split_train_val,
)
from plateai.model import MyNetOcrColor, load_pretrained

LOG = logging.getLogger("plateai.train")


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", required=True, help="CSV/Excel of (label, image-source). Header row is auto-detected.")
    parser.add_argument("--output", required=True, help="Output ONNX file path.")
    parser.add_argument("--cache-dir", default="/workspace/cache", help="Directory used to cache downloaded images.")
    parser.add_argument("--checkpoint-dir", default="/workspace/checkpoints", help="Directory for intermediate .pth files.")
    parser.add_argument("--pretrained", default="/workspace/weights/plate_rec_color.pth", help="Pretrained .pth to start from.")
    parser.add_argument("--url-prefix", default=None, help="Optional URL prefix used when CSV column 2 starts with '/'.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hard-case-repeat", type=int, default=4, help="Repeat each CSV row this many times before mixing with the train set.")
    parser.add_argument("--no-color-loss", action="store_true", help="Skip the color head loss (useful when CSV has no color labels).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-rows", type=int, default=None, help="Limit the number of CSV rows used (debugging).")


def run(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading samples from %s", args.csv)
    csv_kwargs = {"cache_dir": args.cache_dir, "max_rows": args.max_rows}
    if args.url_prefix:
        csv_kwargs["url_prefix"] = args.url_prefix
    samples = load_csv_samples(args.csv, **csv_kwargs)
    LOG.info("Loaded %d samples", len(samples))
    if len(samples) < 4:
        raise RuntimeError(f"Need at least 4 samples to train, got {len(samples)}")

    train_samples, val_samples = split_train_val(samples, val_ratio=args.val_ratio, seed=args.seed)
    train_samples = repeat_hard_cases(train_samples, args.hard_case_repeat)
    LOG.info("Train=%d (after %dx repeat) Val=%d", len(train_samples), args.hard_case_repeat, len(val_samples))

    train_ds = PlateDataset(train_samples, is_train=True)
    val_ds = PlateDataset(val_samples, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    LOG.info("Building model (num_classes=%d, color_num=%d)", NUM_CLASSES, NUM_COLORS)
    model = MyNetOcrColor(num_classes=NUM_CLASSES, color_num=NUM_COLORS, export=False)

    if args.pretrained and os.path.exists(args.pretrained):
        info = load_pretrained(model, args.pretrained, strict=False)
        LOG.info("Loaded pretrained weights from %s (missing=%d unexpected=%d)",
                 args.pretrained, len(info["missing"]), len(info["unexpected"]))
    else:
        LOG.warning("No pretrained checkpoint at %s; training from scratch (will likely be poor)", args.pretrained)

    device = torch.device(args.device)
    model = model.to(device)

    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_acc = -1.0
    best_path = Path(args.checkpoint_dir) / "best.pth"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0
        for img, label, label_len, _weight in train_loader:
            img = img.to(device)
            label = label.to(device)
            label_len = label_len.to(device)

            log_probs, _color = model(img)  # log_probs shape: (T, B, C)
            T = log_probs.size(0)
            B = log_probs.size(1)
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long, device=device)

            # Flatten the padded labels into a 1D tensor of true characters
            # (CTCLoss requires concatenated targets when targets are 1D).
            flat_targets = []
            for b in range(B):
                flat_targets.extend(label[b, : label_len[b].item()].tolist())
            target_tensor = torch.tensor(flat_targets, dtype=torch.long, device=device)

            loss = ctc_loss(log_probs, target_tensor, input_lengths, label_len)
            if not torch.isfinite(loss):
                LOG.warning("Non-finite loss at epoch %d; skipping batch", epoch)
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(1, n_batches)

        acc = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        LOG.info("Epoch %d/%d  loss=%.4f  val_acc=%.4f  lr=%.2e  time=%.1fs",
                 epoch, args.epochs, avg_loss, acc, scheduler.get_last_lr()[0], elapsed)

        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "cfg": model.cfg, "best_acc": best_acc}, best_path)
            LOG.info("  ↳ new best, saved to %s", best_path)

    LOG.info("Training done. Best val acc=%.4f", best_acc)

    # Reload best checkpoint and export ONNX.
    if best_path.exists():
        load_pretrained(model, str(best_path), strict=False)

    # Re-export with export=True to get the ONNX-friendly graph.
    export_model = MyNetOcrColor(num_classes=NUM_CLASSES, color_num=NUM_COLORS, export=True)
    export_model.load_state_dict(model.state_dict())
    export_model.eval()
    sample = torch.zeros(1, 3, 48, 168)
    torch.onnx.export(
        export_model,
        sample,
        args.output,
        opset_version=12,
        input_names=["input"],
        output_names=["plate", "color"],
        dynamic_axes=None,
    )
    LOG.info("Exported ONNX to %s", args.output)
    return 0


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label, label_len, _weight in loader:
            img = img.to(device)
            log_probs, _ = model(img)  # (T, B, C)
            preds = log_probs.argmax(dim=-1).t()  # (B, T)
            for b in range(preds.size(0)):
                seq = _ctc_collapse(preds[b].tolist())
                gt = label[b, : label_len[b].item()].tolist()
                if seq == gt:
                    correct += 1
                total += 1
    return correct / max(1, total)


def _ctc_collapse(idxs: list[int]) -> list[int]:
    out: list[int] = []
    prev = -1
    for i in idxs:
        if i != 0 and i != prev:
            out.append(i)
        prev = i
    return out
