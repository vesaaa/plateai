"""Standalone ONNX export utility (re-export an existing .pth without training)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from plateai.alphabets import NUM_CLASSES, NUM_COLORS
from plateai.model import MyNetOcrColor, detect_cfg_from_checkpoint, load_pretrained

LOG = logging.getLogger("plateai.export")


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth file produced by training.")
    parser.add_argument("--output", required=True, help="Output ONNX file path.")
    parser.add_argument("--no-color", action="store_true", help="Export without the color head.")


def run(args: argparse.Namespace) -> int:
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    color_num = None if args.no_color else NUM_COLORS
    cfg = detect_cfg_from_checkpoint(args.checkpoint)
    if cfg is not None:
        LOG.info("Detected model cfg: %s", cfg)
    model = MyNetOcrColor(num_classes=NUM_CLASSES, color_num=color_num, export=True, cfg=cfg)
    info = load_pretrained(model, args.checkpoint, strict=False)
    LOG.info("Loaded checkpoint %s (missing=%d unexpected=%d)",
             args.checkpoint, len(info["missing"]), len(info["unexpected"]))
    model.eval()
    sample = torch.zeros(1, 3, 48, 168)
    output_names = ["plate", "color"] if color_num else ["plate"]
    torch.onnx.export(
        model,
        sample,
        args.output,
        opset_version=12,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=None,
    )
    LOG.info("Exported ONNX to %s", args.output)
    return 0
