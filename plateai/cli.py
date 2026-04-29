"""plateai CLI entry point.

Usage:

    plateai train  --csv hard.csv --output out.onnx
    plateai export --checkpoint best.pth --output plate_rec_color.onnx
    plateai info
"""
from __future__ import annotations

import argparse
import logging
import sys

from plateai import __version__
from plateai.alphabets import NUM_CLASSES, NUM_COLORS, PLATE_CHR
from plateai.export import add_export_arguments, run as run_export
from plateai.train import add_train_arguments, run as run_train


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _info(_args: argparse.Namespace) -> int:
    print(f"plateai {__version__}")
    print(f"  num_classes (CTC): {NUM_CLASSES}")
    print(f"  num_colors:        {NUM_COLORS}")
    print(f"  charset preview:   {PLATE_CHR[:40]}...")
    try:
        import torch  # type: ignore

        print(f"  torch:             {torch.__version__}")
        print(f"  cuda available:    {torch.cuda.is_available()}")
    except Exception as exc:  # pragma: no cover - import safety only
        print(f"  torch:             unavailable ({exc})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plateai",
        description="Fine-tune the platex plate recognizer with CSV-defined hard cases.",
    )
    parser.add_argument("--log-level", default="info", help="Python logging level (debug/info/warn).")
    sub = parser.add_subparsers(dest="command", required=False)

    train_p = sub.add_parser("train", help="Run a fine-tuning loop on a CSV/Excel of hard cases.")
    add_train_arguments(train_p)

    export_p = sub.add_parser("export", help="Export an existing .pth to ONNX.")
    add_export_arguments(export_p)

    sub.add_parser("info", help="Print version and runtime information.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    if args.command == "train":
        return run_train(args)
    if args.command == "export":
        return run_export(args)
    if args.command == "info" or args.command is None:
        return _info(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
