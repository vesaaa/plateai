# PlateAI — fine-tune the platex plate recognizer with your own hard cases

PlateAI is a small training pipeline that takes a CSV (or Excel) file of
license-plate ground truth labels paired with image URLs, fine-tunes the
plate_rec_color recognizer on your hard cases, and outputs a fresh
`plate_rec_color.onnx` ready to drop into [`platex`](https://github.com/vesaaa/platex).

The intended workflow is a continuous-improvement loop:

```
platex serves traffic → collects misrecognised plates → you fix the labels
        ↓
  feed them as a CSV to plateai → output a new ONNX
        ↓
  swap the new ONNX into platex/models/, restart, repeat
```

## Quickstart with Docker

```bash
docker pull ghcr.io/vesaaa/plateai:latest

# Sanity check
docker run --rm ghcr.io/vesaaa/plateai:latest info

# Train on your CSV. expected layout:
#   col 1  车牌真值   (e.g. 粤BAE6196)
#   col 2  图片URL或本地路径
docker run --rm \
  -v $(pwd)/data:/data:ro \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/cache:/workspace/cache \
  ghcr.io/vesaaa/plateai:latest \
  train \
    --csv /data/hard.csv \
    --output /workspace/output/plate_rec_color.onnx \
    --epochs 10 \
    --batch-size 32

# Re-export an existing checkpoint without training
docker run --rm \
  -v $(pwd)/output:/workspace/output \
  ghcr.io/vesaaa/plateai:latest \
  export --checkpoint /workspace/output/best.pth --output /workspace/output/plate_rec_color.onnx
```

The trained ONNX is byte-compatible with the `plate_rec_color.onnx` slot in
platex's `models/` directory; just replace the file and restart the service.

## CSV / Excel format

PlateAI parses the input table with the following rules:

* The first column is the **plate label** (`粤BAE6196`, `京A12345`, ...).
* The second column is **either an HTTP(S) URL or a local file path**. A
  leading `/` (no scheme) is interpreted as a path under the prefix passed
  with `--url-prefix` (defaults to the platex sample bucket).
* A header row is auto-detected and skipped if the first cell is not a
  plate-shaped string.
* `.csv` files are decoded with `utf-8-sig` / `utf-8` / `gb18030` / `gbk` in
  that order. `.xlsx` files require `openpyxl` (already bundled in the
  image).
* Downloaded images are cached under `/workspace/cache` (mount it as a
  volume to share across runs).

Minimal example:

```csv
plate,image
粤BAE6196,https://example.com/imgs/0001.jpg
粤LDD7691,/SNTDA-500-LS19030650/.../plate.bmp
```

See [`examples/sample.csv`](examples/sample.csv) for the format produced by
platex's failure logs.

## Local development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[torch]"

plateai info
plateai train --csv examples/sample.csv --output output/plate_rec_color.onnx --epochs 2
```

Pretrained weights ship in `weights/plate_rec_color.pth` (the upstream
we0091234 checkpoint). The trainer always starts from this file unless you
pass `--pretrained`.

## How fine-tuning is sized

Every training run starts from the bundled pretrained recognizer rather
than from scratch — this is critical because:

* the pretrained model was trained on hundreds of thousands of plates and
  already speaks the platex character set perfectly,
* the goal is just to nudge it toward your specific image distribution and
  hard-case patterns, not to teach it Chinese plates from zero.

A typical loop on an 8C / 16G CPU:

| samples | epochs | wall time |
| ---: | ---: | ---: |
| ~100 | 10 | a few minutes |
| ~1 000 | 10 | ~30–60 minutes |
| ~10 000 | 10 | a few hours |

Hard cases are oversampled (`--hard-case-repeat`) so that even a small CSV
gets enough gradient signal to bend the model.

## Verifying the new ONNX in platex

After training:

```bash
cp output/plate_rec_color.onnx /path/to/platex/models/plate_rec_color.onnx
docker restart platex
```

Then re-run your usual benchmark. If the new ONNX loaded successfully you
should see `Dual model session pool initialized` for `plate_rec_color.onnx`
and a `WE recognizer loaded` line in platex logs.

## Roadmap

* [ ] Optional CCPD bootstrap data when only a tiny CSV is available.
* [ ] Confusion-pair-aware loss weighting for high-frequency mismatches.
* [ ] Built-in eval against a held-out CSV at the end of training.
