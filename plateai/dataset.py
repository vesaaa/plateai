"""Dataset adapter for plateai.

The trainer accepts two input formats:

1. CSV / Excel (one row per sample) with two columns. The first column is
   the ground-truth plate label (e.g. ``粤BAE6196``), the second column is
   either an image URL or a relative/absolute file path.
2. A directory of images named ``<label>_*.jpg`` for very quick experiments.

URLs are resolved on first use and cached in a local directory so that
repeated training runs stay offline-friendly.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import os
import random
import threading
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import torch
import torch.utils.data as data

from plateai.alphabets import PLATE_CHR

LOG = logging.getLogger("plateai.dataset")

# Common URL-prefix shorthand for the platex production sample set: a CSV
# whose URL column starts with ``/`` is interpreted as a path under this host.
DEFAULT_URL_PREFIX = "https://huizhoupark.obs.cn-south-1.myhuaweicloud.com"

_DOWNLOAD_LOCK = threading.Lock()


class PlateSample:
    __slots__ = ("label", "image_path", "raw_source", "weight")

    def __init__(self, label: str, image_path: str, raw_source: str, weight: float = 1.0):
        self.label = label
        self.image_path = image_path
        self.raw_source = raw_source
        self.weight = weight


class PlateDataset(data.Dataset):
    """Loads (image, plate_label) pairs for CTC training.

    Args:
        samples: list of PlateSample.
        img_h, img_w: input tensor size for the recognizer (default 48x168).
        is_train: enables light data augmentation.
        max_label_len: pad/truncate label lists to this length.
        mean, std: per-pixel normalization.
    """

    def __init__(
        self,
        samples: Sequence[PlateSample],
        img_h: int = 48,
        img_w: int = 168,
        is_train: bool = True,
        max_label_len: int = 12,
        mean: float = 0.588,
        std: float = 0.193,
    ):
        self.samples = list(samples)
        self.img_h = img_h
        self.img_w = img_w
        self.is_train = is_train
        self.max_label_len = max_label_len
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = _safe_read_image(s, max_retry=1)
        if img is None:
            LOG.warning("Skip undecodable sample after retry: %s", s.raw_source)
            return None

        if self.is_train:
            img = _augment(img)

        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        # CTC label encoding (skip unknown chars rather than crashing).
        label_idxs = [PLATE_CHR.index(c) for c in s.label if c in PLATE_CHR]
        label_len = len(label_idxs)
        if label_len == 0:
            raise RuntimeError(f"Empty label for sample: {s.raw_source}")
        if label_len > self.max_label_len:
            label_idxs = label_idxs[: self.max_label_len]
            label_len = self.max_label_len

        # Pad to fixed length so torch can batch the labels.
        padded = label_idxs + [0] * (self.max_label_len - label_len)

        return (
            torch.from_numpy(img.copy()),
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(label_len, dtype=torch.long),
            torch.tensor(s.weight, dtype=torch.float32),
        )


def _safe_read_image(sample: PlateSample, max_retry: int = 1) -> np.ndarray | None:
    """Read image with one automatic cache-heal retry.

    Why:
    - Remote cache files can occasionally be zero-byte or corrupted when
      network fetch fails/interrupted.
    - A single bad sample should not permanently poison subsequent runs.
    """
    attempts = 0
    while attempts <= max_retry:
        attempts += 1
        arr = np.fromfile(sample.image_path, dtype=np.uint8)
        if arr.size == 0:
            LOG.warning("Empty image buffer: %s (attempt %d/%d)", sample.image_path, attempts, max_retry+1)
        else:
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            LOG.warning("Failed to decode image: %s (attempt %d/%d)", sample.image_path, attempts, max_retry+1)

        # Heal cache: remove broken file and re-resolve once.
        if attempts <= max_retry:
            try:
                if os.path.exists(sample.image_path):
                    os.remove(sample.image_path)
            except Exception as exc:
                LOG.warning("Failed to remove broken cache file %s: %s", sample.image_path, exc)
            try:
                sample.image_path = _resolve_image(sample.raw_source, Path(sample.image_path).parent, DEFAULT_URL_PREFIX)
            except Exception as exc:
                LOG.warning("Failed to re-resolve image %s: %s", sample.raw_source, exc)
                break
    return None


def _augment(img: np.ndarray) -> np.ndarray:
    """Light augmentation that does not change plate semantics."""
    h, w = img.shape[:2]

    # Mild contrast/brightness jitter.
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-12, 12)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Tiny rotation to simulate camera tilt; never large enough to clip chars.
    if random.random() < 0.4:
        angle = random.uniform(-3.0, 3.0)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(128, 128, 128))

    # Tiny horizontal shift; helps the recognizer learn position invariance.
    if random.random() < 0.3:
        shift = random.randint(-2, 2)
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (w, h), borderValue=(128, 128, 128))

    return img


def load_csv_samples(
    csv_path: str,
    cache_dir: str,
    url_prefix: str = DEFAULT_URL_PREFIX,
    hard_case_weight: float = 1.0,
    max_rows: int | None = None,
) -> list[PlateSample]:
    """Load samples from a CSV/TSV/Excel file.

    The file must have at least 2 columns: ``label, image_source``. Header rows
    are auto-detected and skipped.
    """
    csv_path = str(csv_path)
    rows = _read_table(csv_path)
    if not rows:
        raise RuntimeError(f"No rows parsed from {csv_path}")

    # Auto-skip a header row if the first row's first cell is not a plate-like
    # token (e.g. column header in Chinese).
    if rows and not _looks_like_plate(rows[0][0]):
        LOG.info("Skipping header row: %r", rows[0])
        rows = rows[1:]

    if max_rows is not None:
        rows = rows[:max_rows]

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    samples: list[PlateSample] = []
    for row in rows:
        if len(row) < 2:
            continue
        label = row[0].strip()
        source = row[1].strip()
        if not label or not source:
            continue
        try:
            img_path = _resolve_image(source, cache_path, url_prefix)
        except Exception as exc:
            LOG.warning("Skip %s: %s", source, exc)
            continue
        samples.append(PlateSample(label=label, image_path=img_path, raw_source=source, weight=hard_case_weight))
    return samples


def _read_table(path: str) -> list[list[str]]:
    ext = Path(path).suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return _read_excel(path)
    return _read_csv(path)


def _read_csv(path: str) -> list[list[str]]:
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                rows = [r for r in reader if r]
            return rows
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Failed to decode {path} with any of {encodings}: {last_err}")


def _read_excel(path: str) -> list[list[str]]:
    try:
        import openpyxl  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("openpyxl is required to read Excel files") from exc
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows: list[list[str]] = []
    for r in ws.iter_rows(values_only=True):
        if not r:
            continue
        rows.append([str(c) if c is not None else "" for c in r])
    return rows


def _looks_like_plate(value: str) -> bool:
    """A plate label is at least 5 chars and starts with a Chinese province
    character (rough heuristic; "id"/"plate"/"车牌号码" headers will not match)."""
    if not value:
        return False
    if len(value) < 5 or len(value) > 12:
        return False
    head = value[0]
    return _is_cjk(head) and head not in {"车", "图"}


def _is_cjk(ch: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in ch)


def _resolve_image(source: str, cache_dir: Path, url_prefix: str) -> str:
    """Return a local path for ``source``. Downloads remote files if needed."""
    parsed = urlparse(source)
    is_url = parsed.scheme in {"http", "https"}
    is_relative_url = source.startswith("/") and not os.path.exists(source)

    if not is_url and not is_relative_url:
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        return source

    full_url = source if is_url else f"{url_prefix.rstrip('/')}{source}"
    cached = _cache_path_for(full_url, cache_dir)
    if cached.exists() and cached.stat().st_size > 0:
        return str(cached)

    with _DOWNLOAD_LOCK:
        # Double-check after acquiring the lock to avoid a thundering herd.
        if cached.exists() and cached.stat().st_size > 0:
            return str(cached)
        LOG.info("Downloading %s", full_url)
        resp = requests.get(full_url, timeout=20)
        resp.raise_for_status()
        body = resp.content
        if len(body) < 32:
            raise ValueError(f"Image payload too small ({len(body)} bytes): {full_url}")
        arr = np.frombuffer(body, dtype=np.uint8)
        if cv2.imdecode(arr, cv2.IMREAD_COLOR) is None:
            raise ValueError(f"Downloaded payload is not a decodable image (403/HTML/empty?): {full_url}")
        cached.write_bytes(body)
    return str(cached)


def _cache_path_for(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    suffix = Path(urlparse(url).path).suffix or ".bin"
    return cache_dir / f"{digest}{suffix}"


def split_train_val(
    samples: Sequence[PlateSample],
    val_ratio: float = 0.1,
    seed: int = 1234,
) -> tuple[list[PlateSample], list[PlateSample]]:
    if not samples:
        return [], []
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    if n_val >= len(shuffled):
        n_val = max(1, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def repeat_hard_cases(samples: Iterable[PlateSample], times: int) -> list[PlateSample]:
    """Duplicate the input list ``times`` times for hard-case oversampling."""
    if times <= 1:
        return list(samples)
    out: list[PlateSample] = []
    for _ in range(times):
        out.extend(samples)
    return out
