#!/usr/bin/env python3
"""Filter a plate CSV so only rows whose label matches platex NEV shape remain.

Uses the same structural rules as platex ``looksLikeNewEnergyPlate``:
  - 8 chars, Han province + ASCII letter city prefix
  - Small/large NEV: ``D`` or ``F`` at index 2 (3rd char) or index 7 (last char)
  - Guangzhou-style pure-electric segment: city letter ``A`` + ``P`` + five digits
    (e.g. ``粤AP12345``), matching platex rollout notices.

Input/output: two-column CSV ``plate,path`` (same as ``sample_training_pool.py``).

Example::

    python tools/filter_nev_csv.py \\
      --input /opt/vscc/plateai/data/20万_err.csv \\
      --output /opt/vscc/plateai/data/20万_err_nev.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plateai.alphabets import PLATE_CHR  # noqa: E402

_PROV_CHARS = set(PLATE_CHR[1:35])


def read_pairs(path: Path) -> list[tuple[str, str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    rows: list[list[str]] = []
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                rows = [r for r in csv.reader(f) if r and len(r) >= 2]
            break
        except UnicodeDecodeError:
            continue
    if rows and rows[0][0].strip().lower() in {"test1", "plate", "车牌", "label"}:
        rows = rows[1:]
    out: list[tuple[str, str]] = []
    for r in rows:
        a, b = r[0].strip(), r[1].strip()
        if a and b:
            out.append((a, b))
    return out


def looks_like_mainland_plate_prefix(label: str) -> bool:
    s = label.strip()
    if len(s) < 2:
        return False
    c0, c1 = s[0], s[1]
    # Mirror platex isChineseRune + isASCIILetter
    if not ("\u4e00" <= c0 <= "\u9fff"):
        return False
    o = ord(c1)
    return (65 <= o <= 90) or (97 <= o <= 122)


def looks_like_new_energy_plate(label: str) -> bool:
    """Match platex internal/engine recognizer.go ``looksLikeNewEnergyPlate``."""
    s = label.strip()
    if len(s) != 8 or not looks_like_mainland_plate_prefix(s):
        return False
    mid = s[2].upper()
    last = s[7].upper()
    if mid in {"D", "F"} or last in {"D", "F"}:
        return True
    if s[1].upper() == "A" and mid == "P":
        tail = s[3:8]
        return len(tail) == 5 and all(ch.isdigit() for ch in tail)
    return False


def label_ok_for_we_charset(label: str) -> bool:
    s = label.strip()
    if s[0] not in _PROV_CHARS:
        return False
    return all(c in PLATE_CHR for c in s)


def dedupe_paths(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for lab, p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append((lab, p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--no-dedupe-path",
        action="store_true",
        help="Keep duplicate image paths.",
    )
    ap.add_argument(
        "--allow-noncharset-label",
        action="store_true",
        help="Keep rows whose label contains chars outside WE PLATE_CHR (not recommended for train).",
    )
    args = ap.parse_args()

    raw = read_pairs(args.input)
    nev = [(a, b) for a, b in raw if looks_like_new_energy_plate(a)]
    skipped_charset = sum(
        1 for a, b in raw if looks_like_new_energy_plate(a) and not label_ok_for_we_charset(a)
    )
    if not args.allow_noncharset_label:
        nev = [(a, b) for a, b in nev if label_ok_for_we_charset(a)]

    if not args.no_dedupe_path:
        nev = dedupe_paths(nev)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["plate", "path"])
        w.writerows(nev)

    print(
        f"input_rows={len(raw)} nev_shape={sum(1 for a,b in raw if looks_like_new_energy_plate(a))} "
        f"skipped_non_charset={skipped_charset} output_rows={len(nev)} -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
