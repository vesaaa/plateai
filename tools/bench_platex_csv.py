#!/usr/bin/env python3
"""Benchmark platex: CSV (label, path) vs POST /api/v1/recognize (mode=crop, type=url)."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PREFIX = "https://huizhoupark.obs.cn-south-1.myhuaweicloud.com"


def read_rows(path: str) -> list[tuple[str, str]]:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "gbk")
    rows: list[list[str]] = []
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                rows = [r for r in csv.reader(f) if r and len(r) >= 2]
            break
        except UnicodeDecodeError:
            continue
    if not rows:
        raise RuntimeError(f"empty or unreadable: {path}")
    if rows[0][0].strip().lower() in {"test1", "plate", "车牌", "label"}:
        rows = rows[1:]
    out: list[tuple[str, str]] = []
    for r in rows:
        lab, src = r[0].strip(), r[1].strip()
        if lab and src:
            out.append((lab, src))
    return out


def full_url(prefix: str, src: str) -> str:
    if src.startswith("http://") or src.startswith("https://"):
        return src
    return f"{prefix.rstrip('/')}/{src.lstrip('/')}"


def one_request(
    api: str,
    prefix: str,
    rid: str,
    src: str,
    label: str,
    mode: str,
    timeout: int,
) -> dict[str, Any]:
    u = full_url(prefix, src)
    body = {
        "images": [{"id": rid, "type": "url", "data": u}],
        "mode": mode,
    }
    raw = json.dumps(body).encode("utf-8")
    req = Request(api, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return {
            "id": rid,
            "label": label,
            "source": src,
            "url": u,
            "ok": False,
            "pred": "",
            "error": str(exc),
            "ms": int((time.perf_counter() - t0) * 1000),
        }
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    pred = ""
    err = ""
    if data.get("code") != 0:
        err = data.get("message") or "nonzero_code"
    else:
        results = (data.get("data") or {}).get("results") or []
        if not results:
            err = "no_results"
        else:
            r0 = results[0]
            err = r0.get("error") or ""
            plates = r0.get("plates") or []
            if plates:
                pred = str(plates[0].get("plate_number") or "")
            elif not err:
                err = "no_plates"
    match = bool(pred) and pred == label
    return {
        "id": rid,
        "label": label,
        "source": src,
        "url": u,
        "ok": match,
        "pred": pred,
        "error": err,
        "ms": elapsed_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--url-prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--api", default="http://127.0.0.1:8080/api/v1/recognize")
    ap.add_argument(
        "--mode",
        default="full",
        help="platex RecognizeRequest mode: full (detect+rec, aligns with production) | crop | auto",
    )
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-err", default="", help="write misclassified rows (label,path) for training")
    ap.add_argument("--out-report", default="", help="write JSON lines of each row result")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    if args.max_rows and len(rows) > args.max_rows:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.max_rows]

    ok = 0
    total = 0
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {}
        for i, (label, src) in enumerate(rows):
            rid = f"r{i}"
            fut = ex.submit(one_request, args.api, args.url_prefix, rid, src, label, args.mode, args.timeout)
            futs[fut] = None
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            total += 1
            if r["ok"]:
                ok += 1

    results.sort(key=lambda x: int(x["id"][1:]) if str(x["id"]).startswith("r") else 0)

    acc = ok / max(1, total)
    print(f"bench rows={total} correct={ok} acc={acc:.4f} mode={args.mode}", flush=True)
    print(f"RESULT acc={acc:.6f} total={total} correct={ok}", flush=True)

    if args.out_err:
        misses = [r for r in results if not r["ok"]]
        with open(args.out_err, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["plate", "path"])
            for r in misses:
                w.writerow([r["label"], r["source"]])
        print(f"wrote {len(misses)} err rows -> {args.out_err}", flush=True)

    if args.out_report:
        with open(args.out_report, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote report -> {args.out_report}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
