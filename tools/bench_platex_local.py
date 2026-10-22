#!/usr/bin/env python3
"""Bench a platex server on locally mirrored images (type=path, no URL fetch).

Usage: bench_platex_local.py CSV IMG_DIR {full|crop|auto} OUT.jsonl [API_URL]
IMG_DIR must be readable by the platex process."""
import csv, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

def read_rows(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if len(r) == 1 and "," in r[0]:
                r = r[0].split(",", 1)
            if len(r) >= 2:
                yield r[0].strip(), re.sub(r"^https?://[^/]+", "", r[1].strip())

def one(api, imgdir, mode, label, path):
    full = os.path.abspath(os.path.join(imgdir, path.lstrip("/")))
    body = json.dumps({"images": [{"id": "1", "type": "path", "data": full}], "mode": mode}).encode()
    t = time.time()
    try:
        req = urllib.request.Request(api, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        res = d["data"]["results"][0]
        plates = res.get("plates") or []
        pred = plates[0]["plate_number"] if plates else ""
        conf = plates[0]["confidence"] if plates else 0
        return {"label": label, "path": path, "pred": pred, "conf": conf, "ok": pred == label,
                "n": len(plates), "error": res.get("error", ""), "ms": int((time.time() - t) * 1000)}
    except Exception as e:
        return {"label": label, "path": path, "pred": "", "ok": False, "error": str(e)}

def main():
    csv_path, imgdir, mode, out = sys.argv[1:5]
    api = sys.argv[5] if len(sys.argv) > 5 else "http://127.0.0.1:18080/api/v1/recognize"
    rows = list(read_rows(csv_path))
    ok, t0 = 0, time.time()
    with open(out, "w", encoding="utf-8") as f, ThreadPoolExecutor(16) as ex:
        for n, r in enumerate(ex.map(lambda x: one(api, imgdir, mode, *x), rows), 1):
            ok += r["ok"]
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if n % 20000 == 0:
                print(f"{n}/{len(rows)} acc={ok/n:.4f} {n/(time.time()-t0):.0f}/s", flush=True)
    print(f"mode={mode} rows={len(rows)} correct={ok} acc={ok/len(rows):.4f} qps={len(rows)/(time.time()-t0):.0f}")

if __name__ == "__main__":
    main()
