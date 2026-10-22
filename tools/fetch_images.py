#!/usr/bin/env python3
"""Mirror plate images from OBS into a local dir: <out>/<url path>."""
import csv, os, re, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

PREFIX = "https://huizhoupark.obs.cn-south-1.myhuaweicloud.com"

def read_rows(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if len(r) == 1 and "," in r[0]:
                r = r[0].split(",", 1)
            if len(r) >= 2:
                yield r[0].strip(), re.sub(r"^https?://[^/]+", "", r[1].strip())

def fetch(path, out):
    dst = os.path.join(out, path.lstrip("/"))
    if os.path.exists(dst) and os.path.getsize(dst) > 32:
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = PREFIX + urllib.parse.quote(path)
    for i in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                body = r.read()
            if len(body) < 32:
                return "small"
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, dst)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return f"http{e.code}"
        except Exception:
            time.sleep(1 + i)
    return "fail"

def main():
    csv_path, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    paths = [p for _, p in read_rows(csv_path)]
    stats, t0 = {}, time.time()
    with ThreadPoolExecutor(workers) as ex:
        for n, s in enumerate(ex.map(lambda p: fetch(p, out), paths), 1):
            stats[s] = stats.get(s, 0) + 1
            if n % 5000 == 0 or n == len(paths):
                print(f"{n}/{len(paths)} {stats} {n/(time.time()-t0):.0f}/s", flush=True)

if __name__ == "__main__":
    main()
