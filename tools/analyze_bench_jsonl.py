#!/usr/bin/env python3
"""Categorize errors in a bench JSONL (bench_platex_local.py or plateai.crnn.evaluate output).

Usage: analyze_bench_jsonl.py BENCH.jsonl"""
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
errs = [r for r in rows if not r["ok"]]
print(f"total={len(rows)} ok={len(rows)-len(errs)} ({1-len(errs)/len(rows):.4f}) err={len(errs)}")
cat = collections.Counter(); conf = collections.Counter(); pos = collections.Counter(); lens = collections.Counter()
for r in errs:
    p, l = r["pred"], r["label"]
    if r.get("error"): c = "api_error"
    elif not p: c = "empty"
    elif len(p) != len(l):
        c = f"len {len(l)}->{len(p)}"
        if len(p) == len(l) - 1 and any(l[:i] + l[i+1:] == p for i in range(len(l))): c += " (drop1)"
        elif len(p) == len(l) + 1 and any(p[:i] + p[i+1:] == l for i in range(len(p))): c += " (add1)"
    else:
        d = [i for i in range(len(l)) if p[i] != l[i]]
        c = "1char" if len(d) == 1 else f"{len(d)}char"
        for i in d:
            pos[i] += 1; conf[f"{l[i]}->{p[i]}"] += 1
    cat[c] += 1
    lens[len(l)] += 1
for k, v in cat.most_common(): print(f"  {k:22s} {v:6d} {v/len(errs):.1%}")
print("label len of errors:", dict(lens))
print("pos:", dict(sorted(pos.items())))
print("top confusions:", conf.most_common(25))
