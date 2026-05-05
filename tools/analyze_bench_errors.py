#!/usr/bin/env python3
"""Analyze bench_report JSONL — categorize errors for strategy planning."""
import json, sys, collections
from pathlib import Path

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/opt/vscc/plateai/output/bench_report_r1.jsonl"
    errs, oks = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        (oks if j.get("ok") else errs).append(j)

    total = len(oks) + len(errs)
    print(f"total={total}  ok={len(oks)} ({100*len(oks)/max(1,total):.1f}%)  err={len(errs)} ({100*len(errs)/max(1,total):.1f}%)")

    empty_pred, wrong_len, diff_1, diff_2plus, http_err = 0, 0, 0, 0, 0
    len_shift = collections.Counter()
    char_pos_errors = collections.Counter()
    prov_err = 0

    for e in errs:
        pred = e.get("pred", "") or ""
        label = e.get("label", "") or ""
        err_msg = e.get("error", "") or ""

        if err_msg and ("http" in err_msg.lower() or "timeout" in err_msg.lower() or "fetch" in err_msg.lower()):
            http_err += 1
            continue
        if not pred:
            empty_pred += 1
            continue
        if len(pred) != len(label):
            wrong_len += 1
            delta = len(pred) - len(label)
            len_shift[delta] += 1
        else:
            diffs = 0
            for i, (a, b) in enumerate(zip(pred, label)):
                if a != b:
                    diffs += 1
                    char_pos_errors[i] += 1
            if diffs == 1:
                diff_1 += 1
            else:
                diff_2plus += 1
        if pred and label and pred[0] != label[0]:
            prov_err += 1

    print(f"\n--- Error breakdown ---")
    print(f"http/timeout/fetch: {http_err}")
    print(f"empty/no pred:      {empty_pred}")
    print(f"wrong length:       {wrong_len}")
    for k in sorted(len_shift):
        print(f"  len delta={k:+d}: {len_shift[k]}")
    print(f"same len, 1 char:   {diff_1}")
    print(f"same len, 2+ char:  {diff_2plus}")
    print(f"province (pos 0) wrong: {prov_err}")

    if char_pos_errors:
        print(f"\nchar position error frequency (same-len only):")
        for pos in sorted(char_pos_errors):
            print(f"  pos {pos}: {char_pos_errors[pos]}")

    print(f"\n--- Sample errors (first 20) ---")
    shown = 0
    for e in errs:
        if shown >= 20:
            break
        pred = e.get("pred", "") or ""
        label = e.get("label", "") or ""
        err_msg = e.get("error", "") or ""
        ms = e.get("ms", "")
        if err_msg and ("http" in err_msg.lower() or "timeout" in err_msg.lower()):
            continue
        print(f"  label={label:12s} pred={pred:12s} ms={ms}  err={err_msg[:60]}")
        shown += 1

    # latency analysis
    ok_ms = [o.get("ms", 0) for o in oks if o.get("ms")]
    err_ms = [e.get("ms", 0) for e in errs if e.get("ms")]
    if ok_ms:
        print(f"\n--- Latency ---")
        print(f"ok   avg={sum(ok_ms)/len(ok_ms):.0f}ms  p50={sorted(ok_ms)[len(ok_ms)//2]}ms  p95={sorted(ok_ms)[int(len(ok_ms)*0.95)]}ms")
    if err_ms:
        print(f"err  avg={sum(err_ms)/len(err_ms):.0f}ms  p50={sorted(err_ms)[len(err_ms)//2]}ms  p95={sorted(err_ms)[int(len(err_ms)*0.95)]}ms")

if __name__ == "__main__":
    main()
