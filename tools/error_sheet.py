#!/usr/bin/env python3
"""Render a contact sheet of bench errors with label/pred captions.

Usage: error_sheet.py BENCH.jsonl IMG_DIR OUT.png {many|one|len|err} SEED N
Set FONT=/path/to/cjk.ttc if Noto CJK is not installed."""
import json, os, random, sys
from PIL import Image, ImageDraw, ImageFont
src, imgdir, out, flt, seed, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
def nd(r):
    p, l = r["pred"], r["label"]
    return sum(a != b for a, b in zip(p, l)) if len(p) == len(l) else -1
F = {"many": lambda r: not r["ok"] and nd(r) >= 3, "one": lambda r: not r["ok"] and nd(r) == 1,
     "len": lambda r: not r["ok"] and nd(r) == -1, "err": lambda r: not r["ok"]}[flt]
sel = [r for r in rows if F(r)]
random.Random(seed).shuffle(sel); sel = sel[:n]
font = ImageFont.truetype(os.getenv("FONT", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"), 18)
W, H, cols = 300, 118, 3
sheet = Image.new("RGB", (W * cols, H * ((len(sel) + cols - 1) // cols)), "white")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(sel):
    x, y = (i % cols) * W, (i // cols) * H
    try:
        im = Image.open(imgdir + r["path"]).convert("RGB"); im.thumbnail((W - 6, 88))
        sheet.paste(im, (x + 3, y + 26))
    except Exception: pass
    d.text((x + 3, y + 2), f"{i} 标{r['label']} 识{r['pred']}", fill="red", font=font)
sheet.save(out)
