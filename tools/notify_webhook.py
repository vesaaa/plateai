#!/usr/bin/env python3
"""POST notify JSON. URL from env NOTIFY_WEBHOOK_URL only (never commit secrets)."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_ICON = "https://iphoto.mac89.com/icon/icon/256/20200728/85393/3722232.png"


def main() -> int:
    url = (os.environ.get("NOTIFY_WEBHOOK_URL") or "").strip()
    if not url:
        print("NOTIFY_WEBHOOK_URL unset; skip", file=sys.stderr)
        return 0
    body = sys.argv[1] if len(sys.argv) > 1 else "当前任务已经执行成功"
    title = sys.argv[2] if len(sys.argv) > 2 else "任务执行进展通知"
    group = os.environ.get("NOTIFY_GROUP", "Cursor任务").strip() or "Cursor任务"
    icon = os.environ.get("NOTIFY_ICON", DEFAULT_ICON).strip() or DEFAULT_ICON
    payload = {"title": title, "body": body, "group": group, "icon": icon}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read(256)
    except urllib.error.URLError as exc:
        print(f"notify failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
