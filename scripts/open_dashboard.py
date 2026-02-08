#!/usr/bin/env python3
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Open local daily dashboard in your default browser.")
    ap.add_argument("--path", default="ideas/daily/index.html", help="Dashboard path relative to repo root")
    args = ap.parse_args()

    target = (REPO_ROOT / args.path).resolve()
    if not target.exists():
        print(f"Dashboardが見つかりません: {target}")
        print("先に次を実行してください: python scripts/collect_trend.py")
        return 1

    url = target.as_uri()
    ok = webbrowser.open(url, new=2)
    print(f"Opened: {url} (ok={ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


