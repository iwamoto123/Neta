#!/usr/bin/env python3
"""
Convert cookie exports (e.g. Chrome extensions) to Playwright add_cookies() format.

IN:  cookies_x.json  (or any JSON list/dict with cookies)
OUT: cookies_x_playwright.json

Note: Cookieはログイン情報そのものです。絶対にコミットしないこと。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


IN_PATH = Path("cookies_x.json")
OUT_PATH = Path("cookies_x_playwright.json")


def to_playwright_cookies(data: Any) -> list[dict[str, Any]]:
    cookies = data if isinstance(data, list) else data.get("cookies", data)
    if not isinstance(cookies, list):
        raise ValueError("Unsupported cookie JSON format (expected list or dict with 'cookies').")

    out: list[dict[str, Any]] = []
    for c in cookies:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        value = c.get("value")
        domain = c.get("domain")
        path = c.get("path", "/")
        if not (name and value and domain):
            continue

        expires = c.get("expires")
        if expires is None:
            expires = c.get("expirationDate")

        same_site = c.get("sameSite")
        if isinstance(same_site, str):
            ss = same_site.lower()
            if ss in ["no_restriction", "none"]:
                same_site = "None"
            elif ss == "lax":
                same_site = "Lax"
            elif ss == "strict":
                same_site = "Strict"

        o: dict[str, Any] = {
            "name": name,
            "value": value,
            "domain": domain,
            "path": path,
        }
        if expires is not None:
            try:
                o["expires"] = int(float(expires))
            except Exception:
                pass
        if "httpOnly" in c:
            o["httpOnly"] = bool(c.get("httpOnly"))
        if "secure" in c:
            o["secure"] = bool(c.get("secure"))
        if same_site is not None:
            o["sameSite"] = same_site
        out.append(o)

    return out


def main() -> int:
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    out = to_playwright_cookies(data)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {OUT_PATH} (count={len(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


