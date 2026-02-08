#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

try:
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    print("依存が不足しています: requests")
    print("次を実行してください: python3 -m pip install requests")
    raise SystemExit(2)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES_PATH = REPO_ROOT / "scripts" / "trend_sources.json"
DEFAULT_FEEDBACK_PATH = REPO_ROOT / "scripts" / "feedback.json"


JST = dt.timezone(dt.timedelta(hours=9))


@dataclass(frozen=True)
class Item:
    source: str
    section: str
    title: str
    url: str
    metric: str = ""
    published: str = ""
    tags: list[str] | None = None


class SimpleLinkRelFeedParser(HTMLParser):
    """
    Extract RSS/Atom <link rel="alternate" type="application/rss+xml|application/atom+xml" href="...">
    and some minimal <title>.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feed_links: list[str] = []
        self._in_title = False
        self.title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.lower(): (v or "") for k, v in attrs}
        if tag.lower() == "link":
            rel = attr.get("rel", "").lower()
            typ = attr.get("type", "").lower()
            href = attr.get("href", "")
            if rel == "alternate" and href and typ in ("application/rss+xml", "application/atom+xml"):
                self.feed_links.append(href)
        elif tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)

    @property
    def title(self) -> str:
        return "".join(self.title_parts).strip()


class NoteFollowingsParser(HTMLParser):
    """
    Best-effort extraction of usernames from note followings page HTML.
    We target patterns like:
      https://note.com/<username>
      href="/<username>"
    and keep plausible usernames.
    """

    def __init__(self) -> None:
        super().__init__()
        self._users: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr = {k.lower(): (v or "") for k, v in attrs}
        href = attr.get("href", "")
        if not href:
            return
        if "note.com/" in href:
            try:
                u = urllib.parse.urlparse(href)
                if u.netloc.endswith("note.com"):
                    seg = u.path.strip("/").split("/")[0] if u.path.strip("/") else ""
                    self._maybe_add(seg)
            except Exception:
                return
        elif href.startswith("/"):
            seg = href.strip("/").split("/")[0]
            self._maybe_add(seg)

    def _maybe_add(self, seg: str) -> None:
        # note usernames are typically [A-Za-z0-9_]+, keep conservative to avoid noise
        if re.fullmatch(r"[A-Za-z0-9_]{3,32}", seg or ""):
            self._users.add(seg)

    @property
    def users(self) -> list[str]:
        return sorted(self._users)


def jst_today() -> dt.date:
    return dt.datetime.now(tz=JST).date()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def http_get(url: str, *, timeout: int = 20, headers: dict[str, str] | None = None) -> requests.Response:
    h = {
        "User-Agent": "neta-trend-collector/1.0 (+https://github.com/)",
        "Accept": "*/*",
    }
    if headers:
        h.update(headers)
    return requests.get(url, headers=h, timeout=timeout)


def safe_urljoin(base: str, href: str) -> str:
    return urllib.parse.urljoin(base, href)


def discover_feed_url(site_url: str) -> str | None:
    """
    RSS/Atom feed discovery:
    - if the URL itself looks like a feed, accept it
    - else, fetch HTML and look for <link rel="alternate" ...>
    """
    if re.search(r"(rss|atom)(\.xml)?(\?|$)", site_url, re.IGNORECASE):
        return site_url
    try:
        res = http_get(site_url)
        if res.status_code >= 400:
            return None
        p = SimpleLinkRelFeedParser()
        p.feed(res.text)
        if not p.feed_links:
            return None
        # pick first
        return safe_urljoin(site_url, p.feed_links[0])
    except Exception:
        return None


def parse_rss(feed_xml: str, *, base_url: str = "") -> list[dict[str, str]]:
    """
    Parse RSS 2.0 / Atom (best-effort).
    Returns list of {title, link, published}.
    """
    # Some feeds have leading BOM or whitespace
    feed_xml = feed_xml.lstrip("\ufeff").strip()
    root = ET.fromstring(feed_xml)

    def text(el: Any, path: str) -> str:
        n = el.find(path)
        return (n.text or "").strip() if n is not None and n.text else ""

    items: list[dict[str, str]] = []
    tag = root.tag.lower()

    # Atom
    if tag.endswith("feed"):
        ns = ""
        if "}" in root.tag:
            ns = root.tag.split("}")[0] + "}"
        for entry in root.findall(f"{ns}entry"):
            title = text(entry, f"{ns}title")
            link = ""
            for ln in entry.findall(f"{ns}link"):
                rel = ln.attrib.get("rel", "alternate")
                if rel == "alternate" and ln.attrib.get("href"):
                    link = ln.attrib["href"]
                    break
            published = text(entry, f"{ns}published") or text(entry, f"{ns}updated")
            if title and link:
                items.append({"title": title, "link": safe_urljoin(base_url, link), "published": published})
        return items

    # RSS
    channel = root.find("channel")
    if channel is None:
        # Some RSS include namespaces; do a fallback scan
        channel = next((c for c in root if c.tag.lower().endswith("channel")), None)
    if channel is None:
        return items

    for it in channel.findall("item"):
        title = text(it, "title")
        link = text(it, "link")
        published = text(it, "pubDate") or text(it, "dc:date")
        if title and link:
            items.append({"title": title, "link": safe_urljoin(base_url, link), "published": published})
    return items


def fetch_feed_items(feed_url: str) -> list[dict[str, str]]:
    res = http_get(feed_url, timeout=25, headers={"Accept": "application/rss+xml, application/atom+xml, text/xml;q=0.9, */*;q=0.8"})
    res.raise_for_status()
    return parse_rss(res.text, base_url=feed_url)


def note_tag_feed_url(tag: str) -> str:
    # note hashtag pages are URL-encoded. RSS endpoint is not officially documented; we try common patterns.
    enc = urllib.parse.quote(tag)
    # Try /hashtag/<tag>/rss first (works for many tags)
    return f"https://note.com/hashtag/{enc}/rss"


def note_user_feed_url(user: str) -> str:
    return f"https://note.com/{user}/rss"


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def tokenize_keywords(text: str) -> list[str]:
    """
    Minimal tokenizer without extra deps:
    - English-ish tokens: A-Za-z0-9_-/ length>=3
    - Japanese tokens: Hiragana/Katakana/Kanji sequences length>=2
    """
    t = normalize_space(text)
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-/]{2,}|[\u3040-\u30ff\u4e00-\u9fff]{2,}", t)
    # Dedup but keep order
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        k = tok.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(tok)
    return out


def compute_interest_score(title: str, *, base_keywords: list[str], keyword_weights: dict[str, float]) -> tuple[float, str, str]:
    """
    Returns (score, stars, category).
    """
    t = normalize_space(title)
    score = 0.0
    matched: list[str] = []

    for kw in base_keywords:
        if kw and kw.lower() in t.lower():
            score += 3.0
            matched.append(kw)

    for tok in tokenize_keywords(t):
        w = keyword_weights.get(tok.lower())
        if w:
            score += float(w)

    # crude category guess
    category = "その他"
    low = t.lower()
    if any(k in low for k in ["m&a", "exit", "buyout", "acqui", "売却", "バイアウト", "買収"]):
        category = "Exit/M&A"
    elif any(k in low for k in ["fund", "series", "資金調達", "投資", "raise"]):
        category = "資金調達"
    elif any(k in low for k in ["ai", "llm", "生成ai", "model", "agent"]):
        category = "AI"
    elif any(k in low for k in ["product hunt", "launch", "beta", "プロダクト"]):
        category = "プロダクト"
    elif any(k in low for k in ["哲学", "科学", "心理", "脳", "宇宙"]):
        category = "科学・哲学"
    elif any(k in low for k in ["仕事", "キャリア", "働き方", "マネジメント"]):
        category = "仕事論"

    if score >= 6:
        stars = "★★★"
    elif score >= 3:
        stars = "★★"
    else:
        stars = "★"
    return score, stars, category


def update_keyword_weights_from_feedback(feedback: dict[str, Any]) -> dict[str, float]:
    weights: dict[str, float] = {k: float(v) for k, v in (feedback.get("keyword_weights") or {}).items()}
    liked_urls: list[str] = feedback.get("liked_urls") or []
    if not liked_urls:
        return weights

    # Best-effort: fetch title from HTML <title> for each URL
    for url in liked_urls[-50:]:
        try:
            res = http_get(url, timeout=20)
            if res.status_code >= 400:
                continue
            p = SimpleLinkRelFeedParser()
            p.feed(res.text[:200_000])
            title = p.title
            if not title:
                continue
            for tok in tokenize_keywords(title):
                k = tok.lower()
                weights[k] = min(10.0, weights.get(k, 0.0) + 0.5)
        except Exception:
            continue
        time.sleep(0.2)
    return weights


def ensure_note_users_initialized(sources: dict[str, Any], *, force: bool = False) -> tuple[dict[str, Any], list[str]]:
    note = sources.get("note") or {}
    users: list[str] = note.get("users") or []
    followings_url = note.get("followings_url")
    init_mode = note.get("followings_init_mode", "once")
    initialized_at = note.get("followings_initialized_at")

    if not followings_url:
        return sources, users

    should_init = False
    if force:
        should_init = True
    elif init_mode == "once":
        should_init = (not users) and (not initialized_at)
    elif init_mode == "always":
        should_init = True

    if not should_init:
        return sources, users

    try:
        res = http_get(followings_url, timeout=25)
        if res.status_code >= 400:
            return sources, users
        p = NoteFollowingsParser()
        p.feed(res.text)
        extracted = p.users
        if extracted:
            users = sorted(set(users).union(extracted))
            note["users"] = users
            note["followings_initialized_at"] = dt.datetime.now(tz=JST).isoformat()
            sources["note"] = note
    except Exception:
        return sources, users

    return sources, users


def collect_producthunt(sources: dict[str, Any]) -> list[Item]:
    ph = (sources.get("producthunt") or {})
    if not ph.get("enabled", True):
        return []
    feed_url = ph.get("feed_url") or "https://www.producthunt.com/feed"
    max_items = int(ph.get("max_items", 20))

    try:
        entries = fetch_feed_items(feed_url)[:max_items]
    except Exception:
        return []

    out: list[Item] = []
    for e in entries:
        out.append(
            Item(
                source="Product Hunt",
                section="Product Hunt",
                title=normalize_space(e["title"]),
                url=e["link"],
                published=normalize_space(e.get("published", "")),
            )
        )
    return out


def collect_note(sources: dict[str, Any]) -> list[Item]:
    note = (sources.get("note") or {})
    tags: list[str] = note.get("tags") or []
    users: list[str] = note.get("users") or []

    out: list[Item] = []

    # Tags
    for tag in tags[:50]:
        feed_url = note_tag_feed_url(tag)
        try:
            entries = fetch_feed_items(feed_url)[:10]
        except Exception:
            continue
        for e in entries:
            out.append(
                Item(
                    source=f"note #{tag}",
                    section="note",
                    title=normalize_space(e["title"]),
                    url=e["link"],
                    published=normalize_space(e.get("published", "")),
                    tags=[tag],
                )
            )
        time.sleep(0.2)

    # Users
    for user in users[:200]:
        feed_url = note_user_feed_url(user)
        try:
            entries = fetch_feed_items(feed_url)[:5]
        except Exception:
            continue
        for e in entries:
            out.append(
                Item(
                    source=f"note @{user}",
                    section="note",
                    title=normalize_space(e["title"]),
                    url=e["link"],
                    published=normalize_space(e.get("published", "")),
                )
            )
        time.sleep(0.2)

    return out


def collect_rss_sites(sources: dict[str, Any]) -> list[Item]:
    sites: list[dict[str, Any]] = sources.get("rss_sites") or []
    out: list[Item] = []

    for s in sites[:50]:
        name = s.get("name") or "RSS"
        category = s.get("category") or "その他"
        url = s.get("feed_url") or s.get("url") or ""
        if not url:
            continue
        feed_url = s.get("feed_url") or discover_feed_url(url)
        if not feed_url:
            continue
        try:
            entries = fetch_feed_items(feed_url)[:15]
        except Exception:
            continue
        for e in entries:
            out.append(
                Item(
                    source=name,
                    section=category,
                    title=normalize_space(e["title"]),
                    url=e["link"],
                    published=normalize_space(e.get("published", "")),
                )
            )
        time.sleep(0.2)
    return out


def collect_x_articles(sources: dict[str, Any], *, cookies_json: str | None) -> list[Item]:
    x = sources.get("x") or {}
    if not x.get("enabled", True):
        return []
    if not cookies_json:
        return []

    article_urls: list[str] = x.get("article_urls") or []
    authors: list[str] = x.get("article_authors") or []
    max_items = int(x.get("max_items", 10))

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return []

    def _to_playwright_cookies(any_data: Any) -> list[dict[str, Any]]:
        cookies_any = any_data if isinstance(any_data, list) else (any_data.get("cookies") if isinstance(any_data, dict) else None)
        if not isinstance(cookies_any, list):
            return []
        out: list[dict[str, Any]] = []
        for c in cookies_any:
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

            o: dict[str, Any] = {"name": name, "value": value, "domain": domain, "path": path}
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

    try:
        cookies = _to_playwright_cookies(json.loads(cookies_json))
        if not cookies:
            return []
    except Exception:
        return []

    found_urls: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        try:
            context.add_cookies(cookies)
        except Exception:
            pass
        page = context.new_page()

        def collect_from_page(url: str) -> list[str]:
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(1500)
                # Extract article URLs
                hrefs = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.map(e => e.getAttribute('href')).filter(Boolean)",
                )
                out2: list[str] = []
                for h in hrefs:
                    if "/i/articles/" in h:
                        out2.append(safe_urljoin("https://x.com/", h))
                # de-dup
                uniq: list[str] = []
                seen: set[str] = set()
                for u in out2:
                    if u in seen:
                        continue
                    seen.add(u)
                    uniq.append(u)
                return uniq[: max_items]
            except Exception:
                return []

        # 1) Explicit URLs
        for u in article_urls:
            if u.startswith("http"):
                found_urls.append(u)

        # 2) Authors pages (best-effort)
        for a in authors:
            a = a.lstrip("@").strip()
            if not a:
                continue
            # Try known route first; fallback to profile
            urls_to_try = [f"https://x.com/{a}/articles", f"https://x.com/{a}"]
            for tu in urls_to_try:
                got = collect_from_page(tu)
                if got:
                    found_urls.extend(got)
                    break

        browser.close()

    # Fetch titles (best-effort via Playwright by visiting each URL)
    out_items: list[Item] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        try:
            context.add_cookies(cookies)
        except Exception:
            pass
        page = context.new_page()
        for u in found_urls:
            if len(out_items) >= max_items:
                break
            try:
                page.goto(u, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(1500)
                title = page.title().strip()
                if not title:
                    title = "X Article"
                out_items.append(Item(source="X", section="X", title=normalize_space(title), url=u))
            except Exception:
                continue
        browser.close()

    # De-dup by URL
    uniq2: list[Item] = []
    seen2: set[str] = set()
    for it in out_items:
        if it.url in seen2:
            continue
        seen2.add(it.url)
        uniq2.append(it)
    return uniq2


def unique_items(items: Iterable[Item]) -> list[Item]:
    seen: set[str] = set()
    out: list[Item] = []
    for it in items:
        key = hashlib.sha256(f"{it.section}|{it.title}|{it.url}".encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def build_markdown(
    *,
    day: dt.date,
    sections: dict[str, list[Item]],
    base_keywords: list[str],
    keyword_weights: dict[str, float],
    max_featured: int,
    max_all: int,
) -> str:
    date_str = day.strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append(f"# トレンドネタ: {date_str}")
    lines.append("")

    for section_name in ["Product Hunt", "note", "資金調達", "Exit/M&A", "科学・哲学", "X"]:
        items = sections.get(section_name) or []
        if not items:
            continue

        scored: list[tuple[float, Item, str, str]] = []
        for it in items:
            score, stars, cat = compute_interest_score(it.title, base_keywords=base_keywords, keyword_weights=keyword_weights)
            scored.append((score, it, stars, cat))
        scored.sort(key=lambda x: (x[0], x[1].published, x[1].title), reverse=True)

        featured = scored[:max_featured]
        all_items = scored[:max_all]

        lines.append(f"## {section_name}")
        lines.append("")
        lines.append("### 注目トピック")
        lines.append("")
        lines.append("| タイトル | 指標 | 興味度 | カテゴリ | メモ |")
        lines.append("|---------|------|--------|----------|------|")
        for score, it, stars, cat in featured:
            metric = it.metric or (it.published[:10] if it.published else "")
            memo = f"{it.source}".strip()
            lines.append(f"| [{escape_md(it.title)}]({it.url}) | {escape_md(metric)} | {stars} | {escape_md(cat)} | {escape_md(memo)} |")
        lines.append("")

        lines.append("### 全エントリー")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>クリックで展開</summary>")
        lines.append("")
        for idx, (score, it, stars, cat) in enumerate(all_items, start=1):
            metric = it.metric or (it.published[:10] if it.published else "")
            extra = f"{metric} - {it.source}".strip(" -")
            lines.append(f"{idx}. [{escape_md(it.title)}]({it.url}) - {escape_md(extra)}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def escape_md(s: str) -> str:
    # Keep it minimal for tables
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def sectionize(items: list[Item]) -> dict[str, list[Item]]:
    sections: dict[str, list[Item]] = {}
    for it in items:
        sections.setdefault(it.section, []).append(it)

    # Map note into one section, and try to route RSS sites into specific sections
    # (already set by collectors)
    # Also attempt to put science/philosophy into that section by score later; here keep original.
    return sections


def ensure_dirs(sources: dict[str, Any]) -> Path:
    out_dir = (sources.get("output") or {}).get("dir") or "ideas/daily"
    p = REPO_ROOT / out_dir
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == content:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect daily trend and write ideas/daily/YYYYMMDD-trend.md")
    ap.add_argument("--sources", default=str(DEFAULT_SOURCES_PATH))
    ap.add_argument("--feedback", default=str(DEFAULT_FEEDBACK_PATH))
    ap.add_argument("--date", default="", help="Override date (YYYY-MM-DD) in JST")
    ap.add_argument("--force-note-init", action="store_true", help="Force re-init note users from followings_url")
    args = ap.parse_args()

    sources_path = Path(args.sources)
    feedback_path = Path(args.feedback)

    sources = read_json(sources_path)
    feedback = read_json(feedback_path) if feedback_path.exists() else {"version": 1, "liked_urls": [], "keyword_weights": {}, "last_updated": None}

    # Day (JST)
    if args.date:
        day = dt.date.fromisoformat(args.date)
    else:
        day = jst_today()

    # Update keyword weights from feedback (best-effort)
    keyword_weights = update_keyword_weights_from_feedback(feedback)
    feedback["keyword_weights"] = keyword_weights
    feedback["last_updated"] = dt.datetime.now(tz=JST).isoformat()
    write_json(feedback_path, feedback)

    # Initialize note users once, if configured
    sources, _note_users = ensure_note_users_initialized(sources, force=args.force_note_init)
    write_json(sources_path, sources)

    base_keywords = [
        "新規事業",
        "起業",
        "AI",
        "生成AI",
        "プロダクト",
        "資金調達",
        "M&A",
        "企業売却",
        "バイアウト",
        "仕事論",
        "科学",
        "哲学",
        "buyout",
        "acquisition",
        "startup",
    ]

    # Collect
    items: list[Item] = []
    items.extend(collect_producthunt(sources))
    items.extend(collect_note(sources))
    items.extend(collect_rss_sites(sources))
    items.extend(collect_x_articles(sources, cookies_json=os.environ.get("X_COOKIES_JSON")))
    items = unique_items(items)

    # Route RSS site sections by category already; additionally, put note into "note"
    sections = sectionize(items)

    # Build markdown
    out_dir = ensure_dirs(sources)
    yyyymmdd = day.strftime("%Y%m%d")
    out_path = out_dir / f"{yyyymmdd}-trend.md"
    max_featured = int((sources.get("output") or {}).get("max_featured_per_section", 5))
    max_all = int((sources.get("output") or {}).get("max_all_per_section", 20))
    md = build_markdown(
        day=day,
        sections=sections,
        base_keywords=base_keywords,
        keyword_weights=keyword_weights,
        max_featured=max_featured,
        max_all=max_all,
    )

    print("ネタ収集完了。")
    changed = write_if_changed(out_path, md)
    print(f"Wrote: {out_path.relative_to(REPO_ROOT)} (changed={changed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


