#!/usr/bin/env python3
"""commons_searcher.py (v3 Step3) — Wikimedia Commons 限定の画像取得

2026-06-12 安福: 収益化チャンネルの権利リスク対策。web_photo の取得元を
Wikimedia Commons API に限定し、**許可ライセンスのみ採用**＋ライセンス/クレジットを
構造化記録する。CC BY 系のクレジット表記義務に対応するため credits.txt も出す。

- API: commons.wikimedia.org/w/api.php (generator=search + imageinfo + extmetadata)
- 採用条件: LicenseShortName が許可リスト(Public domain / CC0 / CC BY / CC BY-SA)のみ。
  非営利(NC)・改変禁止(ND)は除外。
- 日本語クエリで 0 件なら英訳して再検索（Claude 小呼び出し・バッチ）。
- run_web_search_for_selections と同じ item_callback(info) 形で結果を返す
  （info に license / license_url / attribution / commons_page_url を追加）。
"""

import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from utils import claude_query, parse_json_object

_API = "https://commons.wikimedia.org/w/api.php"
_UA = "sentence-tsukuru/1.0 (economics-education; contact: noreply)"

# 許可ライセンス（LicenseShortName を小文字化して判定）。NC/ND は除外。
def _license_ok(short_name: str) -> bool:
    s = (short_name or "").strip().lower()
    if not s:
        return False
    if "-nc" in s or "-nd" in s or "noncommercial" in s:
        return False
    if s.startswith("cc0"):
        return True
    if s.startswith("cc by"):  # CC BY / CC BY-SA（NC/ND は上で除外済み）
        return True
    if "public domain" in s or s in ("pd", "pdm", "cc-pd-mark"):
        return True
    return False


def _strip_html(v: str) -> str:
    v = re.sub(r"<[^>]+>", "", v or "")
    return re.sub(r"\s+", " ", v).strip()


def _clean_attribution(ext: dict) -> str:
    """Artist + Credit を整形（HTML除去・重複排除）してクレジット文字列にする。"""
    def field(k):
        return _strip_html((ext.get(k) or {}).get("value", ""))
    parts = [p for p in (field("Artist"), field("Credit")) if p]
    # 重複・極端に長いものを抑制
    seen, out = set(), []
    for p in parts:
        p = p[:120]
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return " / ".join(out) or "Wikimedia Commons"


def search_commons_one(query: str, limit: int = 12, timeout: int = 20) -> Optional[dict]:
    """Commons を検索し、許可ライセンスの画像 1 枚のメタを返す（無ければ None）。"""
    q = (query or "").strip()
    if not q:
        return None
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": q, "gsrnamespace": "6", "gsrlimit": str(limit),
        "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": "1280",
    }
    url = _API + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    pages = (data.get("query") or {}).get("pages") or {}
    # 検索順（index）でソート
    items = sorted(pages.values(), key=lambda p: p.get("index", 9999))
    for p in items:
        ii_list = p.get("imageinfo") or []
        if not ii_list:
            continue
        ii = ii_list[0]
        mime = ii.get("mime", "")
        if not mime.startswith("image/") or mime == "image/svg+xml":
            continue  # 写真でないもの（SVG等）は除外
        ext = ii.get("extmetadata") or {}
        short = (ext.get("LicenseShortName") or {}).get("value", "")
        if not _license_ok(short):
            continue
        w, h = ii.get("width", 0) or 0, ii.get("height", 0) or 0
        if w < 400 or h < 300:
            continue  # 小さすぎ（ロゴ・アイコン）を除外
        return {
            "url": ii.get("url", ""),
            "thumb_url": ii.get("thumburl") or ii.get("url", ""),
            "license": short,
            "license_url": (ext.get("LicenseUrl") or {}).get("value", ""),
            "attribution": _clean_attribution(ext),
            "commons_page_url": ii.get("descriptionurl", ""),
            "title": (p.get("title", "") or "").replace("File:", ""),
        }
    return None


def _translate_queries(client, queries: list, log=None) -> dict:
    """日本語クエリを Commons 検索用の英語に一括翻訳（0件時の再検索用）。"""
    log = log or (lambda *a, **kw: None)
    queries = [q for q in dict.fromkeys(queries) if q]
    if not queries or client is None:
        return {}
    listing = "\n".join(f"- {q}" for q in queries)
    system = "あなたは画像検索クエリの翻訳係です。JSON オブジェクトのみ返す。"
    query = f"""次の日本語の画像検索クエリを、Wikimedia Commons 検索に適した簡潔な英語（固有名詞は英語表記）に訳してください。

対象:
{listing}

出力は JSON オブジェクトのみ:
{{"<日本語クエリ>": "<English query>", ...}}"""
    try:
        text = claude_query(client, query, system, max_tokens=2000)
        obj = parse_json_object(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def run_commons_search_for_selections(
    client, selections: list, max_workers: int = 8,
    log: Optional[Callable] = None, item_callback: Optional[Callable] = None,
) -> dict:
    """各 selection(query) で Commons を検索。日本語0件は英訳して再検索。

    item_callback(info) に結果を渡す。info = {no, thumb_url, source_url, source_title,
    topic, license, license_url, attribution, commons_page_url}。
    戻り値: {no: info}（採用できたもののみ）。
    """
    log = log or (lambda *a, **kw: None)
    cb = item_callback or (lambda info: None)
    if not selections:
        return {}

    def _emit(sel, res):
        info = {
            "no": sel["no"],
            "thumb_url": res["thumb_url"],
            "source_url": res["commons_page_url"],
            "source_title": res["title"],
            "topic": sel.get("topic", ""),
            "license": res["license"],
            "license_url": res["license_url"],
            "attribution": res["attribution"],
            "commons_page_url": res["commons_page_url"],
        }
        cb(info)
        return info

    results = {}
    pending = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(search_commons_one, s.get("query", "")): s for s in selections}
        for f in as_completed(futs):
            s = futs[f]
            try:
                res = f.result()
            except Exception:
                res = None
            if res:
                results[s["no"]] = _emit(s, res)
            else:
                pending.append(s)

    # 0 件だったものを英訳して再検索
    if pending:
        trans = _translate_queries(client, [s.get("query", "") for s in pending], log)
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futs = {}
            for s in pending:
                en = (trans.get(s.get("query", "")) or "").strip()
                if en and en != s.get("query", ""):
                    futs[ex.submit(search_commons_one, en)] = s
            for f in as_completed(futs):
                s = futs[f]
                try:
                    res = f.result()
                except Exception:
                    res = None
                if res:
                    results[s["no"]] = _emit(s, res)

    log("websearch",
        f"Commons: {len(results)}/{len(selections)} 件取得（許可ライセンスのみ・"
        f"英訳再検索含む）")
    return results


def build_credits_text(items: list) -> str:
    """採用した Commons 画像のクレジット一覧（概要欄貼り付け用）を組み立てる。

    items: [{title, attribution, license, commons_page_url}, ...]
    """
    lines = [
        "■ 画像クレジット（Wikimedia Commons）",
        "本動画で使用した画像のライセンス・出典です。",
        "",
    ]
    seen = set()
    n = 0
    for it in items:
        page = (it.get("commons_page_url") or "").strip()
        lic = (it.get("license") or "").strip()
        if not page or not lic:
            continue
        key = page
        if key in seen:
            continue
        seen.add(key)
        n += 1
        title = (it.get("title") or it.get("source_title") or "").strip() or "(file)"
        attr = (it.get("attribution") or "").strip()
        lines.append(f"{n}. {title}")
        if attr:
            lines.append(f"   作者/クレジット: {attr}")
        lines.append(f"   ライセンス: {lic}")
        lines.append(f"   出典: {page}")
        lines.append("")
    if n == 0:
        lines.append("（Wikimedia Commons の画像は使用していません）")
    return "\n".join(lines)
