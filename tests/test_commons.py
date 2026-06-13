#!/usr/bin/env python3
"""commons_searcher.py（v3 Step3）の pytest。

受け入れ基準:
- ライセンス許可リスト外（NC/ND/全権利留保）の画像が採用されない（APIモック）
- credits.txt（クレジット一覧）が組み立てられる
"""
import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import commons_searcher as cs  # noqa: E402


def _fake_urlopen(payload):
    class _R:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps(payload).encode("utf-8")
    return _R()


def _page(idx, title, lic, w=1200, h=800, artist=None, mime="image/jpeg"):
    ext = {"LicenseShortName": {"value": lic}}
    if artist:
        ext["Artist"] = {"value": artist}
    return {
        "index": idx, "title": f"File:{title}",
        "imageinfo": [{
            "mime": mime, "url": f"http://x/{title}", "thumburl": f"http://x/t_{title}",
            "width": w, "height": h, "descriptionurl": f"http://commons/{title}",
            "extmetadata": ext,
        }],
    }


def test_license_allowlist():
    for ok in ("CC BY 4.0", "CC BY-SA 3.0", "CC0", "Public domain", "CC0 1.0"):
        assert cs._license_ok(ok), ok
    for bad in ("CC BY-NC 4.0", "CC BY-ND 4.0", "CC BY-NC-SA 3.0",
                "All rights reserved", "Fair use", ""):
        assert not cs._license_ok(bad), bad


def test_rejects_disallowed_picks_allowed():
    """1件目が NC(不許可)・2件目が CC BY-SA(許可) → 許可の2件目が採用される。"""
    pages = {
        "1": _page(1, "Bad.jpg", "CC BY-NC 4.0"),
        "2": _page(2, "Good.jpg", "CC BY-SA 4.0", artist="<a href='x'>Jane Doe</a>"),
    }
    payload = {"query": {"pages": pages}}
    with mock.patch.object(cs.urllib.request, "urlopen", return_value=_fake_urlopen(payload)):
        r = cs.search_commons_one("test")
    assert r is not None
    assert r["license"] == "CC BY-SA 4.0"
    assert "Good" in r["commons_page_url"]
    assert r["attribution"] == "Jane Doe"  # HTML タグが除去される


def test_all_disallowed_returns_none():
    pages = {
        "1": _page(1, "A.jpg", "All rights reserved"),
        "2": _page(2, "B.jpg", "CC BY-ND 4.0"),
    }
    payload = {"query": {"pages": pages}}
    with mock.patch.object(cs.urllib.request, "urlopen", return_value=_fake_urlopen(payload)):
        assert cs.search_commons_one("test") is None


def test_too_small_rejected():
    """小さすぎる画像（ロゴ等）は許可ライセンスでも不採用。"""
    pages = {"1": _page(1, "Logo.png", "CC0", w=120, h=80, mime="image/png")}
    payload = {"query": {"pages": pages}}
    with mock.patch.object(cs.urllib.request, "urlopen", return_value=_fake_urlopen(payload)):
        assert cs.search_commons_one("test") is None


def test_credits_text():
    items = [
        {"title": "K.jpg", "attribution": "John", "license": "CC BY-SA 4.0",
         "commons_page_url": "http://commons/K"},
        {"no": 5},  # license/page 無し → スキップ
    ]
    t = cs.build_credits_text(items)
    assert "CC BY-SA 4.0" in t
    assert "John" in t
    assert "http://commons/K" in t
