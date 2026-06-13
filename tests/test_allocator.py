#!/usr/bin/env python3
"""allocator.py（v3 Step4）の pytest（LLM不使用・決定論）。

受け入れ基準:
- 250 文で beat_id / est_start / display が全行に付く
- 画像配分が max_diagrams を超えない
- beat_mode=False で v2 互換（display を付けない）
- importance が高いビートが優先される
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from allocator import allocate  # noqa: E402

_ROUTES = ["chart", "map", "diagram", "illustration", "web_photo", "skip"]


def _make(n=250):
    rows, routes = [], {}
    for i in range(1, n + 1):
        rows.append({"no": i, "sentence": "これはテスト用のセンテンスです。" * ((i % 3) + 1)})
        routes[i] = {
            "route": _ROUTES[i % len(_ROUTES)],
            "importance": (i % 5) + 1,
            "beat": "new" if i % 4 == 0 else "continue",
            "entities": [],
        }
    return rows, routes


def test_all_rows_get_fields_250():
    rows, routes = _make(250)
    a = allocate(rows, routes, max_diagrams=100, chars_per_sec=5.5, beat_mode=True)
    assert len(a) == 250
    for r in rows:
        info = a[r["no"]]
        assert ":" in info["est_start"], "est_start が mm:ss でない"
        if routes[r["no"]]["route"] == "skip":
            assert info["display"] == "none" and info["beat_id"] is None
        else:
            assert info["beat_id"] is not None, "非skip文に beat_id が無い"
            assert info["display"] in ("image", "hold"), "display が image/hold でない"


def test_budget_not_exceeded():
    rows, routes = _make(250)
    for mx in (5, 20, 50, 100):
        a = allocate(rows, routes, max_diagrams=mx, beat_mode=True)
        imgs = sum(1 for v in a.values() if v["display"] == "image")
        assert imgs <= mx, f"配分 {imgs} 枚 > 上限 {mx} 枚"


def test_beat_mode_false_v2_compat():
    rows, routes = _make(40)
    a = allocate(rows, routes, max_diagrams=20, beat_mode=False)
    # v2 互換: display は付けない（none のまま）。beat_id / est_start は付く。
    for r in rows:
        info = a[r["no"]]
        assert info["display"] == "none"
        assert ":" in info["est_start"]
        if routes[r["no"]]["route"] != "skip":
            assert info["beat_id"] is not None


def test_high_importance_prioritized():
    # 予算1。importance5 のビートと importance1 のビート → 5 が画像になる。
    rows = [
        {"no": 1, "sentence": "重要な核心の主張です。"},
        {"no": 2, "sentence": "ただの繋ぎの文。"},
    ]
    routes = {
        1: {"route": "diagram", "importance": 5, "beat": "new", "entities": []},
        2: {"route": "diagram", "importance": 1, "beat": "new", "entities": []},
    }
    a = allocate(rows, routes, max_diagrams=1, beat_mode=True)
    assert a[1]["display"] == "image"
    assert a[2]["display"] == "hold"


def test_timecode_accumulates():
    rows = [{"no": 1, "sentence": "あ" * 55}, {"no": 2, "sentence": "い" * 55}]
    routes = {1: {"route": "diagram", "importance": 3, "beat": "new"},
              2: {"route": "diagram", "importance": 3, "beat": "continue"}}
    a = allocate(rows, routes, max_diagrams=5, chars_per_sec=5.5, beat_mode=True)
    assert a[1]["est_start"] == "00:00"
    assert a[2]["est_start"] == "00:10"  # 55/5.5 = 10秒
