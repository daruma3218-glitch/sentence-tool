#!/usr/bin/env python3
"""renderer.py の map 部（v3 Step2）の pytest。

受け入れ基準:
- highlight / route / neighbors の 3 型が 1920x1080 PNG にレンダリングされる
- 未知の国コード（GeoJSON に無い）は render_map が False を返す（→ illustration 降格）
"""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import renderer  # noqa: E402

T = {"bg": "#FFFDF7", "main": "#1E40AF", "accent": "#C2410C",
     "grid": "#E5E7EB", "text": "#1F2937", "font": "NotoSansJP"}

MAP_SPECS = [
    {"map_type": "highlight", "title": "ロシア連邦",
     "focus_countries": ["RUS"], "extent": "former_ussr"},
    {"map_type": "route", "title": "ロシアからのガス輸出",
     "focus_countries": ["RUS"], "secondary_countries": ["DEU"], "extent": "europe",
     "arrows": [{"from": "RUS", "to": "DEU", "label": "ガス輸出"}],
     "labels": [{"country": "RUS", "text": "ロシア"}, {"country": "DEU", "text": "ドイツ"}]},
    {"map_type": "neighbors", "title": "ロシアと周辺国",
     "focus_countries": ["RUS"], "secondary_countries": ["UKR", "BLR", "FIN", "KAZ"],
     "extent": "europe"},
]


def test_geo_loaded():
    """同梱 GeoJSON が読み込め、主要国が引けること。"""
    geo = renderer._load_geo()
    assert geo, "GeoJSON が読めない"
    for iso in ("RUS", "DEU", "UKR", "JPN", "USA", "CHN"):
        assert iso in geo, f"{iso} が GeoJSON に無い"


@pytest.mark.parametrize("idx,spec", list(enumerate(MAP_SPECS)))
def test_render_map(idx, spec, tmp_path):
    out = tmp_path / f"map_{idx}.png"
    assert renderer.render_map(spec, out, theme=T) is True, f"map[{idx}] 描画失敗"
    assert out.exists() and out.stat().st_size > 3000
    with Image.open(out) as im:
        assert im.size == (1920, 1080), f"map[{idx}] が16:9でない: {im.size}"


@pytest.mark.parametrize("bad", [
    None,
    "not a dict",
    {"map_type": "highlight", "focus_countries": []},          # focus 空
    {"map_type": "highlight", "focus_countries": ["XXX", "ZZZ"]},  # 未知コード
])
def test_map_degrade(bad, tmp_path):
    """解決できない map_spec は False（→ illustration 降格）。例外で落ちない。"""
    assert renderer.render_map(bad, tmp_path / "bad.png", theme=T) is False
