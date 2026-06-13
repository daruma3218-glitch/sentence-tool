#!/usr/bin/env python3
"""巨大数字の表示崩れ対策（renderer）の pytest。

受け入れ基準:
- _fmt_num_jp: 大きな数を兆/億/万で短く正確に。端数は崩さずカンマにフォールバック
- _fit_fontsize: 幅に収まらない文字列はフォントを縮小、収まる文字列は基準サイズ
- 桁数が大きく違う comparison（例: 300円 vs 3億円）が崩れず 1920x1080 PNG になる
"""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import renderer  # noqa: E402

THEME = {
    "bg": "#FFFDF7", "main": "#1E40AF", "accent": "#C2410C",
    "grid": "#E5E7EB", "text": "#1F2937", "font": "NotoSansJP",
}


# ---------- _fmt_num_jp ----------

def test_fmt_num_jp_basic():
    assert renderer._fmt_num_jp(300000000) == "3億"
    assert renderer._fmt_num_jp(300) == "300"
    assert renderer._fmt_num_jp(1200000) == "120万"
    assert renderer._fmt_num_jp(1000000000000) == "1兆"
    assert renderer._fmt_num_jp(305000000) == "3億500万"
    assert renderer._fmt_num_jp(123450000) == "1億2345万"


def test_fmt_num_jp_falls_back_for_remainder():
    # 万の倍数でない端数は崩さずカンマ表記（正確さ優先）
    assert renderer._fmt_num_jp(1234000) == "1,234,000"
    assert renderer._fmt_num_jp(6.3) == "6.3"


def test_fmt_val_jp_with_unit():
    assert renderer._fmt_val_jp(300000000, "円") == "3億円"
    assert renderer._fmt_val_jp(300, "円") == "300円"


# ---------- _fit_fontsize ----------

def test_fit_fontsize_shrinks_long_text():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(renderer._W_IN, renderer._H_IN), dpi=renderer._DPI)
    try:
        base = 150.0
        long_size = renderer._fit_fontsize(fig, "0" * 40, 0.40, base, 20)
        short_size = renderer._fit_fontsize(fig, "3億円", 0.40, base, 20)
        assert long_size < base, "長い文字列はフォントを縮小すべき"
        assert long_size >= 20, "最小サイズは下回らない"
        assert short_size == base, "短い文字列は基準サイズのまま"
    finally:
        plt.close(fig)


# ---------- comparison（桁差が大きい）----------

def test_comparison_huge_number_renders(tmp_path):
    spec = {
        "chart_type": "comparison", "title": "宝くじの購入価格と1等賞金",
        "unit": "円", "highlight_index": 1,
        "series": [
            {"label": "購入価格（1枚）", "value": 300},
            {"label": "1等賞金", "value": 300000000},
        ],
    }
    out = tmp_path / "lottery.png"
    assert renderer.render_chart(spec, out, theme=THEME) is True
    with Image.open(out) as im:
        assert im.size == (1920, 1080)


def test_big_number_huge_renders(tmp_path):
    spec = {"chart_type": "big_number", "title": "1等賞金", "unit": "円", "value": 300000000}
    out = tmp_path / "big.png"
    assert renderer.render_chart(spec, out, theme=THEME) is True
    with Image.open(out) as im:
        assert im.size == (1920, 1080)


def test_clear_geo_cache_frees_memory():
    """地図キャッシュを読み込み→解放できる（512MB環境のOOM緩和）。空でも例外なし。"""
    # 空の状態で呼んでも落ちない
    renderer.clear_geo_cache()
    assert renderer._GEO_CACHE is None
    # 読み込み→解放→Noneを確認（geojson/shapely が無い環境でも _load_geo は {} を返す）
    renderer._load_geo()
    assert renderer._GEO_CACHE is not None  # ロード後はキャッシュが入る（{}含む）
    renderer.clear_geo_cache()
    assert renderer._GEO_CACHE is None  # 解放後は None（次回再読込）
