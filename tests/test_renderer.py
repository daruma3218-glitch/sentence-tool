#!/usr/bin/env python3
"""renderer.py (v3 Step1) の pytest。

受け入れ基準:
- サンプル chart_spec 10 種が、文字化けなく 1920x1080(16:9) PNG になる
- spec 不正（数値が無い等）は render_chart が False を返す（→ engine:ai へ降格）
"""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import renderer  # noqa: E402

KEIZAI_THEME = {
    "bg": "#FFFDF7", "main": "#1E40AF", "accent": "#C2410C",
    "grid": "#E5E7EB", "text": "#1F2937", "font": "NotoSansJP",
}

# ===== 10 種のサンプル spec（日本語ラベル必須）=====
SAMPLE_SPECS = [
    {"chart_type": "bar", "title": "軍事費の対GDP比", "unit": "%", "highlight_index": 0,
     "series": [{"label": "ロシア", "value": 6.3}, {"label": "NATO平均", "value": 2.1}],
     "source_note": "SIPRI 2025"},
    {"chart_type": "bar", "title": "各国の人口", "unit": "人",
     "series": [{"label": "日本", "value": 124000000}, {"label": "ドイツ", "value": 83000000},
                {"label": "フランス", "value": 68000000}]},
    {"chart_type": "line", "title": "実質GDP成長率の推移", "unit": "%",
     "series": [{"label": "2020年", "value": -4.5}, {"label": "2021年", "value": 2.6},
                {"label": "2022年", "value": 1.0}, {"label": "2023年", "value": 1.9}]},
    {"chart_type": "pie", "title": "発電の電源構成", "unit": "%", "highlight_index": 3,
     "series": [{"label": "石炭", "value": 30}, {"label": "天然ガス", "value": 25},
                {"label": "原子力", "value": 20}, {"label": "再生可能エネルギー", "value": 25}]},
    {"chart_type": "big_number", "title": "ロシアの軍事費（対GDP比）", "unit": "%",
     "series": [{"label": "2024年 推計", "value": 6.3}]},
    {"chart_type": "big_number", "title": "国境を接する国の数", "unit": "か国", "value": 14},
    {"chart_type": "comparison", "title": "軍事費 対GDP比の比較", "unit": "%", "highlight_index": 0,
     "series": [{"label": "ロシア", "value": 6.3}, {"label": "NATO平均", "value": 2.1}]},
    {"chart_type": "comparison", "title": "GDP規模（兆ドル）", "unit": "兆ドル",
     "series": [{"label": "日本", "value": 4.2}, {"label": "アメリカ", "value": 27.4},
                {"label": "中国", "value": 17.8}]},
    {"chart_type": "timeline", "title": "ロシアの東方進出の歴史",
     "series": [{"label": "1858年", "value": "アイグン条約"}, {"label": "1860年", "value": "北京条約"},
                {"label": "1945年", "value": "第二次大戦 終結"}]},
    {"chart_type": "line", "title": "貿易収支の推移", "unit": "億ドル",
     "series": [{"label": "1月", "value": 120}, {"label": "2月", "value": -45},
                {"label": "3月", "value": 88}, {"label": "4月", "value": 210}]},
]


def test_fonts_registered():
    """日本語フォントが登録されていること（= 文字化け＝豆腐を構造的に排除）。"""
    assert renderer.FONTS_AVAILABLE, "同梱フォントが登録されていない（文字化けの恐れ）"


@pytest.mark.parametrize("idx,spec", list(enumerate(SAMPLE_SPECS)))
def test_render_sample(idx, spec, tmp_path):
    out = tmp_path / f"chart_{idx}.png"
    ok = renderer.render_chart(spec, out, theme=KEIZAI_THEME)
    assert ok is True, f"spec[{idx}] ({spec['chart_type']}) の描画に失敗"
    assert out.exists() and out.stat().st_size > 2000, f"spec[{idx}] のPNGが空"
    with Image.open(out) as im:
        assert im.size == (1920, 1080), f"spec[{idx}] が16:9(1920x1080)でない: {im.size}"


@pytest.mark.parametrize("bad", [
    None,
    "not a dict",
    {"chart_type": "bar", "series": []},                       # 数値なし
    {"chart_type": "bar", "series": [{"label": "x", "value": "abc"}]},  # 数値化不能
    {"chart_type": "big_number"},                              # 値なし
    {"chart_type": "comparison", "series": [{"label": "a", "value": 1}]},  # 1値のみ
])
def test_degrade_on_bad_spec(bad, tmp_path):
    """不正な spec は False（→ engine:ai へ降格）。例外で落ちない。"""
    out = tmp_path / "bad.png"
    assert renderer.render_chart(bad, out, theme=KEIZAI_THEME) is False


def test_unknown_type_falls_back(tmp_path):
    """未知の chart_type でも数値があれば bar にフォールバックして描画。"""
    spec = {"chart_type": "donut3d", "title": "未知タイプ",
            "series": [{"label": "A", "value": 3}, {"label": "B", "value": 7}]}
    out = tmp_path / "u.png"
    assert renderer.render_chart(spec, out, theme=KEIZAI_THEME) is True
