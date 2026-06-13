#!/usr/bin/env python3
"""chart_spec 抽出の原文照合（ハルシネーション・ガード）の pytest。

数値が原文にあれば採用、無ければ降格(None→engine:ai) になることを確認する。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router import _chart_numbers_in_source  # noqa: E402


def test_numbers_present_pass():
    spec = {"series": [{"label": "ロシア", "value": 6.3}, {"label": "NATO平均", "value": 2.1}]}
    src = "ロシアの軍事費は対GDP比6.3%、NATO平均は2.1%である。"
    assert _chart_numbers_in_source(spec, src) is True


def test_hallucinated_numbers_degrade():
    """原文に無い数値（創作）は False → 降格。"""
    spec = {"series": [{"label": "A", "value": 999}, {"label": "B", "value": 888}]}
    src = "需要が増えると価格は上昇する。"
    assert _chart_numbers_in_source(spec, src) is False


def test_big_number_value_field():
    spec = {"value": 14}
    assert _chart_numbers_in_source(spec, "ロシアは14か国と国境を接する。") is True
    assert _chart_numbers_in_source(spec, "ロシアは多くの国と国境を接する。") is False


def test_comma_grouped_source():
    """原文がカンマ区切りでも照合できる。"""
    spec = {"series": [{"label": "人口", "value": 124000000}]}
    assert _chart_numbers_in_source(spec, "日本の人口は124,000,000人。") is True


def test_no_numbers_at_all():
    assert _chart_numbers_in_source({"series": []}, "数値のない文。") is False
