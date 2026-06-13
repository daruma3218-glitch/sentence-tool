#!/usr/bin/env python3
"""v3 Step7: 原稿パイプライン final.json 直結の pytest。

受け入れ基準:
- parse_final_json: 本物の final.json（final が十分な長さの文字列）だけを dict と判定し、
  生原稿テキスト / final 無しJSON / 短い final / 壊れたJSON は None
- extract_from_final_json: final→本文 / tentative_title→タイトル / fact_report+reference_list→文脈
- extract_chart_specs(extra_context=...): 検証済み出典をプロンプトに注入（既定は注入しない＝挙動不変）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import router  # noqa: E402
from utils import parse_final_json, extract_from_final_json  # noqa: E402


_LONG = "ロシアの軍事費はGDP比6.3%に達した。" * 5  # 50文字以上の本文


# ---------- parse_final_json ----------

def test_valid_final_json():
    obj = {"phase": "E", "final": _LONG, "tentative_title": "ロシア経済の崩壊"}
    got = parse_final_json(json.dumps(obj, ensure_ascii=False))
    assert got is not None
    assert got["final"] == _LONG


def test_plain_text_is_not_final_json():
    assert parse_final_json("これはただの原稿テキストです。{ではじまらない。") is None
    assert parse_final_json(_LONG) is None


def test_json_without_final_key():
    assert parse_final_json(json.dumps({"foo": "bar", "baz": 1})) is None


def test_json_with_short_final():
    assert parse_final_json(json.dumps({"final": "短い"})) is None


def test_broken_json_starting_with_brace():
    assert parse_final_json('{ "final": "...' ) is None


def test_non_string_final():
    assert parse_final_json(json.dumps({"final": [1, 2, 3]})) is None


def test_empty_and_none():
    assert parse_final_json("") is None
    assert parse_final_json(None) is None


# ---------- extract_from_final_json ----------

def test_extract_fields():
    obj = {
        "final": _LONG,
        "tentative_title": "  ロシア経済の崩壊  ",
        "purpose": "視聴者に経済制裁の影響を伝える",
        "fact_report": "### ✅ 確認済\n- 軍事費GDP比6.3%: 出典 SIPRI 2025",
        "reference_list": ["https://sipri.org/...", "https://example.org/..."],
    }
    info = extract_from_final_json(obj)
    assert info["manuscript"] == _LONG
    assert info["title"] == "ロシア経済の崩壊"  # strip 済み
    assert info["purpose"] == "視聴者に経済制裁の影響を伝える"
    assert "SIPRI 2025" in info["fact_context"]
    assert "sipri.org" in info["fact_context"]  # reference_list も結合


def test_extract_missing_optionals():
    info = extract_from_final_json({"final": _LONG})
    assert info["manuscript"] == _LONG
    assert info["title"] == "" and info["purpose"] == "" and info["fact_context"] == ""


# ---------- extract_chart_specs(extra_context=...) ----------

def _capture_chart_query(monkeypatch):
    captured = {}

    def fake_claude_query(client, query, system, **kw):
        captured["query"] = query
        return "[]"

    monkeypatch.setattr(router, "claude_query", fake_claude_query)
    return captured


def test_chart_extract_injects_extra_context(monkeypatch):
    captured = _capture_chart_query(monkeypatch)
    rows = [{"no": 1, "sentence": "軍事費はGDP比6.3%。", "block_text": "出典 SIPRI 2025。"}]
    router.extract_chart_specs(None, rows, extra_context="検証済み: 軍事費6.3% (SIPRI 2025)")
    q = captured["query"]
    assert "検証済みの数値・出典情報" in q
    assert "SIPRI 2025" in q


def test_chart_extract_no_context_by_default(monkeypatch):
    captured = _capture_chart_query(monkeypatch)
    rows = [{"no": 1, "sentence": "軍事費はGDP比6.3%。", "block_text": ""}]
    router.extract_chart_specs(None, rows)  # extra_context 無し
    assert "検証済みの数値・出典情報" not in captured["query"]
