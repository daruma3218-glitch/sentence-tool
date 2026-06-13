#!/usr/bin/env python3
"""v3 Step5c: ルート違いフィードバック → ルーター few-shot 注入の pytest。

受け入れ基準:
- _route_chunk が few_shot を受け取ると、クエリに「過去の編集者フィードバック」と
  各事例（文・正しいルート）が含まれる（claude_query は monkeypatch で捕捉）
- few_shot 無し/空なら、その節はクエリに現れない
- pipeline._load_route_feedback が route_feedback.jsonl を読み、
  同じチャンネルだけ・新しい順に最大N件返す（壊れた行・他チャンネルは無視）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import router  # noqa: E402
from pipeline import SentencePipeline  # noqa: E402


def _capture_query(monkeypatch):
    """router.claude_query を差し替えて、組み立てられたクエリ文字列を捕捉する。"""
    captured = {}

    def fake_claude_query(client, query, system, **kw):
        captured["query"] = query
        captured["system"] = system
        return "[]"  # parse_json_array("[]") -> []（API は呼ばない）

    monkeypatch.setattr(router, "claude_query", fake_claude_query)
    return captured


def test_route_chunk_injects_fewshot(monkeypatch):
    captured = _capture_query(monkeypatch)
    rows = [{"no": 1, "sentence": "軍事費はGDP比6.3%に達した。"}]
    few_shot = [
        {"sentence": "軍事費はGDP比6.3%に達した。", "given_route": "diagram", "correct_route": "chart"},
        {"sentence": "シベリア鉄道が東西を結んだ。", "given_route": "illustration", "correct_route": "map"},
    ]
    router._route_chunk(None, rows, "テスト動画", few_shot=few_shot)
    q = captured["query"]
    assert "過去の編集者フィードバック" in q
    assert "chart" in q and "map" in q
    assert "シベリア鉄道" in q


def test_route_chunk_without_fewshot(monkeypatch):
    captured = _capture_query(monkeypatch)
    rows = [{"no": 1, "sentence": "これはテスト用の文です。"}]
    router._route_chunk(None, rows, "テスト動画", few_shot=None)
    assert "過去の編集者フィードバック" not in captured["query"]
    # 空リストでも節は出さない
    captured.clear()
    router._route_chunk(None, rows, "テスト動画", few_shot=[])
    assert "過去の編集者フィードバック" not in captured["query"]


def test_load_route_feedback_filters_and_limits(tmp_path):
    out_root = tmp_path / "output"
    out_root.mkdir()
    fb = out_root / "route_feedback.jsonl"

    lines = []
    # keizai を 15 件
    for i in range(15):
        lines.append(json.dumps({
            "channel_id": "keizai", "sentence": f"文{i}", "given_route": "diagram",
            "correct_route": "chart", "no": i, "date": "2026-06-13 00:00:00",
        }, ensure_ascii=False))
    # 別チャンネル default を 3 件（除外されるべき）
    for i in range(3):
        lines.append(json.dumps({
            "channel_id": "default", "sentence": f"別{i}", "given_route": "map",
            "correct_route": "illustration",
        }, ensure_ascii=False))
    # 壊れた行（無視されるべき）
    lines.append("{ this is broken json")
    fb.write_text("\n".join(lines) + "\n", encoding="utf-8")

    job_dir = out_root / "job_20260613_000000"
    pipe = SentencePipeline(manuscript_text="x" * 200, output_dir=job_dir, channel_id="keizai")

    fewshot = pipe._load_route_feedback(limit=12)
    assert len(fewshot) == 12, "最大件数で頭打ちになっていない"
    assert all(r["channel_id"] == "keizai" for r in fewshot), "他チャンネルが混ざっている"
    sents = {r["sentence"] for r in fewshot}
    assert "文14" in sents, "新しい記録が含まれていない"
    assert "文0" not in sents, "古い記録が切り落とされていない"


def test_load_route_feedback_missing_file(tmp_path):
    out_root = tmp_path / "output"
    out_root.mkdir()
    job_dir = out_root / "job_x"
    pipe = SentencePipeline(manuscript_text="x" * 200, output_dir=job_dir, channel_id="keizai")
    # ファイルが無くても落ちず空リスト
    assert pipe._load_route_feedback() == []
