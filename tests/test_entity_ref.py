#!/usr/bin/env python3
"""v3 Step6: エンティティ参照（一貫性ロック）の pytest。

受け入れ基準（allocator.assign_entity_refs / 決定論）:
- 3 回以上登場する被写体は canonical 1 + follower N に割り当てられる（初出が canonical）
- 3 回未満の被写体は割り当てられない
- 同じ文に複数の対象被写体があるとき、登場回数の多い方が優先（グリーディ）
- 入力が同じなら結果も同じ（決定論）

受け入れ基準（generator: canonical→follower の依存順生成 / nanobanana）:
- canonical が follower より先に生成される
- follower は canonical の生成画像（バイト列）を参照として受け取る
- canonical 自身は参照なし。デッドロックせず全件完了する
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from allocator import assign_entity_refs  # noqa: E402
from generator import ParallelImageGenerator, PROVIDER_NANOBANANA  # noqa: E402


# ---------- allocator.assign_entity_refs ----------

def _routes(mapping):
    return {no: {"entities": ents} for no, ents in mapping.items()}


def test_three_occurrences_make_one_canonical():
    image_nos = [1, 2, 3, 4]
    routes = _routes({1: ["ロシア"], 2: ["ロシア"], 3: ["欧州"], 4: ["ロシア"]})
    a = assign_entity_refs(image_nos, routes)
    # ロシアは3回 → 1,2,4 がチェーン。初出1がcanonical、2と4がfollower
    assert a[1]["role"] == "canonical" and a[1]["entity"] == "ロシア"
    assert a[2]["role"] == "follower" and a[2]["canon_no"] == 1
    assert a[4]["role"] == "follower" and a[4]["canon_no"] == 1
    # 欧州は1回 → 割り当てなし
    assert 3 not in a


def test_under_threshold_not_assigned():
    image_nos = [1, 2, 3]
    routes = _routes({1: ["A"], 2: ["A"], 3: ["B"]})  # Aは2回のみ
    a = assign_entity_refs(image_nos, routes)
    assert a == {}


def test_greedy_priority_higher_count_wins():
    # 文3 は X と Y の両方を持つ。X は4回、Y は3回 → 文3 は X チェーンに入る
    image_nos = [1, 2, 3, 4, 5, 6]
    routes = _routes({
        1: ["X"], 2: ["X"], 3: ["X", "Y"], 4: ["X"],
        5: ["Y"], 6: ["Y"],
    })
    a = assign_entity_refs(image_nos, routes)
    # X(4回): 1 canonical, 2/3/4 follower
    assert a[1]["role"] == "canonical" and a[1]["entity"] == "X"
    assert a[3]["entity"] == "X"  # 高頻度Xが文3を取る
    # Y は 5,6 しか残らない（文3はXが取った）→ 2件で canonical+follower 成立
    assert a[5]["entity"] == "Y" and a[5]["role"] == "canonical"
    assert a[6]["entity"] == "Y" and a[6]["role"] == "follower" and a[6]["canon_no"] == 5


def test_deterministic():
    image_nos = [1, 2, 3, 4, 5]
    routes = _routes({1: ["国"], 2: ["国"], 3: ["国"], 4: ["国"], 5: ["別"]})
    a1 = assign_entity_refs(image_nos, routes)
    a2 = assign_entity_refs(image_nos, routes)
    assert a1 == a2


# ---------- generator: canonical→follower 依存順生成 ----------

def _tiny_png(path):
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)


def test_generator_canonical_before_followers(tmp_path, monkeypatch):
    gen = ParallelImageGenerator(provider=PROVIDER_NANOBANANA, gemini_api_key="dummy", concurrency=4)

    order = []
    calls = {}

    def fake_dispatch(full_prompt, output_path, use_reference=False,
                      ref_bytes_override=None, ref_mime_override=None):
        idx = int(Path(output_path).stem)
        order.append(idx)
        calls[idx] = {
            "has_ref": ref_bytes_override is not None,
            "ref_len": len(ref_bytes_override) if ref_bytes_override else 0,
            "prompt_has_lock": "CONSISTENCY REFERENCE" in full_prompt,
        }
        _tiny_png(output_path)  # canonical の read_bytes() を成立させる
        return True, ""

    monkeypatch.setattr(gen, "_dispatch_sync_generate", fake_dispatch)

    prompts = [
        {"index": 1, "prompt": "p1", "type": "illustration",
         "entity_role": "canonical", "entity_name": "ロシア"},
        {"index": 2, "prompt": "p2", "type": "illustration",
         "entity_role": "follower", "entity_ref_of": 1, "entity_name": "ロシア"},
        {"index": 3, "prompt": "p3", "type": "illustration",
         "entity_role": "follower", "entity_ref_of": 1, "entity_name": "ロシア"},
    ]
    results = asyncio.run(gen.generate_all(prompts, tmp_path))

    assert all(r["success"] for r in results), "全件成功すべき"
    # canonical(1) が両 follower より先
    assert order.index(1) < order.index(2)
    assert order.index(1) < order.index(3)
    # follower は canonical のバイト列を参照に受け取り、ロック文言が入る
    assert calls[2]["has_ref"] and calls[2]["ref_len"] > 0 and calls[2]["prompt_has_lock"]
    assert calls[3]["has_ref"] and calls[3]["ref_len"] > 0 and calls[3]["prompt_has_lock"]
    # canonical 自身は参照なし
    assert not calls[1]["has_ref"]


def test_generator_follower_proceeds_if_canonical_fails(tmp_path, monkeypatch):
    """canonical が失敗しても follower は無限待機せず、参照なしで進む（壊さない）。"""
    gen = ParallelImageGenerator(provider=PROVIDER_NANOBANANA, gemini_api_key="dummy", concurrency=4)

    def fake_dispatch(full_prompt, output_path, use_reference=False,
                      ref_bytes_override=None, ref_mime_override=None):
        idx = int(Path(output_path).stem)
        if idx == 1:
            return False, "canonical failed"  # 画像は書かない
        _tiny_png(output_path)
        return True, ""

    monkeypatch.setattr(gen, "_dispatch_sync_generate", fake_dispatch)

    prompts = [
        {"index": 1, "prompt": "p1", "type": "illustration",
         "entity_role": "canonical", "entity_name": "E"},
        {"index": 2, "prompt": "p2", "type": "illustration",
         "entity_role": "follower", "entity_ref_of": 1, "entity_name": "E"},
    ]
    results = asyncio.run(gen.generate_all(prompts, tmp_path))
    by_idx = {r["index"]: r for r in results}
    assert by_idx[2]["success"] is True  # follower は参照なしでも成功
