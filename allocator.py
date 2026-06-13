#!/usr/bin/env python3
"""allocator.py (v3 Step4) — ビート確定・タイムコード・画像予算の加重配分（LLM不使用）

2026-06-12 安福: max_diagrams の機械的な均等配置をやめ、ビート(視覚的まとまり)単位で
importance(重要度)に加重して画像予算を配分する。各文に推定タイムコード(est_start)も付与。

入力: rows(順序通り) + routes({no: {route, importance, beat, ...}})
出力: {no: {beat_id, est_start, display(image|hold|none), engine, importance}}
     + allocation.json（監査用）
LLM は一切使わない（決定論的）。
"""

import json
from pathlib import Path

# importance 4-5 のビートで、長い(≥この秒数)ものは progressive で複数枚に分割
_PROGRESSIVE_SEC = 25.0


def _fmt_tc(sec: float) -> str:
    s = int(round(max(0.0, sec)))
    return f"{s // 60:02d}:{s % 60:02d}"


def _route_engine(route: str) -> str:
    """route → engine の既定（pipeline の設定で上書きされ得るヒント）。"""
    if route in ("chart", "map"):
        return "render"
    if route == "web_photo":
        return "commons"
    if route == "skip":
        return "none"
    return "ai"


def _pick_image_rows(nos: list, n: int, importance_of) -> list:
    """ビート内から画像化する文を n 件選ぶ（importance 上位・同点は先頭）。表示は時系列順。"""
    if n <= 0:
        return []
    ranked = sorted(nos, key=lambda x: (-importance_of[x], nos.index(x)))
    picks = ranked[:n]
    return sorted(picks, key=lambda x: nos.index(x))  # 元の出現順に戻す


def allocate(rows: list, routes: dict, max_diagrams: int,
             chars_per_sec: float = 5.5, beat_mode: bool = True) -> dict:
    """ビート/タイムコード/表示(display)を決定する。

    戻り値: {no: {beat_id, est_start, display, engine, importance}}
    - beat_mode=True: importance 加重でビート単位に画像予算(max_diagrams)を配分。
      画像が付く文は display="image"、同ビートの他文は "hold"（前の画像を継続表示）、
      skip は "none"。
    - beat_mode=False: v2 互換。display は付けず（pipeline 側の均等間引きに委ねる）、
      beat_id / est_start のみ付与する。
    """
    chars_per_sec = max(1.0, float(chars_per_sec or 5.5))
    out = {}
    importance_of = {}
    sec_of = {}
    beat_rows = {}   # beat_id -> [no...]（順序保持）
    beat_id = -1
    cum = 0.0

    for r in rows:
        no = r["no"]
        rt = routes.get(no, {}) or {}
        route = rt.get("route", "illustration")
        imp = int(rt.get("importance", 3) or 3)
        importance_of[no] = imp
        sent = r.get("sentence", "") or ""
        sec = max(0.5, len(sent) / chars_per_sec)
        sec_of[no] = sec
        info = {
            "beat_id": None,
            "est_start": _fmt_tc(cum),
            "display": "none",
            "engine": _route_engine(route),
            "importance": imp,
        }
        cum += sec
        out[no] = info
        if route == "skip":
            continue
        beat = rt.get("beat", "continue")
        if beat == "new" or beat_id < 0:
            beat_id += 1
        info["beat_id"] = beat_id
        beat_rows.setdefault(beat_id, []).append(no)

    if not beat_mode:
        # v2 互換: display は pipeline の均等間引きに委ねる（ここでは付けない）
        return out

    # ===== ビート単位の重要度加重配分 =====
    beats = []
    for bid, nos in beat_rows.items():
        beats.append({
            "bid": bid, "nos": nos,
            "score": max(importance_of[n] for n in nos),
            "dur": sum(sec_of[n] for n in nos),
            "order": min(nos),
        })

    budget = max(0, int(max_diagrams))
    chosen = {}  # bid -> [image_nos]

    def _assign(beat, n):
        nonlocal budget
        n = min(n, len(beat["nos"]), budget)
        if n <= 0:
            return
        chosen[beat["bid"]] = _pick_image_rows(beat["nos"], n, importance_of)
        budget -= n

    order = sorted(beats, key=lambda b: (-b["score"], b["order"]))
    # 1) score 4-5: 必ず1枚（長ければ progressive で 2-3枚）
    for b in order:
        if budget <= 0:
            break
        if b["score"] >= 4:
            n = 1
            if b["dur"] >= _PROGRESSIVE_SEC:
                n = min(3, 1 + int(b["dur"] // _PROGRESSIVE_SEC))
            _assign(b, n)
    # 2) score 3: 予算が許す限り1枚
    for b in order:
        if budget <= 0:
            break
        if b["bid"] not in chosen and b["score"] == 3:
            _assign(b, 1)
    # 3) score 1-2: 余れば1枚
    for b in order:
        if budget <= 0:
            break
        if b["bid"] not in chosen:
            _assign(b, 1)

    # display 反映
    for bid, nos in beat_rows.items():
        img = set(chosen.get(bid, []))
        for n in nos:
            out[n]["display"] = "image" if n in img else "hold"

    return out


def write_allocation(path, rows: list, routes: dict, alloc: dict) -> None:
    """allocation.json（監査用）を書き出す。"""
    items = []
    for r in rows:
        no = r["no"]
        a = alloc.get(no, {})
        rt = routes.get(no, {}) or {}
        items.append({
            "no": no,
            "sentence": (r.get("sentence", "") or "")[:60],
            "route": rt.get("route", ""),
            "importance": a.get("importance"),
            "beat_id": a.get("beat_id"),
            "est_start": a.get("est_start"),
            "display": a.get("display"),
            "engine": a.get("engine"),
            "entities": rt.get("entities", []),
        })
    try:
        Path(path).write_text(
            json.dumps({"items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


# v3 Step6: エンティティ参照（一貫性ロック）。同じ被写体が 3 回以上 AI 画像に登場する
# とき、初出を canonical とし、後続は canonical 画像を参照画像にして見た目を揃える。
_ENTITY_MIN_OCCURRENCE = 3


def assign_entity_refs(image_nos: list, routes: dict,
                       min_occurrence: int = _ENTITY_MIN_OCCURRENCE) -> dict:
    """AI 画像化する文の並び(image_nos)から、エンティティ参照チェーンを決める（決定論）。

    入力:
      image_nos: 実際に AI 生成する文番号のリスト（時系列＝生成順の基準）
      routes:    {no: {entities: [...], ...}}
    出力:
      {no: {"role": "canonical"|"follower", "entity": str, "canon_no": int}}
      （チェーンに属さない文は結果に現れない）

    規則:
      - 各エンティティの登場回数を image_nos 内で数える
      - min_occurrence 回以上のものだけ対象
      - 優先度（登場回数の多い順 → 名前昇順）でグリーディに割り当て、
        各文は最大 1 チェーンにのみ属する（generator が参照する canonical は 1 つ）
      - チェーンは初出(最小 no)を canonical、それ以外を follower とする
      - 未割り当てが 2 件未満（canonical＋follower が作れない）のエンティティはスキップ
    """
    pos = {no: i for i, no in enumerate(image_nos)}  # 時系列順の安定ソート用

    # エンティティ → 登場 no リスト（image_nos の順序を維持）
    ent_rows: dict = {}
    for no in image_nos:
        ents = (routes.get(no, {}) or {}).get("entities", []) or []
        seen = set()
        for e in ents:
            if not isinstance(e, str):
                continue
            e = e.strip()
            if not e or e in seen:
                continue
            seen.add(e)
            ent_rows.setdefault(e, []).append(no)

    # 対象エンティティ（min_occurrence 以上）を優先度順に
    candidates = [(e, nos) for e, nos in ent_rows.items() if len(nos) >= min_occurrence]
    candidates.sort(key=lambda kv: (-len(kv[1]), kv[0]))

    assigned: dict = {}
    for entity, nos in candidates:
        free = [n for n in nos if n not in assigned]
        if len(free) < 2:
            continue  # canonical + follower が作れない
        free.sort(key=lambda n: pos.get(n, 0))  # 時系列順 → 初出が canonical
        canon = free[0]
        assigned[canon] = {"role": "canonical", "entity": entity, "canon_no": canon}
        for n in free[1:]:
            assigned[n] = {"role": "follower", "entity": entity, "canon_no": canon}

    return assigned
