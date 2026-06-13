#!/usr/bin/env python3
"""Phase 2 (v2): ルーター Agent

全センテンスを受け取り、各文に最適な route を 1 つ付与する。
DESIGN_v2.md 第 3 章の仕様に準拠。

route 種別:
  web_photo    : 実在の歴史人物・事件・建造物 → Web検索で本物の写真URL
  map          : 地理・国境・領土・移動経路 → AI生成（航空写真風）
  diagram      : 概念・仕組み・因果・対比 → AI生成（図解）
  chart        : 数値・統計・割合・推移 → AI生成（グラフ）
  illustration : 抽象シーン・比喩・心情 → AI生成（イラスト）
  skip         : 接続詞・挨拶・繋ぎ → 画像なし
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import anthropic

from utils import claude_query, parse_json_array


CLAUDE_MODEL = "claude-sonnet-4-6"
CHUNK_SIZE = 40  # 1 リクエストで分類する文数

VALID_ROUTES = ("web_photo", "realphoto", "map", "diagram", "chart", "illustration", "skip")
AI_ROUTES = ("realphoto", "map", "diagram", "chart", "illustration")  # AI生成班が担当


# ===== v3 Step4: importance/entities/beat の正規化 =====
def _clamp_importance(v) -> int:
    """importance を 1〜5 に正規化（不正なら 3）。"""
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return 3


def _clean_entities(v) -> list:
    """entities を文字列・最大3件に正規化。"""
    if not isinstance(v, list):
        return []
    out = []
    for e in v:
        if isinstance(e, str) and e.strip():
            out.append(e.strip()[:30])
    return out[:3]


def _route_chunk(
    client: anthropic.Anthropic,
    rows_chunk: list,
    title: str,
    user_instructions: str = "",
    propaganda_mix: bool = False,
    few_shot: list = None,
) -> list:
    """1 チャンク（最大 CHUNK_SIZE 文）を route 分類する。

    propaganda_mix=True の場合、各文に propaganda (true/false) も付与する
    （ドラマチック/歴史的/政治的に重い文だけプロパガンダ様式に昇格させるため）。
    """
    inputs = [
        {
            "no": r["no"],
            "chapter": r.get("chapter_title", ""),
            "sentence": r.get("sentence", "")[:200],
        }
        for r in rows_chunk
    ]
    inputs_json = json.dumps(inputs, ensure_ascii=False, indent=2)

    user_block = ""
    if user_instructions.strip():
        user_block = f"\n【ユーザーからの指示（最優先）】\n{user_instructions.strip()}\n"

    propaganda_block = ""
    propaganda_field = ""
    if propaganda_mix:
        propaganda_block = (
            "\n【プロパガンダ判定（propaganda_mix モード）】\n"
            "各文について、ソ連プロパガンダ風の様式が映えるかを propaganda(true/false) で判定すること:\n"
            "- true: 歴史的転換点・権力者・イデオロギー・国家の興亡・思想対立・"
            "ドラマチックで重い歴史的瞬間（例: 革命、粛清、冷戦、指導者の決断、帝国の崩壊）\n"
            "- false: 中立的な説明・数値の解説・一般的な概念・軽い繋ぎ（通常様式のままが良い）\n"
            "skip の文は propaganda=false でよい。\n"
        )
        propaganda_field = ',\n    "propaganda": true/false（上記基準で判定）'

    # v3 Step5: 過去の編集者フィードバック（ルート違い）を few-shot として注入
    few_shot_block = ""
    if few_shot:
        lines = []
        for ex in few_shot[:12]:
            s = (ex.get("sentence", "") or "")[:50]
            g = ex.get("given_route", "")
            c = ex.get("correct_route", "")
            if s and c:
                lines.append(f"- 「{s}」は {g} ではなく {c} が正しい")
        if lines:
            few_shot_block = (
                "\n【過去の編集者フィードバック（同じ判定ミスを避けること）】\n"
                + "\n".join(lines) + "\n"
            )

    system = (
        "あなたは動画ディレクターです。原稿の各センテンスに、最適な画像ソースの種別（route）を"
        "1 つ割り当てます。結果は必ず JSON 配列のみで返してください。"
    )

    query = f"""動画「{title}」の各センテンスに、最適な画像ソース種別（route）を 1 つ割り当ててください。

センテンス一覧:
{inputs_json}
{user_block}{propaganda_block}{few_shot_block}

【route の種別と判定基準】
1. web_photo … **実在の特定の**歴史人物・事件・建造物で「本物の写真/絵画」が見たいもの
   例:「スターリンは大粛清を断行した」「ニクソンが訪中した」「ベルリンの壁が崩壊した」
2. realphoto … **都市・建物・施設・インフラ・事件・戦争・人々の生活**などの実写風シーン。
   特定の歴史的瞬間ではないが「リアルな写真」が合うもの（AIで実写風生成）。
   例:「天然ガス施設が稼働していた」「この航路が貿易を支えた」「政府庁舎前に市民が集まった」
   「工場は閑古鳥が鳴いていた」「住民が法案に抗議した」「夜の都市に灯りがともる」
3. map … 地理的位置・国境・領土・移動経路・地名どうしの関係
   例:「ソ連は14か国と国境を接していた」「シベリア鉄道が東西を結んだ」
4. diagram … 概念・仕組み・因果関係・対比・フロー（数値が主役でないもの）
   例:「大陸国家と海洋国家では戦略が異なる」「三権分立の仕組み」
5. chart … 数値・統計・割合・推移・比較データが主役
   例:「軍事費はGDP比6.3%に達した」「人口は3億人を超えた」
6. illustration … 抽象的シーン・比喩・心情・一般的な描写（実在の物でも実写でもない）
   例:「緊張が静かに高まっていった」「希望の光が差した」
7. skip … 接続詞・挨拶・問いかけ・内容のない繋ぎ（視覚化する意味がない）
   例:「では、見ていきましょう」「皆さんこんにちは」「次の章へ進みます」

【判定のコツ】
- **特定の**実在人物・事件（本物の写真でないと意味がない）→ web_photo
- 都市・建物・施設・事件・戦争・生活のシーン（実写が映えるが特定の1枚でなくてよい）→ realphoto
- 「本物が見たい」より「概念・仕組みを説明したい」なら diagram
- 数字が主役なら chart
- 物理的シーンでなく抽象・比喩なら illustration
- 繋ぎ・挨拶は遠慮なく skip（あとで均等配置の対象から外れる）

【v3: 各文に importance / entities / beat も付与すること】
- importance: 1〜5（動画の主張にとっての重要度）。5=章の核心主張・驚きのデータ / 3=主張を支える説明 / 1=繋ぎに近い
- entities: 繰り返し描かれ得る被写体（国・地域・繰り返す概念・組織）。最大3つ。無ければ []
- beat: "new"（話題・被写体が直前の文から切り替わった＝新しい視覚的まとまり） / "continue"（直前と同じまとまり）。
  各チャンクの先頭文は "new" でよい。

【出力 JSON（必ずこの形式のみ）】
[
  {{
    "no": 1,
    "route": "web_photo",
    "reason": "判定理由を15字以内で",
    "search_query": "Web検索クエリ（web_photoのときのみ、日本語30字以内、固有名詞を含む）",
    "topic": "トピック名（web_photoのときのみ、10〜20字）",
    "importance": 4, "entities": ["ロシア"], "beat": "new"{propaganda_field}
  }},
  {{
    "no": 2,
    "route": "diagram",
    "reason": "概念の対比のため",
    "importance": 3, "entities": [], "beat": "continue"
  }}
]

必ず {len(rows_chunk)} 件すべてに route / importance / entities / beat を付与すること。JSON 配列のみ返す。"""

    result = claude_query(client, query, system, max_tokens=8000, model=CLAUDE_MODEL)
    parsed = parse_json_array(result)
    return parsed


def route_all_sentences(
    client: anthropic.Anthropic,
    rows: list,
    title: str,
    user_instructions: str = "",
    propaganda_mix: bool = False,
    max_workers: int = 4,
    log: Optional[Callable] = None,
    few_shot: list = None,
) -> dict:
    """全センテンスを route 分類する。

    戻り値: {no: {"route", "reason", "search_query", "topic", "propaganda"}}
    propaganda_mix=True のとき各文に propaganda(bool) が入る。
    """
    log = log or (lambda *a, **kw: None)

    # チャンク分割
    chunks = [rows[i:i + CHUNK_SIZE] for i in range(0, len(rows), CHUNK_SIZE)]
    log("router", f"{len(rows)} 文を {len(chunks)} チャンクに分割して分類（同時 {max_workers} 並列）"
                  + ("／プロパガンダ・ミックス ON" if propaganda_mix else ""))

    routes_by_no: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_route_chunk, client, chunk, title, user_instructions, propaganda_mix, few_shot): i
            for i, chunk in enumerate(chunks)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results = future.result()
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    no = item.get("no")
                    route = item.get("route", "")
                    if no is None or route not in VALID_ROUTES:
                        continue
                    routes_by_no[no] = {
                        "route": route,
                        "reason": item.get("reason", ""),
                        "search_query": item.get("search_query", ""),
                        "topic": item.get("topic", ""),
                        "propaganda": bool(item.get("propaganda", False)) if propaganda_mix else False,
                        # v3 Step4
                        "importance": _clamp_importance(item.get("importance")),
                        "entities": _clean_entities(item.get("entities")),
                        "beat": "new" if item.get("beat") == "new" else "continue",
                    }
                completed += 1
                log("router", f"チャンク {completed}/{len(chunks)} 分類完了")
            except Exception as e:
                log("error", f"ルーターチャンク {idx} 失敗: {str(e)[:120]}")

    # フォールバック: 未分類の文は illustration 扱い
    for r in rows:
        no = r["no"]
        if no not in routes_by_no:
            routes_by_no[no] = {
                "route": "illustration",
                "reason": "（自動フォールバック）",
                "search_query": "",
                "topic": "",
                "propaganda": False,
                "importance": 2, "entities": [], "beat": "new",
            }

    # 集計ログ
    from collections import Counter
    counts = Counter(v["route"] for v in routes_by_no.values())
    summary = " / ".join(f"{k}:{counts.get(k, 0)}" for k in VALID_ROUTES)
    if propaganda_mix:
        prop_count = sum(1 for v in routes_by_no.values() if v.get("propaganda"))
        summary += f"  [プロパガンダ昇格: {prop_count}]"
    log("router", f"分類結果: {summary}")

    return routes_by_no


# ===== v3 Step1: chart_spec 抽出（router 第2段・LLM使用）=====
# 2026-06-12 安福: chart を matplotlib で正確描画するため、原稿の数値を構造化抽出する。
# 数値は原文にあるものだけ。抽出後にコードで原文照合し、ハルシネーションは降格させる。
_CHART_TYPES = ("bar", "line", "pie", "big_number", "comparison", "timeline")


def _chart_numbers_in_source(spec: dict, source_text: str) -> bool:
    """spec の数値が原文(文+block_context)に部分一致するか（ハルシネーション・ガード）。

    数値の半数以上が原文に見当たらなければ False（呼び出し側で chart→diagram 降格）。
    区切り(カンマ/空白)は無視して照合する。万/億等の表記揺れは許容寄り。
    """
    import re as _re
    src = _re.sub(r"[,\s　]", "", source_text or "")
    vals = []
    for it in (spec.get("series") or []):
        if isinstance(it, dict) and it.get("value") is not None:
            vals.append(it.get("value"))
    if spec.get("value") is not None:
        vals.append(spec.get("value"))
    nums = []
    for v in vals:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return False
    matched = 0
    for n in nums:
        cand = set()
        s = f"{n:.4f}".rstrip("0").rstrip(".")
        cand.add(s)
        cand.add(s.replace(".", ""))
        if n == int(n):
            cand.add(str(int(n)))
        if any(c and c in src for c in cand):
            matched += 1
    return matched * 2 >= len(nums)


def extract_chart_specs(client, chart_rows: list, log: Optional[Callable] = None,
                        extra_context: str = "") -> dict:
    """route=chart の文から chart_spec を抽出する（router 第2段）。

    数値は文と block_context にあるものだけ。創作禁止。抽出不能や数値が原文に
    無い場合は None を返す（呼び出し側で chart→diagram(engine:ai) へ降格）。
    extra_context（v3 Step7: final.json の fact_report 等）があれば、source_note の
    精度向上のための参考として渡す（数値の新規持ち込みは禁止のまま）。
    戻り値: {no: chart_spec(dict) or None}
    """
    log = log or (lambda *a, **kw: None)
    if not chart_rows:
        return {}
    # v3 Step7: 検証済みの数値・出典（fact_report 等）。source_note を補うための参考に限る。
    ctx_block = ""
    ec = (extra_context or "").strip()
    if ec:
        ctx_block = (
            "\n\n【検証済みの数値・出典情報（source_note 精度向上の参考）】\n"
            "※原稿の事実確認レポート等です。グラフの数値は必ず各文/block_context の値を使い、"
            "ここから新しい数値を持ち込まないこと。対応する数値の出典を補える場合のみ source_note に使う。\n"
            + ec[:2000]
        )
    out = {}
    for i in range(0, len(chart_rows), CHUNK_SIZE):
        batch = chart_rows[i:i + CHUNK_SIZE]
        inputs = [{
            "no": r["no"],
            "sentence": r.get("sentence", ""),
            "block_context": (r.get("block_text") or "")[:400],
        } for r in batch]
        inputs_json = json.dumps(inputs, ensure_ascii=False, indent=1)
        system = (
            "あなたは動画原稿の数値からグラフ仕様(chart_spec)を構造化抽出する係です。"
            "数値は与えられた文と block_context に書かれているものだけを使い、"
            "推測・補完・創作は絶対にしないこと。JSON 配列のみを返す。"
        )
        query = f"""次の各文(route=chart)について、グラフ化のための chart_spec を抽出してください。

入力:
{inputs_json}

【厳守ルール】
1. 数値は sentence と block_context に**実際に書かれている数値のみ**。推測・補完・創作は禁止。
2. 数値が曖昧・文に無い → その no は {{"no": N, "chart_type": null}}（抽出不能）。
3. 比較対象が1つしかない(単一の値) → "big_number" 型にする。
4. chart_type は bar|line|pie|big_number|comparison|timeline のいずれか。
5. title は短く（その図が何を示すか）。
6. source_note は文/block_context（または下記の検証済み出典情報）に出典が**ある場合のみ**。無ければ省略（創作禁止）。{ctx_block}

【出力 JSON（各 no につき1オブジェクト・入力と同数）】
[
  {{"no": 12, "chart_type": "bar", "title": "軍事費の対GDP比",
    "series": [{{"label": "ロシア", "value": 6.3}}, {{"label": "NATO平均", "value": 2.1}}],
    "unit": "%", "highlight_index": 0, "source_note": "SIPRI 2025"}},
  {{"no": 13, "chart_type": null}}
]
- big_number: series に1要素 {{"label": ラベル, "value": 数値}} か "value": 数値
- timeline: series=[{{"label": 時点, "value": 出来事(文字でも可)}}]
JSON 配列のみ。"""
        text = claude_query(client, query, system, max_tokens=6000)
        specs = parse_json_array(text)
        by_no = {}
        for s in specs:
            if isinstance(s, dict) and s.get("no") is not None:
                by_no[s["no"]] = s
        for r in batch:
            no = r["no"]
            spec = by_no.get(no)
            src = f"{r.get('sentence', '')} {r.get('block_text') or ''}"
            if (not spec) or (spec.get("chart_type") not in _CHART_TYPES):
                out[no] = None
                continue
            if not _chart_numbers_in_source(spec, src):
                out[no] = None  # 原文に無い数値 → 降格
                continue
            sn = (spec.get("source_note") or "").strip()
            if sn and sn not in src:
                spec.pop("source_note", None)  # 出典の創作防止
            out[no] = spec
    n_ok = sum(1 for v in out.values() if v)
    log("renderer", f"chart_spec 抽出: {n_ok} 件 / 降格(ai) {len(out) - n_ok} 件")
    return out


# ===== v3 Step2: map_spec 抽出（router 第2段・LLM使用）=====
# 2026-06-12 安福: map を Natural Earth で正確描画するため、地名→ISO3 を抽出する。
# 国レベルに落とせない（都市・地形が主役）場合は None→illustration 降格。
_MAP_TYPES = ("highlight", "route", "neighbors")


def extract_map_specs(client, map_rows: list, log: Optional[Callable] = None) -> dict:
    """route=map の文から map_spec を抽出する（router 第2段）。

    国は ISO 3166-1 alpha-3。国レベルに落とせない（都市・地形が主役）→ None を返し、
    呼び出し側で route を illustration(engine:ai) へ降格する。
    戻り値: {no: map_spec(dict) or None}
    """
    log = log or (lambda *a, **kw: None)
    if not map_rows:
        return {}
    out = {}
    for i in range(0, len(map_rows), CHUNK_SIZE):
        batch = map_rows[i:i + CHUNK_SIZE]
        inputs = [{
            "no": r["no"],
            "sentence": r.get("sentence", ""),
            "block_context": (r.get("block_text") or "")[:400],
        } for r in batch]
        inputs_json = json.dumps(inputs, ensure_ascii=False, indent=1)
        system = (
            "あなたは動画原稿の地理情報から地図仕様(map_spec)を構造化抽出する係です。"
            "国は ISO 3166-1 alpha-3 コードで表す。JSON 配列のみを返す。"
        )
        query = f"""次の各文(route=map)について、地図化のための map_spec を抽出してください。

入力:
{inputs_json}

【ルール】
1. 国は ISO 3166-1 alpha-3（例: ロシア=RUS, ウクライナ=UKR, ドイツ=DEU, 中国=CHN,
   日本=JPN, アメリカ=USA, カザフスタン=KAZ, ベラルーシ=BLR, フィンランド=FIN, モンゴル=MNG）。
2. 主役が国・地域でなく**都市・地形・建造物**（例: ウラジオストク、シベリア平原）の場合は
   {{"no": N, "map_type": null}}（国レベルに落とせない→降格）。
3. map_type: highlight(国を強調) | route(国から国への経路・輸出入。arrows必須) | neighbors(隣接関係)
4. extent: world | europe | asia | former_ussr | custom（文脈から最適なもの）。
5. focus_countries = 主役の国(ISO3・最大3)。secondary_countries = 関連/隣接国。
6. labels = {{"text": 表示名, "country": ISO3}}。arrows は route 型のみ {{"from": ISO3, "to": ISO3, "label": 短い説明}}。

【出力 JSON（各 no につき1オブジェクト）】
[
  {{"no": 7, "map_type": "route", "title": "ロシアからのガス輸出",
    "focus_countries": ["RUS"], "secondary_countries": ["DEU"],
    "labels": [{{"text": "ロシア", "country": "RUS"}}, {{"text": "ドイツ", "country": "DEU"}}],
    "arrows": [{{"from": "RUS", "to": "DEU", "label": "ガス輸出"}}], "extent": "europe"}},
  {{"no": 8, "map_type": null}}
]
JSON 配列のみ。"""
        text = claude_query(client, query, system, max_tokens=5000)
        specs = parse_json_array(text)
        by_no = {}
        for s in specs:
            if isinstance(s, dict) and s.get("no") is not None:
                by_no[s["no"]] = s
        for r in batch:
            no = r["no"]
            spec = by_no.get(no)
            if (not spec) or (spec.get("map_type") not in _MAP_TYPES) or (not spec.get("focus_countries")):
                out[no] = None
                continue
            out[no] = spec
    n_ok = sum(1 for v in out.values() if v)
    log("renderer", f"map_spec 抽出: {n_ok} 件 / 降格(ai) {len(out) - n_ok} 件")
    return out
