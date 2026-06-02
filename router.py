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


def _route_chunk(
    client: anthropic.Anthropic,
    rows_chunk: list,
    title: str,
    user_instructions: str = "",
    propaganda_mix: bool = False,
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

    system = (
        "あなたは動画ディレクターです。原稿の各センテンスに、最適な画像ソースの種別（route）を"
        "1 つ割り当てます。結果は必ず JSON 配列のみで返してください。"
    )

    query = f"""動画「{title}」の各センテンスに、最適な画像ソース種別（route）を 1 つ割り当ててください。

センテンス一覧:
{inputs_json}
{user_block}{propaganda_block}

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

【出力 JSON（必ずこの形式のみ）】
[
  {{
    "no": 1,
    "route": "web_photo",
    "reason": "判定理由を15字以内で",
    "search_query": "Web検索クエリ（web_photoのときのみ、日本語30字以内、固有名詞を含む）",
    "topic": "トピック名（web_photoのときのみ、10〜20字）"{propaganda_field}
  }},
  {{
    "no": 2,
    "route": "diagram",
    "reason": "概念の対比のため"
  }}
]

必ず {len(rows_chunk)} 件すべてに route を付与すること。JSON 配列のみ返す。"""

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
            executor.submit(_route_chunk, client, chunk, title, user_instructions, propaganda_mix): i
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
