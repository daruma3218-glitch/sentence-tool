#!/usr/bin/env python3
"""Phase 2: センテンス → 英文画像プロンプト（並列バッチ）

各センテンスをシンプルな「フラットインフォグラフィック」に変換する。
原稿の数値・年代・固有名詞は積極的にホワイトリスト化して画像内テキストとして使う。
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import anthropic

from utils import claude_query, parse_json_array


CLAUDE_MODEL = "claude-sonnet-4-6"
BATCH_SIZE = 8


# ===== 安全な自動抽出（Claude の遠慮を補完） =====
# 原稿に出てくる以下のパターンは無条件で allowed_terms に追加してよい:
#   年代 (1858年, 2024年, 19世紀)
#   数値 (100年, 14か国, 1.5億, 65万8千人, 4,380km, 6.3%, 75%, 1兆530億ドル)
#   括弧内の固有名詞 (「大陸帝国」「ヴィア」)
SAFE_PATTERNS = [
    # 年代
    r'\d{1,4}年(?:代)?',                  # 1858年, 1858年代
    r'\d{1,2}世紀',                       # 19世紀
    # 数値+単位
    r'\d+(?:,\d{3})*(?:\.\d+)?(?:%|％)',  # 6.3%, 75%
    r'\d+(?:,\d{3})*(?:\.\d+)?(?:億|兆|万|千)?(?:円|ドル|人|km|キロメートル|平方キロメートル|か国)',
    r'\d+(?:,\d{3})*年(?:以上|間|前)',    # 100年以上
    # 括弧内のキーワード（「」『』内の短い語）
    r'「([^」]{2,15})」',
    r'『([^』]{2,15})』',
]


def _auto_extract_terms(sentence: str) -> list:
    """センテンスから安全な語句を自動抽出（Claude の補完用）"""
    terms = []
    seen = set()
    for pat in SAFE_PATTERNS:
        for m in re.finditer(pat, sentence):
            # キャプチャグループがあればそれを使う
            t = m.group(1) if m.groups() else m.group(0)
            t = t.strip()
            if t and t not in seen and t in sentence:
                seen.add(t)
                terms.append(t)
    return terms


def _build_user_block(user_instructions: str, style_preset: str) -> str:
    blocks = []
    if user_instructions.strip():
        blocks.append(f"""【ユーザーからの画像指示（最優先で従うこと）】
{user_instructions.strip()}""")

    style_descriptions = {
        "flat_infographic": (
            "【スタイル: フラットインフォグラフィック（最優先で守ること）】\n"
            "- 2〜3 色のフラットカラー（ナビーブルー #1E40AF / 白 / ライトグレー / 1色アクセント）\n"
            "- アイコン・記号ベース（人型シルエット、国旗、矢印、円グラフ、棒グラフなど）\n"
            "- 大きな数字・短い見出しを画面の主役にする（テロップ的に）\n"
            "- 写実画・寓意（動物の擬人化）・劇的演出は避ける\n"
            "- ニュース番組のテロップ・統計レポートのような「説明画面」を目指す"
        ),
        "pictogram": (
            "【スタイル: ピクトグラム調】\n"
            "- 単色シルエット（黒または1色）の人型・物のアイコンのみ\n"
            "- 余計な装飾なし、最大シンプル\n"
            "- 公共表示・トイレマーク的な明快さ"
        ),
        "comic": (
            "【スタイル: コミックストリップ調】\n"
            "- マンガ風セル割り（ただし吹き出しのテキストは allowed_terms 内のみ）\n"
            "- 4色程度のフラットカラー\n"
            "- 線がはっきり太く、表情の分かるキャラクター"
        ),
        "whiteboard": (
            "【スタイル: 手描きホワイトボード調】\n"
            "- 白背景に黒マジック手描き風\n"
            "- ラフな矢印・囲み・吹き出し\n"
            "- TED チャンネル・Sketchnoting のような図解"
        ),
    }
    blocks.append(style_descriptions.get(style_preset, style_descriptions["flat_infographic"]))
    return "\n\n".join(blocks)


def generate_prompts_batch(
    client: anthropic.Anthropic,
    rows_batch: list,
    title: str,
    user_instructions: str = "",
    style_preset: str = "flat_infographic",
    worldview_desc: str = "",
) -> list:
    """1 バッチのセンテンスを英文プロンプト化"""
    user_block = _build_user_block(user_instructions, style_preset)
    # 世界観・キャラ統一の指示（illustration/diagram/decorative に適用）
    worldview_block = ""
    if worldview_desc.strip():
        worldview_block = f"""

【世界観・キャラクター統一（最重要・illustration / diagram / decorative にのみ適用）】
人物や情景を描くイラストでは、以下の世界観・キャラクター設定を**毎回一貫して**反映すること。
登場人物・画風・色調・タッチを動画全体で統一し、シーンが変わっても同じ世界観に見せる:
---
{worldview_desc.strip()}
---
※ realphoto（実写）・map（衛星地図）・chart（グラフ）にはこの世界観を適用しない（実写・地図・数値はそのまま）。
※ 人物が登場する illustration では必ず上のキャラクター設定の人物を使う。"""

    # Claude に渡す入力 + 自動抽出済み terms をヒントとして同梱
    # 各行の route（ルーター判定）を type として固定で渡す
    inputs = []
    for r in rows_batch:
        sent = r.get("sentence", "")
        hints = _auto_extract_terms(sent)
        row_type = r.get("route") or r.get("type") or "illustration"
        inputs.append({
            "no": r["no"],
            "type": row_type,  # ★この type を厳守すること（変更禁止）
            "chapter": r.get("chapter_title", ""),
            "block_context": r.get("block_text", "")[:400],
            "sentence": sent,
            "auto_extracted_terms": hints,  # ヒント
        })
    inputs_json = json.dumps(inputs, ensure_ascii=False, indent=2)

    system = (
        "You are a visual director. You convert each Japanese sentence from a video "
        "manuscript into a precise English image prompt. Each item has a fixed 'type' "
        "you MUST honor: realphoto = a realistic documentary PHOTOGRAPH, map = a realistic "
        "satellite/terrain MAP, and illustration/diagram/chart = the specified graphic style. "
        "Return only a JSON array. No markdown, no commentary."
    )

    query = f"""動画原稿「{title}」の各センテンス（1文）に対応する英文画像プロンプトを書いてください。
**各項目の type は厳守**（変更禁止）。type ごとに描き方が違います。

入力（type=その項目の描画種別。auto_extracted_terms は機械抽出した数値・年代・固有名詞のヒント）:
{inputs_json}

{user_block}{worldview_block}

【最重要: type 別の描き方】
- **realphoto**: 実写写真。"photorealistic documentary photograph, real photo, natural lighting,
  realistic textures, cinematic" を必ず含める。**フラット/アイコン/イラストには絶対しない**。
  上で指定したグラフィックスタイル（フラット等）は realphoto には適用しないこと。
  **日本語ラベルは入れない**。画面内の看板・標識は描かれている場所の現地語
  （ロシア/ソ連のシーンならロシア語＝キリル文字）にすること。
  プロンプトに "signs and text in the local language (Russian/Cyrillic for Russia), no Japanese labels" と明記。
- **map**: リアルな衛星・地形図。"realistic satellite map, terrain, natural earth colors" を含める。
  フラットな地図にはしない。上で指定したグラフィックスタイルは map には適用しないこと。
- **illustration / diagram / chart / decorative**: 上で指定したグラフィックスタイルに従って描く。

【最重要ルール】
1. プロンプトは英語で書く
2. メタファー（クマ＝ロシア など寓意）は**禁止**。国は国旗・国名・地図で直接表現する
3. **画像内テキストは allowed_terms にあるものだけ**（厳格）
4. 出力の type は入力の type を**そのまま返す**（勝手に変えない）
5. **character フラグ**: 上の世界観設定に「繰り返し登場する固定キャラ（先生／教授／解説役）」
   がある場合、そのキャラが実際に画面に描かれる illustration のときだけ "character": true。
   図表(diagram/chart)・写真(realphoto)・地図(map)・人物のいないシーン・装飾は必ず false。

【allowed_terms 抽出方針（積極的に入れる）】
- センテンスに登場する以下は**すべて** allowed_terms に入れること:
  * 年代 (1858年, 19世紀, 1945年)
  * 数値 (100年以上, 14か国, 65万8千人, 6.3%, 1兆530億ドル)
  * 国名 (ロシア, 中国, アメリカ, 日本, ソ連, モンゴル)
  * 地名 (ウラジオストク, 北京, アイグン, 満州)
  * 人名 (スターリン, 毛沢東, ニクソン)
  * 重要なキーワード (大陸帝国, 二正面作戦, アヘン戦争, 北京条約)
- **auto_extracted_terms** はすでに抽出済みなので必ず取り込み、加えてセンテンスからも追加抽出する
- 一般語（「国」「時」「これ」など）は除外

【画像内テキストの記述例】
- allowed_terms = ["ロシア", "中国", "100年"] の場合:
  "Insert these Japanese labels prominently in the image: ロシア, 中国, 100年. Do NOT add any other Japanese or English text."
- allowed_terms = [] の場合:
  "No text in image, no labels, no numbers. Use icons only."

【type の選び方】
- illustration: 人物・物・出来事のイラスト（**シンプルアイコン調**）
- realphoto: **実写風の写真**。都市・建物・施設・インフラ・事件・戦争・人々の生活など
  物理的なシーンを、ドキュメンタリー写真のようにリアルに描く（イラストではなく本物の写真）。
  自然光・実在感のある質感・映画的構図。報道写真／ドキュメンタリー品質を目指す。
- map: 地理関係。**フラットな図ではなく、衛星写真／航空写真のようなリアルな地図**にすること。
  上空から見た本物の地球表面（青い海・緑の森林・茶色の山岳・白い雪原・リアルな海岸線）を描き、
  地形の起伏（レリーフシェーディング）も表現する。対象の国・地域は半透明の色で塗り分け、
  国境は細い線で示す。Google Earth / NASA衛星画像 / ナショナルジオグラフィック品質を目指す。
- diagram: 概念図・フロー図（アイコン + 矢印 + ラベル）
- chart: 数値比較（棒グラフ・円グラフ・大きな数字）
- decorative: 接続詞・挨拶・抽象表現（背景パターン）

【出力 JSON】
[
  {{
    "no": (元のno),
    "prompt": "英語プロンプト（スタイル指示・テキスト制約を必ず含む）",
    "type": "illustration | realphoto | map | diagram | chart | decorative",
    "allowed_terms": ["積極的に抽出した語"],
    "character": true または false（ルール5。固定キャラ＝先生/教授/解説役が描かれる illustration のみ true）
  }},
  ...
]

必ず {len(rows_batch)} 件返すこと（順序は入力と同じ）。JSON のみ。"""

    result = claude_query(client, query, system, max_tokens=8000, model=CLAUDE_MODEL)
    prompts = parse_json_array(result)

    # 入力情報をマージ + allowed_terms をセンテンス検証
    prompts_by_no = {p.get("no"): p for p in prompts if p.get("prompt")}
    merged = []
    for r in rows_batch:
        no = r["no"]
        sent = r.get("sentence", "")
        auto_terms = _auto_extract_terms(sent)
        if no in prompts_by_no:
            p = prompts_by_no[no]
            # allowed_terms 検証 + auto_terms を追加
            terms = p.get("allowed_terms", [])
            if not isinstance(terms, list):
                terms = []
            # 既存 + auto をマージ
            merged_terms = list(terms) + auto_terms
            verified = []
            seen = set()
            for t in merged_terms:
                if not isinstance(t, str):
                    continue
                t = t.strip()
                if t and t in sent and t not in seen:
                    seen.add(t)
                    verified.append(t)
            p["allowed_terms"] = verified
            # type はルーターの route を最優先で固定（Claudeが勝手に変えても上書き）
            forced_type = r.get("route") or r.get("type")
            if forced_type in ("illustration", "realphoto", "map", "diagram", "chart", "decorative"):
                p["type"] = forced_type
            elif p.get("type") not in ("illustration", "realphoto", "map", "diagram", "chart", "decorative"):
                p["type"] = "illustration"
            # character フラグは illustration のときだけ有効（図表/写真/地図/装飾では必ず False）
            p["character"] = bool(p.get("character", False)) and p["type"] == "illustration"
            merged.append(p)
        else:
            # フォールバック
            short = sent[:80]
            fallback_prompt = (
                f"Flat infographic explaining: {short}. "
                "Simple icons, 2-3 flat colors (navy blue, white, light gray). "
                "Bold layout. No metaphors. "
                "No text in image, no labels, no numbers. "
                "16:9 landscape orientation, no title text."
            )
            merged.append({
                "no": no,
                "prompt": fallback_prompt,
                "type": "decorative",
                "allowed_terms": auto_terms,
                "character": False,
            })
    return merged


def generate_all_prompts(
    client: anthropic.Anthropic,
    rows: list,
    title: str,
    user_instructions: str = "",
    style_preset: str = "flat_infographic",
    worldview_desc: str = "",
    max_workers: int = 6,
    log: Optional[Callable] = None,
) -> list:
    """全センテンスを並列バッチで英文プロンプト化"""
    log = log or (lambda *a, **kw: None)

    batches = []
    for i in range(0, len(rows), BATCH_SIZE):
        batches.append(rows[i:i + BATCH_SIZE])

    log("prompter", f"{len(rows)} センテンスを {len(batches)} バッチに分割（同時 {max_workers} 並列）/ style={style_preset}"
                    + ("／世界観統一ON" if worldview_desc.strip() else ""))

    prompts_by_no = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(generate_prompts_batch, client, batch, title, user_instructions, style_preset, worldview_desc): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results = future.result()
                for p in results:
                    if p.get("no") is not None:
                        prompts_by_no[p["no"]] = p
                completed += 1
                log("prompter", f"バッチ {completed}/{len(batches)} 完了（{len(results)} 件）")
            except Exception as e:
                log("error", f"バッチ {idx} 失敗: {str(e)[:120]}")

    out_rows = []
    for r in rows:
        no = r["no"]
        p = prompts_by_no.get(no, {})
        merged_row = dict(r)
        merged_row["prompt"] = p.get("prompt", "")
        merged_row["type"] = p.get("type", "illustration")
        merged_row["allowed_terms"] = p.get("allowed_terms", [])
        merged_row["character"] = bool(p.get("character", False))  # キャラ固定フラグを引き継ぐ
        out_rows.append(merged_row)

    return out_rows
