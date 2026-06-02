#!/usr/bin/env python3
"""Phase 1: 原稿を 章 / ブロック / センテンス に自動分解

スプレッドシート構造に合わせた階層分割:
  原稿
   └── 章 (chapter): オープニング / 大陸帝国の掟 / ...
       └── ブロック (block): 段落 = 意味のかたまり
           └── センテンス (sentence): 1 文（句点で区切る）

【方針】
1. まずシンプルな機械的分割 (改行 + 句点) でベース分解
2. 章名は Claude に名付けてもらう（自動命名・原稿から推測）
3. 出力: 章/ブロック/センテンスのフラットなリスト（№付き）
"""

import json
import re
from typing import Callable, Optional

import anthropic

from utils import claude_query, parse_json_object


CLAUDE_MODEL = "claude-sonnet-4-6"

# 文末記号
SENTENCE_END_PATTERN = re.compile(r'(?<=[。！？])\s*')

# 引用や括弧内の句点では分割しないよう簡易的に保護
QUOTE_PAIRS = [('「', '」'), ('『', '』'), ('（', '）'), ('(', ')')]


def split_into_sentences(text: str) -> list:
    """1段落のテキストを句点でセンテンスに分割（引用内は保護）"""
    if not text or not text.strip():
        return []

    sentences = []
    buf = []
    depth = 0  # 括弧の入れ子深さ

    open_chars = {p[0] for p in QUOTE_PAIRS}
    close_chars = {p[1] for p in QUOTE_PAIRS}

    for ch in text:
        buf.append(ch)
        if ch in open_chars:
            depth += 1
        elif ch in close_chars and depth > 0:
            depth -= 1
        elif ch in '。！？' and depth == 0:
            s = ''.join(buf).strip()
            if s:
                sentences.append(s)
            buf = []

    # 残り（句点で終わっていない場合）
    rest = ''.join(buf).strip()
    if rest:
        sentences.append(rest)

    return sentences


def mechanical_split(manuscript_text: str) -> list:
    """機械的な分割（章/ブロック/センテンスのツリー）

    戻り値:
      [
        {
          "title": "（未分類）",  # 後で Claude が命名
          "blocks": [
            {"text": "段落原文", "sentences": ["文1", "文2", ...]},
            ...
          ]
        },
        ...
      ]
    """
    # 章は「3行以上の連続改行」で区切る（明示的な大区切り）
    # ブロックは「1〜2行の連続改行」で区切る
    text = manuscript_text.replace('\r\n', '\n').replace('\r', '\n')

    # 3行以上の改行 → 章区切り
    chapter_chunks = re.split(r'\n{3,}', text)
    # 章が 1 つしかない場合は 2 行改行も章区切り扱いにする（フォールバック）
    if len(chapter_chunks) <= 1:
        chapter_chunks = [c for c in re.split(r'\n{2,}', text) if c.strip()]
        # それでも 1 つなら全体が 1 章
        if len(chapter_chunks) <= 1:
            chapter_chunks = [text]

    chapters = []
    for ch_text in chapter_chunks:
        ch_text = ch_text.strip()
        if not ch_text:
            continue

        # ブロックは 1 行以上の連続改行で分割
        block_chunks = re.split(r'\n+', ch_text)
        blocks = []
        for b_text in block_chunks:
            b_text = b_text.strip()
            if not b_text:
                continue
            sentences = split_into_sentences(b_text)
            if sentences:
                blocks.append({"text": b_text, "sentences": sentences})

        if not blocks:
            continue

        # 先頭ブロックが「見出しっぽい」場合は章タイトルとして抜き出す
        # 条件: 1 センテンスのみ AND 30 文字以内 AND 文末記号で終わらない
        title_hint = ""
        first = blocks[0]
        if (len(first["sentences"]) == 1
                and len(first["text"]) <= 30
                and not first["text"].rstrip().endswith(('。', '！', '？'))):
            title_hint = first["text"].strip()
            blocks = blocks[1:]  # 本文から除外

        # 章タイトルだけで本文がない場合はスキップ
        if not blocks:
            continue

        chapters.append({"title": title_hint, "blocks": blocks})

    return chapters


def name_chapters_with_claude(
    client: anthropic.Anthropic,
    chapters: list,
    log: Optional[Callable] = None,
) -> list:
    """各章に Claude で見出しを付ける。
    ただし mechanical_split で既にタイトルが取れている章はスキップ。
    """
    log = log or (lambda *a, **kw: None)

    # 未命名（title が空）の章だけ Claude にお願いする
    unnamed_indices = [i for i, ch in enumerate(chapters) if not ch.get("title", "").strip()]
    if not unnamed_indices:
        log("splitter", "全章に見出しあり（命名スキップ）")
        return chapters

    # 各章の冒頭サンプルを Claude に渡す
    samples = []
    for i in unnamed_indices:
        first_block = chapters[i]["blocks"][0]["text"] if chapters[i]["blocks"] else ""
        samples.append({
            "index": i,
            "preview": first_block[:300],
        })
    samples_json = json.dumps(samples, ensure_ascii=False, indent=2)

    system = (
        "あなたは編集者です。動画原稿の各章に短い見出し（5〜15文字）をつけます。"
        "結果は必ず JSON オブジェクトのみで返してください。"
    )
    query = f"""以下は動画原稿の {len(unnamed_indices)} 個の未命名章の冒頭プレビューです。
各章にふさわしい短い見出し（5〜15文字）を付けてください。

章一覧（index は元の章番号、order に依存しない）:
{samples_json}

【見出しのつけ方】
- 冒頭で挨拶・自己紹介・本日のテーマ紹介 → 「オープニング」
- 末尾で振り返り・告知・締め → 「エンディング」「お知らせ」など
- 雑談・余談 → 「雑談コーナー」など
- それ以外 → 本文から要約した短いタイトル（5〜15文字）

【出力 JSON】
{{
  "titles": {{
    "0": "章0のタイトル",
    "3": "章3のタイトル"
  }}
}}

JSON のみで返すこと。キーは元の章 index（文字列）。"""

    log("splitter", f"Claude で {len(unnamed_indices)} 章にタイトル付与中...")
    result = claude_query(client, query, system, max_tokens=1500, model=CLAUDE_MODEL)
    data = parse_json_object(result)
    titles_map = data.get("titles", {}) if isinstance(data, dict) else {}
    if not isinstance(titles_map, dict):
        titles_map = {}

    for i in unnamed_indices:
        key = str(i)
        if key in titles_map and isinstance(titles_map[key], str) and titles_map[key].strip():
            chapters[i]["title"] = titles_map[key].strip()
        else:
            chapters[i]["title"] = f"章 {i + 1}"

    return chapters


def flatten_to_rows(chapters: list) -> list:
    """章/ブロック/センテンスを 1 行 1 センテンスのフラットリストに変換

    スプレッドシート構造に合わせた行データ:
      {
        "no": 1,                  # 通し番号
        "chapter_index": 0,
        "chapter_title": "オープニング",
        "block_index": 0,         # 章内のブロック番号 (0始まり)
        "block_text": "段落全文",
        "sentence_index": 0,      # ブロック内のセンテンス番号
        "sentence": "原稿1文",
        "status": "pending",      # pending / generating / ok / failed
      }
    """
    rows = []
    no = 0
    for ci, ch in enumerate(chapters):
        for bi, block in enumerate(ch["blocks"]):
            for si, sent in enumerate(block["sentences"]):
                no += 1
                rows.append({
                    "no": no,
                    "chapter_index": ci,
                    "chapter_title": ch["title"],
                    "block_index": bi,
                    "block_text": block["text"],
                    "sentence_index": si,
                    "sentence": sent,
                    "status": "pending",
                })
    return rows


def analyze_manuscript_summary(
    client: anthropic.Anthropic,
    manuscript_text: str,
    log: Optional[Callable] = None,
) -> dict:
    """原稿の全体メタ情報（タイトル・要約・キーワード）を取得"""
    log = log or (lambda *a, **kw: None)
    system = (
        "あなたは編集者です。動画原稿の全体構造を JSON で返します。"
    )
    head = manuscript_text[:6000]
    query = f"""以下の原稿の全体メタ情報を JSON で返してください。

{head}

【出力 JSON】
{{
  "title": "原稿のメインテーマを表すタイトル（30文字以内）",
  "summary": "原稿全体の要約（200文字以内）",
  "keywords": ["キーワード1", "..."（5〜10個）]
}}

JSON のみで返すこと。"""
    log("splitter", "Claude で原稿全体を分析中...")
    result = claude_query(client, query, system, max_tokens=800, model=CLAUDE_MODEL)
    data = parse_json_object(result) or {}
    return {
        "title": data.get("title", "無題"),
        "summary": data.get("summary", manuscript_text[:200]),
        "keywords": data.get("keywords", []),
    }


def parse_docx_to_chapters(file_bytes: bytes) -> tuple:
    """Word(.docx) を解析し、見出しスタイルを章タイトルとして章/ブロック構造を作る。

    Google ドキュメントを「ファイル→ダウンロード→Word(.docx)」した想定。
    「見出し1/2/Heading/Title/タイトル」スタイルの段落を章タイトルとして扱う。

    戻り値: (full_text, chapters)
      chapters は mechanical_split と同じ形式（title 付き）。
    """
    from io import BytesIO
    from docx import Document

    doc = Document(BytesIO(file_bytes))
    chapters = []
    current = None
    text_parts = []

    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        style_name = ""
        try:
            style_name = (para.style.name or "").lower()
        except Exception:
            style_name = ""
        is_heading = (
            style_name.startswith("heading")
            or style_name.startswith("title")
            or "見出し" in style_name
            or "タイトル" in style_name
        )
        if is_heading:
            current = {"title": text, "blocks": []}
            chapters.append(current)
            text_parts.append("\n\n\n" + text)
        else:
            if current is None:
                current = {"title": "", "blocks": []}
                chapters.append(current)
            sentences = split_into_sentences(text)
            if sentences:
                current["blocks"].append({"text": text, "sentences": sentences})
            text_parts.append(text)

    # 本文のない章（見出しだけ）は除去
    chapters = [c for c in chapters if c["blocks"]]
    full_text = "\n".join(text_parts).strip()
    return full_text, chapters


def split_manuscript(
    client: anthropic.Anthropic,
    manuscript_text: str,
    log: Optional[Callable] = None,
    prebuilt_chapters: Optional[list] = None,
) -> dict:
    """エントリポイント: 原稿を分解してフラットリストを返す

    prebuilt_chapters を渡すと機械分割をスキップし、その章構造を使う
    （.docx の見出しから作った章など）。タイトル未設定の章は Claude が命名。

    戻り値:
      {
        "analysis": {"title": ..., "summary": ..., "keywords": [...]},
        "chapters": [...],   # ツリー構造
        "rows": [...],       # フラットリスト（1行=1センテンス）
        "total_sentences": int,
      }
    """
    log = log or (lambda *a, **kw: None)

    if not manuscript_text or not manuscript_text.strip():
        raise ValueError("原稿が空です")

    if prebuilt_chapters:
        chapters = prebuilt_chapters
        log("splitter", f".docx の見出しから章 {len(chapters)} 個を検出（見出しスタイル使用）")
    else:
        log("splitter", f"原稿（{len(manuscript_text)}文字）を機械的に分割中...")
        chapters = mechanical_split(manuscript_text)
        log("splitter", f"章 {len(chapters)} 個 / ブロック {sum(len(c['blocks']) for c in chapters)} 個 検出")

    # 章タイトルを Claude が命名（タイトル未設定のものだけ補完される）
    chapters = name_chapters_with_claude(client, chapters, log=log)

    # 全体メタ情報
    analysis = analyze_manuscript_summary(client, manuscript_text, log=log)

    # フラットリスト化
    rows = flatten_to_rows(chapters)
    log("splitter", f"分解完了: 全 {len(rows)} センテンス")

    return {
        "analysis": analysis,
        "chapters": chapters,
        "rows": rows,
        "total_sentences": len(rows),
    }
