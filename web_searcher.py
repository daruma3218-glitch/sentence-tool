#!/usr/bin/env python3
"""Phase 2b: Web 画像 URL 取得（Claude Web Search 経由）

センテンスから「歴史人物・地名・固有事件」を検出し、
Wikipedia 等のソース URL とサムネイル画像 URL を取得する。

ユーザーはこの URL を見て手動で画像を選定・ダウンロードする想定。
"""

import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import anthropic

from utils import parse_json_array


CLAUDE_MODEL = "claude-sonnet-4-6"
SEARCH_BATCH_SIZE = 6  # 1 リクエストで複数センテンスをまとめて検索


def _claude_research_call(client, query: str, system: str, max_tokens: int = 4096, max_uses: int = 5) -> tuple:
    """Claude Web Search を実行して (text, real_urls) を返す"""
    real_urls = []
    text_parts = []
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_uses,
            }],
            messages=[{"role": "user", "content": query}],
        )
        for block in response.content:
            block_type = getattr(block, "type", "")
            if hasattr(block, "text"):
                text_parts.append(block.text)
            if block_type == "web_search_tool_result":
                content = getattr(block, "content", [])
                for r in content:
                    if getattr(r, "type", "") == "web_search_result":
                        u = getattr(r, "url", "")
                        t = getattr(r, "title", "")
                        if u:
                            real_urls.append({"url": u, "title": t})
    except Exception as e:
        print(f"  [WebSearch ERROR] {e}", flush=True)
    return "\n".join(text_parts), real_urls


def _wikipedia_image_url(article_url: str) -> str:
    """Wikipedia 記事 URL からサムネイル画像 URL を取得（公式 API）

    例: https://ja.wikipedia.org/wiki/ヨシフ・スターリン
        → https://...wikipedia/commons/thumb/.../Stalin.jpg/220px-Stalin.jpg
    """
    try:
        m = re.match(r'https?://([a-z]+)\.wikipedia\.org/wiki/(.+)', article_url)
        if not m:
            return ""
        lang = m.group(1)
        title = urllib.parse.unquote(m.group(2))
        api = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": "400",
            "redirects": "1",
        }
        url = f"{api}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "sentence-tool/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for _pid, pg in pages.items():
            thumb = pg.get("thumbnail", {})
            if thumb.get("source"):
                return thumb["source"]
    except Exception as e:
        print(f"  [Wikipedia thumb ERROR] {e}", flush=True)
    return ""


def _select_chunk(
    client: anthropic.Anthropic,
    candidates: list,
    target_count: int,
    exclude_nos: set,
    log: Callable,
) -> list:
    """1 回の Claude 呼び出しで指定数を選定（チャンク単位）"""
    if target_count <= 0:
        return []

    # 既に選ばれた no は候補から除外
    chunk_candidates = [c for c in candidates if c["no"] not in exclude_nos]
    if not chunk_candidates:
        return []

    inputs_json = json.dumps(chunk_candidates, ensure_ascii=False, indent=2)

    system = (
        "あなたは動画素材リサーチャーです。"
        "原稿センテンスから Web 検索で参考画像が見つかりそうなものを選びます。"
        "結果は必ず JSON 配列のみで返してください。前置きや説明は不要です。"
    )
    exclude_note = f"\n\n【除外: 以下の no はすでに選定済みなので絶対に選ばないこと】\n{sorted(exclude_nos)[:200]}" if exclude_nos else ""
    query = f"""以下のセンテンスから、Web で参考画像（写真・絵画・歴史画像）が見つかりやすい候補を**{target_count}件**選んでください。

候補センテンス:
{inputs_json}{exclude_note}

【選定基準（広めに採用）】
- 歴史人物名（スターリン、毛沢東、ニクソン、ゴルバチョフ など）
- 固有歴史事件名（アイグン条約、朝鮮戦争、ニクソン訪中、シベリア抑留 など）
- 具体的地名・建造物（ウラジオストク、満州、バイカル湖、紫禁城 など）
- 文書・条約・著書（ヤルタ協定、防共協定、毛沢東語録 など）
- 兵器・物（T-34戦車、原子爆弾、戦闘機、ロケット など）
- 国名・国旗（アメリカ・中国・ロシア・ソ連 など固有の国名）
- 地理的特徴（バイカル湖、シベリア、太平洋 など）
- 統計データの背景となるもの（GDP、軍事費、原油生産 → 関連写真）

【選定方針】
- できるだけ多く選ぶ。{target_count}件に満たない場合は、候補センテンスから関連画像が見つかりそうなものを広く拾う
- 「やや関連がある程度」でも採用してよい

【除外】
- 純粋に抽象的な接続詞・挨拶のみ
- 「では」「次に」だけの内容のない文

【出力 JSON（必ずこの形式のみ）】
[
  {{"no": 元のno, "topic": "短い検索トピック名(10〜30文字)", "query": "Web検索クエリ(日本語30文字以内、固有名詞含む)"}}
]

必ず可能な限り {target_count} 件を出力。出力は JSON 配列のみ、前置き禁止。"""

    # 必要 token 数を推算（1 件あたり約 150 token）
    needed_tokens = max(4000, target_count * 200 + 2000)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=min(needed_tokens, 16000),  # 上限 16k
        system=system,
        messages=[{"role": "user", "content": query}],
    )
    text = ""
    for b in response.content:
        if hasattr(b, "text"):
            text += b.text

    # デバッグ: stop_reason / token 数
    stop_reason = getattr(response, "stop_reason", "")
    usage = getattr(response, "usage", None)
    out_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    log("websearch",
        f"  chunk: max_tokens={needed_tokens}, 応答 {out_tokens} tokens, stop={stop_reason}")

    parsed = parse_json_array(text)
    if not parsed:
        log("websearch", f"  ⚠ JSON パース失敗。応答先頭: {text[:200]}")
        return []
    return parsed


def select_search_worthy_sentences(
    client: anthropic.Anthropic,
    rows: list,
    target_count: int,
    log: Optional[Callable] = None,
) -> list:
    """Claude で「Web 画像が役立つセンテンス」を target_count 件選ぶ

    target_count が大きい (>30) 場合は内部で複数チャンクに分割して
    トークン制限と「Claude が一度に大量返却を渋る」問題の両方を回避する。
    """
    log = log or (lambda *a, **kw: None)
    if target_count <= 0 or not rows:
        return []

    # 候補センテンス（短文や記号のみを除外）
    candidates = []
    for r in rows:
        sent = r.get("sentence", "")
        if len(sent.strip()) < 8:
            continue
        if not re.search(r'[一-龥ァ-ヴーa-zA-Z]', sent):
            continue
        candidates.append({"no": r["no"], "sentence": sent[:200]})

    if not candidates:
        log("websearch", "候補センテンスが 0 件です（原稿が短すぎる）")
        return []

    log("websearch",
        f"Claude で Web 検索対象を選定中: 候補 {len(candidates)} / 目標 {target_count} 件")

    # チャンク分割: 1 リクエストあたり最大 30 件まで（Claude が大量返却を渋るため）
    CHUNK_SIZE = 30
    no_set = {r["no"] for r in rows}
    all_valid = []
    selected_nos: set = set()
    remaining = target_count
    chunk_idx = 0

    while remaining > 0 and chunk_idx < 10:  # 最大 10 チャンク
        chunk_idx += 1
        request_n = min(remaining, CHUNK_SIZE)
        log("websearch", f"  チャンク {chunk_idx}: {request_n} 件要求（残り {remaining} 件）")

        try:
            selected = _select_chunk(client, candidates, request_n, selected_nos, log)
        except Exception as e:
            log("error", f"  チャンク {chunk_idx} 失敗: {str(e)[:120]}")
            break

        if not selected:
            log("websearch", f"  チャンク {chunk_idx}: 選定 0 件（候補不足の可能性）→ 中断")
            break

        added = 0
        for s in selected:
            if not isinstance(s, dict):
                continue
            no = s.get("no")
            if no not in no_set or no in selected_nos:
                continue
            if not s.get("query"):
                continue
            selected_nos.add(no)
            all_valid.append(s)
            added += 1
            if len(all_valid) >= target_count:
                break

        log("websearch", f"  チャンク {chunk_idx}: {added} 件追加 → 累計 {len(all_valid)}/{target_count}")
        remaining = target_count - len(all_valid)

        # チャンクで何も追加できなかった場合は打ち切り（同じ候補を繰り返すだけ）
        if added == 0:
            break

    log("websearch", f"選定完了: {len(all_valid)} 件（{chunk_idx} チャンク使用）")
    return all_valid[:target_count]


def download_thumbnail(thumb_url: str, output_path) -> bool:
    """Wikimedia 等のサムネイル画像をローカルに保存する。

    参考用の低解像度サムネイル（Wikimedia Commons は大半が CC/PD）を
    動画素材の下調べ用にダウンロードする。
    壊れたURL・極小画像（ロゴ/アイコン）は検証して弾く。
    """
    from pathlib import Path
    from io import BytesIO
    if not thumb_url:
        return False
    output_path = Path(output_path)
    try:
        req = urllib.request.Request(
            thumb_url,
            headers={"User-Agent": "sentence-tool/1.0 (educational video material research)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            if getattr(resp, "status", 200) != 200:
                return False
            ctype = resp.headers.get("Content-Type", "")
            data = resp.read()
        # Content-Type が画像でない（HTMLエラーページ等）→ 弾く
        if "image" not in ctype.lower():
            return False
        if not data or len(data) < 500:
            return False
        # PIL で実画像か検証 + サイズチェック（極小ロゴ/アイコンを除外）
        try:
            from PIL import Image
            img = Image.open(BytesIO(data))
            img.load()
            w, h = img.size
            if w < 80 or h < 80:
                return False  # 小さすぎ（ロゴ・アイコンの可能性）
        except Exception:
            return False  # 画像として開けない＝壊れURL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"  [thumb download ERROR] {str(e)[:100]}", flush=True)
        return False


def search_single_sentence(
    client: anthropic.Anthropic,
    selection: dict,
) -> dict:
    """1 センテンスに対する Web 画像 URL を取得"""
    no = selection["no"]
    query_text = selection.get("query", "")
    topic = selection.get("topic", "")

    system = (
        "あなたはリサーチャーです。Web 検索で指定トピックの参考画像が掲載されたページを探します。"
        "Wikipedia や公的機関、報道機関のサイトを優先してください。"
    )
    query = f"""「{topic}」に関する参考画像が載っているページを Web 検索で探してください。
検索クエリ: {query_text}

【最重要】
- 必ず Web 検索を実行すること
- 画像が含まれるページ（Wikipedia 等）を優先
- 公的機関・報道機関・百科事典のサイトを優先
- 数件で OK。最も信頼性の高い 1 件を選んで返す

回答後に Web 検索結果のリストもそのまま記述してください。"""

    text, urls = _claude_research_call(client, query, system, max_tokens=2000, max_uses=3)

    # 最良の URL を 1 件選ぶ
    best_url = ""
    best_title = ""
    for u in urls:
        # Wikipedia 優先
        if "wikipedia.org" in u["url"]:
            best_url = u["url"]
            best_title = u["title"]
            break
    if not best_url and urls:
        best_url = urls[0]["url"]
        best_title = urls[0]["title"]

    # サムネイル取得（Wikipedia の場合のみ）
    thumb_url = ""
    if "wikipedia.org/wiki/" in best_url:
        thumb_url = _wikipedia_image_url(best_url)

    return {
        "no": no,
        "topic": topic,
        "query": query_text,
        "source_url": best_url,
        "source_title": best_title,
        "thumb_url": thumb_url,
        "all_urls": urls[:5],  # 候補も残す
    }


def run_web_search_for_selections(
    client: anthropic.Anthropic,
    selections: list,
    max_workers: int = 8,
    log: Optional[Callable] = None,
    item_callback: Optional[Callable] = None,
) -> list:
    """v2 用: ルーターが選んだ web_photo 文を検索する（選定ステップなし）。

    selections: [{"no": int, "query": str, "topic": str}, ...]
        ルーターの search_query / topic をそのまま使うので、ここでは選定しない。
    """
    log = log or (lambda *a, **kw: None)
    item_callback = item_callback or (lambda info: None)

    if not selections:
        log("websearch", "Web 検索対象（web_photo）がありません")
        return []

    log("websearch", f"{len(selections)} 件の Web 検索を並列実行中（同時 {max_workers}）...")
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sel = {
            executor.submit(search_single_sentence, client, sel): sel
            for sel in selections
        }
        for future in as_completed(future_to_sel):
            sel = future_to_sel[future]
            try:
                r = future.result()
                results.append(r)
                completed += 1
                log("websearch", f"検索 {completed}/{len(selections)}: 「{r.get('topic', '')[:30]}」")
                item_callback(r)
            except Exception as e:
                log("error", f"検索エラー（no={sel.get('no')}）: {str(e)[:100]}")

    results.sort(key=lambda x: x.get("no", 0))
    log("websearch", f"Web 検索完了: {len(results)} 件")
    return results


def run_web_search(
    client: anthropic.Anthropic,
    rows: list,
    target_count: int = 50,
    max_workers: int = 4,
    log: Optional[Callable] = None,
    item_callback: Optional[Callable] = None,
) -> list:
    """エントリポイント: Web 画像 URL を取得して返す（v1 互換: 内部で選定する）

    戻り値:
      [
        {
          "no": int,
          "topic": str,
          "source_url": str,
          "source_title": str,
          "thumb_url": str,
          "all_urls": [{"url": str, "title": str}, ...]
        },
        ...
      ]
    """
    log = log or (lambda *a, **kw: None)
    item_callback = item_callback or (lambda info: None)

    if target_count <= 0:
        return []

    # Step 1: 検索対象センテンスを Claude が選定
    selections = select_search_worthy_sentences(client, rows, target_count, log=log)
    if not selections:
        log("websearch", "Web 検索対象センテンスが見つかりません")
        return []

    # Step 2: 並列に Web 検索を実行
    log("websearch", f"{len(selections)} 件の Web 検索を並列実行中（同時 {max_workers}）...")
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sel = {
            executor.submit(search_single_sentence, client, sel): sel
            for sel in selections
        }
        for future in as_completed(future_to_sel):
            sel = future_to_sel[future]
            try:
                r = future.result()
                results.append(r)
                completed += 1
                log("websearch", f"検索 {completed}/{len(selections)}: 「{r.get('topic', '')[:30]}」")
                item_callback(r)
            except Exception as e:
                log("error", f"検索エラー（no={sel.get('no')}）: {str(e)[:100]}")

    # no 順にソート
    results.sort(key=lambda x: x.get("no", 0))
    log("websearch", f"Web 検索完了: {len(results)} 件")
    return results
