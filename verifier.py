#!/usr/bin/env python3
"""Phase 3b: 図解の意味を Claude Vision で自動検証

生成された画像（主に diagram / chart）が、対応するセンテンスの意味を
正しく・分かりやすく表せているかを Claude のビジョン機能で判定する。
ズレている場合は改善指示（fix_hint）を返し、パイプラインが再生成に使う。
"""

import base64
from pathlib import Path
from typing import Optional

import anthropic

from utils import parse_json_object


# v3 Step5: chart/map が renderer 化されたため、Vision 検品は diagram のみに縮小。
# モデルも Haiku に変更してコストを約 1/5 に（判定失敗時 ok=True の安全設計は維持）。
CLAUDE_MODEL = "claude-haiku-4-5"

# 検証対象にするデフォルトの type（chart は決定論レンダリングのため検品不要）
DEFAULT_VERIFY_TYPES = ("diagram",)


def verify_image(
    client: anthropic.Anthropic,
    image_path,
    sentence: str,
    img_type: str = "diagram",
    allowed_terms: Optional[list] = None,
    block_context: str = "",
    chapter: str = "",
    theme: str = "",
) -> dict:
    """1 枚の画像が、原稿の文脈の中で文の意味を正しく表しているか検証する。

    単独の文だけでなく、動画テーマ・章・前後段落（block_context）も渡して
    「文脈の中でこの文が本当に意味すること」と図が合っているかを判定する。

    戻り値: {"ok": bool, "reason": str, "fix_hint": str}
        ok=False のとき fix_hint（英語の改善指示）が入る。
    判定に失敗した場合は安全側に倒して ok=True（再生成しない）。
    """
    p = Path(image_path)
    try:
        img_bytes = p.read_bytes()
    except Exception:
        return {"ok": True, "reason": "画像読込失敗（検証スキップ）", "fix_hint": ""}

    if not img_bytes or len(img_bytes) < 200:
        return {"ok": True, "reason": "画像が空（スキップ）", "fix_hint": ""}

    # 検証用に縮小（長辺1024px・JPEG）してから送る。
    # メモリ・アップロード時間・Vision のトークン/コストを大幅に削減する。
    # 図解の良し悪し・文字化け判定には 1024px で十分。失敗時は元画像にフォールバック。
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(img_bytes)) as im:
            im = im.convert("RGB")
            im.thumbnail((1024, 1024))  # 長辺1024pxへ（アスペクト維持）
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            small = buf.getvalue()
        b64 = base64.standard_b64encode(small).decode("ascii")
        media_type = "image/jpeg"
    except Exception:
        b64 = base64.standard_b64encode(img_bytes).decode("ascii")
        ext = p.suffix.lower()
        media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else ("image/webp" if ext == ".webp" else "image/png")
    finally:
        img_bytes = None  # 元のバイト列を早期解放（メモリ削減）

    terms_note = ""
    terms = [t for t in (allowed_terms or []) if isinstance(t, str) and t.strip()]
    if terms:
        terms_note = f"\n画像に入ってよい日本語ラベル: {', '.join(terms)}"

    # 文脈ブロック（テーマ・章・前後段落）を組み立てる
    ctx_parts = []
    if theme.strip():
        ctx_parts.append(f"【動画全体のテーマ】{theme.strip()[:120]}")
    if chapter.strip():
        ctx_parts.append(f"【この図が属する章】{chapter.strip()}")
    if block_context.strip():
        ctx_parts.append(f"【前後の文脈（この文を含む段落）】\n{block_context.strip()[:500]}")
    context_block = "\n".join(ctx_parts)
    if context_block:
        context_block = "===== 原稿の文脈 =====\n" + context_block + "\n=====================\n\n"

    system = (
        "あなたは厳しい図解レビュアーです。動画原稿の文脈を踏まえて、画像がその文の"
        "意味を正しく・分かりやすく表現できているかを評価します。"
        "結果は JSON オブジェクトのみで返してください。"
    )
    query = f"""{context_block}この画像は、上の文脈の中の次の1文の図解（type={img_type}）として生成されました:
「{sentence}」{terms_note}

**文脈を踏まえて**、次の観点で厳しくチェックしてください:
1. 文脈の中でこの文が「本当に伝えたい内容」を正しく表しているか
   （例: 主語・対象・何が変化したか等が文脈で決まる場合、それを正しく描けているか。
    文単独では曖昧でも、文脈から読み取れる正しい内容と図がズレていないか）
2. 図の内容が原稿の主張と矛盾・的外れになっていないか
3. 画像内の文字に文字化け・誤字・読めない崩れた文字がないか
4. 重要な要素（数値・関係・対比・フロー・主体）が抜けていないか
5. ぱっと見て内容が伝わるか

以下の JSON のみで返答:
{{"ok": true または false, "reason": "判定理由を40字以内の日本語で", "fix_hint": "再生成時の改善指示を英語で80字以内（okがfalseのとき必須・文脈の正しい内容を反映）"}}

文脈と図がズレている／文字が崩れている場合は ok=false にしてください。"""

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=500,
            system=system,
            timeout=45,  # 検証が固まって全体を止めないようタイムアウト（短めに）
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": query},
                ],
            }],
        )
    except Exception as e:
        print(f"  [verifier ERROR] {str(e)[:120]}", flush=True)
        return {"ok": True, "reason": "検証API失敗（スキップ）", "fix_hint": ""}

    text = "".join(getattr(b, "text", "") for b in response.content if hasattr(b, "text"))
    data = parse_json_object(text)
    if not data:
        return {"ok": True, "reason": "検証パース失敗（スキップ）", "fix_hint": ""}

    return {
        "ok": bool(data.get("ok", True)),
        "reason": str(data.get("reason", ""))[:60],
        "fix_hint": str(data.get("fix_hint", ""))[:200],
    }
