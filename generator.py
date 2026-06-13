#!/usr/bin/env python3
"""Phase 3: 並列画像生成エージェント（マルチプロバイダ対応）

asyncio + Semaphore で同時 N 枚を並列生成する。
プロバイダは 2 種類から選択可:
  - "nanobanana": Google Gemini Flash Image（高速・安価・16:9 自然）
  - "gpt-image":  OpenAI gpt-image-2（高品質・テキスト精度高）

両 SDK は同期 API なので loop.run_in_executor で thread pool に委譲する。
"""

import asyncio
import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

# Gemini SDK
from google import genai
from google.genai import types

# Pillow（クロップ用）
from PIL import Image


# ===== モデル設定 =====
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_OPENAI_MODEL = "gpt-image-2"
DEFAULT_CONCURRENCY = 12
MAX_RETRIES = 2                # リトライ回数（多いとレート制限時に1枚が長時間スレッドを占有する）
API_TIMEOUT_SEC = 120          # 1 回の画像生成 API 呼び出しの上限（応答停止対策）
PER_IMAGE_HARD_TIMEOUT = 360   # 1 枚あたりの全体上限（リトライ込み・asyncio 側の最終防衛線）
# 重要: MAX_RETRIES × API_TIMEOUT_SEC + リトライsleep合計 < PER_IMAGE_HARD_TIMEOUT を必ず満たす。
# 満たさないと wait_for が先に発火し、実行中スレッドが残留（ゾンビ化）→ 実効並列が枯渇する。
# 現状: 2×120 + (12+20) = 272s < 360s ✓

# 出力アスペクト比（16:9 に統一）
TARGET_RATIO = 16 / 9


def _detect_background_color(img: "Image.Image") -> tuple:
    """画像の四隅から背景色を推定する（最頻色）。図解の余白色になじませる用。"""
    w, h = img.size
    # 四隅 + 各辺中央の計 8 点をサンプリング
    points = [
        (1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2),
        (w // 2, 1), (w // 2, h - 2), (1, h // 2), (w - 2, h // 2),
    ]
    samples = []
    px = img.load()
    for x, y in points:
        try:
            c = px[x, y]
            if isinstance(c, int):  # グレースケール
                c = (c, c, c)
            samples.append(c[:3])
        except Exception:
            pass
    if not samples:
        return (255, 255, 255)
    # 最頻色を返す
    from collections import Counter
    most_common = Counter(samples).most_common(1)[0][0]
    return most_common


def _save_as_16_9(image_bytes: bytes, output_path: Path) -> None:
    """画像バイト列を 16:9 で保存する（レターボックス方式）。

    クロップせず画像全体を 16:9 フレーム内に収め、はみ出さないようにする。
    余白は四隅から検出した背景色で埋めるので、図解の端が欠けない。

    - 既に 16:9 ならそのまま保存
    - 横長すぎ（例 OpenAI 3:2）→ 上下に余白を足す
    - 縦長すぎ → 左右に余白を足す
    """
    img = Image.open(BytesIO(image_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    w, h = img.size
    current = w / h if h else 1.0

    if abs(current - TARGET_RATIO) < 0.01:
        # 既にほぼ 16:9 → そのまま
        img.save(output_path, format="PNG")
        return

    # 背景色を検出（余白に使う）
    bg = _detect_background_color(img)

    if current > TARGET_RATIO:
        # 横長すぎ → 幅基準でキャンバスを作り、上下に余白
        canvas_w = w
        canvas_h = int(round(w / TARGET_RATIO))
    else:
        # 縦長すぎ → 高さ基準でキャンバスを作り、左右に余白
        canvas_h = h
        canvas_w = int(round(h * TARGET_RATIO))

    # 16:9 キャンバスを背景色で作成し、元画像を中央に配置（縮小なし＝全体保持）
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    offset_x = (canvas_w - w) // 2
    offset_y = (canvas_h - h) // 2
    if img.mode == "RGBA":
        canvas.paste(img, (offset_x, offset_y), img)
    else:
        canvas.paste(img, (offset_x, offset_y))

    canvas.save(output_path, format="PNG")


# ===== v3 Step5: realphoto キャプション焼き込み（報道映像との誤認防止）=====
_CAPTION_FONT_DIR = Path(__file__).parent / "assets" / "fonts"
_caption_font_cache = {}


def _caption_font(size: int):
    from PIL import ImageFont
    if size in _caption_font_cache:
        return _caption_font_cache[size]
    f = None
    p = _CAPTION_FONT_DIR / "NotoSansJP-Bold.ttf"
    try:
        if p.exists():
            f = ImageFont.truetype(str(p), size)
    except Exception:
        f = None
    if f is None:
        f = ImageFont.load_default()
    _caption_font_cache[size] = f
    return f


def add_image_caption(path, text: str = "イメージ") -> bool:
    """画像の右下に半透明の「イメージ」キャプションを焼き込む（realphoto 用）。"""
    try:
        from PIL import Image, ImageDraw
        im = Image.open(path).convert("RGBA")
        W, H = im.size
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        fsize = max(22, H // 30)
        font = _caption_font(fsize)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = max(8, fsize // 3)
        m = max(10, H // 60)
        x1 = W - tw - pad * 2 - m
        y1 = H - th - pad * 2 - m
        d.rounded_rectangle([x1, y1, W - m, H - m], radius=pad, fill=(0, 0, 0, 130))
        d.text((x1 + pad - bbox[0], y1 + pad - bbox[1]), text, font=font,
               fill=(255, 255, 255, 230))
        Image.alpha_composite(im, overlay).convert("RGB").save(path)
        return True
    except Exception as e:
        print(f"  [caption ERROR] {str(e)[:80]}", flush=True)
        return False


# プロバイダ識別子
PROVIDER_NANOBANANA = "nanobanana"
PROVIDER_GPT_IMAGE = "gpt-image"
VALID_PROVIDERS = (PROVIDER_NANOBANANA, PROVIDER_GPT_IMAGE)

# 参照画像（キャラ固定）を使うときにプロンプト先頭へ付ける指示。
# 参照画像の「人物」と「絵柄トーン」を新しいシーンでも忠実に再現させる。
_CHARACTER_LOCK_INSTRUCTION = (
    "CHARACTER & STYLE REFERENCE: The attached reference image shows the EXACT recurring "
    "character (the teacher / professor) and the EXACT art style for this video series. "
    "Reproduce the SAME person — identical face shape, hairstyle, glasses, skin tone and outfit "
    "(gray tweed blazer over a dark red V-neck sweater) — and the SAME simple, clean, "
    "thick-outline FLAT CARTOON tone, line weight and flat coloring as the reference image. "
    "Only change the pose, expression and background to fit the new scene described below. "
    "Do NOT redesign the character, and do NOT switch to a detailed, anime, or realistic style."
)

# v3 Step6: エンティティ参照（一貫性ロック）。同じ被写体が繰り返し出るとき、初出画像を
# 参照にして見た目を揃える。nanobanana は参照画像つき、gpt-image は文言のみ（v3.0）。
_ENTITY_LOCK_INSTRUCTION = (
    "CONSISTENCY REFERENCE: The attached reference image shows the SAME recurring subject that "
    "appears earlier in this video. Maintain the SAME visual design, colors, shapes and overall "
    "look as the reference image for this subject, changing only the composition to fit the new "
    "scene described below. Keep it visually consistent so the video feels like one series."
)
_ENTITY_LOCK_TEXT_ONLY = (
    "CONSISTENCY NOTE: This scene shows a recurring subject that appears multiple times in this "
    "video. Keep its visual design, colors and overall look consistent with a clean, unified "
    "series style, so repeated appearances of the same subject feel coherent."
)


def _build_full_prompt(
    user_prompt: str,
    prompt_type: str = "illustration",
    allowed_terms: Optional[list] = None,
    style_preset: str = "",
) -> str:
    """画像生成用のシステム接頭辞を付与

    allowed_terms には「画像内に入れて良い日本語の語句」のホワイトリストを渡す。
    リストに無い文字・ラベル・数値はすべて画像から除外するよう AI に厳格指示する。
    """
    style_hints = {
        "illustration": (
            "Style: choose the most fitting illustration style for the content "
            "(watercolor, flat, line art, paper-cut, 3D rendered, comic, or minimal). "
        ),
        "realphoto": (
            "Style: PHOTOREALISTIC photograph, documentary / photojournalism quality. "
            "Looks like a real photo taken with a professional camera: natural lighting, "
            "realistic textures, depth of field, true-to-life colors. "
            "Cinematic composition suitable for a documentary video. "
            "Depict the actual physical scene (city, building, facility, infrastructure, "
            "event, war scene, or people's daily life) realistically. "
            "NOT an illustration, NOT a cartoon, NOT a flat graphic — a real photograph. "
        ),
        "map": (
            "Style: PHOTOREALISTIC satellite / aerial map, like NASA Blue Marble, "
            "Google Earth, or National Geographic cartography. "
            "Real-looking Earth surface seen from above: accurate natural colors "
            "(deep blue oceans, green forests, brown/tan deserts and mountains, "
            "white snow and ice, realistic coastlines and rivers). "
            "Add subtle relief shading and terrain texture for a 3D sense of landforms. "
            "Render country/region borders as thin clean lines and shade relevant "
            "territories with semi-transparent color overlays so they stand out. "
            "Professional, high-resolution, documentary-quality geographic map. "
            "NOT a flat cartoon, NOT a simplified illustration — make it look like a real map. "
        ),
        "diagram": (
            "Style: clean conceptual diagram with arrows and 3-5 boxes. "
            "Minimal lines, clear structure, easy to understand at a glance. "
        ),
        "chart": (
            "Style: clean chart (bar / pie / line graph) with 3-5 data elements. "
        ),
    }
    style = style_hints.get(prompt_type, style_hints["illustration"])

    # 画像内テキストのホワイトリスト指示（最重要）
    terms = [t for t in (allowed_terms or []) if isinstance(t, str) and t.strip()]
    if prompt_type == "realphoto":
        # 実写写真は「現地のリアルな看板・標識」が映える。日本語ラベルは載せない。
        text_policy = (
            "*** TEXT POLICY for a REALISTIC PHOTO (CRITICAL) ***\n"
            "- Do NOT add Japanese text, informational labels, captions, titles, or annotations.\n"
            "- Any signage, shop signs, street signs, billboards, or text that naturally appears "
            "in the scene MUST be in the LOCAL LANGUAGE of the depicted real-world location "
            "(e.g., Russian / Cyrillic for a scene in Russia or the USSR; the local language "
            "for other countries). NEVER use Japanese on signs in a foreign scene.\n"
            "- Keep such incidental text minimal and natural, like in a real documentary photo.\n"
        )
    elif terms:
        terms_str = ", ".join(terms)
        text_policy = (
            "*** TEXT POLICY (CRITICAL — VIOLATION IS UNACCEPTABLE) ***\n"
            f"- The ONLY Japanese text/labels/numbers allowed in this image are EXACTLY these: {terms_str}\n"
            "- DO NOT invent or add ANY other text, labels, place names, numbers, captions, or annotations.\n"
            "- DO NOT translate or paraphrase the allowed terms; render them character-for-character.\n"
            "- DO NOT include English text of any kind.\n"
            "- If unsure whether a piece of text is in the allowed list, OMIT it.\n"
            "- DO NOT add any title text or heading at the top of the image.\n"
        )
    else:
        text_policy = (
            "*** TEXT POLICY (CRITICAL) ***\n"
            "- NO text in this image. Purely visual representation only.\n"
            "- DO NOT add any labels, numbers, place names, captions, or annotations.\n"
            "- DO NOT include any Japanese or English text.\n"
            "- DO NOT add any title text or heading.\n"
        )

    common = (
        "OTHER REQUIREMENTS:\n"
        "- 16:9 landscape aspect ratio for video presentation (WIDE, not square, not tall).\n"
        "- COMPOSITION (CRITICAL): The ENTIRE subject/diagram/figure MUST be fully visible inside the frame.\n"
        "  Keep a generous safe margin (at least 10% padding) around all edges.\n"
        "  Do NOT let any part of the figure, text, icon, or chart touch or extend beyond the edges.\n"
        "  Zoom out / use a wider view so nothing is cropped or cut off.\n"
        "- Center the main content with comfortable empty space around it.\n"
        "- Simple, clear, professional. Avoid clutter.\n"
    )
    return f"{style}\n{text_policy}{common}\nContent to visualize:\n{user_prompt}"


# ===== Gemini (nanobanana) =====
def _sync_generate_image_gemini(
    client: genai.Client,
    full_prompt: str,
    output_path: Path,
    model_name: str = DEFAULT_GEMINI_MODEL,
    reference_bytes: Optional[bytes] = None,  # キャラ固定の参照画像
    reference_mime: str = "image/png",
) -> tuple:
    """1 枚の画像を同期生成（Gemini）。

    reference_bytes を渡すと参照画像 + テキストのマルチモーダル入力で生成し、
    参照画像のキャラ・絵柄を新シーンに反映する（キャラ固定）。
    """
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            if reference_bytes:
                contents = [
                    types.Part.from_bytes(data=reference_bytes, mime_type=reference_mime),
                    full_prompt,
                ]
            else:
                contents = full_prompt
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )

            parts_iter = []
            if hasattr(response, "parts") and response.parts:
                parts_iter = response.parts
            elif hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if cand.content and cand.content.parts:
                    parts_iter = cand.content.parts

            for part in parts_iter:
                inline = getattr(part, "inline_data", None)
                if inline and inline.mime_type and inline.mime_type.startswith("image/"):
                    img_data = inline.data
                    if isinstance(img_data, str):
                        img_data = base64.b64decode(img_data)
                    _save_as_16_9(img_data, output_path)
                    return True, ""

            last_error = "no image in response"
            for part in parts_iter:
                txt = getattr(part, "text", None)
                if txt:
                    last_error = f"text only: {txt[:120]}"
                    break

        except Exception as e:
            err = str(e)
            last_error = err[:200]
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                time.sleep(30 * (attempt + 1))
                continue
            if "safety" in err.lower() or "block" in err.lower():
                return False, f"safety blocked: {err[:120]}"
            if "not found" in err.lower() or "404" in err:
                return False, f"model not available: {model_name}"
            time.sleep(3 + 2 * attempt)

    return False, last_error or "max retries exceeded"


# ===== OpenAI (gpt-image-2) =====
def _sync_generate_image_openai(
    client,  # openai.OpenAI
    full_prompt: str,
    output_path: Path,
    model_name: str = DEFAULT_OPENAI_MODEL,
    size: str = "1536x1024",  # 3:2 横長（16:9 に最も近い）
    quality: str = "medium",  # low / medium / high
    reference_bytes: Optional[bytes] = None,  # キャラ固定の参照画像（あれば images.edit）
) -> tuple:
    """1 枚の画像を同期生成（OpenAI gpt-image）。

    reference_bytes を渡すと images.edit を使い、参照画像のキャラ・絵柄を
    反映した新シーンを生成する（キャラ固定）。無ければ通常の images.generate。
    """
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            if reference_bytes:
                import io
                bio = io.BytesIO(reference_bytes)
                bio.name = "reference.png"  # SDK が拡張子から MIME を判定
                response = client.images.edit(
                    model=model_name,
                    image=bio,
                    prompt=full_prompt,
                    n=1,
                    size=size,
                    quality=quality,
                )
            else:
                response = client.images.generate(
                    model=model_name,
                    prompt=full_prompt,
                    n=1,
                    size=size,
                    quality=quality,
                )

            if not response or not response.data:
                last_error = "no data in response"
                continue

            datum = response.data[0]
            # 新 API: b64_json で base64 が返る
            b64 = getattr(datum, "b64_json", None)
            url = getattr(datum, "url", None)

            if b64:
                img_data = base64.b64decode(b64)
                _save_as_16_9(img_data, output_path)
                return True, ""
            elif url:
                # URL なら fetch
                import urllib.request
                with urllib.request.urlopen(url, timeout=60) as r:
                    _save_as_16_9(r.read(), output_path)
                return True, ""
            else:
                last_error = "neither b64_json nor url in response"

        except Exception as e:
            err = str(e)
            last_error = err[:200]
            err_lower = err.lower()
            if "429" in err or "rate" in err_lower or "limit" in err_lower:
                # レート制限。待ちすぎるとスレッドを長時間占有するので上限20秒。
                time.sleep(min(20, 12 * (attempt + 1)))
                continue
            if "safety" in err_lower or "policy" in err_lower or "moderation" in err_lower:
                return False, f"content policy blocked: {err[:120]}"
            if "404" in err or "model_not_found" in err_lower or "does not exist" in err_lower:
                return False, f"model not available: {model_name} ({err[:80]})"
            if "401" in err or "invalid api key" in err_lower:
                return False, f"invalid OpenAI API key: {err[:80]}"
            time.sleep(3 + 2 * attempt)

    return False, last_error or "max retries exceeded"


# ===== 並列ジェネレータ =====
class ParallelImageGenerator:
    """asyncio + Semaphore による並列画像生成器（マルチプロバイダ）"""

    def __init__(
        self,
        provider: str,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        openai_quality: str = "medium",
        openai_size: str = "1536x1024",
        concurrency: int = DEFAULT_CONCURRENCY,
        style_preset: str = "",
        progress_callback: Optional[Callable[[dict], None]] = None,
        reference_image_path: Optional[str] = None,  # キャラ固定の参照画像パス
        realphoto_watermark: bool = False,  # v3 Step5: realphoto に「イメージ」焼き込み
    ):
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"unknown provider: {provider} (valid: {VALID_PROVIDERS})")
        self.provider = provider
        self.openai_quality = openai_quality
        self.openai_size = openai_size
        self.style_preset = style_preset
        self.realphoto_watermark = bool(realphoto_watermark)

        # 参照画像（キャラ固定用）。character=True のシーンでのみ使用。
        # ファイルが無ければ None（=テキスト方式に自動フォールバック、壊れない）。
        self.reference_bytes = None
        self.reference_mime = "image/png"
        if reference_image_path:
            try:
                rp = Path(reference_image_path)
                if rp.exists() and rp.stat().st_size > 200:
                    self.reference_bytes = rp.read_bytes()
                    self.reference_mime = "image/jpeg" if rp.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            except Exception:
                self.reference_bytes = None

        # クライアント初期化（必要な分だけ）
        self.gemini_client = None
        self.openai_client = None
        self.gemini_model = gemini_model or DEFAULT_GEMINI_MODEL
        self.openai_model = openai_model or DEFAULT_OPENAI_MODEL

        if provider == PROVIDER_NANOBANANA:
            if not gemini_api_key:
                raise RuntimeError("nanobanana を使うには GEMINI_API_KEY が必要です")
            # HTTP タイムアウト（ミリ秒）を設定し、応答が止まった呼び出しを必ず失敗させる
            try:
                self.gemini_client = genai.Client(
                    api_key=gemini_api_key,
                    http_options=types.HttpOptions(timeout=API_TIMEOUT_SEC * 1000),
                )
            except Exception:
                # 古い SDK で http_options 非対応の場合は従来通り
                self.gemini_client = genai.Client(api_key=gemini_api_key)
        elif provider == PROVIDER_GPT_IMAGE:
            if not openai_api_key:
                raise RuntimeError("gpt-image を使うには OPENAI_API_KEY が必要です")
            import openai  # 遅延 import
            # タイムアウト + リトライ0（リトライは自前で制御）
            self.openai_client = openai.OpenAI(api_key=openai_api_key, timeout=API_TIMEOUT_SEC, max_retries=0)

        self.concurrency = max(1, min(concurrency, 32))
        self.progress_callback = progress_callback or (lambda info: None)
        self._executor = None  # generate_all で専用 ThreadPoolExecutor を割り当てる
        # Lock は async 関数内で生成する（Python 3.9 対策）
        self._counter_lock: Optional[asyncio.Lock] = None
        # v3 Step6: エンティティ canonical の完成イベント／確保バイト列（generate_all で構築）
        self._canonical_events: dict = {}
        self._canonical_bytes: dict = {}
        self._completed = 0
        self._failed = 0
        self._total = 0

    def _dispatch_sync_generate(self, full_prompt: str, output_path: Path,
                                use_reference: bool = False,
                                ref_bytes_override: Optional[bytes] = None,
                                ref_mime_override: Optional[str] = None) -> tuple:
        """provider に応じた同期生成関数を呼び分ける。

        use_reference=True かつ参照画像があれば、キャラ固定モードで生成する。
        ref_bytes_override を渡すと、その画像（v3 Step6 のエンティティ canonical 画像など）を
        参照として使う（self.reference_bytes より優先）。
        """
        if ref_bytes_override is not None:
            ref = ref_bytes_override
            ref_mime = ref_mime_override or "image/png"
        else:
            ref = self.reference_bytes if use_reference else None
            ref_mime = self.reference_mime
        if self.provider == PROVIDER_NANOBANANA:
            return _sync_generate_image_gemini(
                self.gemini_client, full_prompt, output_path, self.gemini_model,
                reference_bytes=ref, reference_mime=ref_mime,
            )
        else:  # gpt-image
            return _sync_generate_image_openai(
                self.openai_client, full_prompt, output_path,
                model_name=self.openai_model,
                size=self.openai_size,
                quality=self.openai_quality,
                reference_bytes=ref,
            )

    async def _generate_one(
        self,
        prompt_entry: dict,
        output_dir: Path,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        """1 枚を生成（semaphore で並列度制御）"""
        idx = prompt_entry.get("index", 0)
        prompt_text = prompt_entry.get("prompt", "")
        prompt_type = prompt_entry.get("type", "illustration")
        section = prompt_entry.get("section", "")
        excerpt = prompt_entry.get("excerpt", "")
        keypoint = prompt_entry.get("keypoint", "")
        allowed_terms = prompt_entry.get("allowed_terms", [])
        # 行ごとのスタイル指定があればそれを優先（プロパガンダ・ミックス用）
        row_style = prompt_entry.get("style") or self.style_preset
        filename = f"{idx}.png"  # 数字だけのファイル名（№と一致）
        output_path = output_dir / filename

        # v3 Step6: エンティティ follower は、参照する canonical 画像の完成を待ってから
        # セマフォを取りに行く（待機中はスロットを占有しない＝canonical が確実に進める）。
        # ここで待つことで「直列化はエンティティ内のみ」を満たし、デッドロックを避ける。
        entity_role = prompt_entry.get("entity_role")
        entity_ref_bytes = None
        # 参照画像の受け渡しは nanobanana のみ（gpt-image は文言のみ＝待機不要）。
        if entity_role == "follower" and self.provider == PROVIDER_NANOBANANA:
            canon_idx = prompt_entry.get("entity_ref_of")
            ev = self._canonical_events.get(canon_idx) if self._canonical_events else None
            if ev is not None:
                try:
                    await asyncio.wait_for(ev.wait(), timeout=PER_IMAGE_HARD_TIMEOUT)
                except asyncio.TimeoutError:
                    pass  # canonical が間に合わなければ参照なしで進む（壊さない）
                entity_ref_bytes = (self._canonical_bytes or {}).get(canon_idx)

        async with semaphore:
            self.progress_callback({
                "index": idx,
                "status": "generating",
                "section": section,
                "keypoint": keypoint,
                "excerpt": excerpt,
                "provider": self.provider,
            })

            full_prompt = _build_full_prompt(prompt_text, prompt_type, allowed_terms=allowed_terms, style_preset=row_style)
            # キャラ固定: character=True かつ参照画像があるシーンだけ参照モードで生成
            use_reference = bool(prompt_entry.get("character")) and self.reference_bytes is not None
            if use_reference:
                full_prompt = _CHARACTER_LOCK_INSTRUCTION + "\n\n" + full_prompt

            # v3 Step6: エンティティ follower の一貫性ロック（キャラ固定シーンとは排他）。
            # nanobanana は canonical 画像を参照に渡す。gpt-image は文言のみ（v3.0）。
            ref_override = None
            if entity_role == "follower" and not use_reference:
                if self.provider == PROVIDER_NANOBANANA and entity_ref_bytes:
                    ref_override = entity_ref_bytes
                    full_prompt = _ENTITY_LOCK_INSTRUCTION + "\n\n" + full_prompt
                else:
                    full_prompt = _ENTITY_LOCK_TEXT_ONLY + "\n\n" + full_prompt

            loop = asyncio.get_running_loop()

            try:
                # 専用 executor + asyncio.wait_for で、1 枚が固まっても
                # 全体（asyncio.gather）を巻き込まないようハード上限を設ける
                success, error = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        self._dispatch_sync_generate,
                        full_prompt,
                        output_path,
                        use_reference,
                        ref_override,
                    ),
                    timeout=PER_IMAGE_HARD_TIMEOUT,
                )
            except asyncio.TimeoutError:
                success, error = False, f"timeout (> {PER_IMAGE_HARD_TIMEOUT}s)"
            except Exception as e:
                success, error = False, str(e)[:200]

            # v3 Step5: realphoto は「イメージ」を焼き込む（報道映像との誤認防止）
            if success and prompt_type == "realphoto" and self.realphoto_watermark:
                try:
                    await loop.run_in_executor(self._executor, add_image_caption, output_path)
                except Exception:
                    pass

            # v3 Step6: canonical は完成画像を後続(follower)の参照用に確保し、待機を解除する。
            # 失敗時も必ず解除して follower を無限待機させない（参照なしで進む）。nanobanana のみ。
            if entity_role == "canonical" and self.provider == PROVIDER_NANOBANANA:
                if success and self._canonical_bytes is not None:
                    try:
                        self._canonical_bytes[idx] = output_path.read_bytes()
                    except Exception:
                        self._canonical_bytes[idx] = None
                ev = self._canonical_events.get(idx) if self._canonical_events else None
                if ev is not None:
                    ev.set()

            async with self._counter_lock:
                if success:
                    self._completed += 1
                else:
                    self._failed += 1
                completed_now = self._completed
                failed_now = self._failed

            # メモリ解放: 一定枚数ごとに GC（512MB 環境での OOM 緩和。ロック外で実行）
            if (completed_now + failed_now) % 10 == 0:
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass

            result = {
                "index": idx,
                "filename": filename if success else None,
                "section": section,
                "excerpt": excerpt,
                "keypoint": keypoint,
                "allowed_terms": allowed_terms,
                "type": prompt_type,
                "prompt": prompt_text,
                "provider": self.provider,
                "success": success,
                "error": error if not success else "",
            }

            self.progress_callback({
                "index": idx,
                "status": "ok" if success else "failed",
                "section": section,
                "keypoint": keypoint,
                "excerpt": excerpt,
                "filename": filename if success else None,
                "error": error if not success else "",
                "provider": self.provider,
                "completed_total": completed_now,
                "failed_total": failed_now,
                "grand_total": self._total,
            })

            return result

    async def generate_all(self, prompts: list, output_dir: Path) -> list:
        """全プロンプトを並列生成"""
        from concurrent.futures import ThreadPoolExecutor
        output_dir.mkdir(parents=True, exist_ok=True)
        self._completed = 0
        self._failed = 0
        self._total = len(prompts)

        # asyncio オブジェクトは running loop の中で生成する
        self._counter_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.concurrency)
        # v3 Step6: エンティティ canonical の完成イベント群を running loop 内で生成。
        # follower は対応する canonical のイベントを待ってから生成する（nanobanana のみ）。
        self._canonical_events = {}
        self._canonical_bytes = {}
        if self.provider == PROVIDER_NANOBANANA:
            for p in prompts:
                if p.get("entity_role") == "canonical":
                    ci = p.get("index")
                    if ci is not None:
                        self._canonical_events[ci] = asyncio.Event()
                        self._canonical_bytes[ci] = None
        # 専用スレッドプール（デフォルト executor のスレッド枯渇を防ぐ）。
        # 固まったスレッドが居ても新しい画像生成が進められるよう余裕を持たせる
        self._executor = ThreadPoolExecutor(max_workers=self.concurrency + 4)
        # 全体の時間予算（安全網）: 超えたら未完了を打ち切り、必ず完了させる。
        # 自然完了の理論上限（バッチ数 × 1枚ハード上限）＋余裕に設定するので、
        # 正常な「遅いだけ」のジョブは切らず、真の暴走（無限ハング）だけを止める。最大4時間。
        batches = (len(prompts) + self.concurrency - 1) // max(1, self.concurrency)
        overall_budget = min(14400, max(1800, batches * (PER_IMAGE_HARD_TIMEOUT + 90)))
        try:
            tasks = [
                asyncio.ensure_future(self._generate_one(p, output_dir, semaphore))
                for p in prompts
            ]
            done, pending = await asyncio.wait(tasks, timeout=overall_budget)
            if pending:
                print(f"  [generator] 時間予算({overall_budget}s)超過: 未完了 {len(pending)} 枚を打ち切り", flush=True)
        finally:
            self._executor.shutdown(wait=False)

        # 入力順に正規化。未完了はキャンセルして「失敗」確定（行の🌀を必ず消す）。
        results = []
        for i, t in enumerate(tasks):
            idx = prompts[i].get("index", i + 1) if i < len(prompts) else i + 1
            if t in done:
                try:
                    r = t.result()
                    results.append(r if isinstance(r, dict) else {
                        "index": idx, "filename": None, "success": False,
                        "error": f"task error: {str(r)[:120]}"})
                except Exception as e:
                    results.append({"index": idx, "filename": None, "success": False,
                                    "error": f"task error: {str(e)[:120]}"})
            else:
                t.cancel()
                try:
                    self.progress_callback({
                        "index": idx, "status": "failed", "provider": self.provider,
                        "error": "時間上限により打ち切り（並列度を下げる/枚数を分割してください）",
                    })
                except Exception:
                    pass
                results.append({"index": idx, "filename": None, "success": False,
                                "error": "時間上限により打ち切り"})
        return results


def run_parallel_generation(
    prompts: list,
    output_dir: Path,
    provider: str = PROVIDER_NANOBANANA,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_quality: str = "medium",
    openai_size: str = "1536x1024",
    concurrency: int = DEFAULT_CONCURRENCY,
    style_preset: str = "",
    progress_callback: Optional[Callable[[dict], None]] = None,
    reference_image_path: Optional[str] = None,  # キャラ固定の参照画像
    realphoto_watermark: bool = False,  # v3 Step5: realphoto に「イメージ」焼き込み
) -> list:
    """同期エントリポイント: pipeline から呼び出す"""
    # 環境変数からデフォルト補完
    if gemini_api_key is None:
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if gemini_model is None:
        gemini_model = os.environ.get("GEMINI_IMAGE_MODEL", DEFAULT_GEMINI_MODEL)
    if openai_model is None:
        openai_model = os.environ.get("OPENAI_IMAGE_MODEL", DEFAULT_OPENAI_MODEL)

    generator = ParallelImageGenerator(
        provider=provider,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        gemini_model=gemini_model,
        openai_model=openai_model,
        openai_quality=openai_quality,
        openai_size=openai_size,
        concurrency=concurrency,
        style_preset=style_preset,
        progress_callback=progress_callback,
        reference_image_path=reference_image_path,
        realphoto_watermark=realphoto_watermark,
    )
    return asyncio.run(generator.generate_all(prompts, output_dir))
