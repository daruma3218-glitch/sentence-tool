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
MAX_RETRIES = 3

# 出力アスペクト比（16:9 に統一）
TARGET_RATIO = 16 / 9


def _save_as_16_9(image_bytes: bytes, output_path: Path) -> None:
    """画像バイト列を中央クロップして 16:9 で保存する。

    - 既に 16:9 ならそのまま保存
    - 横長すぎ（例 OpenAI 3:2）→ 左右をクロップ
    - 縦長すぎ → 上下をクロップ
    """
    img = Image.open(BytesIO(image_bytes))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    w, h = img.size
    current = w / h if h else 1.0

    if abs(current - TARGET_RATIO) < 0.01:
        # 既にほぼ 16:9
        img.save(output_path, format="PNG")
        return

    if current > TARGET_RATIO:
        # 横長すぎ → 左右をクロップ
        new_w = int(round(h * TARGET_RATIO))
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # 縦長すぎ → 上下をクロップ
        new_h = int(round(w / TARGET_RATIO))
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    img.save(output_path, format="PNG")

# プロバイダ識別子
PROVIDER_NANOBANANA = "nanobanana"
PROVIDER_GPT_IMAGE = "gpt-image"
VALID_PROVIDERS = (PROVIDER_NANOBANANA, PROVIDER_GPT_IMAGE)


def _build_full_prompt(
    user_prompt: str,
    prompt_type: str = "illustration",
    allowed_terms: Optional[list] = None,
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
        "map": (
            "Style: aerial photograph / satellite imagery style map. "
            "Show geographical features clearly. "
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
    if terms:
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
        "- 16:9 landscape aspect ratio for video presentation.\n"
        "- Simple, clear, professional. Avoid clutter.\n"
    )
    return f"{style}\n{text_policy}{common}\nContent to visualize:\n{user_prompt}"


# ===== Gemini (nanobanana) =====
def _sync_generate_image_gemini(
    client: genai.Client,
    full_prompt: str,
    output_path: Path,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> tuple:
    """1 枚の画像を同期生成（Gemini）"""
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
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
) -> tuple:
    """1 枚の画像を同期生成（OpenAI gpt-image-2）"""
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
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
                time.sleep(30 * (attempt + 1))
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
        progress_callback: Optional[Callable[[dict], None]] = None,
    ):
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"unknown provider: {provider} (valid: {VALID_PROVIDERS})")
        self.provider = provider
        self.openai_quality = openai_quality
        self.openai_size = openai_size

        # クライアント初期化（必要な分だけ）
        self.gemini_client = None
        self.openai_client = None
        self.gemini_model = gemini_model or DEFAULT_GEMINI_MODEL
        self.openai_model = openai_model or DEFAULT_OPENAI_MODEL

        if provider == PROVIDER_NANOBANANA:
            if not gemini_api_key:
                raise RuntimeError("nanobanana を使うには GEMINI_API_KEY が必要です")
            self.gemini_client = genai.Client(api_key=gemini_api_key)
        elif provider == PROVIDER_GPT_IMAGE:
            if not openai_api_key:
                raise RuntimeError("gpt-image を使うには OPENAI_API_KEY が必要です")
            import openai  # 遅延 import
            self.openai_client = openai.OpenAI(api_key=openai_api_key)

        self.concurrency = max(1, min(concurrency, 32))
        self.progress_callback = progress_callback or (lambda info: None)
        # Lock は async 関数内で生成する（Python 3.9 対策）
        self._counter_lock: Optional[asyncio.Lock] = None
        self._completed = 0
        self._failed = 0
        self._total = 0

    def _dispatch_sync_generate(self, full_prompt: str, output_path: Path) -> tuple:
        """provider に応じた同期生成関数を呼び分ける"""
        if self.provider == PROVIDER_NANOBANANA:
            return _sync_generate_image_gemini(
                self.gemini_client, full_prompt, output_path, self.gemini_model
            )
        else:  # gpt-image
            return _sync_generate_image_openai(
                self.openai_client, full_prompt, output_path,
                model_name=self.openai_model,
                size=self.openai_size,
                quality=self.openai_quality,
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
        filename = f"diagram_{idx:03d}.png"
        output_path = output_dir / filename

        async with semaphore:
            self.progress_callback({
                "index": idx,
                "status": "generating",
                "section": section,
                "keypoint": keypoint,
                "excerpt": excerpt,
                "provider": self.provider,
            })

            full_prompt = _build_full_prompt(prompt_text, prompt_type, allowed_terms=allowed_terms)
            loop = asyncio.get_running_loop()

            try:
                success, error = await loop.run_in_executor(
                    None,
                    self._dispatch_sync_generate,
                    full_prompt,
                    output_path,
                )
            except Exception as e:
                success, error = False, str(e)[:200]

            async with self._counter_lock:
                if success:
                    self._completed += 1
                else:
                    self._failed += 1
                completed_now = self._completed
                failed_now = self._failed

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
        output_dir.mkdir(parents=True, exist_ok=True)
        self._completed = 0
        self._failed = 0
        self._total = len(prompts)

        # asyncio オブジェクトは running loop の中で生成する
        self._counter_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [
            self._generate_one(p, output_dir, semaphore)
            for p in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
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
    progress_callback: Optional[Callable[[dict], None]] = None,
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
        progress_callback=progress_callback,
    )
    return asyncio.run(generator.generate_all(prompts, output_dir))
