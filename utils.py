#!/usr/bin/env python3
"""共通ユーティリティ - JSON パースと Claude API 呼び出し"""

import json
import os
import time
from pathlib import Path

import anthropic


def load_env(project_root: Path) -> None:
    """.env ファイルから環境変数を読み込む

    既存の環境変数が空文字列のときも .env で上書きする
    （setdefault だと空文字を「設定済み」と判定してしまうため）。
    """
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        if val and val != "your_api_key_here":
            existing = os.environ.get(key, "")
            if not existing:  # 未設定 or 空文字列なら上書き
                os.environ[key] = val


def get_anthropic_client() -> anthropic.Anthropic:
    """Anthropic クライアントを取得（API キー未設定時はエラー）"""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("ANTHROPIC_API_KEY が設定されていません。.env を確認してください。")
    return anthropic.Anthropic(api_key=api_key)


def claude_query(
    client: anthropic.Anthropic,
    query: str,
    system: str,
    max_tokens: int = 4096,
    model: str = "claude-sonnet-4-6",
    max_retries: int = 3,
) -> str:
    """Claude API（Web 検索なし）でクエリを実行"""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": query}],
            )
            if not response or not response.content:
                if attempt == max_retries - 1:
                    return ""
                time.sleep(3)
                continue
            text_parts = [getattr(b, "text", "") for b in response.content if hasattr(b, "text")]
            return "\n".join(text_parts)
        except anthropic.RateLimitError:
            wait = 15 * (attempt + 1)
            print(f"  [RATE LIMIT] {wait}s 待機します...", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"  [ERROR] Claude API: {e}", flush=True)
            if attempt == max_retries - 1:
                return ""
            time.sleep(3)
    return ""


def parse_json_array(text: str) -> list:
    """テキストからJSON配列を抽出（コードブロック対応）"""
    if not text:
        return []
    text = text.strip()
    # ```json ... ``` を剥がす
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        text = text.strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 末尾欠損の修復試行
    for suffix in ['"}]', '}]', ']']:
        try:
            return json.loads(candidate + suffix)
        except json.JSONDecodeError:
            continue

    # 修復: シングルクォートをダブルクォートに置換（最終手段）
    try:
        return json.loads(candidate.replace("'", '"'))
    except json.JSONDecodeError:
        pass

    return []


def parse_json_object(text: str) -> dict:
    """テキストからJSONオブジェクトを抽出"""
    if not text:
        return {}
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        text = text.strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    for suffix in ['"}]}', '"}}', '}']:
        try:
            return json.loads(candidate + suffix)
        except json.JSONDecodeError:
            continue

    return {}


def save_json(path: Path, data) -> None:
    """JSON ファイルを保存（ディレクトリも作成）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: Path, default=None):
    """JSON ファイルを読み込み"""
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default if default is not None else {}
