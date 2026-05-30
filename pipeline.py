#!/usr/bin/env python3
"""メインパイプライン: 4 フェーズ統合

Phase 1: 原稿 → 章/ブロック/センテンス分解（Claude）
Phase 2a: センテンス → 英文画像プロンプト（Claude、並列バッチ）
Phase 2b: Web 画像 URL 取得（Claude Web Search、並列）※オプション
Phase 3: 英文プロンプト → 画像（gpt-image / nanobanana、asyncio 並列）
"""

import csv
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from utils import get_anthropic_client, save_json, load_json
from splitter import split_manuscript
from prompter import generate_all_prompts
from web_searcher import run_web_search, run_web_search_for_selections
from router import route_all_sentences, AI_ROUTES
from generator import (
    run_parallel_generation,
    DEFAULT_CONCURRENCY,
    PROVIDER_NANOBANANA,
    PROVIDER_GPT_IMAGE,
    VALID_PROVIDERS,
)


VALID_STYLES = ("flat_infographic", "pictogram", "comic", "whiteboard", "soviet_propaganda")
VALID_ROUTE_MODES = ("auto", "all_ai")


class SentencePipeline:
    """センテンス単位の図解生成パイプライン"""

    def __init__(
        self,
        manuscript_text: str,
        output_dir: Path,
        user_instructions: str = "",
        concurrency: int = DEFAULT_CONCURRENCY,
        provider: str = PROVIDER_NANOBANANA,
        openai_quality: str = "medium",
        style_preset: str = "flat_infographic",
        skip_decorative: bool = False,
        web_image_count: int = 0,
        max_diagrams: int = 150,
        route_mode: str = "auto",
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        item_callback: Optional[Callable] = None,
    ):
        self.manuscript_text = manuscript_text
        self.output_dir = Path(output_dir)
        self.user_instructions = user_instructions
        self.concurrency = concurrency
        self.provider = provider if provider in VALID_PROVIDERS else PROVIDER_NANOBANANA
        self.openai_quality = openai_quality
        self.style_preset = style_preset if style_preset in VALID_STYLES else "flat_infographic"
        self.skip_decorative = skip_decorative
        self.web_image_count = max(0, min(web_image_count, 200))
        self.max_diagrams = max(1, min(max_diagrams, 300))
        self.route_mode = route_mode if route_mode in VALID_ROUTE_MODES else "auto"
        self.progress_callback = progress_callback or (lambda phase, msg, pct: None)
        self.log_callback = log_callback or (lambda *a, **kw: None)
        self.item_callback = item_callback or (lambda info: None)

        self.images_dir = self.output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._rows_state: dict = {}
        self._rows_lock = threading.Lock()

    # ---- ヘルパ ----
    def _log(self, category: str, message: str, detail: str = ""):
        print(f"  [{category}] {message}" + (f" - {detail}" if detail else ""), flush=True)
        try:
            self.log_callback(category, message, detail)
        except Exception:
            pass

    def _progress(self, phase: int, message: str, percent: int):
        print(f"  [Phase {phase}] {message} ({percent}%)", flush=True)
        try:
            self.progress_callback(phase, message, percent)
        except Exception:
            pass

    def _update_row(self, no: int, **fields):
        with self._rows_lock:
            r = self._rows_state.get(no, {})
            r.update(fields)
            self._rows_state[no] = r
        self._dump_snapshot()
        try:
            self.item_callback({"no": no, **fields})
        except Exception:
            pass

    def _dump_snapshot(self):
        with self._rows_lock:
            rows = sorted(self._rows_state.values(), key=lambda x: x.get("no", 0))
        snapshot = {
            "rows": rows,
            "updated_at": datetime.now().isoformat(),
        }
        try:
            (self.output_dir / "rows_progress.json").write_text(
                json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    # ---- 画像配置ロジック ----
    @staticmethod
    def _select_evenly_distributed(candidates: list, max_count: int) -> set:
        """候補センテンスから max_count 個を全文均等に間引いて選定する。

        - 候補数 <= max_count: 全部選ぶ
        - 候補数 > max_count: 順序を保ったまま等間隔でサンプリング
          例: 候補 250, max 50 → 5 ステップごとに 1 つ選ぶ
              実装は浮動小数演算で「最も均等な分布」を実現

        戻り値: 選定された row["no"] の set
        """
        n_cand = len(candidates)
        if n_cand == 0 or max_count <= 0:
            return set()
        if n_cand <= max_count:
            return {r["no"] for r in candidates}

        # 等間隔サンプリング: index i (0..max-1) → round((i + 0.5) * n / max)
        # (i + 0.5) を使うことで「先頭・末尾に寄らず中央付近にも均等配置」される
        step = n_cand / max_count
        selected: set = set()
        for i in range(max_count):
            idx = int((i + 0.5) * step)
            if idx >= n_cand:
                idx = n_cand - 1
            selected.add(candidates[idx]["no"])

        # 万一重複でズレた分を補充（小さい数なので O(n) で十分）
        if len(selected) < max_count:
            for r in candidates:
                if r["no"] not in selected:
                    selected.add(r["no"])
                    if len(selected) >= max_count:
                        break

        return selected

    # ---- メインフロー ----
    def run(self) -> dict:
        client = get_anthropic_client()
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")

        if self.provider == PROVIDER_NANOBANANA and not gemini_key:
            raise RuntimeError("nanobanana を使うには GEMINI_API_KEY が必要です。")
        if self.provider == PROVIDER_GPT_IMAGE and not openai_key:
            raise RuntimeError("gpt-image を使うには OPENAI_API_KEY が必要です。")

        # Phase 0
        self._progress(0, "原稿を保存中...", 1)
        manuscript_path = self.output_dir / "manuscript.txt"
        manuscript_path.write_text(self.manuscript_text, encoding="utf-8")
        self._log("setup", f"原稿を保存しました（{len(self.manuscript_text)}文字）")

        # ===== Phase 1: 分解 =====
        self._progress(1, "原稿を章/ブロック/センテンスに分解中...", 5)
        self._log("splitter", "Claude で原稿を分解しています...")
        split_result = split_manuscript(client, self.manuscript_text, log=self._log)
        analysis = split_result["analysis"]
        chapters = split_result["chapters"]
        rows = split_result["rows"]
        total_sentences = split_result["total_sentences"]
        title = analysis.get("title", "無題")

        # 上限を超えるなら警告して切り詰める
        if total_sentences > self.max_diagrams:
            self._log("warn",
                      f"センテンス {total_sentences} 個が上限 {self.max_diagrams} を超過。先頭 {self.max_diagrams} 件のみ生成します。",
                      "それ以降のセンテンスはテーブルには出るが画像なし扱い")

        self._log("splitter", f"分解完了: {title}", f"章 {len(chapters)} / センテンス {total_sentences}")
        save_json(self.output_dir / "split_result.json", split_result)

        with self._rows_lock:
            for r in rows:
                self._rows_state[r["no"]] = {
                    **r,
                    "status": "pending",
                    "filename": None,
                    "prompt": "",
                    "allowed_terms": [],
                    "type": "",
                    "route": "",
                    "route_reason": "",
                    "web_source_url": "",
                    "web_thumb_url": "",
                    "web_topic": "",
                }
        self._dump_snapshot()
        self._progress(1, f"分解完了: {total_sentences} センテンス検出", 15)

        # ===== Phase 2-router: 各文のソースを判定 =====
        if self.route_mode == "auto":
            self._progress(2, "各文のソースを判定中（ルーター）...", 16)
            self._log("router", "ルーターが各文の最適なソースを判定します")
            routes = route_all_sentences(
                client, rows, title,
                user_instructions=self.user_instructions,
                max_workers=4, log=self._log,
            )
        else:  # all_ai: v1 互換（全文 AI 生成）
            self._log("router", "route_mode=all_ai: 全文を AI 生成に回します")
            routes = {
                r["no"]: {"route": "illustration", "reason": "all_ai モード", "search_query": "", "topic": ""}
                for r in rows
            }
        save_json(self.output_dir / "routes.json", routes)

        # route を各行に反映
        for no, rt in routes.items():
            self._update_row(no, route=rt.get("route", "illustration"), route_reason=rt.get("reason", ""))

        # route で 3 分類
        web_photo_rows = [r for r in rows if routes.get(r["no"], {}).get("route") == "web_photo"]
        ai_rows = [r for r in rows if routes.get(r["no"], {}).get("route") in AI_ROUTES]
        skip_rows = [r for r in rows if routes.get(r["no"], {}).get("route") == "skip"]

        # skip 文をマーク
        for r in skip_rows:
            self._update_row(r["no"], status="skipped")

        self._log("router",
                  f"振り分け: AI生成 {len(ai_rows)} / Web写真 {len(web_photo_rows)} / skip {len(skip_rows)}")

        # ===== Phase 2a: 英文プロンプト（AI 行のみ） =====
        self._progress(2, f"英文プロンプトを並列生成中（style={self.style_preset}）...", 22)
        self._log("prompter", f"{len(ai_rows)} 件（AI生成対象）のプロンプトを生成します")
        rows_with_prompts = generate_all_prompts(
            client,
            ai_rows,
            title=title,
            user_instructions=self.user_instructions,
            style_preset=self.style_preset,
            max_workers=6,
            log=self._log,
        )
        save_json(self.output_dir / "prompts.json", {"rows": rows_with_prompts})
        self._log("prompter", f"プロンプト生成完了: {len(rows_with_prompts)} 件")

        for r in rows_with_prompts:
            self._update_row(
                r["no"],
                prompt=r.get("prompt", ""),
                allowed_terms=r.get("allowed_terms", []),
                type=r.get("type", "illustration"),
            )
        self._progress(2, "プロンプト生成完了", 35)

        # ===== Phase 2b: Web 画像 URL 取得（並列実行） =====
        # 部分結果を保持する list（タイムアウト時にも参照できる）
        web_results_accumulator: list = []

        def _web_on_item(info):
            web_results_accumulator.append(info)
            self._update_row(
                info["no"],
                web_source_url=info.get("source_url", ""),
                web_thumb_url=info.get("thumb_url", ""),
                web_topic=info.get("topic", ""),
                web_source_title=info.get("source_title", ""),
            )
            try:
                save_json(
                    self.output_dir / "web_results.json",
                    {"items": list(web_results_accumulator)},
                )
            except Exception:
                pass

        def _web_save_final():
            try:
                save_json(
                    self.output_dir / "web_results.json",
                    {"items": list(web_results_accumulator)},
                )
            except Exception:
                pass

        web_thread = None

        if self.route_mode == "auto" and web_photo_rows:
            # ルーターが web_photo に振った文を検索（選定済み）
            selections = []
            for r in web_photo_rows:
                rt = routes.get(r["no"], {})
                selections.append({
                    "no": r["no"],
                    "query": rt.get("search_query") or r.get("sentence", "")[:30],
                    "topic": rt.get("topic") or r.get("sentence", "")[:20],
                })
            self._log("websearch",
                      f"Web 画像取得を並列起動: {len(selections)} 件（ルーター選定・同時 8 並列）")

            def web_task_auto():
                try:
                    run_web_search_for_selections(
                        client, selections, max_workers=8,
                        log=self._log, item_callback=_web_on_item,
                    )
                except Exception as e:
                    self._log("error", f"Web 画像取得失敗: {str(e)[:120]}")
                _web_save_final()

            web_thread = threading.Thread(target=web_task_auto, daemon=True)

        elif self.route_mode == "all_ai" and self.web_image_count > 0:
            # v1 互換: web_image_count で内部選定
            self._log("websearch",
                      f"Web 画像取得を並列起動: 目標 {self.web_image_count} 件（v1 選定・同時 8 並列）")

            def web_task_v1():
                try:
                    run_web_search(
                        client, rows_with_prompts,
                        target_count=self.web_image_count, max_workers=8,
                        log=self._log, item_callback=_web_on_item,
                    )
                except Exception as e:
                    self._log("error", f"Web 画像取得失敗: {str(e)[:120]}")
                _web_save_final()

            web_thread = threading.Thread(target=web_task_v1, daemon=True)

        if web_thread:
            web_thread.start()

        # ===== Phase 3: 画像生成（全文均等配置で選定） =====
        # Step A: skip_decorative なら decorative 行を先に除外（候補から外す）
        candidates = []
        skipped_decorative = 0
        for r in rows_with_prompts:
            if self.skip_decorative and r.get("type") == "decorative":
                self._update_row(r["no"], status="skipped")
                skipped_decorative += 1
                continue
            candidates.append(r)

        # Step B: 候補数が max_diagrams 以下ならそのまま全部、超えていれば均等間引き
        selected_nos = self._select_evenly_distributed(candidates, self.max_diagrams)
        self._log("generator",
                  f"画像配置方式: 全文均等配置 "
                  f"(候補 {len(candidates)} / 選定 {len(selected_nos)} / 上限 {self.max_diagrams})")

        # Step C: 各 row のステータスを「選定済み（pending）」or「間引き」にマーク
        generation_targets = []
        thinned_count = 0
        for r in rows_with_prompts:
            no = r["no"]
            if self.skip_decorative and r.get("type") == "decorative":
                continue  # 既に skipped
            if no in selected_nos:
                generation_targets.append({
                    "index": no,
                    "prompt": r.get("prompt", ""),
                    "type": r.get("type", "illustration"),
                    "section": r.get("chapter_title", ""),
                    "excerpt": r.get("sentence", ""),
                    "keypoint": r.get("sentence", "")[:30],
                    "allowed_terms": r.get("allowed_terms", []),
                })
            else:
                # 候補だったが均等配置から外れた → 「間引き」
                self._update_row(no, status="thinned")
                thinned_count += 1

        provider_label = ("nanobanana (Gemini)" if self.provider == PROVIDER_NANOBANANA
                          else f"gpt-image ({self.openai_quality})")
        self._progress(3,
                       f"画像を並列生成中（{provider_label} / 同時 {self.concurrency} 枚 / {len(generation_targets)} 枚）...",
                       40)
        self._log("generator",
                  f"{provider_label} で {len(generation_targets)} 枚を並列生成します",
                  f"スタイル: {self.style_preset}")

        def on_item_event(info: dict):
            no = info.get("index", 0)
            status = info.get("status", "")
            update = {"status": status}
            if status == "ok":
                update["filename"] = info.get("filename")
            if info.get("error"):
                update["error"] = info["error"]
            self._update_row(no, **update)

        results = run_parallel_generation(
            prompts=generation_targets,
            output_dir=self.images_dir,
            provider=self.provider,
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            openai_quality=self.openai_quality,
            concurrency=self.concurrency,
            style_preset=self.style_preset,
            progress_callback=on_item_event,
        )

        success_count = sum(1 for r in results if r.get("success"))
        fail_count = len(results) - success_count
        self._log("generator", f"画像生成完了: 成功 {success_count} / 失敗 {fail_count}")

        # Web 検索の完了を待つ（タイムアウト 20 分）
        # Web 検索は I/O bound + Claude Web Search のレート制限により遅い:
        # 1 件あたり 5〜15 秒 × 100 件 ÷ 並列 8 ≈ 1〜3 分が目安
        # 余裕を見て 20 分に延長
        if web_thread:
            self._progress(3, "Web 画像取得の完了を待機中...", 92)
            wait_minutes = 20
            self._log("websearch",
                      f"Web 画像取得の完了を最大 {wait_minutes} 分待機します...")
            web_thread.join(timeout=wait_minutes * 60)
            if web_thread.is_alive():
                self._log("warn",
                          f"Web 画像取得が {wait_minutes} 分以内に完了しませんでした。"
                          f"部分結果（{len(web_results_accumulator)} 件）で続行します。")

        # ===== マニフェスト =====
        with self._rows_lock:
            final_rows = sorted(self._rows_state.values(), key=lambda x: x.get("no", 0))

        # rows_progress から Web URL がついた行数を再カウント（accumulator と二重チェック）
        web_count_from_rows = sum(1 for r in final_rows if r.get("web_source_url"))
        web_count_from_acc = len(web_results_accumulator)
        web_count_final = max(web_count_from_rows, web_count_from_acc)

        self._log("websearch",
                  f"Web 画像取得集計: accumulator={web_count_from_acc} / rows={web_count_from_rows}")

        manifest = {
            "title": title,
            "summary": analysis.get("summary", ""),
            "keywords": analysis.get("keywords", []),
            "user_instructions": self.user_instructions,
            "provider": self.provider,
            "openai_quality": self.openai_quality if self.provider == PROVIDER_GPT_IMAGE else None,
            "style_preset": self.style_preset,
            "route_mode": self.route_mode,
            "concurrency": self.concurrency,
            "total_sentences": total_sentences,
            "max_diagrams": self.max_diagrams,
            "web_image_count": self.web_image_count,
            "ai_route_count": len(ai_rows),
            "web_photo_count": len(web_photo_rows),
            "skip_route_count": len(skip_rows),
            "generated": success_count,
            "failed": fail_count,
            "skipped_decorative": skipped_decorative,
            "thinned": thinned_count,  # 均等配置のため間引かれた数
            "web_results_count": web_count_final,
            "rows": final_rows,
            "chapters": [{"title": c["title"], "block_count": len(c["blocks"])} for c in chapters],
            "completed_at": datetime.now().isoformat(),
        }
        save_json(self.output_dir / "manifest.json", manifest)

        # CSV
        self._write_csv(self.output_dir / "result.csv", final_rows)

        self._progress(4, f"完了: 図解 {success_count} / Web {web_count_final} / 全 {total_sentences} 文", 100)
        return manifest

    # ルート → 日本語ラベル
    ROUTE_LABELS = {
        "web_photo": "Web写真",
        "map": "地図",
        "diagram": "図解",
        "chart": "グラフ",
        "illustration": "イラスト",
        "skip": "スキップ",
        "": "",
    }

    def _write_csv(self, path: Path, rows: list):
        """CSV を書き出す（スプレッドシートと同構造）"""
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["章", "ブロック", "センテンス", "№", "ソース", "画像", "URL", "Web トピック"])
            for r in rows:
                block_text = ""
                if r.get("sentence_index") == 0:
                    block_text = r.get("block_text", "")
                chapter = ""
                if r.get("block_index") == 0 and r.get("sentence_index") == 0:
                    chapter = r.get("chapter_title", "")
                route_label = self.ROUTE_LABELS.get(r.get("route", ""), r.get("route", ""))
                w.writerow([
                    chapter,
                    block_text,
                    r.get("sentence", ""),
                    r.get("no", ""),
                    route_label,
                    r.get("filename", "") or "",
                    r.get("web_source_url", "") or "",
                    r.get("web_topic", "") or "",
                ])
