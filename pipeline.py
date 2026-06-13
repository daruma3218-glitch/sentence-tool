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


VALID_STYLES = ("flat_infographic", "pictogram", "comic", "whiteboard")
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
        worldview_desc: str = "",
        verify_diagrams: bool = True,
        channel_id: str = "default",
        anthropic_key: str = "",
        gemini_key: str = "",
        openai_key: str = "",
        character_ref_path: str = "",
        skip_decorative: bool = False,
        web_image_count: int = 0,
        max_diagrams: int = 150,
        route_mode: str = "auto",
        chart_engine: str = "ai",          # v3: render で chart を matplotlib 描画
        map_engine: str = "ai",            # v3: render で map を GeoJSON 描画
        photo_source: str = "web",         # v3: commons で Wikimedia Commons 限定（権利安全）
        beat_mode: bool = False,           # v3: ビート単位で重要度加重配分（False=v2均等）
        chars_per_sec: float = 5.5,        # v3: 読み上げ速度（推定タイムコード用）
        realphoto_watermark: bool = False,  # v3: realphoto に「イメージ」焼き込み
        chart_theme: Optional[dict] = None,  # v3: チャンネル別チャート/地図配色
        title_override: str = "",           # v3 Step7: final.json の tentative_title
        fact_context: str = "",             # v3 Step7: final.json の検証済み数値・出典
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
        self.worldview_desc = worldview_desc or ""
        self.verify_diagrams = bool(verify_diagrams)
        self.channel_id = channel_id or "default"
        self.anthropic_key = anthropic_key or ""
        self.gemini_key = gemini_key or ""
        self.openai_key = openai_key or ""
        self.character_ref_path = character_ref_path or ""
        self.skip_decorative = skip_decorative
        self.web_image_count = max(0, min(web_image_count, 200))
        self.max_diagrams = max(1, min(max_diagrams, 300))
        self.route_mode = route_mode if route_mode in VALID_ROUTE_MODES else "auto"
        self.chart_engine = (chart_engine or "ai").strip()  # "render" で matplotlib 描画
        self.map_engine = (map_engine or "ai").strip()       # "render" で GeoJSON 描画
        self.photo_source = (photo_source or "web").strip()  # "commons" で Commons 限定
        self.beat_mode = bool(beat_mode)                     # v3: ビート加重配分
        self.chars_per_sec = float(chars_per_sec or 5.5)
        self.realphoto_watermark = bool(realphoto_watermark)  # v3: 「イメージ」焼き込み
        self.chart_theme = chart_theme or None
        self.title_override = (title_override or "").strip()   # v3 Step7
        self.fact_context = (fact_context or "").strip()       # v3 Step7
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
    def _wants_image(self, r) -> bool:
        """この文を今回のジョブで画像化するか。

        beat_mode=True のとき allocator が選んだ display=image の文のみ。
        False は v2（全候補→均等間引きに委ねる）。
        """
        if self.beat_mode:
            return r.get("display") == "image"
        return True

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

    def _load_route_feedback(self, limit: int = 12) -> list:
        """v3 Step5: 過去の「ルート違い」フィードバックを読み、ルーターに渡す few-shot を作る。

        route_feedback.jsonl（output ルート直下＝全ジョブ共通）から、同じチャンネルの
        記録だけを新しい順に最大 limit 件返す。ファイルが無い／壊れていても落とさない。
        """
        fb_path = self.output_dir.parent / "route_feedback.jsonl"
        if not fb_path.exists():
            return []
        records = []
        try:
            with open(fb_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("channel_id", "default") != self.channel_id:
                        continue
                    if rec.get("sentence") and rec.get("correct_route"):
                        records.append(rec)
        except Exception:
            return []
        # 新しい順（末尾優先）に limit 件
        return records[-limit:]

    # ---- メインフロー ----
    def run(self) -> dict:
        # チャンネル別キーがあれば優先、無ければ共通（環境変数）
        client = get_anthropic_client(self.anthropic_key)
        gemini_key = (self.gemini_key or "").strip() or os.environ.get("GEMINI_API_KEY", "")
        openai_key = (self.openai_key or "").strip() or os.environ.get("OPENAI_API_KEY", "")

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
        # .docx の見出しから作った章構造があれば使う
        prebuilt = None
        pc_path = self.output_dir / "prebuilt_chapters.json"
        if pc_path.exists():
            try:
                prebuilt = load_json(pc_path, {}).get("chapters") or None
            except Exception:
                prebuilt = None
        split_result = split_manuscript(client, self.manuscript_text, log=self._log,
                                        prebuilt_chapters=prebuilt)
        analysis = split_result["analysis"]
        chapters = split_result["chapters"]
        rows = split_result["rows"]
        total_sentences = split_result["total_sentences"]
        # v3 Step7: final.json の tentative_title があれば最優先（無ければ分解で推定）
        title = self.title_override or analysis.get("title", "無題")

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
            few_shot = self._load_route_feedback()
            if few_shot:
                self._log("router", f"過去のルート違いフィードバック {len(few_shot)} 件を学習に反映します")
            routes = route_all_sentences(
                client, rows, title,
                user_instructions=self.user_instructions,
                max_workers=4, log=self._log,
                few_shot=few_shot,
            )
        else:  # all_ai: v1 互換（全文 AI 生成）
            self._log("router", "route_mode=all_ai: 全文を AI 生成に回します")
            routes = {
                r["no"]: {"route": "illustration", "reason": "all_ai モード", "search_query": "", "topic": "", "propaganda": False}
                for r in rows
            }
        save_json(self.output_dir / "routes.json", routes)

        # route を各行に反映（row dict 自体にも route を入れる＝prompter が type 判定に使う）
        for r in rows:
            rt = routes.get(r["no"], {})
            r["route"] = rt.get("route", "illustration")
            r["route_reason"] = rt.get("reason", "")
        for no, rt in routes.items():
            self._update_row(no, route=rt.get("route", "illustration"), route_reason=rt.get("reason", ""))

        # ===== v3 Step4: ビート確定・タイムコード・重要度加重配分（LLM不使用）=====
        # beat_mode=True のとき max_diagrams をビート単位で重要度加重配分し、画像を付ける
        # 文(display=image)だけを画像化する。False は v2（均等間引き）。失敗時は v2 に倒す。
        try:
            from allocator import allocate, write_allocation
            alloc = allocate(rows, routes, self.max_diagrams,
                             chars_per_sec=self.chars_per_sec, beat_mode=self.beat_mode)
            for r in rows:
                a = alloc.get(r["no"], {})
                r["beat_id"] = a.get("beat_id")
                r["est_start"] = a.get("est_start", "")
                r["display"] = a.get("display", "none")
                r["importance"] = a.get("importance", 3)
                self._update_row(r["no"], beat_id=r["beat_id"],
                                 est_start=r["est_start"], importance=r["importance"],
                                 display=r["display"])
            write_allocation(self.output_dir / "allocation.json", rows, routes, alloc)
            if self.beat_mode:
                n_img = sum(1 for r in rows if r.get("display") == "image")
                n_hold = sum(1 for r in rows if r.get("display") == "hold")
                self._log("allocator",
                          f"ビート配分: 画像 {n_img} 枚 / 継続(hold) {n_hold} 文"
                          f"（重要度加重・上限 {self.max_diagrams}）")
                for r in rows:
                    if r.get("display") == "hold":
                        self._update_row(r["no"], status="hold")
        except Exception as e:
            self._log("error", f"allocator をスキップ（{str(e)[:80]}）。v2 配分にフォールバック。")
            self.beat_mode = False

        # ===== v3 Step1: chart を matplotlib で決定論レンダリング（engine:render）=====
        # chart_engine=render のとき chart 文の数値を抽出し正確に描画。抽出不能/描画失敗は
        # diagram(engine:ai) へ降格して v2 同様 AI 生成へ。何が起きても v2 にフォールバック。
        for r in rows:
            r.setdefault("engine", "ai")
        if self.chart_engine == "render":
            try:
                from router import extract_chart_specs
                chart_rows = [r for r in rows if r.get("route") == "chart" and self._wants_image(r)]
                if chart_rows:
                    self._progress(2, "chart の数値を抽出して図を描画中...", 20)
                    specs = extract_chart_specs(client, chart_rows, log=self._log,
                                                extra_context=self.fact_context)
                    to_render = []
                    for r in chart_rows:
                        spec = specs.get(r["no"])
                        if spec:
                            r["engine"] = "render"
                            r["chart_spec"] = spec
                            to_render.append(r)
                        else:
                            r["route"] = "diagram"  # 降格（chart→diagram, engine:ai）
                            r["engine"] = "ai"
                            self._update_row(r["no"], route="diagram")
                    if to_render:
                        self._render_charts(to_render)
            except Exception as e:
                self._log("error", f"chartレンダリングをスキップ（{str(e)[:80]}）。AI生成に回します。")
                for r in rows:
                    if r.get("route") == "chart" and r.get("engine") == "render":
                        r["engine"] = "ai"
        # ----- map（Step2）: route=map を Natural Earth GeoJSON で正確描画。降格先は illustration -----
        if self.map_engine == "render":
            try:
                from router import extract_map_specs
                map_rows = [r for r in rows if r.get("route") == "map" and self._wants_image(r)]
                if map_rows:
                    self._progress(2, "地図データを抽出して描画中...", 21)
                    specs = extract_map_specs(client, map_rows, log=self._log)
                    to_render = []
                    for r in map_rows:
                        spec = specs.get(r["no"])
                        if spec:
                            r["engine"] = "render"
                            r["map_spec"] = spec
                            to_render.append(r)
                        else:
                            r["route"] = "illustration"  # 降格（map→illustration, engine:ai）
                            r["engine"] = "ai"
                            self._update_row(r["no"], route="illustration")
                    if to_render:
                        self._render_maps(to_render)
            except Exception as e:
                self._log("error", f"地図レンダリングをスキップ（{str(e)[:80]}）。AI生成に回します。")
                for r in rows:
                    if r.get("route") == "map" and r.get("engine") == "render":
                        r["engine"] = "ai"
        for r in rows:
            self._update_row(r["no"], engine=r.get("engine", "ai"))

        # route で分類（engine:render はレンダリング済みなので AI 対象から除外）
        web_photo_rows = [r for r in rows if r.get("route") == "web_photo"]
        ai_rows = [r for r in rows if r.get("route") in AI_ROUTES
                   and r.get("engine") != "render" and self._wants_image(r)]
        skip_rows = [r for r in rows if r.get("route") == "skip"]

        # skip 文をマーク
        for r in skip_rows:
            self._update_row(r["no"], status="skipped")

        self._log("router",
                  f"振り分け: AI生成 {len(ai_rows)} / Web写真 {len(web_photo_rows)} / skip {len(skip_rows)}")

        # ===== Phase 2a: 英文プロンプト（AI 行のみ） =====
        self._progress(2, f"英文プロンプトを並列生成中（style={self.style_preset}）...", 22)
        self._log("prompter", f"{len(ai_rows)} 件（AI生成対象）のプロンプトを生成します")
        rows_with_prompts = generate_all_prompts(
            client, ai_rows, title=title,
            user_instructions=self.user_instructions,
            style_preset=self.style_preset, worldview_desc=self.worldview_desc,
            max_workers=6, log=self._log,
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
            # サムネをローカルに DL して実画像として表示・ZIP 同梱できるようにする
            from web_searcher import download_thumbnail
            local_file = ""
            thumb_url = info.get("thumb_url", "")
            if thumb_url:
                fname = f"{info['no']}.jpg"  # 数字だけのファイル名（№と一致）
                if download_thumbnail(thumb_url, self.images_dir / fname):
                    local_file = fname
            self._update_row(
                info["no"],
                web_source_url=info.get("source_url", ""),
                web_thumb_url=info.get("thumb_url", ""),
                web_local_file=local_file,
                web_topic=info.get("topic", ""),
                web_source_title=info.get("source_title", ""),
                # v3 Step3: Commons のライセンス・クレジット（CSV / credits.txt 用）
                license=info.get("license", ""),
                attribution=info.get("attribution", ""),
                commons_page_url=info.get("commons_page_url", ""),
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
                # v3 Step3: 採用した Commons 画像のクレジット一覧（概要欄用）
                from commons_searcher import build_credits_text
                txt = build_credits_text(list(web_results_accumulator))
                (self.output_dir / "credits.txt").write_text(txt, encoding="utf-8")
            except Exception:
                pass

        web_thread = None

        if self.route_mode == "auto" and web_photo_rows:
            # ルーターが web_photo に振った文を検索（beat_mode 時は display=image のみ）
            selections = []
            for r in web_photo_rows:
                if not self._wants_image(r):
                    continue
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
                    if self.photo_source == "commons":
                        # v3 Step3: Wikimedia Commons 限定（許可ライセンスのみ・権利安全）
                        from commons_searcher import run_commons_search_for_selections
                        run_commons_search_for_selections(
                            client, selections, max_workers=8,
                            log=self._log, item_callback=_web_on_item,
                        )
                    else:
                        run_web_search_for_selections(
                            client, selections, max_workers=8,
                            log=self._log, item_callback=_web_on_item,
                        )
                except Exception as e:
                    self._log("error", f"Web/Commons 画像取得失敗: {str(e)[:120]}")
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

        # Step B: 選定。beat_mode は allocator が配分済み（candidates 全部）。
        # v2 は候補が max_diagrams 超なら均等間引き。
        if self.beat_mode:
            selected_nos = set(r["no"] for r in candidates)
        else:
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
                # 画像 type はルーターの route を最優先（realphoto/map 等を確実に反映）
                route_type = routes.get(no, {}).get("route", "")
                img_type = route_type if route_type in AI_ROUTES else r.get("type", "illustration")
                generation_targets.append({
                    "index": no,
                    "prompt": r.get("prompt", ""),
                    "type": img_type,
                    "section": r.get("chapter_title", ""),
                    "excerpt": r.get("sentence", ""),
                    "block_text": r.get("block_text", ""),  # 検証の文脈用（前後段落）
                    "keypoint": r.get("sentence", "")[:30],
                    "allowed_terms": r.get("allowed_terms", []),
                    "style": self.style_preset,
                    # キャラ固定: 先生が描かれる illustration のみ参照画像を使う
                    "character": bool(r.get("character", False)) and img_type == "illustration",
                })
            else:
                # 候補だったが均等配置から外れた → 「間引き」
                self._update_row(no, status="thinned")
                thinned_count += 1

        # ===== v3 Step6: エンティティ参照（一貫性ロック）=====
        # 同じ被写体（国・組織・繰り返す概念）が 3 回以上 AI 画像に登場するとき、
        # 初出を canonical とし、後続は canonical 画像を参照に見た目を揃える。
        # beat_mode（v3 チャンネル）のときのみ。失敗しても生成は止めない。
        if self.beat_mode and generation_targets:
            try:
                from allocator import assign_entity_refs
                image_nos = [t["index"] for t in generation_targets]
                ent_assign = assign_entity_refs(image_nos, routes)
                if ent_assign:
                    targets_by_no = {t["index"]: t for t in generation_targets}
                    for no, a in ent_assign.items():
                        t = targets_by_no.get(no)
                        if not t:
                            continue
                        t["entity_role"] = a["role"]
                        t["entity_name"] = a["entity"]
                        if a["role"] == "follower":
                            t["entity_ref_of"] = a["canon_no"]
                    n_canon = sum(1 for v in ent_assign.values() if v["role"] == "canonical")
                    n_follow = sum(1 for v in ent_assign.values() if v["role"] == "follower")
                    n_kind = len({v["entity"] for v in ent_assign.values()})
                    self._log("generator",
                              f"エンティティ参照ロック: 初出 {n_canon} / 後続 {n_follow}（{n_kind} 種の繰り返し被写体）",
                              "後続は初出画像を参照して見た目を統一します")
            except Exception as e:
                self._log("generator", f"エンティティ参照の割当をスキップ（{str(e)[:80]}）")

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
            # 生成が1枚進むたびにプログレスバー(40→85%)とメッセージを更新し、
            # 長時間ジョブでも「止まって見えない」ようにする。
            if status in ("ok", "failed"):
                gt = info.get("grand_total") or 0
                done = (info.get("completed_total") or 0) + (info.get("failed_total") or 0)
                if gt:
                    pct = 40 + int(done / gt * 45)
                    self._progress(3,
                                   f"画像を並列生成中（{done}/{gt} 枚 / {provider_label}）...",
                                   min(85, pct))

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
            reference_image_path=self.character_ref_path,
            realphoto_watermark=self.realphoto_watermark,
        )

        success_count = sum(1 for r in results if r.get("success"))
        fail_count = len(results) - success_count
        self._log("generator", f"画像生成完了: 成功 {success_count} / 失敗 {fail_count}")

        # ===== Phase 3b: 図解の意味を自動検証 → ズレてたら再生成 =====
        # 検証は「あれば嬉しい」機能。何があってもジョブ完了を止めない（必ず先へ進む）。
        if self.verify_diagrams:
            # 生成フェーズのバッファを解放してから検証に入る（512MB環境のOOM対策）。
            import gc as _gc
            _gc.collect()
            theme = title
            _sum = analysis.get("summary", "")
            if _sum:
                theme = f"{title}（{_sum}）"
            try:
                self._verify_and_fix(results, generation_targets, gemini_key, openai_key, theme=theme)
            except Exception as e:
                self._log("error", f"検証フェーズをスキップしました（{str(e)[:80]}）。生成画像はそのまま使えます。")

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
            "channel_id": self.channel_id,
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

        # HTML ギャラリー（画像を埋め込んだ表。開けばぱっと全体を見渡せる）
        self._write_gallery(self.output_dir / "result.html", final_rows, title)

        self._progress(4, f"完了: 図解 {success_count} / Web {web_count_final} / 全 {total_sentences} 文", 100)
        return manifest

    def _write_gallery(self, path: Path, rows: list, title: str):
        """画像を埋め込んだ HTML ギャラリーを出力（相対パスで自己完結）"""
        import html as _html

        def cell_img(r):
            fn = r.get("filename") or r.get("web_local_file") or ""
            if fn:
                return f'<img src="images/{_html.escape(fn)}" loading="lazy">'
            st = r.get("status", "")
            if st == "skipped" or r.get("route") == "skip":
                return '<span class="no">—（スキップ）</span>'
            return '<span class="no">（画像なし）</span>'

        parts = [f"""<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)} - 画像ギャラリー</title>
<style>
 body{{font-family:'Hiragino Sans','Noto Sans JP',sans-serif;margin:0;padding:24px;background:#f8fafc;color:#1f2937}}
 h1{{font-size:20px;margin:0 0 16px}}
 table{{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
 th,td{{border:1px solid #e5e7eb;padding:8px;vertical-align:top;font-size:13px}}
 th{{background:#f1f5f9;position:sticky;top:0;text-align:left}}
 td.no{{text-align:center;color:#9ca3af;font-family:monospace;width:48px}}
 td.chap{{font-weight:600;color:#0e7490;width:120px}}
 td.sent{{max-width:340px}}
 td.img{{width:340px;text-align:center}}
 td.img img{{max-width:320px;max-height:180px;border-radius:6px;border:1px solid #e5e7eb}}
 .src{{display:inline-block;font-size:11px;padding:1px 6px;border-radius:4px;background:#eef2ff;color:#4338ca}}
 .no{{color:#cbd5e1;font-size:12px}}
 a{{color:#7c3aed}}
</style></head><body>
<h1>{_html.escape(title)} — 画像ギャラリー（全 {len(rows)} 文）</h1>
<table><thead><tr>
<th>№</th><th>章</th><th>センテンス</th><th>ソース</th><th>画像</th>
</tr></thead><tbody>"""]

        # 冒頭の固定画像（あれば先頭に）
        def _find_fixed(slot):
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                if (self.images_dir / f"{slot}{ext}").exists():
                    return f"{slot}{ext}"
            return None
        intro_fn = _find_fixed("intro")
        outro_fn = _find_fixed("outro")
        if intro_fn:
            parts.append(
                f'<tr style="background:#ecfeff"><td class="no">▶</td><td class="chap">冒頭固定</td>'
                f'<td class="sent">（差し込み画像・冒頭）</td><td><span class="src">固定</span></td>'
                f'<td class="img"><img src="images/{intro_fn}" loading="lazy"></td></tr>'
            )

        last_chap = None
        for r in rows:
            chap = r.get("chapter_title", "")
            chap_show = chap if chap != last_chap else ""
            last_chap = chap
            route_label = self.ROUTE_LABELS.get(r.get("route", ""), r.get("route", ""))
            web_link = ""
            if r.get("web_source_url"):
                web_link = f'<br><a href="{_html.escape(r["web_source_url"])}" target="_blank">出典</a>'
            parts.append(
                f'<tr><td class="no">{r.get("no","")}</td>'
                f'<td class="chap">{_html.escape(chap_show)}</td>'
                f'<td class="sent">{_html.escape(r.get("sentence",""))}</td>'
                f'<td><span class="src">{_html.escape(route_label)}</span>{web_link}</td>'
                f'<td class="img">{cell_img(r)}</td></tr>'
            )
        # 終わりの固定画像（あれば末尾に）
        if outro_fn:
            parts.append(
                f'<tr style="background:#fef2f2"><td class="no">■</td><td class="chap">終わり固定</td>'
                f'<td class="sent">（差し込み画像・終わり/CTA）</td><td><span class="src">固定</span></td>'
                f'<td class="img"><img src="images/{outro_fn}" loading="lazy"></td></tr>'
            )
        parts.append("</tbody></table></body></html>")
        try:
            path.write_text("\n".join(parts), encoding="utf-8")
        except Exception as e:
            self._log("warn", f"ギャラリー出力に失敗: {str(e)[:80]}")

    # ルート → 日本語ラベル
    ROUTE_LABELS = {
        "web_photo": "Web写真",
        "realphoto": "実写風",
        "map": "地図",
        "diagram": "図解",
        "chart": "グラフ",
        "illustration": "イラスト",
        "skip": "スキップ",
        "": "",
    }

    def _render_charts(self, render_rows):
        """chart_spec を matplotlib で 1920x1080 PNG に描画（engine:render・LLM不使用）。

        数値の狂い・文字化けが構造的にゼロ。描画失敗は diagram(engine:ai) へ降格する。
        """
        try:
            from renderer import render_chart
        except Exception as e:
            self._log("error", f"renderer 読込失敗（{str(e)[:60]}）。chart は AI 生成へ降格。")
            for r in render_rows:
                r["route"] = "diagram"
                r["engine"] = "ai"
                self._update_row(r["no"], route="diagram", engine="ai")
            return
        done = 0
        for r in render_rows:
            no = r["no"]
            self._update_row(no, status="generating", engine="render")
            ok = render_chart(r.get("chart_spec"), self.images_dir / f"{no}.png", theme=self.chart_theme)
            if ok:
                # chart_spec を保存しておく（再生成で確実に同じグラフを描き直すため）
                self._update_row(no, status="ok", filename=f"{no}.png", engine="render",
                                 chart_spec=r.get("chart_spec"))
                done += 1
            else:
                r["route"] = "diagram"
                r["engine"] = "ai"
                self._update_row(no, route="diagram", engine="ai")
        self._log("renderer", f"chart レンダリング完了: {done} 枚（決定論・文字化けゼロ）")

    def _render_maps(self, render_rows):
        """map_spec を Natural Earth GeoJSON + matplotlib で描画（engine:render・LLM不使用）。

        国境が正確（AI の航空写真風のデタラメ国境を排除）。失敗は illustration(ai) へ降格。
        """
        try:
            from renderer import render_map
        except Exception as e:
            self._log("error", f"renderer 読込失敗（{str(e)[:60]}）。map は AI 生成へ降格。")
            for r in render_rows:
                r["route"] = "illustration"
                r["engine"] = "ai"
                self._update_row(r["no"], route="illustration", engine="ai")
            return
        done = 0
        for r in render_rows:
            no = r["no"]
            self._update_row(no, status="generating", engine="render")
            ok = render_map(r.get("map_spec"), self.images_dir / f"{no}.png", theme=self.chart_theme)
            if ok:
                # map_spec を保存しておく（再生成で確実に同じ地図を描き直すため）
                self._update_row(no, status="ok", filename=f"{no}.png", engine="render",
                                 map_spec=r.get("map_spec"))
                done += 1
            else:
                r["route"] = "illustration"
                r["engine"] = "ai"
                self._update_row(no, route="illustration", engine="ai")
        self._log("renderer", f"map レンダリング完了: {done} 枚（正確な国境）")

    def _verify_and_fix(self, results, generation_targets, gemini_key, openai_key, theme=""):
        """生成済み diagram/chart を Claude Vision で検証し、ズレてたら1回だけ再生成する。

        検証は「あれば嬉しい」機能なので、絶対にジョブ完了をブロックしない:
        - 全体に時間予算（budget）を設け、超過したら残りはスキップして先へ進む
        - 1 枚ごとにもタイムアウト（固まった検証で全体が止まらない）
        - チャンネル別 Anthropic キーを使い、リトライを抑えて素早く諦める
        """
        import time as _time
        from concurrent.futures import (
            ThreadPoolExecutor, as_completed, TimeoutError as _FutTimeout)
        from verifier import verify_image, DEFAULT_VERIFY_TYPES

        # チャンネル別 Anthropic キーを使う。固まっても素早く諦めるためリトライ抑制。
        try:
            client = get_anthropic_client(self.anthropic_key).with_options(
                max_retries=1, timeout=60.0)
        except Exception as e:
            self._log("verify", f"検証をスキップ（APIクライアント初期化失敗: {str(e)[:60]}）")
            return

        # 検証対象: 生成成功した diagram / chart（targets と results を突合）
        targets_by_no = {t["index"]: t for t in generation_targets}
        verify_list = []
        for r in results:
            if not r.get("success"):
                continue
            t = targets_by_no.get(r.get("index"))
            if t and t.get("type") in DEFAULT_VERIFY_TYPES:
                verify_list.append(t)

        if not verify_list:
            return

        # 時間予算: 1 枚 ~12 秒 ÷ 4 並列 を目安に、最大 10 分。超えたら打ち切って先へ進む。
        budget = min(600, max(90, int(len(verify_list) / 4 * 12) + 60))
        self._progress(3, f"図解の意味を検証中（0/{len(verify_list)} 枚）...", 90)
        self._log("verify",
                  f"diagram/chart {len(verify_list)} 枚の意味を Claude Vision で検証します"
                  f"（最大 {budget // 60} 分・超過分はスキップ）")

        # 並列で検証（原稿の文脈＝テーマ・章・前後段落 を渡す）
        def _do_verify(t):
            no = t["index"]
            img_path = self.images_dir / f"{no}.png"
            v = verify_image(
                client, img_path, t.get("excerpt", ""), t.get("type", "diagram"),
                allowed_terms=t.get("allowed_terms"),
                block_context=t.get("block_text", ""),
                chapter=t.get("section", ""),
                theme=theme,
            )
            return (t, v)

        ng = []
        checked = 0
        timed_out = False
        deadline = _time.monotonic() + budget
        # 並列は 3 に抑える（縮小済み画像 + 512MB 環境でのピークメモリ削減）
        ex = ThreadPoolExecutor(max_workers=3)
        try:
            futs = {ex.submit(_do_verify, t): t for t in verify_list}
            try:
                for f in as_completed(futs, timeout=budget):
                    remaining = deadline - _time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        t, v = f.result(timeout=max(1.0, remaining))
                        checked += 1
                        if checked % 20 == 0:
                            self._progress(3, f"図解の意味を検証中（{checked}/{len(verify_list)} 枚）...", 90)
                        if not v.get("ok"):
                            ng.append((t, v))
                            self._log("verify", f"№{t['index']} 要修正: {v.get('reason','')}")
                    except _FutTimeout:
                        timed_out = True
                        break
                    except Exception as e:
                        self._log("error", f"検証エラー: {str(e)[:80]}")
            except _FutTimeout:
                timed_out = True
        finally:
            # 残りの未実行タスクはキャンセル。実行中スレッドは待たずに先へ進む。
            ex.shutdown(wait=False, cancel_futures=True)
            import gc as _gc
            _gc.collect()  # 検証で使った画像バッファを解放

        if timed_out:
            self._log("verify",
                      f"検証は時間上限({budget // 60}分)に達したため打ち切りました"
                      f"（{checked}/{len(verify_list)} 枚確認）。生成画像はそのまま使えます。")

        if not ng:
            if not timed_out:
                self._log("verify", "検証完了: 全て意味OK ✓")
            return

        # 改善指示を付けて再生成（1回）
        self._log("verify", f"{len(ng)} 枚を改善指示付きで再生成します")
        fix_targets = []
        for t, v in ng:
            entry = dict(t)
            hint = v.get("fix_hint", "")
            if hint:
                entry["prompt"] = f"{t.get('prompt','')}\n\nIMPROVE: {hint}"
            self._update_row(t["index"], status="generating")
            fix_targets.append(entry)

        def on_fix_event(info):
            no = info.get("index", 0)
            st = info.get("status", "")
            upd = {"status": st}
            if st == "ok":
                upd["filename"] = info.get("filename")
            self._update_row(no, **upd)

        run_parallel_generation(
            prompts=fix_targets,
            output_dir=self.images_dir,
            provider=self.provider,
            gemini_api_key=gemini_key,
            openai_api_key=openai_key,
            openai_quality=self.openai_quality,
            concurrency=self.concurrency,
            style_preset=self.style_preset,
            progress_callback=on_fix_event,
            reference_image_path=self.character_ref_path,
            realphoto_watermark=self.realphoto_watermark,
        )
        self._log("verify", f"再生成完了（{len(fix_targets)} 枚を作り直しました）")

    def _write_csv(self, path: Path, rows: list):
        """CSV を書き出す（スプレッドシートと同構造）"""
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            # v3: ビート/推定開始/エンジン/重要度/表示/ライセンス/クレジット 列を追加
            w.writerow(["章", "ブロック", "センテンス", "№", "ビート", "推定開始", "ソース",
                        "エンジン", "重要度", "表示", "画像", "URL", "ライセンス", "クレジット"])
            _disp = {"image": "画像", "hold": "継続", "none": "なし"}
            for r in rows:
                block_text = ""
                if r.get("sentence_index") == 0:
                    block_text = r.get("block_text", "")
                chapter = ""
                if r.get("block_index") == 0 and r.get("sentence_index") == 0:
                    chapter = r.get("chapter_title", "")
                route_label = self.ROUTE_LABELS.get(r.get("route", ""), r.get("route", ""))
                beat = r.get("beat_id")
                disp_label = _disp.get(r.get("display", ""), "")
                if not disp_label and r.get("filename"):
                    disp_label = "画像"  # v2(beat_mode無し)でも画像があれば「画像」
                w.writerow([
                    chapter,
                    block_text,
                    r.get("sentence", ""),
                    r.get("no", ""),
                    "" if beat is None else beat,
                    r.get("est_start", "") or "",
                    route_label,
                    r.get("engine", "") or "",
                    r.get("importance", "") or "",
                    disp_label,
                    r.get("filename", "") or "",
                    r.get("web_source_url", "") or r.get("commons_page_url", "") or "",
                    r.get("license", "") or "",
                    r.get("attribution", "") or "",
                ])
