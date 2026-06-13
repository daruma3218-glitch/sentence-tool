#!/usr/bin/env python3
"""センテンスつくーる - Flask Web アプリ

原稿をセンテンス単位に分割して、各文に対応する図解を一括生成する。
出力: テーブル表示（Web）+ CSV ダウンロード（Excel / Sheets 用）
"""

import functools
import io
import json
import os
import secrets
import threading
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)

from utils import load_env, load_json
from pipeline import SentencePipeline, VALID_STYLES
from generator import PROVIDER_NANOBANANA, PROVIDER_GPT_IMAGE, VALID_PROVIDERS


PROJECT_ROOT = Path(__file__).parent
# 出力先は環境変数 DATA_DIR で永続ディスク(Render Disk 等)に向けられる。
# 未設定ならプロジェクト直下（ローカル/ディスク無し運用）。
# ※ Render は DATA_DIR=/data に Disk をマウントすると、デプロイをまたいで
#   過去ジョブ（画像・manifest 等）が消えなくなる。
_DATA_DIR = os.environ.get("DATA_DIR", "").strip()
OUTPUT_DIR = (Path(_DATA_DIR) if _DATA_DIR else PROJECT_ROOT) / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_env(PROJECT_ROOT)


def load_channels() -> list:
    """channels.json を読み込みチャンネル一覧を返す（無ければ default 1件）。"""
    data = load_json(PROJECT_ROOT / "channels.json", {})
    chans = data.get("channels", []) if isinstance(data, dict) else []
    if not chans:
        chans = [{"id": "default", "name": "共通（デフォルト）", "api_env_prefix": "", "defaults": {}}]
    return chans


def get_channel(channel_id: str) -> dict:
    for c in load_channels():
        if c.get("id") == channel_id:
            return c
    return load_channels()[0]


def resolve_channel_keys(channel: dict) -> dict:
    """チャンネルの api_env_prefix から各APIキーを解決（無ければ共通キーにフォールバック）。"""
    prefix = (channel.get("api_env_prefix") or "").strip()
    def pick(name):
        if prefix:
            v = os.environ.get(f"{prefix}_{name}", "").strip()
            if v:
                return v
        return os.environ.get(name, "").strip()
    return {
        "anthropic": pick("ANTHROPIC_API_KEY"),
        "gemini": pick("GEMINI_API_KEY"),
        "openai": pick("OPENAI_API_KEY"),
    }


def _resolve_secret_key() -> str:
    """安定した SECRET_KEY を取得する。

    優先順: 環境変数 SECRET_KEY → 永続ファイル(.secret_key) → 新規生成して永続化。
    こうすることでサーバー再起動やワーカー間でも同じ鍵を使い、
    セッション（ログイン状態）が無効化されない。
    """
    env_key = os.environ.get("SECRET_KEY", "").strip()
    if env_key:
        return env_key
    key_file = PROJECT_ROOT / ".secret_key"
    try:
        if key_file.exists():
            saved = key_file.read_text(encoding="utf-8").strip()
            if saved:
                return saved
        new_key = secrets.token_hex(32)
        key_file.write_text(new_key, encoding="utf-8")
        return new_key
    except Exception:
        # ファイルに書けない環境では一応ランダム（最終手段）
        return secrets.token_hex(32)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB
app.secret_key = _resolve_secret_key()
# セッションを永続化（ブラウザを閉じても・長時間ダウンロード中でも切れない）
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=14)

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# ジョブ状態（メモリ）
_jobs: dict = {}
_job_logs: dict = {}
_jobs_lock = threading.Lock()
# route_feedback.jsonl への追記を直列化（同時フィードバックでも行が壊れないように）
FEEDBACK_LOCK = threading.Lock()


# ====== 認証 ======
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not APP_PASSWORD:
            return f(*args, **kwargs)
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return redirect(url_for("index"))
    if session.get("authenticated"):
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        if request.form.get("password", "") == APP_PASSWORD:
            session.permanent = True  # 14日間有効（PERMANENT_SESSION_LIFETIME）
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "パスワードが正しくありません"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ====== ジョブ管理 ======
def _set_job_state(job_id: str, **kwargs):
    with _jobs_lock:
        state = _jobs.setdefault(job_id, {})
        state.update(kwargs)
        state["updated_at"] = datetime.now().isoformat()
        try:
            (OUTPUT_DIR / job_id / "job.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def _get_job_state(job_id: str) -> dict:
    with _jobs_lock:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    job_path = OUTPUT_DIR / job_id / "job.json"
    if job_path.exists():
        try:
            return json.loads(job_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _add_log(job_id: str, category: str, message: str, detail: str = ""):
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "category": category,
        "message": message,
        "detail": detail,
    }
    with _jobs_lock:
        logs = _job_logs.setdefault(job_id, [])
        logs.append(entry)
    try:
        (OUTPUT_DIR / job_id / "logs.json").write_text(
            json.dumps(_job_logs[job_id], ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def _run_pipeline_thread(job_id: str, manuscript_text: str, user_instructions: str,
                         concurrency: int, provider: str, openai_quality: str,
                         skip_decorative: bool, style_preset: str,
                         web_image_count: int, max_diagrams: int, route_mode: str,
                         worldview_desc: str = "", verify_diagrams: bool = True,
                         channel_id: str = "default", ch_keys: dict = None,
                         character_ref_path: str = "",
                         title_override: str = "", fact_context: str = ""):
    job_dir = OUTPUT_DIR / job_id
    ch_keys = ch_keys or {}
    provider_label = ("nanobanana (Gemini)" if provider == PROVIDER_NANOBANANA
                      else f"gpt-image ({openai_quality})")
    try:
        _set_job_state(job_id, status="running", phase=0, message="開始しています...", percent=0)
        _add_log(job_id, "system",
                 f"ジョブ {job_id} を開始（ch={channel_id} / {provider_label} / 並列 {concurrency} / style={style_preset} / route={route_mode} / Web画像 {web_image_count}）")

        def on_progress(phase, msg, pct):
            _set_job_state(job_id, status="running", phase=phase, message=msg, percent=pct)

        def on_log(category, message, detail=""):
            _add_log(job_id, category, message, detail)

        def on_item(info):
            pass  # rows_progress.json 経由でフロントへ

        pipeline = SentencePipeline(
            manuscript_text=manuscript_text,
            output_dir=job_dir,
            user_instructions=user_instructions,
            concurrency=concurrency,
            provider=provider,
            openai_quality=openai_quality,
            style_preset=style_preset,
            worldview_desc=worldview_desc,
            verify_diagrams=verify_diagrams,
            channel_id=channel_id,
            anthropic_key=ch_keys.get("anthropic", ""),
            gemini_key=ch_keys.get("gemini", ""),
            openai_key=ch_keys.get("openai", ""),
            character_ref_path=character_ref_path,
            skip_decorative=skip_decorative,
            web_image_count=web_image_count,
            max_diagrams=max_diagrams,
            route_mode=route_mode,
            chart_engine=(get_channel(channel_id).get("defaults") or {}).get("chart_engine", "ai"),
            map_engine=(get_channel(channel_id).get("defaults") or {}).get("map_engine", "ai"),
            photo_source=(get_channel(channel_id).get("defaults") or {}).get("photo_source", "web"),
            beat_mode=bool((get_channel(channel_id).get("defaults") or {}).get("beat_mode", False)),
            chars_per_sec=(get_channel(channel_id).get("defaults") or {}).get("chars_per_sec", 5.5),
            realphoto_watermark=bool((get_channel(channel_id).get("defaults") or {}).get("realphoto_watermark", False)),
            chart_theme=(get_channel(channel_id).get("defaults") or {}).get("chart_theme"),
            title_override=title_override,
            fact_context=fact_context,
            progress_callback=on_progress,
            log_callback=on_log,
            item_callback=on_item,
        )
        manifest = pipeline.run()
        _set_job_state(
            job_id,
            status="completed",
            phase=4,
            message=f"完了: 生成 {manifest['generated']} / 全 {manifest['total_sentences']} 文",
            percent=100,
            title=manifest.get("title", ""),
            generated=manifest.get("generated", 0),
            failed=manifest.get("failed", 0),
            total_sentences=manifest.get("total_sentences", 0),
        )
        _add_log(job_id, "system",
                 f"全フェーズ完了（成功 {manifest['generated']} / 失敗 {manifest['failed']}）")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _set_job_state(job_id, status="error", message=str(e)[:200], percent=0)
        _add_log(job_id, "error", "パイプライン実行エラー", str(e)[:300])


# ====== ルート ======
@app.route("/")
@login_required
def index():
    past_jobs = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            manifest = load_json(d / "manifest.json", {})
            job_state = load_json(d / "job.json", {})
            if not manifest and not job_state:
                continue
            ch_id = manifest.get("channel_id", job_state.get("channel_id", ""))
            past_jobs.append({
                "id": d.name,
                "title": manifest.get("title", job_state.get("title", d.name)),
                "status": job_state.get("status", "unknown"),
                "generated": manifest.get("generated", job_state.get("generated", 0)),
                "total": manifest.get("total_sentences", job_state.get("total_sentences", 0)),
                "channel": ch_id,
                "date": d.name[:8] if len(d.name) >= 8 else "",
            })
    # 各チャンネルのキー設定状況（UI 表示用）
    channels = load_channels()
    for c in channels:
        keys = resolve_channel_keys(c)
        c["_has_gemini"] = bool(keys["gemini"])
        c["_has_openai"] = bool(keys["openai"])
        c["_has_anthropic"] = bool(keys["anthropic"])
    return render_template(
        "upload.html",
        past_jobs=past_jobs[:30],
        channels=channels,
        has_anthropic=bool(os.environ.get("ANTHROPIC_API_KEY")),
        has_gemini=bool(os.environ.get("GEMINI_API_KEY")),
        has_openai=bool(os.environ.get("OPENAI_API_KEY")),
    )


@app.route("/start", methods=["POST"])
@login_required
def start_job():
    # チャンネル選択 → そのチャンネルの API キーを解決
    channel_id = request.form.get("channel_id", "default")
    channel = get_channel(channel_id)
    channel_id = channel.get("id", "default")
    ch_keys = resolve_channel_keys(channel)

    provider = request.form.get("provider", PROVIDER_NANOBANANA)
    if provider not in VALID_PROVIDERS:
        provider = PROVIDER_NANOBANANA
    openai_quality = request.form.get("openai_quality", "medium")
    if openai_quality not in ("low", "medium", "high"):
        openai_quality = "medium"
    skip_decorative = request.form.get("skip_decorative", "off") == "on"
    style_preset = request.form.get("style_preset", "flat_infographic")
    if style_preset not in VALID_STYLES:
        style_preset = "flat_infographic"
    route_mode = request.form.get("route_mode", "auto")
    if route_mode not in ("auto", "all_ai"):
        route_mode = "auto"
    # 世界観統一モード（チェックON時のみ description を有効化）
    worldview_on = request.form.get("worldview_mode", "off") == "on"
    worldview_desc = request.form.get("worldview_desc", "").strip() if worldview_on else ""
    # ON なのに本文が空（フォーム未入力など）なら、チャンネル既定の世界観へフォールバック。
    # これで「先生キャラ等の設定が空欄で効かない」事故を防ぐ。
    if worldview_on and not worldview_desc:
        worldview_desc = (channel.get("defaults", {}) or {}).get("worldview_desc", "").strip()
    # キャラ固定の参照画像（チャンネル設定 character_ref）。存在すれば絶対パスを渡す。
    # 世界観モードON のときだけ有効（キャラ統一の一部）。
    character_ref_path = ""
    if worldview_on:
        _cref = (channel.get("defaults", {}) or {}).get("character_ref", "").strip()
        if _cref:
            _crp = PROJECT_ROOT / _cref
            if _crp.exists():
                character_ref_path = str(_crp)
    verify_diagrams = request.form.get("verify_diagrams", "off") == "on"
    try:
        web_image_count = int(request.form.get("web_image_count", "0"))
    except ValueError:
        web_image_count = 0
    web_image_count = max(0, min(web_image_count, 200))
    try:
        max_diagrams = int(request.form.get("max_diagrams", "150"))
    except ValueError:
        max_diagrams = 150
    max_diagrams = max(1, min(max_diagrams, 300))

    # API キー確認（チャンネルのキー＝個別 or 共通フォールバック）
    missing = []
    if not ch_keys["anthropic"]:
        missing.append("ANTHROPIC_API_KEY")
    if provider == PROVIDER_NANOBANANA and not ch_keys["gemini"]:
        missing.append("GEMINI_API_KEY")
    if provider == PROVIDER_GPT_IMAGE and not ch_keys["openai"]:
        missing.append("OPENAI_API_KEY")
    if missing:
        pfx = channel.get("api_env_prefix", "")
        hint = f"（チャンネル「{channel.get('name','')}」用に {pfx}_... を設定するか共通キーを設定）" if pfx else ""
        return jsonify({"error": f"{', '.join(missing)} が設定されていません{hint}"}), 400

    # 原稿取得（.docx は見出しを章として解析 / .json または貼り付けJSONは原稿パイプライン final.json 直結）
    from utils import parse_final_json, extract_from_final_json
    manuscript_text = ""
    prebuilt_chapters = None
    title_override = ""      # v3 Step7: final.json の tentative_title をタイトルに
    fact_context = ""        # v3 Step7: final.json の fact_report 等を chart 抽出の文脈に
    _final_obj = None
    if "manuscript_file" in request.files and request.files["manuscript_file"].filename:
        f = request.files["manuscript_file"]
        fname = f.filename.lower()
        raw = f.read()
        if fname.endswith(".docx"):
            try:
                from splitter import parse_docx_to_chapters
                manuscript_text, prebuilt_chapters = parse_docx_to_chapters(raw)
                if not prebuilt_chapters:
                    return jsonify({"error": ".docx から本文を抽出できませんでした"}), 400
            except Exception as e:
                return jsonify({"error": f".docx の解析に失敗: {str(e)[:120]}"}), 400
        else:
            decoded = raw.decode("utf-8", errors="ignore")
            _final_obj = parse_final_json(decoded)
            if fname.endswith(".json") and _final_obj is None:
                return jsonify({"error": "final.json の形式が不正です（本文の final キーが見つかりません）"}), 400
            if _final_obj is None:
                manuscript_text = decoded
    elif request.form.get("manuscript_text"):
        pasted = request.form["manuscript_text"]
        _final_obj = parse_final_json(pasted)  # JSON を貼り付けても final.json として扱う
        if _final_obj is None:
            manuscript_text = pasted
    else:
        return jsonify({"error": "原稿が入力されていません"}), 400

    # final.json から本文・タイトル・検証文脈を取り出す（存在しないキーは任意扱い）
    if _final_obj is not None:
        _info = extract_from_final_json(_final_obj)
        manuscript_text = _info["manuscript"]
        title_override = _info["title"]
        fact_context = _info["fact_context"]

    if len(manuscript_text.strip()) < 100:
        return jsonify({"error": "原稿が短すぎます（100文字以上必要）"}), 400

    try:
        concurrency = int(request.form.get("concurrency", "12"))
    except ValueError:
        concurrency = 12
    concurrency = max(1, min(concurrency, 24))

    user_instructions = request.form.get("user_instructions", "").strip()

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "manuscript.txt").write_text(manuscript_text, encoding="utf-8")
    if user_instructions:
        (job_dir / "user_instructions.txt").write_text(user_instructions, encoding="utf-8")
    # .docx の見出しから作った章構造を保存（pipeline が読み込む）
    if prebuilt_chapters:
        (job_dir / "prebuilt_chapters.json").write_text(
            json.dumps({"chapters": prebuilt_chapters}, ensure_ascii=False), encoding="utf-8")

    # 冒頭・終わりの固定画像（任意）を保存（images/intro.*, images/outro.*）
    (job_dir / "images").mkdir(parents=True, exist_ok=True)
    for slot in ("intro", "outro"):
        f = request.files.get(f"{slot}_image")
        if f and f.filename:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".webp"):
                f.save(str(job_dir / "images" / f"{slot}{ext}"))

    _set_job_state(
        job_id,
        status="queued",
        phase=0,
        message="キューに追加しました",
        percent=0,
        channel_id=channel_id,
        concurrency=concurrency,
        provider=provider,
        openai_quality=openai_quality if provider == PROVIDER_GPT_IMAGE else None,
        skip_decorative=skip_decorative,
        style_preset=style_preset,
        web_image_count=web_image_count,
        max_diagrams=max_diagrams,
    )

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, manuscript_text, user_instructions, concurrency, provider, openai_quality,
              skip_decorative, style_preset, web_image_count, max_diagrams, route_mode, worldview_desc, verify_diagrams,
              channel_id, ch_keys, character_ref_path, title_override, fact_context),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id, "redirect": f"/progress/{job_id}"})


@app.route("/progress/<job_id>")
@login_required
def progress_page(job_id):
    return render_template("progress.html", job_id=job_id)


@app.route("/api/status/<job_id>")
@login_required
def api_status(job_id):
    state = _get_job_state(job_id)
    if not state:
        return jsonify({"status": "not_found"}), 404
    return jsonify(state)


@app.route("/api/rows/<job_id>")
@login_required
def api_rows(job_id):
    """センテンス行の進捗"""
    snapshot = load_json(OUTPUT_DIR / job_id / "rows_progress.json", {"rows": []})
    return jsonify(snapshot)


@app.route("/api/logs/<job_id>")
@login_required
def api_logs(job_id):
    since = int(request.args.get("since", 0))
    with _jobs_lock:
        logs = list(_job_logs.get(job_id, []))
    if not logs:
        logs = load_json(OUTPUT_DIR / job_id / "logs.json", [])
    return jsonify({"logs": logs[since:], "total": len(logs)})


@app.route("/api/manifest/<job_id>")
@login_required
def api_manifest(job_id):
    manifest = load_json(OUTPUT_DIR / job_id / "manifest.json", {})
    return jsonify(manifest)


def _update_regen_snapshot(job_dir, no, ok, filename=None, engine=None):
    """再生成結果を rows_progress.json に反映（chart/map/AI 共通）。"""
    import json as _json
    snap_path = job_dir / "rows_progress.json"
    snap = load_json(snap_path, {"rows": []})
    for r in snap.get("rows", []):
        if r.get("no") == no:
            r["status"] = "ok" if ok else "failed"
            if filename:
                r["filename"] = filename
            if engine:
                r["engine"] = engine
            break
    try:
        snap_path.write_text(_json.dumps(snap, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _regenerate_render_chart(job_dir, no, snap_row, ch_keys, defaults, extra=""):
    """v3: chart 行（決定論レンダ）を再生成。

    1) 保存済み chart_spec があり追加指示が無ければ、その spec をそのまま描き直す
       （決定論・確実に同じグラフ）。
    2) 保存 spec が無い（旧ジョブ）or 追加指示あり → 文（＋指示）から spec を抽出し直す。
       追加指示は block_context に足すので、ユーザーが数値を補えば原文照合も通る。
    """
    from renderer import render_chart
    row = snap_row or {}
    chart_theme = defaults.get("chart_theme")
    out = job_dir / "images" / f"{no}.png"
    saved_spec = row.get("chart_spec")

    # 1) 保存 spec をそのまま再描画（追加指示が無いとき）
    if saved_spec and not extra:
        try:
            if render_chart(saved_spec, out, theme=chart_theme):
                _update_regen_snapshot(job_dir, no, True, filename=f"{no}.png", engine="render")
                return jsonify({"ok": True, "no": no, "filename": f"{no}.png",
                                "ts": datetime.now().strftime("%H%M%S")})
        except Exception:
            pass  # 失敗したら抽出し直しへフォールバック

    # 2) 抽出し直し（旧ジョブ or 追加指示で数値・体裁を変えたいとき）
    from utils import get_anthropic_client
    from router import extract_chart_specs
    sentence = (row.get("sentence") or "").strip()
    if not sentence and not saved_spec:
        return jsonify({"error": "グラフ再生成に必要な文が見つかりません（データが消えた可能性）"}), 404
    ctx = row.get("block_text") or ""
    if extra:
        ctx = (ctx + "\n" + extra).strip()  # 追加指示を文脈に（数値を補える＝原文照合も通る）
    spec = None
    if sentence:
        try:
            client = get_anthropic_client(ch_keys.get("anthropic", ""))
            specs = extract_chart_specs(client, [{"no": no, "sentence": sentence, "block_text": ctx}])
            spec = specs.get(no)
        except Exception as e:
            if not saved_spec:
                return jsonify({"error": f"グラフの数値抽出に失敗: {str(e)[:140]}"}), 500
    spec = spec or saved_spec  # 抽出できなければ保存 spec にフォールバック
    if not spec:
        return jsonify({"error": "この文からグラフ化できる数値が読み取れませんでした。再生成ダイアログに数値（例: ロシア 6.3%, NATO 2.1%）を書くと作り直せます。"}), 422
    try:
        ok = bool(render_chart(spec, out, theme=chart_theme))
    except Exception as e:
        return jsonify({"error": f"グラフ描画に失敗: {str(e)[:140]}"}), 500
    if not ok:
        return jsonify({"error": "グラフ描画に失敗しました"}), 500
    _update_regen_snapshot(job_dir, no, True, filename=f"{no}.png", engine="render")
    return jsonify({"ok": True, "no": no, "filename": f"{no}.png", "ts": datetime.now().strftime("%H%M%S")})


def _regenerate_render_map(job_dir, no, snap_row, ch_keys, defaults, extra=""):
    """v3: map 行（決定論レンダ）を再生成。保存 map_spec を優先、無ければ抽出し直す。"""
    from renderer import render_map
    row = snap_row or {}
    chart_theme = defaults.get("chart_theme")
    out = job_dir / "images" / f"{no}.png"
    saved_spec = row.get("map_spec")

    if saved_spec and not extra:
        try:
            if render_map(saved_spec, out, theme=chart_theme):
                _update_regen_snapshot(job_dir, no, True, filename=f"{no}.png", engine="render")
                return jsonify({"ok": True, "no": no, "filename": f"{no}.png",
                                "ts": datetime.now().strftime("%H%M%S")})
        except Exception:
            pass

    from utils import get_anthropic_client
    from router import extract_map_specs
    sentence = (row.get("sentence") or "").strip()
    if not sentence and not saved_spec:
        return jsonify({"error": "地図再生成に必要な文が見つかりません（データが消えた可能性）"}), 404
    ctx = row.get("block_text") or ""
    if extra:
        ctx = (ctx + "\n" + extra).strip()
    spec = None
    if sentence:
        try:
            client = get_anthropic_client(ch_keys.get("anthropic", ""))
            specs = extract_map_specs(client, [{"no": no, "sentence": sentence, "block_text": ctx}])
            spec = specs.get(no)
        except Exception as e:
            if not saved_spec:
                return jsonify({"error": f"地図の地名抽出に失敗: {str(e)[:140]}"}), 500
    spec = spec or saved_spec
    if not spec:
        return jsonify({"error": "この文から地図化できる国・地域が特定できませんでした。再生成ダイアログに国名を書くと作り直せます。"}), 422
    try:
        ok = bool(render_map(spec, out, theme=chart_theme))
    except Exception as e:
        return jsonify({"error": f"地図描画に失敗: {str(e)[:140]}"}), 500
    if not ok:
        return jsonify({"error": "地図描画に失敗しました（対象の国/地域を特定できませんでした）"}), 500
    _update_regen_snapshot(job_dir, no, True, filename=f"{no}.png", engine="render")
    return jsonify({"ok": True, "no": no, "filename": f"{no}.png", "ts": datetime.now().strftime("%H%M%S")})


@app.route("/api/regenerate/<job_id>/<int:no>", methods=["POST"])
@login_required
def api_regenerate(job_id, no):
    """指定シーン(№)の画像を1枚だけ作り直す。

    - chart / map（決定論レンダ）: 文から spec を再抽出して renderer で描き直す
    - それ以外（illustration / realphoto / diagram など）: AI で作り直す
    任意で extra_instruction（追加指示）を受け取り、プロンプト末尾に足して再生成できる。
    """
    from generator import run_parallel_generation, PROVIDER_NANOBANANA
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return jsonify({"error": "ジョブのデータがサーバー上にありません（再デプロイ等で消えた可能性）。お手数ですが再生成してください。"}), 404

    # manifest があればそれを、無ければ job.json を params に使う（生成中でも動くように）
    manifest = load_json(job_dir / "manifest.json", {})
    job_state = load_json(job_dir / "job.json", {})
    params = manifest or job_state
    prompts = load_json(job_dir / "prompts.json", {"rows": []}).get("rows", [])
    snap_all = load_json(job_dir / "rows_progress.json", {"rows": []}).get("rows", [])
    target = next((r for r in prompts if r.get("no") == no), None)
    snap_row = next((r for r in snap_all if r.get("no") == no), None)
    extra = (request.form.get("extra_instruction", "") or "").strip()

    # チャンネル設定・キーを解決（再生成も該当チャンネルの設定で）
    channel_id = params.get("channel_id", "default")
    channel = get_channel(channel_id)
    ch_keys = resolve_channel_keys(channel)
    defaults = channel.get("defaults", {}) or {}

    # 対象行のルート/エンジン（snapshot 優先：chart/map は render エンジン）
    route = (snap_row or {}).get("route") or (target or {}).get("route") or (target or {}).get("type") or ""
    engine = (snap_row or {}).get("engine") or ""

    # ===== v3: chart / map は決定論レンダ（AIプロンプトを持たない）→ 抽出し直して描き直す =====
    if route == "chart" and (engine == "render" or defaults.get("chart_engine") == "render"):
        return _regenerate_render_chart(job_dir, no, snap_row, ch_keys, defaults, extra)
    if route == "map" and (engine == "render" or defaults.get("map_engine") == "render"):
        return _regenerate_render_map(job_dir, no, snap_row, ch_keys, defaults, extra)

    # ===== AI 生成（illustration / realphoto / diagram など）=====
    if not prompts:
        return jsonify({"error": "プロンプト情報が見つかりません（まだ生成準備中か、データが消えています）"}), 404
    if not target or not target.get("prompt"):
        return jsonify({"error": f"№{no} のAI生成用プロンプトが見つかりません（chart/map は数値・地名のある文のみ再生成可）"}), 404

    route = route or "illustration"
    prompt_text = target.get("prompt", "")
    if extra:
        prompt_text = f"{prompt_text}\n\nAdditional instruction: {extra}"

    provider = params.get("provider", PROVIDER_NANOBANANA)
    openai_quality = params.get("openai_quality") or "medium"
    style_preset = params.get("style_preset", "flat_infographic")

    # キャラ固定の参照画像（チャンネル設定）。先生が描かれる illustration のみ使用。
    character_ref_path = ""
    _cref = defaults.get("character_ref", "").strip()
    if _cref:
        _crp = PROJECT_ROOT / _cref
        if _crp.exists():
            character_ref_path = str(_crp)

    entry = {
        "index": no,
        "prompt": prompt_text,
        "type": route,
        "section": target.get("chapter_title", ""),
        "excerpt": target.get("sentence", ""),
        "keypoint": (target.get("sentence", "") or "")[:30],
        "allowed_terms": target.get("allowed_terms", []),
        "style": style_preset,
        # 元の行が先生シーン(character)なら、再生成でもキャラ固定
        "character": bool(target.get("character", False)) and route == "illustration",
    }

    try:
        results = run_parallel_generation(
            prompts=[entry],
            output_dir=job_dir / "images",
            provider=provider,
            gemini_api_key=ch_keys.get("gemini") or None,
            openai_api_key=ch_keys.get("openai") or None,
            openai_quality=openai_quality,
            concurrency=1,
            reference_image_path=character_ref_path,
        )
    except Exception as e:
        return jsonify({"error": f"再生成に失敗: {str(e)[:150]}"}), 500

    ok = bool(results and results[0].get("success"))
    filename = results[0].get("filename") if ok else None

    # rows_progress.json を更新（AI 生成は engine=ai のまま）
    _update_regen_snapshot(job_dir, no, ok, filename=filename)

    if not ok:
        return jsonify({"error": results[0].get("error", "生成失敗") if results else "生成失敗"}), 500
    # キャッシュ回避用にタイムスタンプ付き URL を返す
    return jsonify({"ok": True, "no": no, "filename": filename, "ts": datetime.now().strftime("%H%M%S")})


# v3 Step5: ルート違いフィードバック。編集者が「この文は別ルートが正しい」と教えると、
# チャンネル別の route_feedback.jsonl に蓄積し、次回以降のルーターに few-shot として渡す。
_FEEDBACK_ROUTES = ("web_photo", "realphoto", "map", "diagram", "chart", "illustration", "skip")


@app.route("/api/feedback/<job_id>/<int:no>", methods=["POST"])
@login_required
def api_feedback(job_id, no):
    """指定シーン(№)のルート判定が間違っていたことを記録する。

    body: correct_route（正しいルート）。文・誤判定ルート・チャンネルは
    サーバー側のジョブデータから補完して route_feedback.jsonl に追記する。
    """
    import json as _json
    correct_route = (request.form.get("correct_route", "") or "").strip()
    if correct_route not in _FEEDBACK_ROUTES:
        return jsonify({"error": f"未知のルート: {correct_route}"}), 400

    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return jsonify({"error": "ジョブのデータが見つかりません"}), 404

    # 文・誤判定ルート・チャンネルをジョブデータから取得
    prompts = load_json(job_dir / "prompts.json", {"rows": []}).get("rows", [])
    target = next((r for r in prompts if r.get("no") == no), None)
    if target is None:
        snap = load_json(job_dir / "rows_progress.json", {"rows": []})
        target = next((r for r in snap.get("rows", []) if r.get("no") == no), None)
    if target is None:
        return jsonify({"error": f"№{no} が見つかりません"}), 404

    sentence = (target.get("sentence", "") or "").strip()
    given_route = (target.get("route") or target.get("type") or "").strip()
    if given_route == correct_route:
        return jsonify({"error": "同じルートです"}), 400

    manifest = load_json(job_dir / "manifest.json", {})
    job_state = load_json(job_dir / "job.json", {})
    channel_id = manifest.get("channel_id") or job_state.get("channel_id") or "default"

    record = {
        "channel_id": channel_id,
        "sentence": sentence[:200],
        "given_route": given_route,
        "correct_route": correct_route,
        "job_id": job_id,
        "no": no,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    fb_path = OUTPUT_DIR / "route_feedback.jsonl"
    try:
        with FEEDBACK_LOCK:
            with open(fb_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        return jsonify({"error": f"保存に失敗: {str(e)[:150]}"}), 500

    return jsonify({"ok": True, "no": no, "given_route": given_route, "correct_route": correct_route})


@app.route("/results/<job_id>/<path:filename>")
@login_required
def serve_results(job_id, filename):
    result_dir = OUTPUT_DIR / job_id
    if not result_dir.exists():
        return "結果が見つかりません", 404
    return send_from_directory(str(result_dir), filename)


@app.route("/download/csv/<job_id>")
@login_required
def download_csv(job_id):
    """CSV を直接ダウンロード（Excel/Sheets 用 UTF-8 BOM 付き）"""
    csv_path = OUTPUT_DIR / job_id / "result.csv"
    if not csv_path.exists():
        return "CSV が見つかりません", 404
    manifest = load_json(OUTPUT_DIR / job_id / "manifest.json", {})
    title = manifest.get("title", job_id)
    safe_title = "".join(c for c in title if c not in r'\/:*?"<>|').strip()[:50] or job_id
    return send_file(
        csv_path,
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name=f"{safe_title}_{job_id}.csv",
    )


@app.route("/download/<job_id>")
@login_required
def download_zip(job_id):
    """画像一式 + CSV + manifest を ZIP でダウンロード"""
    result_dir = OUTPUT_DIR / job_id
    if not result_dir.exists():
        return "結果が見つかりません", 404

    manifest = load_json(result_dir / "manifest.json", {})
    title = manifest.get("title", job_id)
    safe_title = "".join(c for c in title if c not in r'\/:*?"<>|').strip()[:50] or job_id

    # ZIP はメモリ(BytesIO)ではなく一時ファイルに書き出す。
    # 大量画像(100枚超)を BytesIO に展開すると Render Free(512MB)で
    # メモリ超過 → ワーカー再起動 → ダウンロード中断/ログアウトの原因になるため。
    import tempfile
    tmp = tempfile.NamedTemporaryFile(prefix=f"{job_id}_", suffix=".zip", delete=False)
    tmp_path = tmp.name
    tmp.close()
    # 画像は既に圧縮済み(PNG/JPG)なので ZIP_STORED で CPU/メモリを節約
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf:
        images_dir = result_dir / "images"
        if images_dir.exists():
            for img in sorted(images_dir.iterdir()):
                if img.is_file() and img.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                    zf.write(img, f"images/{img.name}")
        for extra, arc in [("result.csv", "result.csv"),
                           ("result.html", "result.html"),
                           ("manifest.json", "manifest.json"),
                           ("manuscript.txt", "manuscript.txt")]:
            p = result_dir / extra
            if p.exists():
                zf.write(p, arc)

    resp = send_file(
        tmp_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_title}_{job_id}.zip",
    )

    # 送信完了後に一時ファイルを削除
    @resp.call_on_close
    def _cleanup():
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3002))
    print("\n" + "=" * 50)
    print("  センテンスつくーる 起動中...")
    print(f"  http://localhost:{port}")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
