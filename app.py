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
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

load_env(PROJECT_ROOT)

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
                         propaganda_mix: bool = False):
    job_dir = OUTPUT_DIR / job_id
    provider_label = ("nanobanana (Gemini)" if provider == PROVIDER_NANOBANANA
                      else f"gpt-image ({openai_quality})")
    try:
        _set_job_state(job_id, status="running", phase=0, message="開始しています...", percent=0)
        _add_log(job_id, "system",
                 f"ジョブ {job_id} を開始（{provider_label} / 並列 {concurrency} / style={style_preset} / route={route_mode} / Web画像 {web_image_count}）")

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
            skip_decorative=skip_decorative,
            web_image_count=web_image_count,
            max_diagrams=max_diagrams,
            route_mode=route_mode,
            propaganda_mix=propaganda_mix,
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
            past_jobs.append({
                "id": d.name,
                "title": manifest.get("title", job_state.get("title", d.name)),
                "status": job_state.get("status", "unknown"),
                "generated": manifest.get("generated", job_state.get("generated", 0)),
                "total": manifest.get("total_sentences", job_state.get("total_sentences", 0)),
                "date": d.name[:8] if len(d.name) >= 8 else "",
            })
    return render_template(
        "upload.html",
        past_jobs=past_jobs[:30],
        has_anthropic=bool(os.environ.get("ANTHROPIC_API_KEY")),
        has_gemini=bool(os.environ.get("GEMINI_API_KEY")),
        has_openai=bool(os.environ.get("OPENAI_API_KEY")),
    )


@app.route("/start", methods=["POST"])
@login_required
def start_job():
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
    propaganda_mix = request.form.get("propaganda_mix", "off") == "on"
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

    # API キー確認
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if provider == PROVIDER_NANOBANANA and not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if provider == PROVIDER_GPT_IMAGE and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        return jsonify({"error": f"{', '.join(missing)} が設定されていません"}), 400

    # 原稿取得（.docx は見出しスタイルを章として解析）
    manuscript_text = ""
    prebuilt_chapters = None
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
            manuscript_text = raw.decode("utf-8", errors="ignore")
    elif request.form.get("manuscript_text"):
        manuscript_text = request.form["manuscript_text"]
    else:
        return jsonify({"error": "原稿が入力されていません"}), 400

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

    _set_job_state(
        job_id,
        status="queued",
        phase=0,
        message="キューに追加しました",
        percent=0,
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
              skip_decorative, style_preset, web_image_count, max_diagrams, route_mode, propaganda_mix),
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
