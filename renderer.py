#!/usr/bin/env python3
"""renderer.py (v3 Step1) — chart を matplotlib で決定論レンダリング（LLM不使用）

2026-06-12 安福: AI 製グラフの「数値ズレ・日本語文字化け」を構造的に排除するため新設。
原稿に書かれた数値をそのまま描画し、日本語は同梱フォントで描く。
verifier による検品（確率的）を不要にする。

入力: chart_spec（dict, §3.2）/ chart_theme（dict, channels.json）
出力: 1920x1080(16:9) PNG。成功で True、失敗で False（呼び出し側で engine:ai へ降格）。
LLM は一切使わない。
"""

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ヘッドレス（サーバ）
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ===== フォント登録（文字化けを構造的に排除）=====
_ASSETS = Path(__file__).parent / "assets"
_FONT_DIR = _ASSETS / "fonts"
_FONT_NAME = "Noto Sans JP"


def _register_fonts() -> bool:
    ok = False
    for fname in ("NotoSansJP-Regular.ttf", "NotoSansJP-Bold.ttf"):
        p = _FONT_DIR / fname
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
                ok = True
            except Exception:
                pass
    if ok:
        matplotlib.rcParams["font.family"] = _FONT_NAME
    matplotlib.rcParams["axes.unicode_minus"] = False  # マイナス記号の豆腐対策
    return ok


FONTS_AVAILABLE = _register_fonts()

# ===== 既定テーマ（channels.json の chart_theme で上書き）=====
DEFAULT_THEME = {
    "bg": "#FFFDF7", "main": "#1E40AF", "accent": "#C2410C",
    "grid": "#E5E7EB", "text": "#1F2937", "font": "NotoSansJP",
}

# 出力サイズ: 19.2 x 10.8 inch * 100dpi = 1920x1080
_W_IN, _H_IN, _DPI = 19.2, 10.8, 100

VALID_CHART_TYPES = ("bar", "line", "pie", "big_number", "comparison", "timeline")


# ===== 数値フォーマット =====
def _fmt_num(v) -> str:
    """数値を見やすく整形（整数はカンマ区切り、小数は無駄な0を落とす）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f):
        return f"{int(f):,}"
    s = f"{f:,.2f}".rstrip("0").rstrip(".")
    return s


def _fmt_val(v, unit: str = "") -> str:
    s = _fmt_num(v)
    return f"{s}{unit}" if unit else s


def _fmt_num_jp(v) -> str:
    """大きな数を 兆/億/万 で読みやすく短く整形（巨大数字の表示崩れ・桁見間違い防止）。

    例: 300000000 → "3億" / 1200000 → "120万" / 1000000000000 → "1兆".
    端数があり崩れる場合（10^4 の倍数でない等）はカンマ区切りにフォールバックして
    正確さを保つ。小数もそのまま（_fmt_num）。
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != int(f):
        return _fmt_num(v)
    n = int(f)
    neg = n < 0
    n = abs(n)
    # 10^4 未満、または 10^4 の倍数でない（万未満の端数がある）→ カンマ表記で正確に
    if n < 10000 or n % 10000 != 0:
        return _fmt_num(v)
    cho = n // 10**12
    oku = (n % 10**12) // 10**8
    man = (n % 10**8) // 10**4
    parts = []
    if cho:
        parts.append(f"{cho:,}兆")  # 兆は桁が大きくなり得るのでカンマ可
    if oku:
        parts.append(f"{oku}億")    # 億・万は 0〜9999 なのでカンマ不要
    if man:
        parts.append(f"{man}万")
    if not parts:
        return _fmt_num(v)
    s = "".join(parts)
    return f"-{s}" if neg else s


def _fmt_val_jp(v, unit: str = "") -> str:
    s = _fmt_num_jp(v)
    return f"{s}{unit}" if unit else s


def _fit_fontsize(fig, s: str, max_w_frac: float, base_size: float,
                  min_size: float, weight: str = "bold") -> float:
    """文字列 s が図幅の max_w_frac 以内に収まる最大フォントサイズを返す（実寸計測）。

    巨大数字（例: 300,000,000円）が固定サイズで隣や中央に食い込む崩れを防ぐ。
    計測に失敗しても落ちないよう、文字数からの概算にフォールバックする。
    """
    if not s:
        return base_size
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    target = max_w_frac * fig_w_px
    w = None
    try:
        renderer = fig.canvas.get_renderer()
        t = fig.text(0.5, 0.5, s, fontsize=base_size, fontweight=weight)
        try:
            w = t.get_window_extent(renderer=renderer).width
        finally:
            t.remove()
    except Exception:
        w = None
    if not w or w <= 0:
        # フォールバック: 全角想定で広めに概算（収めきれず崩れるより安全側）
        w = len(s) * base_size * 0.7
    if w <= target:
        return base_size
    return max(min_size, base_size * target / w)


def _series(spec: dict):
    """series を [(label, value)] に正規化。value は数値化できるものだけ。"""
    out = []
    for it in (spec.get("series") or []):
        if not isinstance(it, dict):
            continue
        out.append((str(it.get("label", "")), it.get("value")))
    return out


def _num_series(spec: dict):
    """数値 series のみ [(label, float)]。数値化できない要素は除外。"""
    out = []
    for label, val in _series(spec):
        try:
            out.append((label, float(val)))
        except (TypeError, ValueError):
            continue
    return out


# ===== 共通の装飾（タイトル・出典）=====
def _draw_title_and_source(fig, spec: dict, theme: dict):
    title = (spec.get("title") or "").strip()
    if title:
        fig.text(0.5, 0.93, title, ha="center", va="top",
                 fontsize=46, fontweight="bold", color=theme["text"])
    src = (spec.get("source_note") or "").strip()
    if src:
        fig.text(0.98, 0.03, f"出典: {src}", ha="right", va="bottom",
                 fontsize=20, color="#6B7280")


def _palette(theme: dict, n: int):
    """main/accent を基点に n 色のパレットを作る（虹色は使わない）。"""
    base = [theme["main"], theme["accent"], "#0F766E", "#7C3AED", "#B45309", "#475569"]
    if n <= len(base):
        return base[:n]
    # 足りなければ薄い繰り返し
    return [base[i % len(base)] for i in range(n)]


# ===== 各 chart_type の描画 =====
def _draw_bar(fig, spec: dict, theme: dict):
    data = _num_series(spec)
    if not data:
        raise ValueError("bar: 数値 series が空")
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    unit = spec.get("unit", "")
    hi = spec.get("highlight_index")
    ax = fig.add_axes([0.10, 0.13, 0.84, 0.68])
    ax.set_facecolor(theme["bg"])
    colors = [theme["main"]] * len(values)
    if isinstance(hi, int) and 0 <= hi < len(values):
        colors[hi] = theme["accent"]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.62, zorder=3)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=28, color=theme["text"])
    ax.tick_params(axis="y", labelsize=22, colors="#6B7280")
    ax.grid(axis="y", color=theme["grid"], linewidth=1, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    # 値ラベル（軸を読ませない）
    top = max(values) if values else 1
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.02,
                _fmt_val(values[i], unit), ha="center", va="bottom",
                fontsize=28, fontweight="bold", color=theme["text"])
    ax.set_ylim(0, top * 1.18)


def _draw_line(fig, spec: dict, theme: dict):
    data = _num_series(spec)
    if not data:
        raise ValueError("line: 数値 series が空")
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    unit = spec.get("unit", "")
    ax = fig.add_axes([0.10, 0.13, 0.84, 0.68])
    ax.set_facecolor(theme["bg"])
    ax.plot(range(len(values)), values, color=theme["main"], linewidth=4,
            marker="o", markersize=12, markerfacecolor=theme["accent"],
            markeredgecolor="white", markeredgewidth=2, zorder=3)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=26, color=theme["text"])
    ax.tick_params(axis="y", labelsize=22, colors="#6B7280")
    ax.grid(axis="y", color=theme["grid"], linewidth=1, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(theme["grid"])
    rng = (max(values) - min(values)) or max(abs(max(values)), 1)
    for i, v in enumerate(values):
        ax.text(i, v + rng * 0.04, _fmt_val(v, unit), ha="center", va="bottom",
                fontsize=24, fontweight="bold", color=theme["text"])
    ax.set_ylim(min(values) - rng * 0.12, max(values) + rng * 0.20)


def _draw_pie(fig, spec: dict, theme: dict):
    data = _num_series(spec)
    if not data:
        raise ValueError("pie: 数値 series が空")
    labels = [d[0] for d in data]
    values = [abs(d[1]) for d in data]
    hi = spec.get("highlight_index")
    explode = [0.06 if (isinstance(hi, int) and i == hi) else 0 for i in range(len(values))]
    ax = fig.add_axes([0.18, 0.10, 0.64, 0.70])
    colors = _palette(theme, len(values))
    wedges, _txts, autotxts = ax.pie(
        values, labels=labels, explode=explode, colors=colors,
        autopct=lambda p: f"{p:.0f}%", startangle=90, counterclock=False,
        textprops={"fontsize": 28, "color": theme["text"]},
        wedgeprops={"edgecolor": theme["bg"], "linewidth": 3},
    )
    for at in autotxts:
        at.set_color("white")
        at.set_fontsize(26)
        at.set_fontweight("bold")
    ax.set_aspect("equal")


def _draw_big_number(fig, spec: dict, theme: dict):
    """巨大数字 + 単位 + ラベル（テロップ的な説明画面）。"""
    data = _series(spec)
    # 数値を1つ取り出す（series[0] または spec['value']）
    val = None
    label = ""
    if data:
        label, val = data[0]
    if val is None:
        val = spec.get("value")
    if val is None:
        raise ValueError("big_number: 値がない")
    unit = spec.get("unit", "")
    # 巨大数字は 億/万 表記＋実寸で図幅90%に収める（桁あふれ・見間違い防止）
    num_str = _fmt_num_jp(val)
    nsize = _fit_fontsize(fig, num_str, 0.90, 260, 70)
    fig.text(0.5, 0.50, num_str, ha="center", va="center",
             fontsize=nsize, fontweight="bold", color=theme["main"])
    if unit:
        usize = _fit_fontsize(fig, unit, 0.5, 64, 30)
        fig.text(0.5, 0.30, unit, ha="center", va="center",
                 fontsize=usize, fontweight="bold", color=theme["accent"])
    if label:
        lsize = _fit_fontsize(fig, label, 0.9, 46, 24, weight="normal")
        fig.text(0.5, 0.20, label, ha="center", va="center",
                 fontsize=lsize, color=theme["text"])


def _draw_comparison(fig, spec: dict, theme: dict):
    """2値（以上）を巨大数字で横並び比較（A vs B）。

    各値は 億/万 表記にし、自分のスロット幅に収まるようフォントを自動縮小する。
    桁数の違う数字（例: 300円 と 3億円）が中央や隣に食い込む崩れを構造的に防ぐ。
    """
    data = _series(spec)
    if len(data) < 2:
        raise ValueError("comparison: 2つ以上の値が必要")
    data = data[:3]  # 最大3
    unit = spec.get("unit", "")
    hi = spec.get("highlight_index")
    n = len(data)
    slot = 1.0 / n
    # 値テキスト（億/万）と、各スロット82%に収める共通フォントサイズ（統一感のため最小に合わせる）
    val_texts = [_fmt_val_jp(val, unit) for (_, val) in data]
    vsize = min(_fit_fontsize(fig, t, slot * 0.82, 150, 40) for t in val_texts)
    # ラベルもスロット90%に収める
    lsize = min(_fit_fontsize(fig, lab, slot * 0.9, 44, 22, weight="normal")
                for (lab, _) in data)
    for i, (label, val) in enumerate(data):
        cx = slot * (i + 0.5)
        color = theme["accent"] if (isinstance(hi, int) and i == hi) else theme["main"]
        fig.text(cx, 0.54, val_texts[i], ha="center", va="center",
                 fontsize=vsize, fontweight="bold", color=color)
        fig.text(cx, 0.30, label, ha="center", va="center",
                 fontsize=lsize, color=theme["text"])
    # スロット境界に "vs"（数値とは別レイヤ・小さめ。値が82%なので境界に隙間が残る）
    for i in range(n - 1):
        fig.text(slot * (i + 1), 0.54, "vs", ha="center", va="center",
                 fontsize=40, color="#9CA3AF", zorder=5)


def _draw_timeline(fig, spec: dict, theme: dict):
    """横一直線のタイムライン。series=[{label: 時点, value: 出来事(文字 or 数値)}]。"""
    data = _series(spec)
    if not data:
        raise ValueError("timeline: series が空")
    data = data[:6]
    ax = fig.add_axes([0.06, 0.18, 0.88, 0.58])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0.04, 0.96], [0.5, 0.5], color=theme["grid"], linewidth=4, zorder=1)
    n = len(data)
    xs = [0.08 + (0.84) * (i / max(1, n - 1)) for i in range(n)]
    for i, (label, val) in enumerate(data):
        x = xs[i]
        ax.scatter([x], [0.5], s=420, color=theme["accent"], zorder=3,
                   edgecolors="white", linewidths=3)
        up = (i % 2 == 0)
        ty = 0.72 if up else 0.28
        ax.text(x, 0.5 + (0.10 if up else -0.10), str(label), ha="center",
                va=("bottom" if up else "top"), fontsize=30,
                fontweight="bold", color=theme["main"])
        ax.text(x, ty, str(val), ha="center", va=("bottom" if up else "top"),
                fontsize=26, color=theme["text"], wrap=True)


_DRAWERS = {
    "bar": _draw_bar, "line": _draw_line, "pie": _draw_pie,
    "big_number": _draw_big_number, "comparison": _draw_comparison,
    "timeline": _draw_timeline,
}


def render_chart(spec: dict, output_path, theme: dict = None) -> bool:
    """chart_spec を 1920x1080 PNG に描画する。

    成功で True。spec 不正・描画失敗時は False を返し、図は作らない
    （呼び出し側で engine:ai の従来ルートへ降格すること）。
    """
    if not isinstance(spec, dict):
        return False
    theme = {**DEFAULT_THEME, **(theme or {})}
    ctype = spec.get("chart_type", "bar")
    drawer = _DRAWERS.get(ctype, _draw_bar)
    fig = None
    try:
        fig = plt.figure(figsize=(_W_IN, _H_IN), dpi=_DPI)
        fig.patch.set_facecolor(theme["bg"])
        drawer(fig, spec, theme)
        _draw_title_and_source(fig, spec, theme)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=_DPI, facecolor=theme["bg"])
        return True
    except Exception as e:
        print(f"  [renderer ERROR] chart_type={ctype}: {str(e)[:140]}", flush=True)
        return False
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close("all")


# ============================================================
# Map (v3 Step2) — Natural Earth 1:50m + matplotlib（LLM不使用）
# 2026-06-12 安福: AI製「航空写真風」地図の国境デタラメ事故を防ぐため、
# 正確な GeoJSON を同梱して決定論描画する。海色・国塗り分け・ラベル・矢印。
# ============================================================
_GEO_PATH = _ASSETS / "geo" / "ne_50m_admin_0_countries.geojson"
_GEO_CACHE = None  # {iso3: {"name_ja": str, "geom": shapely, "rings": [[(lon,lat)...]]}}

VALID_MAP_TYPES = ("highlight", "route", "neighbors")

# extent プリセット（lon0, lon1, lat0, lat1）。地政学チャンネルで頻出の範囲を固定。
_EXTENTS = {
    "world": (-168.0, 190.0, -58.0, 84.0),
    "europe": (-26.0, 50.0, 33.0, 72.0),
    "asia": (40.0, 150.0, 3.0, 58.0),
    "former_ussr": (18.0, 180.0, 35.0, 82.0),
}


def _geom_to_rings(geom: dict):
    """GeoJSON geometry → 外環座標 [[(lon,lat),...], ...]（穴は無視）。"""
    t = geom.get("type")
    c = geom.get("coordinates")
    rings = []
    try:
        if t == "Polygon" and c:
            rings.append([(float(x), float(y)) for x, y in c[0]])
        elif t == "MultiPolygon" and c:
            for poly in c:
                if poly:
                    rings.append([(float(x), float(y)) for x, y in poly[0]])
    except Exception:
        return []
    return rings


def _load_geo():
    """同梱 GeoJSON を読み込み iso3 で索引化（初回のみ。shapely で重心算出用）。"""
    global _GEO_CACHE
    if _GEO_CACHE is not None:
        return _GEO_CACHE
    cache = {}
    try:
        import json as _json
        from shapely.geometry import shape
        data = _json.loads(_GEO_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [renderer map] GeoJSON 読込失敗: {str(e)[:100]}", flush=True)
        _GEO_CACHE = {}
        return _GEO_CACHE
    for f in data.get("features", []):
        p = f.get("properties", {}) or {}
        iso = p.get("ISO_A3")
        if not iso or iso == "-99":
            iso = p.get("ADM0_A3") or p.get("ISO_A3_EH")
        if not iso or iso == "-99":
            continue
        geom = f.get("geometry")
        if not geom:
            continue
        rings = _geom_to_rings(geom)
        if not rings:
            continue
        try:
            sgeom = shape(geom)
        except Exception:
            sgeom = None
        cache[iso] = {"name_ja": p.get("NAME_JA") or p.get("NAME") or iso,
                      "geom": sgeom, "rings": rings}
    _GEO_CACHE = cache
    return cache


def _rep_point(info, extent=None):
    """ラベル/矢印用の代表点（内点）。extent を渡すと、表示範囲に交差した
    可視部分の中で代表点を取る（ロシア等の大国でも画面内に配置される）。"""
    g = info.get("geom")
    try:
        if g is None:
            raise ValueError
        if extent is not None:
            from shapely.geometry import box
            x0, x1, y0, y1 = extent
            clipped = g.intersection(box(x0, y0, x1, y1))
            if (not clipped.is_empty) and clipped.area > 0:
                g = clipped
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        pt = g.representative_point()
        return (pt.x, pt.y)
    except Exception:
        ring = max(info["rings"], key=len)
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


def _lighten(hex_color: str, amt: float = 0.55) -> str:
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r + (255 - r) * amt)
        g = int(g + (255 - g) * amt)
        b = int(b + (255 - b) * amt)
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return hex_color


def _resolve_extent(spec: dict, geo: dict, countries: list):
    ext = (spec.get("extent") or "").strip()
    if ext in _EXTENTS:
        return _EXTENTS[ext]
    # custom / 未知 → focus+secondary の bbox から自動算出（余白付き）
    xs0, xs1, ys0, ys1 = 180, -180, 90, -90
    for iso in countries:
        info = geo.get(iso)
        if not info:
            continue
        for ring in info["rings"]:
            for x, y in ring:
                xs0, xs1 = min(xs0, x), max(xs1, x)
                ys0, ys1 = min(ys0, y), max(ys1, y)
    if xs0 > xs1:
        return _EXTENTS["world"]
    padx = max(4.0, (xs1 - xs0) * 0.25)
    pady = max(4.0, (ys1 - ys0) * 0.25)
    return (xs0 - padx, xs1 + padx, ys0 - pady, ys1 + pady)


def render_map(spec: dict, output_path, theme: dict = None) -> bool:
    """map_spec を 1920x1080 PNG に描画する。

    成功で True。解決できる国が無い/描画失敗時は False（呼び出し側で
    route を illustration(engine:ai) へ降格すること）。
    """
    if not isinstance(spec, dict):
        return False
    theme = {**DEFAULT_THEME, **(theme or {})}
    geo = _load_geo()
    if not geo:
        return False
    focus = [c for c in (spec.get("focus_countries") or []) if c in geo]
    if not focus:
        return False  # 国コードが解決できない → 降格
    secondary = [c for c in (spec.get("secondary_countries") or []) if c in geo and c not in focus]
    mtype = spec.get("map_type", "highlight")
    fig = None
    try:
        fig = plt.figure(figsize=(_W_IN, _H_IN), dpi=_DPI)
        fig.patch.set_facecolor(theme["bg"])
        ax = fig.add_axes([0.01, 0.03, 0.98, 0.84])
        ax.set_facecolor("#DCEAF7")  # 海
        ax.axis("off")
        x0, x1, y0, y1 = _resolve_extent(spec, geo, focus + secondary)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        # 簡易メルカトル風: 緯度方向の歪みを抑える
        mean_lat = max(-80, min(80, (y0 + y1) / 2))
        ax.set_aspect(1.0 / max(0.2, math.cos(math.radians(mean_lat))))
        # 全ての国を描画（その他=グレー / focus=アクセント / secondary=薄アクセント）
        light = _lighten(theme["accent"])
        for iso, info in geo.items():
            if iso in focus:
                color = theme["accent"]
            elif iso in secondary:
                color = light
            else:
                color = "#D7DBE0"
            for ring in info["rings"]:
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                ax.fill(xs, ys, facecolor=color, edgecolor="white", linewidth=0.4, zorder=2)
        # ラベル（spec.labels 指定があればそれを、無ければ focus に自動）
        import matplotlib.patheffects as pe
        stroke = [pe.withStroke(linewidth=4, foreground="white")]
        label_targets = []
        if spec.get("labels"):
            for lab in spec["labels"]:
                iso = lab.get("country")
                txt = lab.get("text")
                if iso in geo and txt:
                    label_targets.append((iso, txt))
        else:
            for iso in focus:
                label_targets.append((iso, geo[iso]["name_ja"]))
        for iso, txt in label_targets:
            px, py = _rep_point(geo[iso], extent=(x0, x1, y0, y1))
            if x0 <= px <= x1 and y0 <= py <= y1:
                ax.text(px, py, txt, ha="center", va="center", fontsize=34,
                        fontweight="bold", color=theme["text"], zorder=6,
                        path_effects=stroke)
        # 矢印（route 型）
        if mtype == "route":
            for ar in (spec.get("arrows") or []):
                a, b = ar.get("from"), ar.get("to")
                if a in geo and b in geo:
                    ax_, ay_ = _rep_point(geo[a], extent=(x0, x1, y0, y1))
                    bx_, by_ = _rep_point(geo[b], extent=(x0, x1, y0, y1))
                    ax.annotate("", xy=(bx_, by_), xytext=(ax_, ay_),
                                arrowprops=dict(arrowstyle="-|>", color=theme["main"],
                                                lw=5, connectionstyle="arc3,rad=0.2",
                                                shrinkA=8, shrinkB=8), zorder=7)
                    if ar.get("label"):
                        ax.text((ax_ + bx_) / 2, (ay_ + by_) / 2 + 1.5, ar["label"],
                                ha="center", va="center", fontsize=26,
                                fontweight="bold", color=theme["main"], zorder=8,
                                path_effects=stroke)
        _draw_title_and_source(fig, spec, theme)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=_DPI, facecolor=theme["bg"])
        return True
    except Exception as e:
        print(f"  [renderer map ERROR] {str(e)[:140]}", flush=True)
        return False
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close("all")
