# センテンスつくーる v3 設計書 ― 決定論レンダリング + ビート演出 + 権利処理

最終更新: 2026-06-12
ステータス: 設計フェーズ(実装前)
前提バージョン: v2(ルーター + 専門エージェント)実装済み
対象: `経済探求ラボ/センテンスつくーる/`

この設計書は `DESIGN_v2.md` の続編であり、そのまま Claude Code に渡して実装させる改修マニュアルを兼ねる。
**「11. 実装ステップ」の順に 1 ステップ = 1 ブランチ = 1 PR で進めること。**

---

## 0. 目的(なぜ v3 か)

v2 はルーターで「文ごとに最適なソース」を選べるようになった。
v3 は **「AI 生成に頼る範囲を絞り、保証できるものはコードで保証する」** 段階へ進む。

設計原則: **AI へのお願いは最小限にし、正確性・一貫性・権利は決定論的な仕組みで保証する。**

### v3 で解決したい課題

| 課題(v2) | v3 での解決 |
|---|---|
| AI 製グラフの数値が不正確・日本語が文字化け → verifier で検品再生成 | chart は matplotlib で決定論レンダリング。数値の狂い・文字化けが構造的にゼロに |
| AI 製「航空写真風」地図の国境線がデタラメ | map は同梱 GeoJSON + matplotlib で正確描画 |
| 1 文 = 1 画像の独立判定で似た画像が連発 | ビート(シーン)単位の演出設計 |
| Web 画像のライセンスが不明 | Wikimedia Commons API 限定 + ライセンス・クレジット自動記録 |
| max_diagrams の均等配置が機械的 | 重要度加重配分 + 推定タイムコード |
| 繰り返し登場する被写体が毎回違う見た目 | エンティティ参照画像のロック(character_ref 方式の一般化) |
| realphoto が報道映像と誤認されうる | 「イメージ」キャプションの自動焼き込み |
| 編集者の差し戻しが学習されない | フィードバック蓄積 → ルーター few-shot 注入 |

---

## 1. 用語の整理(v3 で増える軸)

v2 の 2 軸(route / style_preset)に、v3 は **engine(作り方)** と **beat(演出単位)** を加える。

- **route** = どんな種類のビジュアルか(v2 と同じ。文/ビートごと)
- **engine** = その画像をどう作るか(v3 新設。route から自動決定)
  - `ai` … 画像生成モデル(従来通り)
  - `render` … コードで描画(matplotlib 等)。chart / map の既定
  - `commons` … Wikimedia Commons から取得。web_photo の既定
  - `none` … skip
- **beat** = 連続する文の視覚的まとまり(v3 新設)。画像はビートに対して付く
- **style_preset / worldview** = v2 と同じ(engine: ai にのみ作用。render はチャンネルのチャートテーマが作用)

---

## 2. 全体アーキテクチャ(v3)

```
原稿テキスト ──(または)── 原稿パイプライン final.json ★v3: 直結対応
        │
【Phase 1】 splitter.py(変更なし。final.json 入力時は章分解をスキップ)
        ▼
【Phase 2】 router.py v3 ★改修
        │  文ごとに: route + importance(1-5) + entities[] + beat グルーピング
        ▼
【Phase 2.5】 allocator.py ★新規(LLM不使用)
        │  重要度加重で生成予算を配分 / 推定タイムコード付与 / beat 確定
        ▼
   ┌─────────────┬──────────────┬─────────────────────┐
   │ engine:commons│ engine:render │ engine:ai            │
   │ web_photo     │ chart / map   │ illustration/realphoto│
   │               │               │ /diagram             │
【Phase 3】 並行実行(threading は v2 踏襲)
commons_searcher  renderer.py     prompter → generator
.py ★新規       ★新規(LLM     (v2 のまま。entity_ref
ライセンス取得+   不使用。spec    注入 ★改修、realphoto
クレジット記録    抽出のみClaude)  キャプション ★改修)
   └───────────────┴───────┬───────┘
                           ▼
              【Phase 3b】 verifier.py(対象は engine:ai の diagram のみに縮小)
                           ▼
【Phase 4】 表組み立て + CSV + credits.txt ★改修
```

---

## 3. router.py v3 詳細仕様

### 3.1 出力スキーマの拡張

v2 の出力に 3 フィールドを追加(チャンク方式・CHUNK_SIZE=40・フォールバック illustration は v2 踏襲):

```json
{
  "no": 12,
  "route": "chart",
  "reason": "GDP比の数値が主役",
  "importance": 4,
  "entities": ["ロシア", "軍事費"],
  "beat": "continue"
}
```

- **importance**: 1〜5。動画の主張にとっての重要度(画像予算の配分に使う)。5=章の核心主張・驚きのデータ / 3=主張を支える説明 / 1=繋ぎに近い
- **entities**: 文に登場する「繰り返し描かれる可能性のある被写体」。最大 3 つ。一貫性管理(§7)に使う
- **beat**: `new`(新しい視覚的まとまり) / `continue`(直前の文と同じまとまり)。話題・被写体が直前から切り替わったら new。チャンク境界をまたぐ場合、各チャンクの先頭文は new 固定(シンプル優先)

### 3.2 chart / map の spec 抽出(第 2 段呼び出し)

route 分類とは**別の小バッチ呼び出し**として、route=chart の文だけを集めて構造化データを抽出する(分類と抽出を 1 回に混ぜない)。

**chart_spec**(renderer.py への入力):
```json
{
  "no": 12,
  "chart_type": "bar",
  "title": "軍事費の対GDP比",
  "series": [{"label": "ロシア", "value": 6.3}, {"label": "NATO平均", "value": 2.1}],
  "unit": "%",
  "highlight_index": 0,
  "source_note": "SIPRI 2025"
}
```
抽出ルール:
- 数値は文と block_context に書かれているものだけ。推測・補完・創作は絶対禁止
- 比較対象が 1 つしかない → `big_number` 型に倒す
- 抽出不能 → `chart_spec: null` を返し、その文の route を diagram(engine: ai)へ降格

**map_spec**:
```json
{
  "no": 7,
  "map_type": "highlight",
  "focus_countries": ["RUS"],
  "secondary_countries": ["UKR", "BLR"],
  "labels": [{"text": "ロシア", "country": "RUS"}],
  "arrows": [{"from": "RUS", "to": "DEU", "label": "ガス輸出"}],
  "extent": "europe"
}
```
国コードに変換できない地名が主役 → `map_spec: null` → route を illustration へ降格(v3.0 は国レベルのみ)。

---

## 4. renderer.py 詳細仕様(★新規・LLM 不使用)

### 4.1 共通
- 依存: matplotlib + shapely + 同梱データのみ。GDAL / geopandas は使わない
- 日本語フォント: `assets/fonts/NotoSansJP-Regular.otf` / `-Bold.otf` を同梱し `font_manager.fontManager.addfont()` で登録
- 出力: 1920×1080(16:9)PNG。レターボックス不要
- チャンネル別テーマ: channels.json の defaults に `chart_theme`(§9)

### 4.2 chart_renderer
- chart_type ごとの描画関数。タイトル上部太字 / 単位併記 / source_note は右下に「出典: ○○」
- `big_number` 型: 中央に巨大数字(200pt級)+ 単位 + ラベル
- 値ラベルは必ずバー/点の近くに数値表示(軸を読ませない)
- 禁止: 3D・影・虹色・凡例だけで判別

### 4.3 map_renderer
- Natural Earth 1:50m countries GeoJSON を `assets/geo/` に同梱(パブリックドメイン)
- extent プリセットで範囲固定(europe / former_ussr 等は座標ハードコード)
- focus を塗り / secondary 薄塗り / 他グレー。labels は代表点、arrows は曲線矢印
- 係争地は Natural Earth 準拠。`assets/geo/README.md` で人間判断に委ねる

### 4.4 フォールバック
renderer が例外 → その文は engine: ai へ自動降格。降格はログと CSV の engine 列で可視化。

---

## 5. commons_searcher.py 詳細仕様(★新規)
- API: `commons.wikimedia.org/w/api.php` の `generator=search` + `prop=imageinfo&iiprop=url|extmetadata`
- 検索: router の search_query。日本語ヒットなし→英訳クエリ再検索(Claude 小呼び出し・バッチ可)
- 採用条件: LicenseShortName が許可リスト(Public domain / CC0 / CC BY 系 / CC BY-SA 系)に一致のみ
- 0 件: v2 同様 AI イラスト保険(web_fallback)
- row 追加: license, license_url, attribution, commons_page_url
- `credits.txt` 出力(概要欄用・CC BY 系のクレジット義務に対応)
- 既存 web_searcher.py は残す(`photo_source: "web" | "commons"` で切替。既定 commons)

---

## 6. allocator.py 詳細仕様(★新規・LLM 不使用)
- ビート確定: beat: new/continue → beat_id 採番(skip 文は属さない)
- タイムコード: 秒数 = len(sentence)/chars_per_sec(既定 5.5)。累積で est_start(mm:ss)
- 画像予算の加重配分(max_diagrams 上限・ビート単位):
  - ビートスコア = ビート内 importance の最大値
  - 4-5: 必ず 1 枚(≥25秒は progressive で 2-3 枚)
  - 3: 予算が許す限り 1 枚 / 1-2: 余れば
  - 同一ビート内は 1 枚目を代表文に紐付け、残りは `display: hold`
- engine 決定: route→engine(chart/map→render, web_photo→commons, 他→ai)。spec=null は §3.2 降格
- 出力: `allocation.json`(rows に beat_id / est_start / display / engine 追記)

---

## 7. エンティティ参照(一貫性ロック)詳細仕様
- 対象: engine: ai の画像で、entities に同一エンティティが 3 回以上登場するもの
- 動作: allocator が初出画像(canonical)を決定 → canonical 生成完了後に後続を投入し、canonical PNG を nanobanana に参照画像として渡す(character_ref 配管流用)+「maintain the same visual design as the reference image」追記
- provider が gpt-image の場合 v3.0 は参照注入スキップ(文言のみ)
- 並行性: 直列化はエンティティ内のみ。PER_IMAGE_HARD_TIMEOUT 等の既存防衛を壊さない

---

## 8. realphoto ガードレールと verifier 縮小
- realphoto キャプション: engine: ai かつ route: realphoto は生成後 PIL で右下に半透明「イメージ」合成(`realphoto_watermark: true` 既定 ON)
- verifier 縮小: chart/map が renderer 化 → Vision 検品対象は engine: ai の diagram のみ(DEFAULT_VERIFY_TYPES 変更)。モデルは `claude-haiku-4-5` に変更。判定失敗 ok=True の安全設計は維持
- フィードバック: progress.html 各行に「ルート違い」ボタン → `POST /api/feedback/<job_id>/<no>` → `output/route_feedback.jsonl` に追記。router 起動時に同 channel_id の直近 12 件を few-shot 注入

---

## 9. 設定・データモデルの変更

### channels.json defaults 追加キー
```json
{
  "chart_engine": "render",
  "map_engine": "render",
  "photo_source": "commons",
  "beat_mode": true,
  "chars_per_sec": 5.5,
  "realphoto_watermark": true,
  "chart_theme": {
    "bg": "#FFFDF7", "main": "#1E40AF", "accent": "#C2410C",
    "grid": "#E5E7EB", "text": "#1F2937", "font": "NotoSansJP"
  }
}
```

### row 最終形(追加分)
```json
{
  "importance": 4, "entities": ["ロシア"],
  "beat_id": 3, "display": "image", "est_start": "02:34",
  "engine": "render", "chart_spec": {}, "map_spec": {},
  "license": "CC BY-SA 4.0", "attribution": "...", "commons_page_url": "..."
}
```

### CSV 列(v3)
章, ブロック, センテンス, №, ビート, 推定開始, ソース(route), エンジン, 重要度, 表示(画像/継続/なし), 画像ファイル, URL, ライセンス, クレジット

### 新規成果物
- `credits.txt`(Commons クレジット一覧。概要欄用)
- `allocation.json`(配分の監査用)

---

## 10. 原稿パイプライン直結(final.json 入力)
- upload.html に「final.json アップロード」欄追加(テキスト貼り付けと排他)
- `final` キーを本文として使用(章タイトル行で splitter 安定)
- `reference_list` / `fact_report` があれば chart_spec 抽出に「検証済み数値・出典」として渡し source_note 精度向上。commons 検索起点にも活用
- 存在しないキーはすべて任意扱い(原稿側の改修進度に依存しない)

---

## 11. 実装ステップ(段階リリース・各ステップで v2 動作を壊さない)

| Step | ブランチ | 内容 | 受け入れ基準 |
|---|---|---|---|
| 1 | feat/v3-renderer-chart | renderer.py chart 部 + フォント同梱 + chart_spec 抽出 + chart_engine 設定 | サンプル spec 10種が文字化けなく 16:9 PNG になる pytest。spec=null で ai 降格のテスト |
| 2 | feat/v3-renderer-map | map 部 + Natural Earth GeoJSON 同梱 + map_spec 抽出 | highlight/route/neighbors の 3 型描画。未知国コードで ai 降格 |
| 3 | feat/v3-commons | commons_searcher.py + credits.txt + photo_source 設定 | 許可リスト外不採用テスト(API モック)。credits.txt 生成 |
| 4 | feat/v3-router-allocator | router 拡張(importance/entities/beat) + allocator.py + CSV/progress 列 | 250文で beat_id・est_start・display 全行付与。配分が max_diagrams 超えない。beat_mode=false で v2 同一動作 |
| 5 | feat/v3-guardrails | realphoto 焼き込み + verifier 縮小(Haiku化) + feedback API | watermark 合成のユニットテスト。feedback POST→jsonl→few-shot 注入の結合テスト |
| 6 | feat/v3-entity-ref | エンティティ参照ロック(nanobanana のみ) | canonical→後続の依存順生成。タイムアウト不等式維持 |
| 7 | feat/v3-finaljson | final.json 入力対応 | final.json / 生テキスト両入力で完走 |

### 実装上の注意(全 Step 共通)
- 新規依存は matplotlib / shapely のみ。requirements.txt の **httpcore 直接 URL 指定(Render ビルド対策)を壊さない**
- 新規モジュールは `encoding="utf-8"` 明示、パスは pathlib
- `route_mode=all_ai` の v1 互換、各 engine 設定を ai に戻せば v2 完全互換、を最後まで維持
- 既存コメントの流儀(日付 + 指示者 + 理由)を踏襲

---

## 12. コスト・品質の見積もり(250文・chart 30・map 15 の例)

| 項目 | v2 | v3(推定) |
|---|---|---|
| AI 画像生成数 | 〜100 枚 | 〜55 枚 |
| chart の数値正確性 | verifier 頼み | 100% |
| 日本語文字化け | 高頻度 | chart/map でゼロ |
| verifier 呼び出し | diagram+chart 全数(Sonnet) | diagram のみ(Haiku)→ 1/5 以下 |
| 権利リスク | ライセンス不明 | 許可リスト制 + クレジット自動生成 |
| 編集者の手作業 | 目視判断 | est_start とビート列で半自動配置 |

---

## 13. リスクと対策

| リスク | 対策 |
|---|---|
| chart_spec 抽出で数値創作 | 「文と block_context の数値のみ・無ければ null」厳命 + 抽出後にコードで原文照合(部分一致しなければ降格) |
| renderer デザインが worldview と乖離 | chart_theme をチャンネル別定義。初回 keizai テーマを人間レビュー |
| Natural Earth の国境表現 | §4.3 README で人間判断に委ねる。差し替え可能 |
| Commons ヒット率が低い | 英訳再検索 + web_fallback 維持 |
| ビート誤判定で重要文に画像なし | importance 4-5 は必ず 1 枚 + progress で beat 列可視化 + regenerate |
| 直列化(entity canonical 待ち)遅延 | エンティティ内のみ直列。「3 回以上」に限定 |

---

## 14. 将来拡張(v4 候補)
- 都市・地形レベルの地図
- gpt-image 系へのエンティティ参照対応
- progressive ビートの差分指定生成
- est_start を使った Premiere/Resolve マーカー(XML/EDL)出力
- route_feedback.jsonl の定期分析 → ルーター判定基準の自動改訂提案
