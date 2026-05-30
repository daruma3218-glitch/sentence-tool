# センテンスつくーる v2 設計書 ― ルーター + 専門エージェント

最終更新: 2026-05-29
ステータス: 設計フェーズ（実装前）

---

## 0. 目的（なぜ v2 か）

現状（v1）は **全センテンス → 単一の AI 画像生成パイプライン**。
これを **「ルーターが各文に最適なソースを振り分け、複数の専門エージェントが並行で表を埋める」** 方式へ進化させる。

### v2 で解決したい課題

| 課題（v1） | v2 での解決 |
|---|---|
| 歴史人物が AI 生成だと「似た別人」になる | リアル写真は **Web 検索で本物の URL** を取得 |
| 全文に AI 生成 → コスト・時間が無駄 | ルーターが「不要な文」を仕分け、生成数を削減 |
| 文の性質に関わらず同じ処理 | 文ごとに **最適なビジュアル種別** を選択 |
| スタイルが単調 | スタイルプリセット（プロパガンダ等）を全AI画像に適用 |

---

## 1. 用語の整理（重要：2 軸で考える）

v2 は **「ルート（ソース）」** と **「スタイル」** の 2 軸を分離する。

- **ルート（route）= どこから画像を得るか**（文ごとに決まる）
  - `web_photo` / `map` / `diagram` / `chart` / `illustration` / `skip`
- **スタイル（style_preset）= AI 画像の見た目**（ジョブ全体で 1 つ選ぶ、直交）
  - `flat_infographic` / `pictogram` / `comic` / `whiteboard` / `soviet_propaganda`

> 例：「ソ連プロパガンダ風」を選ぶと、`map`/`diagram`/`chart`/`illustration` で
> 生成される画像はすべてプロパガンダ様式になる。`web_photo` は本物写真なので
> スタイルの影響を受けない。

---

## 2. 全体アーキテクチャ

```
┌──────────────────────────────────────────────────────────┐
│ 原稿（完成原稿テキスト）                                    │
└──────────────────────────────────────────────────────────┘
                          │
              【Phase 1】 splitter.py
                          │  Claude が章/ブロック/センテンスに分解
                          ▼
        rows = [{no, chapter, block, sentence, ...}, ...]
                          │
              【Phase 2】 router.py  ★v2 の中核（新規）
                          │  Claude が各文に route を付与
                          ▼
   ┌──────────┬───────────┬──────────┬──────────┬─────────┐
   │ web_photo │ map        │ diagram  │ chart    │ illust  │ skip
   └────┬─────┴─────┬─────┴────┬────┴────┬────┴────┬───┘
        │           │          │         │         │
   【Phase 3】 専門エージェントが並行実行
        │           └──────────┴─────────┴─────────┘
        │                       │
   web_searcher.py        prompter.py → generator.py
   (Web検索班)            (プロンプト生成 → AI画像生成班)
   実URL+サムネ           style_preset を適用して生成
        │                       │
        └───────────┬───────────┘
                    ▼
        【Phase 4】 表組み立て + CSV 出力
        各 row に「画像 or URL」を格納
```

---

## 3. ルーター詳細仕様（`router.py`）

### 3.1 役割
全センテンスを受け取り、各文に最適な `route` を 1 つ付与する。
Claude 1 回（大量時はチャンク分割）で全文を一括分類。

### 3.2 ルート判定基準

| route | 判定条件 | 担当エージェント | 例文 |
|---|---|---|---|
| `web_photo` | 実在の歴史人物・事件・建造物・特定の場所で **本物の写真/絵画**が見たい | Web検索 | 「スターリンは大粛清を断行した」 |
| `map` | 地理的位置・国境・領土・移動経路・地名の関係 | AI生成（航空写真風） | 「ソ連は14か国と国境を接していた」 |
| `diagram` | 概念・仕組み・因果関係・対比・フロー | AI生成（図解） | 「大陸国家と海洋国家の戦略の違い」 |
| `chart` | 数値・統計・割合・推移・比較データ | AI生成（グラフ） | 「軍事費はGDP比6.3%に達した」 |
| `illustration` | 抽象的シーン・比喩・心情・一般描写 | AI生成（イラスト） | 「緊張が静かに高まっていった」 |
| `skip` | 接続詞・挨拶・問いかけ・内容のない繋ぎ | なし（画像生成しない） | 「では、見ていきましょう」 |

### 3.3 入力スキーマ
```json
[
  {"no": 1, "chapter": "序章", "block_context": "...(前後文脈400字)...", "sentence": "スターリンは..."},
  ...
]
```

### 3.4 出力スキーマ
```json
[
  {
    "no": 1,
    "route": "web_photo",
    "reason": "実在の歴史人物（スターリン）の写真が有効",
    "search_query": "ヨシフ・スターリン 1930年代 写真",   // web_photo のみ
    "topic": "スターリンの肖像"                             // web_photo のみ
  },
  {
    "no": 2,
    "route": "map",
    "reason": "ソ連の国境という地理情報"
  },
  ...
]
```

### 3.5 チャンク分割（大量原稿対策）
- 1 リクエスト最大 40 文程度（max_tokens 制約 + Claude の大量返却渋り対策）
- web_searcher.py で実証済みの「チャンク + 除外リスト」方式を流用

### 3.6 フォールバック
- ルーターが失敗/空を返した文 → デフォルト `illustration` 扱い
- 分類不能 → `illustration`

---

## 4. 専門エージェント詳細

### 4.1 Web検索班（`web_searcher.py` を改修）

- **入力**: route == `web_photo` の文（+ router が生成した search_query）
- **処理**: Claude Web Search → Wikipedia 等のソース URL + サムネイル取得
- **出力**: `web_source_url`, `web_thumb_url`, `web_topic`
- **変更点**: 現状は内部で「検索対象センテンス選定」もしているが、
  v2 ではその選定を **router に移譲**。web_searcher は「与えられた文を検索するだけ」に単純化。
- **フォールバック**: Web で 0 件 → 保険として AI イラストを 1 枚生成（後述の設定で ON/OFF）

### 4.2 AI画像生成班（`prompter.py` → `generator.py`）

- **入力**: route ∈ {map, diagram, chart, illustration} の文
- **処理**:
  1. prompter.py が route(=type) と style_preset に基づき英文プロンプト生成（並列バッチ）
  2. generator.py が nanobanana / gpt-image で並列生成（asyncio + Semaphore）
  3. 16:9 レターボックス（実装済み）
- **スタイル適用**: style_preset（propaganda 等）を generator まで伝播（実装済み）

### 4.3 skip
- 画像生成も検索もしない。テーブルには「—」で表示。

---

## 5. データモデル（row の最終形）

各 row は処理を通じて以下のフィールドを蓄積する：

```json
{
  "no": 1,
  "chapter_title": "序章",
  "block_index": 0,
  "sentence_index": 0,
  "block_text": "...",
  "sentence": "スターリンは大粛清を断行した",

  // ルーターが付与
  "route": "web_photo",
  "route_reason": "実在の歴史人物",

  // AI生成班が付与（route が AI 系のとき）
  "prompt": "...英文プロンプト...",
  "allowed_terms": ["スターリン"],
  "filename": "diagram_001.png",     // 生成成功時
  "status": "ok | generating | failed | pending | skipped",

  // Web検索班が付与（route == web_photo のとき）
  "web_source_url": "https://ja.wikipedia.org/wiki/...",
  "web_thumb_url": "https://upload.wikimedia.org/...",
  "web_topic": "スターリンの肖像"
}
```

---

## 6. 処理シーケンス（並行性）

```
時間 →
Phase1 分解        ████
Phase2 ルーティング     ████
Phase3 並行実行              ┌─ Web検索班   ████████████░░  (web_photo 文)
                            └─ AI生成班    ██████████████  (map/diagram/chart/illust 文)
Phase4 表組立+CSV                          ██
```

- Web検索班と AI生成班は **threading で完全並行**（v1 の Web検索並行化を踏襲）
- AI生成班内部は asyncio + Semaphore で同時 N 枚（実装済み）
- 全体の所要時間 ≈ max(Web検索, AI生成) + α

---

## 7. UI / テーブル / CSV の変更

### 7.1 進捗テーブル（progress.html）
列に **「ソース」バッジ** を追加：

| 章 | センテンス | № | ソース | 図解/Web画像 |
|---|---|---|---|---|
| 序章 | スターリンは… | 1 | 🌐 web_photo | [Wikipediaサムネ] |
| 序章 | 14か国と国境… | 2 | 🗺️ map | [AI航空写真] |
| 本論 | 大陸と海洋… | 3 | 📊 diagram | [AI図解] |
| 本論 | GDP比6.3%… | 4 | 📈 chart | [AIグラフ] |
| 結び | では本題に… | 5 | ⊘ skip | — |

### 7.2 CSV 出力
列構成（スプレッドシート貼り付け用）：
```
章, ブロック, センテンス, №, ソース, 画像ファイル, URL, Webトピック
```

---

## 8. ファイル構成と変更点

| ファイル | 変更 | 内容 |
|---|---|---|
| `splitter.py` | 変更なし | 章/ブロック/センテンス分解 |
| **`router.py`** | 🆕 新規 | ルーター Agent（全文を route 分類） |
| `prompter.py` | 改修 | AI 系 route の文のみプロンプト生成 |
| `web_searcher.py` | 改修 | 選定ロジックを router に移譲、検索に専念 |
| `generator.py` | 変更なし | 16:9 レターボックス + style_preset 適用（実装済み） |
| `pipeline.py` | 改修 | router → 2 班並行のオーケストレーション |
| `app.py` | 微修正 | route 関連の集計・状態管理 |
| `templates/progress.html` | 改修 | ソースバッジ列 + 凡例 |
| `templates/upload.html` | 微修正 | （オプション）route の ON/OFF 設定 |
| `DESIGN_v2.md` | 🆕 本書 | 設計書 |

---

## 9. 設定（upload 画面に追加し得るオプション）

| 設定 | 既定 | 説明 |
|---|---|---|
| `route_mode` | `auto` | `auto`=ルーター判定 / `all_ai`=v1互換（全文AI生成） |
| `web_fallback` | ON | web_photo が 0 件なら AI イラストで保険生成 |
| `skip_decorative` | OFF | skip 判定文を完全に省く（既存機能を route と統合） |
| `style_preset` | flat_infographic | AI画像の様式（propaganda 等） |
| `max_diagrams` | 150 | AI生成の上限（全文均等配置・実装済み） |
| `web_image_count` | （廃止予定） | router が自動決定するため不要に |

---

## 10. 実装ステップ（段階リリース）

各ステップで動作確認 → 問題なければ次へ。途中でも v1 は壊さない。

1. **`router.py` 単体実装 + テスト**
   - サンプル原稿で route 分類の精度を目視確認
   - 出力スキーマの安定性を確認
2. **pipeline.py にルーター組み込み（route_mode=auto）**
   - web_photo 班 / AI班の振り分けと並行実行
   - route_mode=all_ai で v1 互換も維持
3. **web_searcher.py をルーター入力対応に改修**
   - 内部選定を削除、router の search_query を使う
4. **progress.html / CSV にソース列追加**
5. **ローカル総合テスト**（複数原稿で精度・速度・コスト確認）
6. **Render デプロイ**

---

## 11. コスト・速度の見積もり（250 文の原稿の例）

| 項目 | v1 | v2（推定） |
|---|---|---|
| Claude 呼び出し | 分解 + プロンプト生成 | + ルーター 1〜数回 |
| AI 画像生成数 | 〜150 枚（均等配置） | 〜100 枚（web_photo/skip を除外） |
| Web 検索数 | 別途指定 | route が自動決定（〜30件） |
| 速度 | — | AI生成減 + 並行で **やや高速化** |
| コスト | — | AI生成減で **やや削減** |

---

## 12. リスクと対策

| リスク | 対策 |
|---|---|
| ルーターの誤分類（重要な文を skip） | reason を出力させ目視可能に。route_mode=all_ai で全生成も選べる |
| Web 検索のヒット率が低い | web_fallback で AI イラスト保険 |
| Claude 呼び出しが 1 段増える | チャンク分割で 1〜数回に抑制、軽微 |
| 既存ユーザーの混乱 | route_mode=auto を既定にしつつ、all_ai で v1 動作を温存 |

---

## 13. 将来拡張（v3 以降の候補）

- **propaganda を route 化**：歴史的・政治的に重い文だけプロパガンダ風、他は通常 → 1 本の動画内でメリハリ
- **地図 API 連携**：OpenStreetMap 静的地図で正確な地図（現状は AI 航空写真風）
- **リアル画像の自動 DL**：著作権フリー素材（Wikimedia Commons の PD/CC 画像）に限定して自動取得
- **複数スタイルの A/B 生成**：1 文を 2 スタイルで生成して見比べ

---

## 付録 A: ソ連プロパガンダ風スタイル仕様（実装済み）

`generator.py` の `SOVIET_PROPAGANDA_STYLE` 定数として実装済み。

- 世界観: 1920-1950年代モスクワ印刷工場、教育省ポスター。Constructivism + Socialist Realism
- 3色厳守: 深紅 `#8B0000-#A6192E` / 純黒 `#1A1A1A` / 肌色オフホワイト `#E8D5B7-#F0E0CC`
- フラット塗り・グラデなし・低視点・対角線・英雄的シルエット・リトグラフ質感
- 参照: Rodchenko / Lissitzky / Mayakovsky / Toidze / Klimashin / Pravda / Krokodil
- 教育文脈: 武器ではなく書物・地球儀・分析装置・建築をシンボルに
- 禁止: 4色以上 / 高彩度赤 / 暴力・武器・ハンマー&鎌・赤い星 / アニメ調 / 笑顔 / カラーコード表示
