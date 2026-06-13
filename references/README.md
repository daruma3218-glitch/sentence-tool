# 参照画像（キャラ固定用）

ロシア解体新書チャンネルの「繰り返し登場するキャラ（ナレーター）」を毎回同じ見た目で描くための参照画像を置きます。

## 使い方
1. キャラのデザインを 1 枚の PNG にして、このフォルダに **`roshia_character.png`** という名前で置く
   - 推奨: 顔・髪型・服装・色がはっきり分かる、正面〜やや斜めの 1 人立ち絵（背景はシンプル）
   - サイズの目安: 長辺 1024px 以上
2. コミットしてデプロイ（Render は `references/roshia_character.png` を読み込む）

## 動作
- `channels.json` の `roshia` チャンネルに `"character_ref": "references/roshia_character.png"` が設定済み。
- この PNG があると、ナレーター等のキャラ登場シーンで**その絵柄・人物に固定**して生成します（provider=nanobanana のとき画像参照、その他は文言で統一）。
- **PNG が無い場合**でも壊れません。worldview の文章設定だけでキャラの雰囲気を統一します（画像ロックは無し）。

> 経済探求ラボの `keizai_professor.png` と同じ仕組みです。
