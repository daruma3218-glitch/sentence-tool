#!/usr/bin/env python3
"""generator.add_image_caption（v3 Step5b: realphoto「イメージ」焼き込み）の pytest。

受け入れ基準:
- 焼き込み後も画像が開け、サイズが変わらない
- 右下領域に半透明の暗いキャプションが合成されている（元の単色から変化）
- 左上領域は変化しない（焼き込みは右下だけ）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from generator import add_image_caption  # noqa: E402


def test_caption_burns_only_bottom_right(tmp_path):
    W, H = 640, 360
    p = tmp_path / "shot.png"
    Image.new("RGB", (W, H), (255, 255, 255)).save(p)

    ok = add_image_caption(str(p), text="イメージ")
    assert ok is True

    im = Image.open(p).convert("RGB")
    assert im.size == (W, H), "焼き込みで画像サイズが変わってはいけない"
    px = im.load()

    # 右下クアドラントに暗いピクセル（半透明黒の帯）が存在する
    dark = 0
    for x in range(W // 2, W):
        for y in range(H // 2, H):
            r, g, b = px[x, y]
            if r < 150 and g < 150 and b < 150:
                dark += 1
    assert dark > 100, f"右下に焼き込みが見当たらない（dark={dark}）"

    # 左上は白のまま（焼き込みは右下だけ）
    assert px[10, 10] == (255, 255, 255), "左上が変化してはいけない"


def test_caption_missing_file_returns_false(tmp_path):
    # 存在しないファイルでも例外を投げず False を返す（安全側）
    assert add_image_caption(str(tmp_path / "nope.png")) is False
