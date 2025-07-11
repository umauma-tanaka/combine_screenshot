<!-- ====================== AGENTS.md ====================== -->
# 👩‍💻 Codex 実装ガイド

> *このファイルは Codex（または他の LLM コード生成エージェント）に対し、プロジェクトの仕様を一意に示すためのものです。*

---

## 1. ゴール概要

複数のスクリーンショットが保存された **ディレクトリ** を入力として、  
**ゲームウィンドウ内のスクロール画像を 1 枚に結合**する **Python 3.9+** パッケージ `stitcher` を作成する。

---

## 2. 入出力仕様

| 項目                | 型 / フラグ      | 説明 |
| ------------------- | ---------------- | ---- |
| `<input_dir>`       | `Path` (必須)    | スクリーンショットが格納されたディレクトリ |
| `--output, -o`      | `Path`           | 出力画像（PNG 推奨）のファイルパス |
| `--roi`             | `x,y,w,h`        | *任意* ウィンドウ領域（座標指定） |
| `--threshold`       | `float`          | *任意* 類似度閾値（デフォルト 0.8） |
| **戻り値 (API)**    | `Path`           | 正常終了時の結合画像パス |
| **エラー (API)**    | `StitchError`    | 信頼できる重複が検出できない場合に送出 |

CLI 終了コード:

```
0   正常終了
10  重複領域が検出できなかった
11  入力ディレクトリ / 画像が読み込めない
12  画像の向きが解決できなかった
```

---

## 3. アーキテクチャ

```
Stitcher
 ├─ detect_window()   # ROI 検出・検証
 ├─ find_overlap()    # ペアごとの平行移動推定
 ├─ solve_layout()    # 全体配置の最適化
 └─ compose()         # キャンバス生成とブレンド
```

`core.py` に主要ロジックを集約し、`cli.py` は薄く保つこと。

---

## 4. 使用ライブラリと制約

* **必須**: `opencv-python`, `numpy`, `Pillow`
* **禁止**: 重量級 DL ライブラリ（TensorFlow, PyTorch など）
* **型ヒント**を全関数に付与
* **PEP 8 / PEP 257** 準拠

---

## 5. アルゴリズム詳細

1. **ROI 決定**  
   - `--roi` 指定があればそれを使用  
   - 未指定の場合、境界の色均一性と内部エッジ密度から最大矩形を推定
2. **重複検出**  
   - ROI 中央 60 % 帯域で `cv2.matchTemplate` を実施  
   - 信頼度低い場合は ORB 特徴点 + BF マッチャで再評価し、RANSAC で外れ値除去  
   - インライア率 > 0.6 かつ平均誤差 < 2px のみ採用
3. **レイアウト最適化**  
   - ペアごとのオフセットで有向グラフを構築  
   - 最小二乗法（`scipy` が望ましいが未導入でも可）で絶対位置決定
4. **合成**  
   - RGBA キャンバスにペースト。初回ペーストは α=1、重複領域は線形ブレンド
5. **失敗条件**  
   - 有効な重複が 2 未満 → `StitchError`  
   - 矛盾する変換（閉路誤差） → `StitchError`

---

## 6. 品質ゲート

| 項目 | ツール / 基準 |
| ---- | ------------- |
| CI   | GitHub Actions（Ubuntu, macOS, Windows） |
| Lint | `ruff` strict |
| Format | `black` (line length 88) |
| Docs | MkDocs ビルド、`README.md` がトップページ |

---

## 7. サンプル実行（Codex テスト向け）

```bash
python -m stitcher tests/assets -o tests/output/stitched.png
```

ユニットテスト例:

```python
from stitcher import Stitcher, StitchError
out = Stitcher().stitch("tests/assets")
assert out.exists() and out.suffix == ".png"
```

---

## 8. 非目標

* パノラマ（屋外風景）ステッチ  
* 透視変換・円筒投影などの高次変換  
* 動画入力やリアルタイム処理  

---

## 9. ロードマップ（将来計画）

| マイルストーン | 内容 |
| -------------- | ---- |
| v0.2 | PySide6 によるドラッグ＆ドロップ GUI |
| v0.3 | OS ネイティブ API を使った画面キャプチャ |
| v1.0 | 複雑な重複領域に対するシームカット |

---

*End of AGENTS.md*
