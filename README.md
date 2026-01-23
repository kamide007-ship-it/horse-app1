# horse-app v1.5.7fix (download)

## 画像「成長予測」(崩れない + ③寄せハイブリッド)

この版では、成長予測画像を **必ず破綻しないベース処理（拡大縮小 + 中央クロップ）** で作ったあと、
OpenAIの **画像Edit（弱め固定・タイル無し）** を1回だけ掛けることで、
「③のような自然な成長・筋肉感」に寄せます。

### 必須/任意 環境変数

- `OPENAI_API_KEY` : 設定されている場合のみ、画像Editを実行します（無い場合は自動スキップ）。
- `GPT_IMAGE_MODEL` : 画像Editモデル（既定: `gpt-image-1.5`）
- `GROWTH_USE_OPENAI` : `0` で画像Editを完全に無効化（既定: 自動）
- `GROWTH_IMG_INPUT_FIDELITY` : 既定 `high`

※画像Editを使わない環境（APIキーなし）でも動作は継続します。
