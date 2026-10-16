# 2026-10-16 CPU修正検証

適用元: `596abc87f4db983aaceea19b437119367f45eb23`。この記録は今回の修正差分のCPU検査であり、GPU推論・実機ComfyUIの動作・画質改善・実測VRAMを証明しません。過去のログと画像は変更していません。

## 実施結果

- 修正前: 68 tests / 5 subtests 成功。
- 修正後: **167 tests / 43 subtests 成功**。実装担当と独立レビューに加え、公開前の別実行でも確認しました。
- compileall、git diffの空白検査: 成功。
- 実機計画内のworkflow生成例: オフラインで7条件のJSONとnode参照を確認。サーバーへは送信していません。
- Python 3.12.14、PyTorch 2.10.0+cpu、pytest 9.1.1、NumPy 2.5.4、Linux CPU環境。
- ComfyUI `15ef24d1c0333a3eba56c5cd153d8db65363ff8f` の取得済みソースを用いてwrapper契約を確認。AST抽出等のCPU検査であり、このComfyUIサーバーを起動した記録ではありません。

```bash
export AFM_COMFYUI_SOURCE_ROOT=/path/to/pinned/ComfyUI-source
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pytest tests -q
python -m compileall -q anima_afm.py sampling_lifecycle.py nodes.py scripts tests
git diff --check
```

## 修正した範囲

- autoの平方数推測を停止し、post_inputの実グリッドを利用。明示寸法とruntimeの矛盾も検出。
- sampling単位の状態・run IDとmodel評価単位のcall番号、終了・失敗・中断の集計。
- 診断対象をQK計算より先に選択し、編集と診断のメモリ見積もり・guardを整備。見積もりは総VRAMの上限保証ではありません。
- 非有限値、originalの二重呼び出し、scopeの過剰選択、等しいQ/K長の真のcross-attention、active autocast下のFP32処理を検査。
- 比較データの来歴、終了状態、欠損・重複、分岐保持の最大絶対差を検査。未定義のcentered rhoは有効なゼロ値として集計しません。

## 残る実機確認と制約

新しいwrapper/post_input APIに対応しないComfyUIでは明示的に停止します。初回はEuler/simpleで検証し、他sampler・非単調sigma・split CFG・他量子化/backendへの一般化はしません。split CFGでの正当な`branch_not_selected`も、現時点のstrict解析は証拠照合未対応として拒否します。

GPUメモリ、速度、描画結果、cached MODEL再利用、実際のhook接続、他ノードとの併用は[実機検証計画](hardware-validation-plan-ja.md)に従って後日確認します。raw rhoの新規測定やKrea2向け総注意量保存は追加していません。
