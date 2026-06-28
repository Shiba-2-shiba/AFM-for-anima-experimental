# Anima AFM 非正方形対応 タスク

最終更新: 2026-06-28

## Phase 0: 設計と観測

- [x] 元論文PDFを確認し、`H x W` 定式化と `512 x 512` 実験条件を整理する。
- [x] 現在の正方形限定実装を確認する。
- [x] 仕様書を作成する。
- [x] 進捗ファイルを作成する。
- [x] タスク一覧を作成する。
- [x] ComfyUI APIで `mode=edit`, `debug_format=jsonl` の16:9短縮実行を行う。
- [x] 観測ログから `q_shape`, `k_shape`, `spatial_shape`, fallback reason を確認する。
- [x] `transformer_options` にブロック識別メタデータが存在することを確認する。
- [x] `rope_emb` または block wrapper の入力 `x` から空間形状を復元できるか調査する。

## Phase 1: 単体設計

- [x] `infer_spatial_shape` のテストを追加する。
- [x] `explicit_latent` のテストを追加する。
- [x] `explicit_pixels` のdownscale一致テストを追加する。
- [x] `explicit_pixels` のaspect比因数推定テストを追加する。
- [x] 完全平方数だが非正方形指定のケースをテストする。
- [x] 寸法不一致時のフォールバック理由をテストする。
- [x] `radial_low_high_masks` の長方形入力テストを追加する。
- [x] `edit_logits_fft` の長方形入力テストを追加する。
- [x] `sampled_spectral_diagnostics` / `hf_ratio_from_concentration` の長方形入力テストを追加する。

## Phase 2: 実装

- [x] `AFMConfig` に空間形状指定フィールドを追加する。
- [x] `nodes.py` の `AnimaAFMModelPatch` にadvanced入力を追加する。
- [x] 矩形関連advanced入力をoptional化し、既存APIワークフロー互換を維持する。
- [x] `infer_square_spatial_shape` を互換関数として残しつつ、新しい `infer_spatial_shape` を追加する。
- [x] `_prepare_context` で新しい形状推定を使う。
- [x] `AttentionCallContext` に `spatial_shape_source` と `spatial_shape_reason` を追加する。
- [x] fallback reasonを `spatial_shape_missing`, `spatial_shape_mismatch`, `spatial_shape_metadata_invalid` に整理する。
- [x] 複数候補が同程度に近い場合の `spatial_shape_ambiguous` 判定を追加する。
- [x] JSONL/textログへ `spatial_shape_source` を出す。
- [x] JSONLログへ明示寸法入力値を出す。
- [x] `max_logits_mib` / `max_peak_mib` の見積もりが長方形でも積ベースで扱えることをAPIログで確認する。

## Phase 3: ランタイム検証

- [x] `16:9` 横長ワークフローをAPIで生成する。
- [x] `9:16` 縦長ワークフローをAPIで生成する。
- [x] `4:3` を追加で1ケース生成する。
- [x] 各ケースでログの `spatial_shape` が期待値と一致することを確認する。
- [x] 各ケースで空間形状由来のフォールバックが対象callに出ていないことを確認する。
- [x] 既存正方形ワークフローでbaseline生成が成功することを確認する。
- [x] 既存正方形ワークフローでAFM edit生成が成功することを確認する。
- [x] 寸法をわざと間違えたAPI実行で安全にフォールバックすることを確認する。
- [x] ComfyUI再起動後、矩形関連入力が `/object_info` でoptional扱いになることを確認する。
- [x] 新規矩形入力を省いた古いAPIワークフローが動くことを確認する。

## Phase 4: 比較画像とドキュメント

- [x] 16:9 baseline / AFM の比較画像を作る。
- [x] 9:16 baseline / AFM の比較画像を作る。
- [x] 4:3 baseline / AFM の比較画像を作る。
- [x] 比較画像manifestを作る。
- [x] 既存記事 `docs/anima-afm-article-ja.md` に非正方形対応後の説明を反映する。
- [x] READMEのLimitationsを更新する。
- [x] 仕様書の実装状況を実装結果で更新する。
- [x] 進捗ファイルに検証ログと生成画像パスを追記する。

## Phase 5: 完了判定

- [x] `python -m pytest` が通る。
- [x] ComfyUI APIで正方形・横長・縦長の生成が通る。
- [x] 新しいログ項目で空間形状の由来を説明できる。
- [x] 明示寸法なしの場合も既存正方形ワークフローは従来通り動く。
- [x] 明示寸法ありの非正方形では誤った正方形推定を行わない。
- [x] 記事、README、仕様書、進捗、タスクが最新状態になっている。

## Phase 6: `auto` 強化と候補曖昧性

- [x] runtime metadata候補を複数集め、一意なら採用する。
- [x] 有効候補が複数あり互いに異なる場合、`spatial_shape_ambiguous` でフォールバックする。
- [x] 4D block input `x` から `afm_spatial_shape_candidates` を渡す。
- [x] dict形式の `rope_emb` からshape候補を渡す。
- [x] 候補注入後に `transformer_options` を元に戻す。
- [x] `python -m pytest` が `68 passed` で通る。
- [x] ComfyUI再起動後、実機APIで `auto` のmetadata候補ログを確認する。
- [ ] 実機Animaで `auto` にshape候補が出ない原因を追加調査する。
- [ ] 非正方形時に `auto` の正方形fallbackを避けるUI/API設計を検討する。
