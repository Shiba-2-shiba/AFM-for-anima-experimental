# Anima AFM 非正方形対応 進捗

最終更新: 2026-06-28

## 現在の状態

ステータス: Phase 1からPhase 6の単体実装まで完了。明示寸法指定による静止画の非正方形AFM editは、ComfyUI APIで `16:9`、`9:16`、`4:3` の生成とログ確認まで通過。runtime metadata候補の自動利用と曖昧候補の拒否も単体テストで確認済み。ただし実機Anima APIでは、現時点の `auto` は16:9の非正方形shape候補を取得できず、`square_legacy` へ戻ることを確認した。

`spatial_shape_mode=explicit_pixels` を使い、生成ピクセル寸法から内部queryグリッドを復元する。`auto` は後方互換の正方形推定を維持し、明示寸法がない非正方形を無理に推測しない。

## 完了したこと

- 元論文PDF `2603.28114v1.pdf` を確認。
- 論文の数式は `H x W` であり、理論上は正方形限定ではないことを確認。
- 論文の実験は `512 x 512` 正方形で、16:9などの非正方形検証は見当たらないことを確認。
- 仕様書 `docs/rectangular-afm/spec-ja.md` を作成。
- タスク一覧 `docs/rectangular-afm/tasks-ja.md` を作成。
- `AFMConfig` に `spatial_shape_mode`, `image_width`, `image_height`, `latent_width`, `latent_height`, `latent_downscale`, `aspect_tolerance` を追加。
- `nodes.py` に上記のadvanced入力を追加し、既存APIワークフロー互換のため任意入力にした。
- `infer_spatial_shape` を追加し、`explicit_latent`, `explicit_pixels`, `auto`, `square_only` を実装。
- `explicit_pixels` では、downscale一致とaspect比による因数推定を実装。
- 長方形edit pathが通る単体テストを追加。
- 形状不一致時に `spatial_shape_mismatch` で安全にフォールバックするテストを追加。
- JSONL/textログに `spatial_shape_source`, `spatial_shape_reason`, `spatial_shape_aspect_error` を追加。
- JSONLログに明示寸法入力値も出るようにした。
- ComfyUI APIで非正方形生成を実行し、比較画像を `docs/assets` に作成。
- ComfyUI再起動後、`scope_*` と矩形関連advanced入力が `/object_info` でoptional扱いになることを確認。
- 新規optional入力を省いた旧API JSON形状で、正方形AFM editが動くことを確認。
- 誤った明示寸法では `spatial_shape_mismatch` で編集せずフォールバックすることを確認。
- `auto` でruntime metadata候補を収集する処理を強化。
  - `afm_spatial_shape`, `afm_spatial_shape_candidates`
  - `latent_height` / `latent_width`
  - `spatial_height` / `spatial_width`
  - 4D block input `x` の中間2軸または末尾2軸
  - dict形式の `rope_emb` に含まれるshape情報
- 複数の有効候補が互いに異なる場合は、正方形推定へ逃げず `spatial_shape_ambiguous` でフォールバックするようにした。

## API検証結果

使用ワークフロー:

- `AFM-APIワークフロー.json`
- モデルは環境にある `capanima_base1.safetensors` に差し替え。

短縮検証:

| ケース | 解像度 | ログ | 結果 |
| --- | --- | --- | --- |
| 横長 | `1024 x 576` | `logs/rect_16x9_edit_api_validation.jsonl` | `spatial_shape=[36,64]`, `spatial_shape_source=explicit_pixels_downscale` |
| 縦長 | `576 x 1024` | `logs/rect_9x16_edit_api_validation.jsonl` | `spatial_shape=[64,36]`, `spatial_shape_source=explicit_pixels_downscale` |

比較画像用生成:

| ケース | 解像度 | ログ | 比較画像 |
| --- | --- | --- | --- |
| rainy neon alley | `1024 x 576` | `logs/rect_16x9_neon_edit.jsonl` | `docs/assets/afm_rect_compare_rect_16x9_neon.png` |
| clockwork library | `576 x 1024` | `logs/rect_9x16_clocktower_edit.jsonl` | `docs/assets/afm_rect_compare_rect_9x16_clocktower.png` |
| glass greenhouse | `896 x 672` | `logs/rect_4x3_greenhouse_edit.jsonl` | `docs/assets/afm_rect_compare_rect_4x3_greenhouse.png` |

ログ確認:

- `16:9`: `q_shape=[2,16,2304,128]`, `spatial_shape=[36,64]`
- `9:16`: `q_shape=[2,16,2304,128]`, `spatial_shape=[64,36]`
- `4:3`: `q_shape=[2,16,2352,128]`, `spatial_shape=[42,56]`
- 対象cross-attentionでは、空間形状不一致由来のフォールバックなし。
- `fallback_reasons` に出る `not_cross_attention` は対象外attention呼び出しの集計であり、矩形推定失敗ではない。

再起動後の互換性検証:

| ケース | 内容 | 結果 |
| --- | --- | --- |
| `/object_info` | `scope_*` と矩形関連入力 | 11入力すべてoptional、requiredは28入力 |
| 旧API JSON互換 | `scope_*` / `spatial_*` を省いたまま送信 | missing required なし、生成成功 |
| 正方形edit | `512 x 512`, `spatial_shape_mode=auto` | `spatial_shape=[32,32]`, `spatial_shape_source=square_legacy` |
| 横長edit | `1024 x 576`, `explicit_pixels` | `spatial_shape=[36,64]`, `spatial_shape_source=explicit_pixels_downscale` |
| 縦長edit | `576 x 1024`, `explicit_pixels` | `spatial_shape=[64,36]`, `spatial_shape_source=explicit_pixels_downscale` |
| 誤寸法 | 生成 `1024 x 576`, 指定 `1024 x 512` | `spatial_shape_mismatch` で編集0、生成成功 |

追加ログと出力:

- `logs/old_api_optional_compat.jsonl`
- `logs/latest_16x9_edit.jsonl`
- `logs/latest_9x16_edit.jsonl`
- `logs/mismatch_safe_fallback.jsonl`
- `C:\ComfyUI\output\AFM_rect_validation\old_api_optional_compat_00001_.png`
- `C:\ComfyUI\output\AFM_rect_validation\square_baseline_off_compat_00001_.png`
- `C:\ComfyUI\output\AFM_rect_validation\latest_16x9_edit_00001_.png`
- `C:\ComfyUI\output\AFM_rect_validation\latest_9x16_edit_00001_.png`
- `C:\ComfyUI\output\AFM_rect_validation\mismatch_safe_fallback_00001_.png`

Phase 6 実機確認:

| ケース | 内容 | 結果 |
| --- | --- | --- |
| auto 16:9 | `1024 x 576`, `spatial_shape_mode=auto`, 明示寸法なし | 生成成功。ただし `spatial_shape=[48,48]`, `spatial_shape_source=square_legacy` |
| metadata候補 | `afm_spatial_shape_candidates` | 実機ログには出現せず |
| explicit 16:9 | `1024 x 576`, `spatial_shape_mode=explicit_pixels` | 生成成功。`spatial_shape=[36,64]`, `spatial_shape_source=explicit_pixels_downscale` |

関連ログと出力:

- `logs/auto_metadata_16x9_after_restart.jsonl`
- `logs/explicit_pixels_16x9_after_auto_check.jsonl`
- `C:\ComfyUI\output\AFM_rect_validation\auto_metadata_16x9_after_restart_00001_.png`
- `C:\ComfyUI\output\AFM_rect_validation\explicit_pixels_16x9_after_auto_check_00001_.png`

## 重要な判断

### 判断1: Phase 1は静止画の非正方形対応に限定する

動画や `time x height x width` のqueryグリッドは別設計が必要。まずはT2Iの `16:9` / `9:16` / `4:3` を安全に通す。

### 判断2: 明示寸法指定を第一実装にする

ノードは `EmptyLatentImage` の `width` / `height` と直接接続されていない。runtime metadataだけに依存すると非正方形を推測できないため、advanced入力で `image_width` / `image_height` または `latent_width` / `latent_height` を指定できるようにした。

### 判断3: 推測よりフォールバックを優先する

誤った `height x width` でFFTすると、画像は変化してもAFMとして意味のある周波数操作ではなくなる。そのため、不明・矛盾・曖昧な場合は編集せず元attentionへ戻す。

### 判断4: 新規advanced入力は任意入力にする

API検証で、古い `AFM-APIワークフロー.json` が新規入力を持たないため、ComfyUIのAPI validationで不足入力として失敗することを確認した。既存ワークフロー互換のため、矩形関連入力は `optional=True` とし、execute側にもデフォルト値を持たせる。

## 次にやること

1. 実機のAnima/ComfyUI Desktopで、block wrapperから到達できるshape情報をさらに調査する。
2. 非正方形では、引き続き `explicit_pixels` を推奨ルートにする。
3. `auto` の正方形fallbackが非正方形の完全平方query_lenで誤適用される問題を避けるUI/API設計を検討する。
4. 動画や複数フレームの `time x height x width` レイアウトを扱う場合は、別仕様として設計する。

## リスク

- Animaモデルの内部query gridが常に `image_width / 16` と `image_height / 16` になるとは限らない。
- `query_len` が複数スケールで現れるモデルでは、ひとつの画像寸法から複数の `height x width` 候補を導く必要がある。
- 長方形FFTのcutoff解釈は数学的には成立するが、正方形サンプルと同じ見え方になるとは限らない。
- 実機Anima APIでは、現時点で `auto` 用のshape metadata候補が出ていない。非正方形では `explicit_pixels` または `explicit_latent` を使う。
- `1024 x 576` のように内部query_lenが `2304=48x48` にも見えるケースでは、`auto` の正方形互換fallbackが誤った正方形として編集する可能性がある。

## 検証状況

実行済み:

- 2026-06-28: 記事作成後の既存テストは `58 passed`。
- 2026-06-28: 非正方形対応の単体テスト追加後 `python -m pytest tests\test_anima_afm.py -q` は `61 passed`。
- 2026-06-28: `python -m py_compile anima_afm.py nodes.py` は成功。
- 2026-06-28: `python -m pytest` は `66 passed`。
- 2026-06-28: ComfyUI APIで `1024 x 576`, `576 x 1024`, `896 x 672` のbaseline/edit比較画像生成に成功。
- 2026-06-28: ComfyUI再起動後、optional入力化の `/object_info` 反映確認に成功。
- 2026-06-28: 新規optional入力を省いた古いAPIワークフロー互換確認に成功。
- 2026-06-28: 最新ロード後、正方形baseline、正方形edit、16:9 edit、9:16 edit、誤寸法fallback確認に成功。
- 2026-06-28: runtime metadata候補と `spatial_shape_ambiguous` の単体テスト追加後、`python -m pytest` は `68 passed`。
- 2026-06-28: ComfyUI再起動後、`auto` 16:9 API確認を実施。生成は成功したが、runtime metadata候補は出ず `square_legacy` へ戻った。
- 2026-06-28: 同じ再起動後環境で `explicit_pixels` 16:9 API確認を実施。`spatial_shape=[36,64]` で編集されることを再確認。
