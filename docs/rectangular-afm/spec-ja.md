# Anima AFM 非正方形アスペクト比対応 仕様書

最終更新: 2026-06-28

## 目的

`Anima AFM Model Patch` を、`1:1` の正方形画像だけでなく、`16:9`、`9:16`、`4:3`、`3:4` などの非正方形アスペクト比でも安全に使えるようにする。

現在のAFMは、クロスアテンションの `query_len` を平方根で `side x side` に戻している。これは `768 x 768` や `1024 x 1024` のような正方形では成立しやすいが、横長・縦長では本当の `height x width` を復元できない。

## 背景

元論文 `Attention Frequency Modulation: Training-Free Spectral Modulation of Diffusion Cross-Attention` は、クロスアテンションを `H x W` の潜在空間グリッドとして定義している。数式上は `H = W` に限定されない。

一方、論文内の実験は `512 x 512` の正方形画像で行われており、16:9などの非正方形解像度を明示的に検証した記述はない。そのため、このリポジトリでの非正方形対応は、論文の考え方をAnima/Cosmos DiT向けに拡張する実装作業として扱う。

## 実装状況

2026-06-28時点で、Phase 1の明示寸法指定は実装済み。

- `spatial_shape_mode=explicit_pixels` で `image_width` / `image_height` から矩形queryグリッドを復元する。
- `spatial_shape_mode=explicit_latent` で `latent_width` / `latent_height` を直接指定できる。
- `auto` はruntime metadata候補を確認した後、従来互換の正方形推定へ戻る。
- 4D block input `x` やdict形式の `rope_emb` から安全に候補化できる場合は、`afm_spatial_shape_candidates` としてattention側へ渡す。
- 複数の有効候補が互いに異なる場合は `spatial_shape_ambiguous` としてフォールバックする。
- `16:9`, `9:16`, `4:3` はComfyUI APIで生成とログ確認済み。
- 新しい矩形関連入力は、既存APIワークフロー互換のためoptional入力として扱う。

## 変更前の制約

変更前の実装では、空間形状推定は次の関数に集約されていた。

```python
def infer_square_spatial_shape(query_len: int) -> tuple[int, int] | None:
    side = math.isqrt(int(query_len))
    if side * side != query_len:
        return None
    return side, side
```

問題点:

- `query_len` が完全平方数でない非正方形グリッドは `cannot_infer_spatial_shape` でフォールバックする。
- 本当は `64 x 36` のような長方形でも、積が完全平方数なら `48 x 48` のような誤った正方形として扱う可能性がある。
- ノードは `EmptyLatentImage` の `width` / `height` と直接つながっていないため、現在はワークフロー解像度を自動的に知らない。
- `AnimaAFMBlockMetadataWrapper` はブロック識別メタデータを `transformer_options` に追加しているが、空間 `height` / `width` は追加していない。

## 対応範囲

### Phase 1で対応するもの

- 静止画T2Iの非正方形アスペクト比。
- Anima/Cosmos DiTのクロスアテンションで、`q.shape[-2] == height * width` と検証できる呼び出し。
- `16:9`、`9:16`、`4:3`、`3:4` を代表ケースとして検証する。
- 既存の正方形ワークフローは後方互換で維持する。

### Phase 1で対応しないもの

- 動画・時系列レイアウト。
- 複数フレームを含む `time x height x width` のqueryグリッド。
- マスク付き attention の新規対応。
- Anima以外のモデルファミリへの一般化。
- 論文と同じSD U-Net向けAFM実装。

## 基本方針

非正方形対応で一番重要なのは、`query_len` だけから形状を推測しないこと。AFMはFFTで空間周波数を扱うため、`height x width` を間違えると、変化が出ても意味のある周波数操作とは言えない。

そのため、形状推定は次の順に行う。

1. 明示的なユーザー指定またはAPI指定。
2. ComfyUI / Anima 実行時メタデータから取得できる実寸グリッド。
3. 正方形互換フォールバック。
4. 不明な場合は編集せず元のattentionにフォールバック。

## 実装済みノード入力

Phase 1では、まず明示指定を優先する。

| 入力 | 種別 | 初期値 | 意味 |
| --- | --- | --- | --- |
| `spatial_shape_mode` | combo | `auto` | `auto`, `square_only`, `explicit_pixels`, `explicit_latent` |
| `image_width` | int | `0` | 生成画像のピクセル幅。`explicit_pixels` 用 |
| `image_height` | int | `0` | 生成画像のピクセル高さ。`explicit_pixels` 用 |
| `latent_width` | int | `0` | attention queryの幅。`explicit_latent` 用 |
| `latent_height` | int | `0` | attention queryの高さ。`explicit_latent` 用 |
| `latent_downscale` | int | `16` | pixel寸法からlatent寸法を計算する時の倍率 |
| `aspect_tolerance` | float | `0.001` | 比率推定の許容誤差 |

入力はすべて advanced 扱いにし、既存ユーザーの正方形ワークフローでは触らなくても動く状態を維持する。ComfyUI APIワークフロー互換のため、矩形関連入力はoptional入力として実装する。

## 形状推定仕様

新しい推定関数は仮に `infer_spatial_shape(query_len, config, transformer_options, metadata)` とする。

### `explicit_latent`

`latent_width > 0` かつ `latent_height > 0` の場合のみ有効。

- `latent_width * latent_height == query_len` なら `(latent_height, latent_width)` を返す。
- 積が一致しない場合は `spatial_shape_mismatch` でフォールバック。

### `explicit_pixels`

`image_width > 0` かつ `image_height > 0` の場合のみ有効。

1. `image_width / latent_downscale` と `image_height / latent_downscale` が整数で、積が `query_len` と一致するなら採用する。
2. 一致しない場合は、`query_len` の因数分解から、`image_width:image_height` に最も近い `(width, height)` を探す。
3. 誤差が `aspect_tolerance` 以内で、かつ積が一致する場合のみ採用する。
4. 見つからない場合は `spatial_shape_mismatch` でフォールバック。

戻り値の順序は常に `(height, width)`。

### `auto`

`auto` は安全優先とする。

1. `transformer_options` やクロスアテンションwrapperから信頼できる `height` / `width` が見つかれば採用する。
2. 複数の有効候補があり、互いに異なる `height x width` を示す場合は `spatial_shape_ambiguous` でフォールバックする。
3. 見つからない場合、`query_len` が完全平方数なら従来と同じ正方形推定を使う。
4. 完全平方数でなければフォールバック。

`auto` では、比率だけから長方形を推測しない。推測に必要な元画像寸法がないため。

現在の候補取得元:

- `afm_spatial_shape`
- `afm_spatial_shape_candidates`
- `latent_height` / `latent_width`
- `spatial_height` / `spatial_width`
- `spatial_shape`
- `latent_shape`
- 4D block input `x` の中間2軸または末尾2軸
- dict形式の `rope_emb` に含まれるshape情報

### `square_only`

従来互換モード。

- `query_len` が完全平方数なら `(side, side)`。
- それ以外はフォールバック。

## 周波数処理仕様

`edit_logits_fft` はすでに `spatial_shape: tuple[int, int]` を受け取る形になっているため、`height != width` でもreshape自体は可能。

確認すべき点:

- `height * width == query_len` を必ず検証する。
- `radial_low_high_masks(height, width, ...)` が長方形でも安定することを単体テストで確認する。
- `hf_ratio_from_concentration` も同じ `height x width` でreshapeできることを確認する。
- DC成分保持、soft mask、cutoff、entropy gate の挙動は正方形と同じ意味で扱う。

## ログ仕様

既存ログの `spatial_shape` は維持し、追加で次を出す。

| フィールド | 意味 |
| --- | --- |
| `spatial_shape_source` | `explicit_latent`, `explicit_pixels_downscale`, `explicit_pixels_aspect`, `runtime_metadata`, `square_legacy`, `none` |
| `spatial_shape_reason` | 採用またはフォールバック理由 |
| `spatial_shape_mode` | 形状推定モード |
| `image_width` / `image_height` | 明示指定された生成画像のピクセル寸法 |
| `latent_width` / `latent_height` | 明示指定されたlatentグリッド寸法 |
| `latent_downscale` | pixel寸法からlatent寸法を計算する倍率 |
| `aspect_tolerance` | aspect比因数推定の許容誤差 |
| `spatial_shape_aspect_error` | 比率推定を使った場合の誤差 |

ログで「誤った正方形推定」を見逃さないため、非正方形指定があるのに正方形推定へ落ちることは禁止する。

## 失敗時の挙動

デフォルトは安全側に倒す。

- `fail_mode=fallback`: 元のattentionを呼ぶ。
- `fail_mode=raise`: 明確な例外を出す。

新しいfallback reason:

- `spatial_shape_missing`
- `spatial_shape_mismatch`
- `spatial_shape_ambiguous`
- `spatial_shape_metadata_invalid`

既存の `cannot_infer_spatial_shape` は互換のため残してよいが、新しい理由に寄せる。

## 受け入れ条件

- 既存の正方形テストがすべて通る。
- `infer_spatial_shape` の単体テストで、`64 x 64`, `84 x 48`, `48 x 84`, `80 x 60`, `60 x 80` が正しく推定される。
- 明示寸法と `query_len` が矛盾する場合、編集せずフォールバックする。
- `query_len` が完全平方数でも、明示されたアスペクト比が非正方形なら誤った正方形推定をしない。
- ComfyUI APIで少なくとも `16:9` と `9:16` のワークフローが成功する。
- `debug_level=summary` または `debug_format=jsonl` で、ログに正しい `spatial_shape` と `spatial_shape_source` が出る。
- 正方形の既存サンプル生成に回帰がない。

## 未解決事項

- 実機Anima API実行時に、4D block input `x` や `rope_emb` から期待どおり候補が出るか。
- `latent_downscale=16` を固定値にしてよいか、モデルごとに違うか。
- 複数解像度のattention callが混ざるモデルで、どのスケールを編集対象にするか。
- 動画対応時に `time x height x width` をどう扱うか。
