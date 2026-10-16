# Anima AFM 修正後の実機検証計画

作成日: 2026-10-16。これは**今後実施する検証手順**であり、この文書の追加は GPU・ComfyUI 実行済み、画質改善、論文再現の証拠ではない。

## 1. 目的と判定の順序

次の順にゲートを通す。前段が失敗・未実施なら、後段の結果を「合格」にしない。

1. CPU 回帰テストと実行環境・入力の固定
2. 小さい正方形で original/observe/manual-zero と編集経路の切り分け
3. 長方形の正しい二次元配置、サンプリングごとの状態、対象ブロック・CFG 分岐の確認
4. ガード・異常終了・非対応 sampler の安全性と、実測時間・ピークメモリ
5. 複数 prompt・seed による盲検画質評価

`logs/`、既存の比較画像、2026-06 の検証文書は当時の記録として保存する。新しい修正の証拠には流用しない。とくに過去の `auto` が `36×64=2304` query を `48×48` と扱った記録は、今回の回帰ケースの根拠であり、成功例ではない。旧仕様書の `auto` による平方数推定より、今回の **auto は信頼できる H/W がなければ original に戻る**仕様を優先する。`square_only` は利用者が正方形を保証する明示的互換モードである。

## 2. 事前確認と保存先

以下は実機担当者が実施する POSIX shell の例。現在の修正作業では、モデルのダウンロード、GPU 利用、ComfyUI 起動、API への生成要求は行わない。実行時には専用・ローカルの検証用 ComfyUI を使い、他の利用者のジョブがないことを確認する。

既存の ComfyUI Python 環境を指定する。モデル名は例に合わせて置き換えず、実際に検証するファイルを明示する。`anima-base-v1.0`、`capanima_base1`、preview 系を同一モデルとしてまとめない。追加インストールが必要なら、先に環境変更を記録してから再度固定する。

```bash
export COMFY_ROOT=/absolute/path/to/ComfyUI
export AFM_ROOT="$COMFY_ROOT/custom_nodes/AFM-for-anima-experimental"
export AFM_COMFYUI_SOURCE_ROOT="$COMFY_ROOT"
export PYTHON="$COMFY_ROOT/.venv/bin/python"
export MODEL_FILE="$COMFY_ROOT/models/diffusion_models/capanima_base1.safetensors"
export CLIP_FILE="$COMFY_ROOT/models/text_encoders/qwen_3_06b_base.safetensors"
export VAE_FILE="$COMFY_ROOT/models/vae/qwen_image_vae.safetensors"
export COMFY_URL=http://127.0.0.1:8188
export EVIDENCE="$AFM_ROOT/validation-results/$(date -u +%Y%m%dT%H%M%SZ)"
test -x "$PYTHON" && test -f "$MODEL_FILE" && test -f "$CLIP_FILE" && test -f "$VAE_FILE"
mkdir -p "$EVIDENCE"/{preflight,cpu,runs,analysis,quality}
cd "$AFM_ROOT"
git rev-parse HEAD > "$EVIDENCE/preflight/afm-commit.txt"
git status --porcelain=v1 > "$EVIDENCE/preflight/afm-status.txt"
git diff HEAD --binary > "$EVIDENCE/preflight/afm-working-tree.patch"
git -C "$COMFY_ROOT" rev-parse HEAD > "$EVIDENCE/preflight/comfy-commit.txt"
git -C "$COMFY_ROOT" status --porcelain=v1 > "$EVIDENCE/preflight/comfy-status.txt"
"$PYTHON" --version > "$EVIDENCE/preflight/python.txt" 2>&1
"$PYTHON" -m pip freeze > "$EVIDENCE/preflight/packages.txt"
sha256sum "$MODEL_FILE" "$CLIP_FILE" "$VAE_FILE" > "$EVIDENCE/preflight/model-sha256.txt"
sha256sum ./*.py scripts/*.py tests/*.py > "$EVIDENCE/preflight/source-sha256.txt"
find logs -type f -print0 | sort -z | xargs -0 sha256sum > "$EVIDENCE/preflight/legacy-log-sha256.txt"
CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
  "$PYTHON" -m pytest -q > "$EVIDENCE/cpu/pytest.txt" 2>&1
CPU_STATUS=$?
printf '%s\n' "$CPU_STATUS" > "$EVIDENCE/cpu/exit-code.txt"
test "$CPU_STATUS" -eq 0
```

CPU テストの失敗後に残りのコマンドをそのまま続けない。テスト数、実行時刻、commit、終了コードを新規 CPU 結果として報告する。CPU の成功は ComfyUI の wrapper 接続や CUDA allocator の検証を代替しない。

`AFM_COMFYUI_SOURCE_ROOT` は実際の ComfyUI ソース root を指定する。ホスト契約テストが source 不在で skip されていないかも確認する。source hash は新規 `sampling_lifecycle.py` を含む全 root Python と scripts/tests を保存する。使った scope map と実機計測 hook も個別に hash を保存する。

Windows/AMD 環境の最小 PowerShell 入口は以下。CPU 専用の別 terminal で実行し、GPU 段階の前にその terminal を閉じる。以降の Bash コマンドは shell 構文を対応させる必要があるが、Python の workflow/manifest 作成部分は同じコードを使える。Paperspace など Linux では上の POSIX 手順を使う。

```powershell
$env:COMFY_ROOT = 'D:\ComfyUI'
$env:AFM_ROOT = Join-Path $env:COMFY_ROOT 'custom_nodes\AFM-for-anima-experimental'
$env:AFM_COMFYUI_SOURCE_ROOT = $env:COMFY_ROOT
$env:PYTHON = Join-Path $env:COMFY_ROOT '.venv\Scripts\python.exe'
$env:EVIDENCE = Join-Path $env:AFM_ROOT ('validation-results\' + (Get-Date -Format 'yyyyMMddTHHmmss'))
New-Item -ItemType Directory -Force "$env:EVIDENCE\cpu", "$env:EVIDENCE\preflight" | Out-Null
Set-Location $env:AFM_ROOT
$env:CUDA_VISIBLE_DEVICES = '-1'; $env:HIP_VISIBLE_DEVICES = '-1'; $env:ROCR_VISIBLE_DEVICES = '-1'
& $env:PYTHON -m pytest -q *> "$env:EVIDENCE\cpu\pytest.txt"
$code = $LASTEXITCODE
$code | Set-Content "$env:EVIDENCE\cpu\exit-code.txt"
Get-ChildItem *.py, scripts\*.py, tests\*.py | Get-FileHash -Algorithm SHA256 | Format-List | Out-File "$env:EVIDENCE\preflight\source-sha256.txt"
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion | Format-List | Out-File "$env:EVIDENCE\preflight\gpu-driver.txt"
if ($code -ne 0) { throw "CPU tests failed: $code" }
```

実機で GPU 段階を開始する際に、GPU 型・VRAM・driver・OS、PyTorch/CUDA または HIP/ROCm/cuDNN、attention backend、dtype、起動引数、追加 custom nodes、VRAM モードを記録する。自動選択された backend/dtype は実ログで確認する。次は read-only の状態確認で、生成は要求しない。`nvidia-smi` は NVIDIA のときだけ使い、AMD では既存の `amd-smi` / `rocm-smi`、または Windows driver 情報を保存する。情報取得だけのために追加ツールをインストールしない。

```bash
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$EVIDENCE/preflight/nvidia-smi.txt"
elif command -v amd-smi >/dev/null 2>&1; then
  amd-smi static > "$EVIDENCE/preflight/amd-smi.txt"
elif command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi > "$EVIDENCE/preflight/rocm-smi.txt"
fi
"$PYTHON" - <<'PY' > "$EVIDENCE/preflight/torch.txt"
import platform, torch
print(platform.platform())
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("hip", getattr(torch.version, "hip", None))
print("cudnn", torch.backends.cudnn.version())
print("deterministic_algorithms", torch.are_deterministic_algorithms_enabled())
print("tf32_matmul", torch.backends.cuda.matmul.allow_tf32)
print(torch.__config__.show())
PY
curl --fail --silent --show-error "$COMFY_URL/system_stats" > "$EVIDENCE/preflight/system-stats.json"
curl --fail --silent --show-error "$COMFY_URL/object_info" > "$EVIDENCE/preflight/object-info.json"
```

`object-info.json` で `AnimaAFMModelPatch` と使用する loader/latent/sampler の schema を確認する。現行 ComfyUI でモデルに適合する latent node を使用し、旧 example の widget 配列をそのまま信用しない。API template は `AFM-APIワークフロー.json` を基礎にするが、旧 `er_sde` のまま初回検証を始めない。

今回の sampling adapter は ComfyUI commit `15ef24d1c0333a3eba56c5cd153d8db65363ff8f` の `OUTER_SAMPLE` / `APPLY_MODEL` / `post_input` 契約を対象にソース確認されている。これはその組み合わせの実機合格を意味しない。異なる版では契約と引数を再確認し、不足 API を無視して旧経路へ戻さない。

並列 multi-GPU evaluation は未対応であり、検出されたら明確に拒否する。最初の実機検証は単一 GPU で行う。複数 GPU の設定を単一 GPU の合格結果から保証しない。

shape を取得する `post_input` AFM hook は、画像/latent の形状を変更する既存 hook より後に実行される必要がある。AFM より後で H/W や flatten 順序を変更する hook は未対応とし、他 node の hook 順序も記録する。trusted metadata があっても、その後に形状が変わっていれば根拠として不十分である。

## 3. 固定する入力と比較条件

各 run に一意な ID、workflow JSON と SHA-256、モデル・text encoder・VAE SHA-256、AFM/ComfyUI commit と未コミット差分、scope map の内容と SHA-256 を保存する。さらに以下を固定・記録する。

- positive/negative prompt の実文字列と UTF-8 SHA-256、seed、batch size、初期 noise と latent の tensor/hash
- sampler、scheduler、steps、CFG、denoise、実際の sigma 列、stochastic sampler の追加乱数/RNG 状態
- image 幅/高さと実測 query の `(H,W)`、dtype、backend、device、実行順序
- cutoff、schedule、entropy_gate、preserve_dc、soft mask、対象 call/block/stage、branch、診断範囲、全 guard 上限

同じ seed だけでは同じ stochastic draw を保証できない。最初は `euler` / `simple` / `denoise=1` / `CFG=4` / batch 1、3 steps、固定 noise で経路確認する。元の sampler の評価は独立した互換性段階で行う。3 steps の結果から画質の結論を出さない。

主要な比較を次のように分ける。observe と manual-zero/nonzero の対象 scope は必ず一致させる。

| ID | 条件 | 何を切り分けるか |
| --- | --- | --- |
| O | AFM node を取り除き loader MODEL を sampler へ直結 | 真の original backend |
| V | `mode=observe`, `strength=0` | original に診断・wrapper を加えた影響 |
| Z | `mode=edit`, `strength=0`, `zero_strength_mode=manual` | 手動 attention backend だけの影響 |
| S10/S20/S35 | `mode=edit`, strength 0.10/0.20/0.35 | 同じ手動 backend 上の AFM 効果 |
| F（補助） | `mode=off` | 完全無効化時の original 保持 |

O 対 V/F は同一経路の parity、O 対 Z は backend 差、Z 対 S は AFM 差、O 対 S は利用者が受ける総変化を表す。Z が O と異なることだけで AFM の効果とは言わない。Z を「無加工画像」の唯一の基準にしない。

### API 入力を作る例（生成要求はまだ送らない）

以下は現リポジトリの API node ID に対応する。モデルがサブフォルダ内なら loader 名をその相対名に変更する。`explicit_pixels` の downscale は本モデルで確認した値を使う。初回は `target_call_indices=0` で割り当て・分岐を抑え、後に同じ組の全条件を `block_scope/early` または検証した厳密 block path に揃えて再生成する。

```bash
export RUN_GROUP=sq512-smoke
export WIDTH=512 HEIGHT=512 STEPS=3 SEED=55675678500
export SCOPE_MODE=target_call_indices TARGET_CALLS=0 STAGE_SCOPE=
"$PYTHON" - <<'PY'
import copy, hashlib, json, os
from pathlib import Path
root = Path(os.environ["AFM_ROOT"])
base = json.loads((root / "AFM-APIワークフロー.json").read_text(encoding="utf-8"))
base["78"]["inputs"]["unet_name"] = Path(os.environ["MODEL_FILE"]).name
base["71"]["inputs"]["clip_name"] = Path(os.environ["CLIP_FILE"]).name
base["72"]["inputs"]["vae_name"] = Path(os.environ["VAE_FILE"]).name
w, h = int(os.environ["WIDTH"]), int(os.environ["HEIGHT"])
base["74"]["inputs"].update(width=w, height=h, batch_size=1)
base["76"]["inputs"].update(seed=int(os.environ["SEED"]), steps=int(os.environ["STEPS"]),
    cfg=4, sampler_name="euler", scheduler="simple", denoise=1)
base["77"]["inputs"]["text"] = "anime illustration, a glass greenhouse, broad leaves, fine veins, soft light"
base["79"]["inputs"].update(scope_mode=os.environ["SCOPE_MODE"],
    target_call_indices=os.environ["TARGET_CALLS"], stage_scope=os.environ["STAGE_SCOPE"],
    block_scope="", scope_map_path="", spatial_shape_mode="explicit_pixels",
    image_width=w, image_height=h, latent_width=0, latent_height=0, latent_downscale=16,
    branch_mode="both", entropy_gate=False, schedule="curve", preserve_dc=True,
    debug_level="summary", debug_format="jsonl", spectral_diag="sampled",
    diagnostic_call_indices=os.environ["TARGET_CALLS"], diagnostic_branch="both_separate",
    diagnostic_include_unselected=True, diagnostic_every_n_steps=1,
    fail_mode="fallback", max_logits_mib=1024, max_peak_mib=4096)
conditions = [("O", None, 0), ("V", "observe", 0), ("Z", "edit", 0),
              ("S10", "edit", .10), ("S20", "edit", .20), ("S35", "edit", .35),
              ("F", "off", 0)]
for label, mode, strength in conditions:
    run = Path(os.environ["EVIDENCE"]) / "runs" / os.environ["RUN_GROUP"] / label
    run.mkdir(parents=True, exist_ok=False)  # 同名 run の上書きを禁止
    workflow = copy.deepcopy(base)
    workflow["46"]["inputs"]["filename_prefix"] = "afm_validation/" + os.environ["RUN_GROUP"] + "/" + label
    if mode is None:
        del workflow["79"]
        workflow["76"]["inputs"]["model"] = ["78", 0]
    else:
        workflow["79"]["inputs"].update(mode=mode, strength=strength,
            zero_strength_mode="manual" if label == "Z" else "observe",
            jsonl_path=str(run / "afm.jsonl"))
    encoded = json.dumps(workflow, ensure_ascii=False, sort_keys=True, indent=2).encode()
    (run / "workflow.json").write_bytes(encoded)
    (run / "workflow.sha256").write_text(hashlib.sha256(encoded).hexdigest() + "\n")
    (run / "request.json").write_text(json.dumps({"prompt": workflow}, ensure_ascii=False))
PY
```

事前確認・実機実行の許可後、1 件ずつ以下の形で投入する。`request.json` の全設定とパスを先に確認する。出力先は ComfyUI サーバーから書けるローカルパスでなければならない。HTTP 200 は生成成功ではない。response の `error` / `node_errors` と、history の terminal status・例外・出力ファイルを確認する。

```bash
export RUN="$EVIDENCE/runs/sq512-smoke/O"
curl --fail --silent --show-error -H 'Content-Type: application/json' \
  --data-binary @"$RUN/request.json" "$COMFY_URL/prompt" > "$RUN/response.json"
export PROMPT_ID=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prompt_id"])' "$RUN/response.json")
curl --fail --silent --show-error "$COMFY_URL/history/$PROMPT_ID" > "$RUN/history.json"
```

history が空・running ならまだ完了ではない。完了まで同じ ID を照会し、同じ request を再送しない。成功時は保存 PNG（workflow metadata を含む）、生成 latent、コンソールログ、計測値、対応する `run_id` を run ディレクトリへ収集する。

## 4. 正しさのゲート

### A. 512×512 の短縮検証

1. O を独立に 2 回実行し、元 backend 自体の再現性を調べる。tensor bit equality と max absolute/relative error を保存する。
2. V/F と O の final latent・decoded image を比較する。入力が完全同一で backend が決定的なら bit equality を要求する。非決定的なら O/O の反復測定に基づく許容誤差を、S の結果を見る前に固定する。PNG ファイル hash は metadata で異なり得るので、画素または tensor を比較する。
3. Z と O の差、Z と S10 の差をそれぞれ保存する。V の `edited=0`、Z の spectral local delta=0、S の対象 call で編集実施を確認する。ガードが全編集を拒否した run は成功した AFM edit と数えない。
4. query の実測 H/W、scope、CFG batch 配置、step/sigma、各 eligible call を確認する。512 pixels だから内部が 512 query と推測しない。downscale 16 が成立する場合は `32×32` だが、実測値を優先する。
5. 診断なしでも同じ比較を繰り返す。診断を有効にしたために元出力が変わる、残る tensor が増える、終了時に state が漏れる場合は停止する。

### B. 長方形と転置（両方 query 数は 2304）

同じ短縮設定で `WIDTH=1024 HEIGHT=576` と `WIDTH=576 HEIGHT=1024` を別 group として作る。まず O/V/Z/S10 を確認し、対象 scope を揃える。

- `explicit_pixels` と `explicit_latent` の一致: 横長は `(H,W)=(36,64)`、縦長は `(64,36)`。明示 latent の値はこの順序に合わせる。`H×W=q_len` だけでは転置誤りを検出できない。
- `auto` のメタデータあり: 正しい H/W と候補の由来を要求する。衝突・不正値は編集しない。
- `auto` のメタデータなし: query 数 2304 が完全平方数でも `48×48` に推定せず、original に戻り理由を記録する。メタデータが取れない実機で `auto` が全 fallback でも安全性は合格し得るが、「auto 編集対応」は未達とする。
- 同じ積で誤った明示 H/W を与えるケース: runtime が独立に検証できない明示値の正しさまで保証できるとは主張しない。誤った明示設定の検出可否と利用者側の責任を記録する。
- 積不一致・曖昧候補・time-folded/video は fail-closed を確認する。矩形に `square_only` を使って「auto が成功した」としない。
- 二次元 topology の検証: 実際の tokenizer/patchify の順序を確認し、flatten 前に row/column ramp・水平/垂直 sinusoid・局所 impulse を与えた GPU tensor fixture で FFT 軸を検証する。`36×64` と転置 `64×36` の転置同変性、DC 保持、既知の低/高周波 gain を検証する。形状ログと自然画像の見た目だけでは topology 合格としない。

### C. キャッシュ MODEL の再利用と終了処理

同じ patched MODEL インスタンスのまま、次の 2 種類を別々に試す。patch node が再実行され新しい MODEL を作るだけの試験では不十分。

1. 一つの workflow 内で同じ patched MODEL を 2 個の sampler に渡す。独立 noise/latent の同条件入力を与える。
2. patch node の入力を変えず 2 回 queue する。sampler の seed を A→B と変えて sampler 自体の全キャッシュヒットを避ける。各 seed の単独 fresh 実行とも比較する。

各 sampling invocation に異なる `run_id`、step 0 / eligible call 0 からの計数、同じ対象 call 選択、必要な最終 summary があることを確認する。`target_call_indices=0` と診断 call 0 を必ず含め、2 回目だけ編集・診断が消えないことを確認する。block wrapper が二重登録されない、終了後に override/metadata が呼出し前の状態へ戻ることも確認する。対応 outer-sampling wrapper API がない ComfyUI 版は未対応として停止し、状態共有のまま続行しない。

### D. CFG と scope

- `both` / `positive_only` / `negative_only` を同じ scope と入力で実行し、`cond_or_uncond` の順序、選択 batch index を記録する。
- 非選択 branch は**同一 call・同一 q/k/v に対する original 出力 tensor**と bit equality を要求する。保存するのは差の max absolute 値と有限性。別 run の negative 最終画像や `rho` の一致では証明しない。
- CFG が連結 batch と別 call に分割される条件を両方試す。低 VRAM での split、batch>1、反復 sigma/model evaluation を追加する。call index の意味や単一 step の計数が変わる場合はその条件を未対応とし、call-index scope の品質集計から除外する。
- `encoder_equivalent` は discovery で確認した exact block path/id を選んだ構造化 map で試す。複数の identity フィールドを指定した entry は全て一致する必要がある。同じ stage を共有する非選択 block を必ず含め、共有 `early` や `encoder_candidate`、config の `stage_scope` だけで編集対象が広がらないことを確認する。stage 全体の選択は map entry に `match_type="stage"` を明示する別試験にする。map の対象数と実測数が一致するまで進めない。
- `early/middle/late` は深さの区分であり、論文の U-Net encoder/decoder と同等とみなさない。候補 map の生成だけで意味的対応が証明されたことにはしない。

### E. sampler の対応範囲

まず実際の sigma が `sample_sigmas` に一致する一評価/step の経路を通す。その後に元 workflow の `er_sde` などを別実験として試す。off-grid sigma、多段 solver、反復評価、途中 denoise、split CFG によって `missing_sigmas` や不安定な call identity が生じた場合は、理由付き original fallback または明確な失敗を要求する。編集率ゼロの run を「その sampler に対応」としない。最も近い step に黙って丸めたり、欠測診断をゼロ補完して品質評価へ進めない。

## 5. メモリ・異常系・時間

`estimated_peak_mib` はモデル本体・allocator・FFT workspace を含む実測ピークではない。把握できる tensor lifetime に基づく下限寄りの見積もりであり、実際の総 VRAM 使用量の上限保証ではない。guard と、実機の余裕の両方を確認する。

今回の見積もりは FP32 full logits の大きさを L とすると、FFT 編集経路で最低 `8L + Q/K/V copies`、診断を含む経路で `12L + Q/K/V copies` を基礎にする。backend workspace などは別途必要であり、この式から空き VRAM の十分性を断定しない。実際の source/JSONL の係数と copy 対象も照合する。

1. 小さい入力で `max_logits_mib` または `max_peak_mib` を必要量未満に設定し、編集前と診断前に割り当てを拒否することを確認する。`observe` / `edit`、診断 `off` / `sampled` / `full` を分ける。sampled が full allocation を隠していないか profiler で見る。
2. guard 上限の直下/直上、both/片 branch、矩形、GQA がある場合の head 展開を測る。logits、complex FFT、逆変換、softmax、診断 tensor の生存区間を profiler で記録する。
3. 非有限 q/k/v/logits/output、CUDA OOM、ユーザー cancel を小さい専用 fixture で注入する。現実の VRAM を故意に使い切る試験はしない。`fail_mode=fallback` と `raise` を区別し、非有限出力を成功保存しない、OOM を隠さない、`finally` で状態が戻ることを確認する。original 自体が非有限の場合は正常出力を捏造しない。
4. cancel/例外直後に新しい通常 run を実行し、run ID・計数・対象 scope が汚染されずに再開できるか確認する。cancel は専用サーバーの対象 run に対してだけ行う。
5. JSONL 書込み不可などの telemetry 故障を注入する。編集済み出力を戻す設計では warning が出ることと、編集を再実行・黙って original に切り替えないことを確認する。画像が保存できても診断証拠を欠く run は検証データとして無効にする。

実測は **ComfyUI と同じプロセス**の sampling 前後に計測 hook を入れる。別 Python プロセスの CUDA peak を測っても生成の peak にならない。hook 自体のコード・diff も保存する。最低限の計測本体は以下。例外時も `status` と計測を記録し、完了を偽装しない。

PyTorch ROCm では互換 API の名前が `torch.cuda` のまま使われるため、名前だけで NVIDIA 実行と判定しない。HIP 版・device 情報を併記する。使用 backend がこの計測 API を提供しない場合は未計測と明記し、ゼロや推定値を実測欄へ入れない。

```python
torch.cuda.synchronize(device)
start_allocated = torch.cuda.memory_allocated(device)
start_reserved = torch.cuda.memory_reserved(device)
torch.cuda.reset_peak_memory_stats(device)
start_time = time.perf_counter()
try:
    result = actual_sampling_call()
finally:
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    # 保存: run_id、status、上記5値、elapsed_seconds、device と条件
```

これは挿入位置を示す instrumentation の例であり、`actual_sampling_call` という既製関数が本 repo にあるわけではない。exception/cancel 時の hook 自身の失敗で元例外を失わない実装にする。GPU 全体の `nvidia-smi` は他プロセス・sampling 間隔の影響があるため補助資料とし、上の process-local peak の代用にしない。

モデル cold load、warm sampling、VAE decode、API queue 待ちを分ける。warmup を 1 回記録後、同条件を少なくとも 3 回測り中央値と範囲を示す。品質用 run は診断 off、診断 overhead は別系列で測る。ピークは全体値と開始値からの増分を併記する。guard が通ったという理由だけで高解像度や full 診断へ拡大しない。

## 6. 診断解析と fail-closed な証拠

新規データは一つの sampling invocation ごとに JSONL を分離する。キャッシュ試験で同じファイルに複数 `run_id` が入る場合、raw ファイルを保持したうえで run ID ごとに分離し、分離元と SHA-256 を manifest に記録する。空・壊れた JSONL、混在 run、重複 key、observe/edit 双方向の欠測、非有限値、provenance 不一致は停止条件とする。

診断 key は `(step_index, model_evaluation_index, eligible_call_index, block_id, block_path, branch)`。同じ sigma で model evaluation が繰り返されても黙って平均しない。比較にはさらに同一のモデル、入力 noise/latent、prompt、sigma 列、sampler、dtype/backend、解像度、scope、diagnostic sampling を確認する。条件ラベルや `run_id` が一致する必要はないが、設定差分が意図した treatment だけであることを明示する。全 workflow hash は treatment により異なるので、その一致だけを要求しない。

まず構文を厳格検証する。

```bash
"$PYTHON" scripts/parse_anima_afm_log.py "$EVIDENCE/runs/sq512-smoke/V/afm.jsonl" \
  --strict --format csv > "$EVIDENCE/analysis/V.parsed.csv"
```

次の manifest 作成例は完了済み V/S20 run の**実ファイルから**最低限の provenance を作る。ログ内 run ID が一つでなければ停止する。SHA-256 を仮の値で埋めない。これは初期 noise、sigma 列、backend、scope の検証を自動化するものではない。それらは第3節の追加証拠で照合する。

```bash
"$PYTHON" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
root = Path(os.environ["AFM_ROOT"])
group = Path(os.environ["EVIDENCE"]) / "runs" / "sq512-smoke"
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
model_hash = sha256(os.environ["MODEL_FILE"])
clip_hash = sha256(os.environ["CLIP_FILE"])
vae_hash = sha256(os.environ["VAE_FILE"])
revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
for label in ("V", "S20"):
    run = group / label
    w = json.loads((run / "workflow.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run / "afm.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    run_ids = {r["run_id"] for r in records if r.get("run_id")}
    assert len(run_ids) == 1, (label, run_ids)
    sampler, shape = w["76"]["inputs"], w["74"]["inputs"]
    settings = dict(seed=sampler["seed"], model=w["78"]["inputs"]["unet_name"],
        model_sha256=model_hash, prompt=w["77"]["inputs"]["text"],
        negative_prompt=w["75"]["inputs"]["text"], sampler=sampler["sampler_name"],
        scheduler=sampler["scheduler"], num_steps=sampler["steps"], cfg=sampler["cfg"],
        denoise=sampler["denoise"], text_encoder_sha256=clip_hash, vae_sha256=vae_hash,
        width=shape["width"], height=shape["height"], batch_size=shape["batch_size"])
    manifest = dict(schema_version=1, run_id=next(iter(run_ids)), source_revision=revision,
        workflow_sha256=sha256(run / "workflow.json"), generation_settings=settings)
    with (run / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
PY

"$PYTHON" scripts/compare_afm_runs.py \
  "$EVIDENCE/runs/sq512-smoke/V/afm.jsonl" \
  "$EVIDENCE/runs/sq512-smoke/S20/afm.jsonl" \
  --strict-validation \
  --observe-manifest "$EVIDENCE/runs/sq512-smoke/V/manifest.json" \
  --edit-manifest "$EVIDENCE/runs/sq512-smoke/S20/manifest.json" \
  --format csv > "$EVIDENCE/analysis/S20-vs-V.csv.tmp" \
  && mv "$EVIDENCE/analysis/S20-vs-V.csv.tmp" "$EVIDENCE/analysis/S20-vs-V.csv"

"$PYTHON" scripts/summarize_afm_experiments.py \
  --compare "S20-vs-V=$EVIDENCE/analysis/S20-vs-V.csv" \
  --strict-validation --report-md --out-dir "$EVIDENCE/analysis/S20-summary"
```

strict CLI の非ゼロ終了では処理を止め、stderr と終了コードを保存する。失敗した `.tmp` を有効な結果として扱わない。ログ取りを最後まで完了したことと missing/duplicate=0 を確認する。manifest の minimum schema に書かれていない項目を「自動検証済み」と言わない。

現行 strict 比較は行・provenance に加え、同一 run ID の `run_final_summary` がちょうど1件、`sampling_status=completed`、sampling error なし、`telemetry_errors=0`、terminal 後の診断なしを要求する。全 step `0..num_steps-1` の terminal summary、正しい last step、各 step の `spectral_emitted_count`（互換時は defined + undefined count）と実際の診断行数も照合する。`not_cross_attention` / `progress_outside_window` 以外の正カウント fallback は拒否する。新 parser はこの完了証拠を `run_completion_json` として CSV に保存するが、終端が不明な legacy CSV は strict に通らない。初回の入力には raw JSONL を推奨する。

fallback 合計と理由別合計の一致も必要であり、strict summary は compare CSV に残された terminal coverage を再照合する。CSV の行を削って coverage を隠せない。現状の保守的 allowlist では `branch_not_selected` も strict 拒否となるため、分割 CFG の片 branch 編集で意味的に正当な除外が起きても、意図した分岐との証拠照合が実装されるまでは strict 合格とはしない。これは明示的な対応範囲の制約として報告する。

それでも**事前に計画した実験の成功を全て自動証明するものではない**。ComfyUI history の成功、事前に期待した evaluation/block/branch の coverage、意図した call の実編集を別途照合する。manifest の検証は提出された値の内部整合性であり、モデルや workflow の実ファイル hash を CLI が再計算するわけではない。上の hash コマンドと実ファイル照合を省かない。失敗・ガード・中断を注入した試験は正常実験と別に保存し、strict 拒否を期待する負例として扱う。

`generation_settings` は追加 key も含むオブジェクト全体を一致比較する。取得済みの noise/latent/sigma hash、実 backend/dtype、diagnostic sampling の設定は追加してよい。treatment として変える scope の fingerprint は manifest top-level に保存し、意図した差分として別に確認する。matched-scope 比較では実対象 block/call の集合が同じことを確認する。

片 branch 保全では、別途作成した positive-only の P と matched observe V の strict 比較 CSV を使用し、次のように明示的に対象名を指定する。両 branch を編集する S20 を preservation run として代用しない。negative-only も別系列で行う。

```bash
"$PYTHON" scripts/summarize_afm_experiments.py \
  --compare "P-vs-V=$EVIDENCE/analysis/P-vs-V.csv" \
  --strict-validation --require-branch-preservation --preservation-run P-vs-V \
  --report-md --out-dir "$EVIDENCE/analysis/P-preservation"
```

strict provenance と正しい scope coverage が揃うまで、legacy 互換モードを新規データの合格判定に使わない。比較した全行の metric と centering 定義も一致させ、現行の `centered_concentration_hf_energy_ratio` / `spatial_mean_subtracted` と、定義のない旧 rho を同一系列に混ぜない。中心化後の全 FFT energy が `1e-8` 以下または非有限の場合は `metric_defined=false`, `metric_undefined_reason=zero_or_nonfinite_centered_energy` として扱う。例えば top-K が全 text tokens を覆う concentration map は定数になり得る。互換性のため数値欄が残っていても、有効な測定に置き換えず strict 拒否・集計除外の理由を残す。

`rho` は attention concentration の内部スペクトル比であり、画像の精細さ・自然さ・画質の尺度ではない。非選択 branch の別 run 間 `delta_rho_vs_observe` は軌道変化で非ゼロになり得る。分岐保全では local の各行、非有限、最大絶対差、coverage を調べ、符号付き平均が相殺されたことを合格にしない。診断 call の標本から未観測の全 block へ一般化しない。

## 7. 正しさの後に行う盲検・複数 seed 評価

### 事前に固定する主実験

- prompt 6 種類以上: 細部/反復柄、建築/直線、顔/手、葉/自然、低周波の空・グラデーション、意図的に柔らかい/低詳細な場面。成功しそうな画像だけで選ばない。
- 各 prompt に 10 seeds、固定 noise を使い、O/Z/S10/S20/S35 の 5 条件を揃える。計 300 枚。全条件が揃わなかった組を黙って除外せず失敗率も報告する。
- main scope・shape・sampler を一つに固定してから始める。横長/縦長、別 scope、片 branch は別事前定義 stratum/ablation とし、混ぜて見かけの標本数を増やさない。
- 24 steps を初期候補とし、正しさ試験後、画質結果を見る前に固定する。cutoff 0.25、curve、entropy off、preserve_dc true、同一 mask を主条件にする。

生成順と提示順を分けて counterbalance する。5 条件の基本順を `O,Z,S35,S10,S20` とし、条件ラベルを巡回置換した 5 順と、それぞれの逆順の 5 順を 10 seeds に割り当てる。prompt ごとに割当を回転させる。この割当表を実行前に保存する。左右位置は独立に均衡化し、盲検 ID に strength・AFM 有無・実行順を含めない。評価完了まで対応表を評価者へ見せない。

比較は O 対各 S と Z 対各 S を区別し、O 対 Z も評価する。各対の「総合好み」「prompt 適合」「過剰な輪郭/halo・模様崩れ・ノイズ」「細部の自然さ」を同点可で記録する。sharpness だけを最大化しない。少なくとも独立した複数評価者、共通の提示サイズ/表示条件を使い、全画像と除外理由を保存する。

統計単位は prompt×seed の生成組であり、step・head・block・画像 crop を独立サンプルとして数えない。prompt/seed を保持した集計・不確実性区間を報告し、複数 strength の探索で最良条件を選んだ場合は別 holdout で再評価する。小規模結果の区間が広ければ「未確定」とする。

### 任意の機序対照

curve strength `s` に対し `schedule=constant`, `strength=s/2`, `entropy_gate=false` を追加する。preserve_dc と対象 scope は同一にする。これは curve 中点の非 DC gain に合わせた**周波数選択性を持たない増幅対照**であり、全軌道の作用量が等しい保証ではない。`s=0.2` なら constant 0.1。cutoff/scope/branch、entropy on/off は一度に一軸ずつ変え、主実験と区別する。

## 8. 証拠一式と終了条件

実験 group ごとに次をまとめる。生ログを後から編集して欠測を埋めない。

- `preflight/`: commits、差分、依存パッケージ、モデル/ソース hash、実機/起動設定
- `cpu/`: CPU テストの全文・終了コード。実機結果と別扱い
- `runs/<group>/<condition>/`: workflow/request/response/history、raw JSONL と console、noise/latent/PNG、run ID と provenance manifest、status、計測値
- `analysis/`: strict 検証のコマンド・終了コード・stderr、照合 coverage、分岐 tensor 差、shape/topology、lifecycle と異常系結果
- `quality/`: 事前固定の設計、順序表、blind ID、評価票、全画像、集計、不確実性、失敗/除外理由
- group 全体の SHA-256 一覧と、`pass / fail / not_run / unsupported` を分けた短い結果文書

**即停止**: 誤った H/W で編集、observe/off 出力破壊、非選択 branch tensor 改変、run を跨ぐ状態漏れ、対象外 block の編集、guard より前の過大 allocation、非有限の成功扱い、出所不明/欠測の strict 解析通過、実測 VRAM の安全余裕不足。

**正しさの完了**: 対象とした ComfyUI/model/sampler/shape/dtype/backend の組で A〜E と異常系が全て合格し、未対応組が明示され、fresh/cached の証拠と実測性能が揃ったとき。CPU のみ、画像が出たのみ、ログに `edited>0` のみでは未完了。

**品質の結論**: 上記を通過した組の事前定義した比較結果からのみ述べる。「変化した」「内部 rho が増えた」「特定例で好み」は「一般的に改善した」や「論文の再現」と分ける。画質改善が支持されなくても、その否定的・未確定結果とコストを残して検証を終了する。
