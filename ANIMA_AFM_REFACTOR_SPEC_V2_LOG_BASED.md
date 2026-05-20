# Anima AFM Refactor / Debug Specification V2

This version incorporates the pre-refactor ComfyUI runtime log captured in `Using split attention in VAE.txt`.

## 0. Executive summary

The log shows that the current AFM node is installed and reaches at least one eligible Anima/Cosmos cross-attention call on every denoising step. The edited call shape is stable:

```text
q=(2, 16, 4096, 128)
k=(2, 16, 512, 128)
v=(2, 16, 512, 128)
spatial=(64, 64)
branch_mode=both
edited_slices=2
steps logged: 0..23, 24 denoising steps
```

This is good news: the current square-grid path is being exercised on a real Anima run. The refactor priority should therefore change slightly:

1. Keep square-grid T2I support as the immediate target.
2. Fix progress indexing, because the current final logged step is `u=0.9583` instead of `u=1.0` for a 24-step run.
3. Add step-level fallback histograms, because fallback count grows to `645` but the log does not say why.
4. Add branch-preserving output merge before testing `positive_only` / `negative_only`.
5. Add post-softmax spectral diagnostics so the log can confirm AFM's paper-level effect, not only raw logit delta.
6. Add original-backend baseline / observe mode because the environment uses PyTorch/AOTriton efficient attention, while the AFM path currently replaces attention with manual matmul + softmax.

Rectangular/video layout support is still important for later Cosmos workflows, but this log confirms the current Anima text-to-image workflow is square `64 x 64`; rectangular/video should move behind the correctness/debugging tasks above.

---

## 1. Findings from the pre-refactor log

### 1.1 The node is installed and active

The log contains:

```text
[AnimaAFM] installed model patch strength=0.2 cutoff=0.25 schedule=curve branch_mode=both
```

Then, during sampling, AFM logs one success line per denoising step. This means the override is installed and at least one attention call per step passes the current eligibility checks.

### 1.2 The current workflow is square-grid text/image cross-attention

The logged eligible call is always:

```text
q=(2, 16, 4096, 128)
k=(2, 16, 512, 128)
v=(2, 16, 512, 128)
spatial=(64, 64)
```

Interpretation:

- `query_len=4096 = 64 * 64`, so the existing square layout inference is valid for this run.
- `text_len=512`, so this is likely image-query to text-token cross-attention.
- `batch=2`, probably CFG cond/uncond batch.
- `heads=16`, `head_dim=128`, and q/k/v head counts match. GQA is not exercised in this log.

### 1.3 The current progress schedule is off by one

The run has 24 denoising iterations, logged as `step=0/24` through `step=23/24`. The final AFM line is:

```text
step=23/24 u=0.9583 ... alpha_lf=1.0083 alpha_hf=1.1917
```

For a 24-step sampler, the paper convention is `u = s / (S - 1)` with `s in {0, ..., S-1}`. Therefore the last model call should be `u=1.0`, `alpha_lf=1.0`, `alpha_hf=1.2` for `strength=0.2` and `schedule=curve`.

Current code uses:

```python
total = max(int(candidates.numel()) - 1, 1)
progress = index / total
```

When `sample_sigmas` contains an extra terminal sigma, `candidates.numel() - 1` is the number of model steps, not the last model index. The denominator should be `num_model_steps - 1`.

### 1.4 Fallback count is high but opaque

Fallback count evolves approximately as:

```text
step 0:  fallbacks=1
step 1:  fallbacks=29
step 2:  fallbacks=57
...
step 23: fallbacks=645
```

This likely means many override calls are intentionally skipped, such as self-attention, unsupported signatures, masked attention, non-cross-attention, or non-target shapes. That is not automatically bad. However, the current summary log only gives cumulative fallback count, not reason histograms or shape categories, so the run is not debuggable.

### 1.5 The log proves numeric editing, but not AFM's intended spectral effect

`max_logit_delta` is nonzero and follows the schedule qualitatively:

```text
step 0:  max_logit_delta=0.457427
step 12: max_logit_delta=0.223543
step 23: max_logit_delta=0.174802
```

This proves the logits are being modified. It does not prove that post-softmax top-K concentration spectra or HF ratio `rho_s` changed in the AFM sense. The paper diagnoses AFM on post-softmax attention-derived concentration maps, not raw logit deltas.

### 1.6 Current runtime uses efficient original attention

The log includes a PyTorch warning from `scaled_dot_product_attention` using an AOTriton backend. Therefore, when `strength=0.0`, returning manual fp32 matmul + softmax is not a strict baseline for the live workflow. The node needs an `observe` or `original_on_zero` mode.

---

## 2. Updated priority list

### P0-A. Fix denoising progress indexing

Current issue:

- For 24 model steps, final progress logs as `u=0.9583`.
- AFM schedule never reaches its intended terminal value.

Required behavior:

```python
@dataclass(frozen=True)
class ProgressInfo:
    index: int              # 0-based model-call index
    num_steps: int          # number of model calls, e.g. 24
    last_index: int         # num_steps - 1, e.g. 23
    progress: float         # index / last_index
    sigma: float
```

Implementation:

```python
num_sigmas = int(candidates.numel())
num_steps = max(num_sigmas - 1, 1)      # terminal sigma is not a model call
last_index = max(num_steps - 1, 1)
index = min(index, num_steps - 1)
progress = min(max(index / last_index, 0.0), 1.0)
return ProgressInfo(index=index, num_steps=num_steps, last_index=last_index, progress=progress, sigma=float(sigma.item()))
```

Logging should use unambiguous names:

```text
step_index=23 num_steps=24 last_index=23 u=1.0000
```

Test updates:

```python
sample_sigmas = torch.tensor([1.0, 0.5, 0.0])
sigma = torch.tensor([0.5])
# This is the second and final model step in a 2-step run.
assert info.index == 1
assert info.num_steps == 2
assert info.last_index == 1
assert info.progress == 1.0
```

Also test:

```python
sample_sigmas = torch.linspace(1, 0, 25)
# index 23 should produce progress 1.0 for a 24-step run.
```

### P0-B. Preserve unselected CFG branches exactly

Current issue:

The code edits only selected logits, but recomputes attention for the entire batch:

```python
logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)
selected_logits = logits.index_select(0, selected)
edited_logits = logits.clone()
edited_selected = edit_logits_fft(selected_logits, spatial_shape, alpha_lf, alpha_hf, self.config)
edited_logits.index_copy_(0, selected, edited_selected)
out = attention_from_edited_logits(edited_logits, v, int(heads), ...)
```

For `branch_mode=positive_only` or `negative_only`, non-selected CFG slices are not AFM-edited, but they are still replaced by manual attention output. This can alter CFG behavior and breaks baseline comparisons.

Required behavior:

- If all batch slices are selected, compute only AFM output and return it.
- If only some slices are selected:
  - Call `original_func(*args, **kwargs)` once.
  - Compute AFM only for selected batch slices.
  - Merge AFM-selected output into the original output along batch dimension.
  - Non-selected output must be exactly the original backend output.

Pseudo-code:

```python
all_selected = selected.numel() == batch
base_out = None if all_selected else original_func(*args, **kwargs)

if all_selected:
    q_sel, k_sel, v_sel = q, k, v
else:
    q_sel = q.index_select(0, selected)
    k_sel = k.index_select(0, selected)
    v_sel = v.index_select(0, selected)

q_sel, k_sel, v_sel, gqa_info = maybe_repeat_gqa(q_sel, k_sel, v_sel, kwargs)
logits_sel = torch.matmul(q_sel.float(), k_sel.float().transpose(-2, -1)) * float(scale)

edited_logits_sel = edit_logits_fft(logits_sel, spatial_shape, alpha_lf, alpha_hf, self.config)
out_sel = attention_from_edited_logits(
    edited_logits_sel,
    v_sel,
    int(q_sel.shape[1]),
    bool(kwargs.get("skip_output_reshape", False)),
)

if base_out is None:
    return out_sel
return base_out.index_copy(0, selected, out_sel.to(dtype=base_out.dtype))
```

Test requirements:

- Use `branch_mode="positive_only"` and `cond_or_uncond=[1, 0]` with batch 4.
- Original function should return a sentinel output that makes exact branch preservation easy to assert.
- Assert non-selected slices exactly match `original_func` output.
- Assert selected slices differ from original when `strength > 0`.
- Repeat for `negative_only`.

### P0-C. Add `mode`: `edit`, `observe`, `off`

Current issue:

README says `strength=0.0` should match baseline, but live ComfyUI uses efficient original attention. Manual fp32 attention is not a strict backend match.

Add config:

```python
AFM_MODES = ["edit", "observe", "off"]
mode: str = "edit"
zero_strength_mode: str = "observe"  # choices: observe, original, manual
```

Behavior:

- `mode="off"`: return `original_func` immediately.
- `mode="observe"`: run eligibility/layout/progress/branch/logging checks, then return `original_func`; do not materialize full logits unless diagnostics explicitly request it.
- `mode="edit"` and `abs(strength) == 0`:
  - `zero_strength_mode="observe"`: behave like observe.
  - `zero_strength_mode="original"`: return original immediately, minimal logging.
  - `zero_strength_mode="manual"`: old test-only behavior.

Recommended node defaults:

```text
mode=edit
zero_strength_mode=observe
spectral_diag=off
```

Validation workflow after this change:

```text
A. No AFM node: true baseline.
B. AFM node mode=observe strength=0.0: should match baseline image and performance very closely.
C. AFM node mode=edit strength=0.1: should show AFM changes.
```

### P0-D. Step-level debug aggregation and fallback histogram

Current issue:

The log shows `fallbacks=645`, but not why. Summary logs are emitted only once per step because `_log_success()` suppresses subsequent success logs for the same `progress.index`. This can hide multiple eligible calls per step.

Add per-step stats:

```python
@dataclass
class StepStats:
    index: int
    num_steps: int
    sigma: float
    total_calls: int = 0
    eligible_calls: int = 0
    edited_calls: int = 0
    observed_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    shape_counts: Counter[str] = field(default_factory=Counter)
    block_counts: Counter[str] = field(default_factory=Counter)
    selected_counts: Counter[str] = field(default_factory=Counter)
    max_logit_delta: float = 0.0
    max_attn_delta: float | None = None
    rho_deltas: list[float] = field(default_factory=list)
    max_estimated_logits_mib: float = 0.0
```

Logging strategy:

- In `summary`, emit:
  - The first edited/observed eligible call per step, including q/k/v shape.
  - A compact step aggregate when the step changes.
- In `verbose`, emit first `N` examples per fallback reason with q/k/v shape and transformer option keys.

Example target summary line:

```text
[AnimaAFM] step_summary step_index=23 num_steps=24 u=1.0000 \
  calls=29 eligible=1 edited=1 observed=0 fallbacks=28 \
  fallback_reasons={not_cross_attention:16, not_skip_reshape:8, mask_shape_unsupported:4} \
  shapes={(q4096,k512,h16,d128):1, (q4096,k4096,h16,d128):16, ...} \
  selected=2/2 cond_or_uncond=[1,0] branch_mode=both \
  max_logit_delta=0.17 rho_delta=-0.04 estimated_logits_mib=256
```

The exact fallback reasons are examples. Do not hard-code assumptions; collect the actual reasons.

### P0-E. Log branch layout and selected indices

Current log only says `edited_slices=2`. For `branch_mode=both` this is fine, but it does not tell whether ComfyUI orders CFG as `[uncond, cond]`, `[cond, uncond]`, or something else.

Add fields:

```text
cond_or_uncond=[...]
branch_mode=...
selected_indices=[...]
selected_count=...
batch=...
```

For safety, if `branch_mode != both` and `cond_or_uncond` is missing or ambiguous, fallback with:

```text
branch_layout_unknown
```

Do not silently select all slices for `positive_only` / `negative_only` when branch layout is unknown.

### P0-F. Add GQA/head mismatch handling

This log does not exercise GQA because q/k/v all have 16 heads. Keep GQA as P0 because Anima/Cosmos variants may switch attention layouts.

Required helper:

```python
@dataclass(frozen=True)
class GQAInfo:
    enabled: bool
    repeats: int
    q_heads: int
    kv_heads_before: int
    kv_heads_after: int


def maybe_repeat_gqa(q, k, v, kwargs):
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    v_heads = int(v.shape[1])
    enable_gqa = bool(kwargs.get("enable_gqa", False))

    if q_heads == k_heads == v_heads:
        return q, k, v, GQAInfo(False, 1, q_heads, k_heads, k_heads)

    if not enable_gqa:
        raise AFMFallback("head_mismatch")
    if k_heads != v_heads or q_heads % k_heads != 0:
        raise AFMFallback("gqa_head_mismatch")

    repeats = q_heads // k_heads
    return (
        q,
        k.repeat_interleave(repeats, dim=1),
        v.repeat_interleave(repeats, dim=1),
        GQAInfo(True, repeats, q_heads, k_heads, q_heads),
    )
```

### P0-G. VRAM estimate and guard

The observed shape requires large temporary tensors.

For the logged edited call:

```text
B * H * Q * T = 2 * 16 * 4096 * 512 = 67,108,864 elements
fp32 logits ~= 256 MiB
complex64 FFT spectrum ~= 512 MiB
```

Current code additionally clones full logits and materializes selected logits. Peak overhead can easily exceed 1 GiB per edited call.

Add:

```python
max_logits_mib: float = 1024.0  # advanced UI input, default conservative
```

Estimate:

```python
def estimate_logits_mib(batch, heads, query_len, text_len, bytes_per=4):
    return batch * heads * query_len * text_len * bytes_per / (1024 ** 2)
```

If estimate exceeds guard:

- `fail_mode="fallback"`: fallback reason `vram_guard_exceeded`.
- `fail_mode="raise"`: raise with shape and estimate.

The branch-preserving selected-only implementation should reduce memory for `positive_only` / `negative_only`.

---

## 3. P1 validity diagnostics

### P1-A. Add post-softmax spectral diagnostics

Current `max_logit_delta` proves logit editing, but not the AFM paper-level effect. Add optional diagnostics:

```python
spectral_diag: str = "off"  # off, sampled, full
diagnostic_top_k: int = 8
diagnostic_max_batches: int = 1
diagnostic_max_heads: int = 4
```

Helper:

```python
def topk_concentration_map_from_logits(logits, top_k):
    probs = torch.softmax(logits.float(), dim=-1)
    k = min(int(top_k), int(probs.shape[-1]))
    top_vals = torch.topk(probs, k=k, dim=-1).values
    return top_vals.mean(dim=-1)  # [B, heads, Q]


def hf_ratio_from_concentration(conc, spatial_shape, cutoff):
    # conc: [B, heads, Q]
    h, w = spatial_shape
    maps = conc.reshape(conc.shape[0], conc.shape[1], h, w)
    maps = maps - maps.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fft2(maps, dim=(-2, -1))
    power = fft.abs().square()
    power = power / power.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    _, high = radial_low_high_masks(h, w, cutoff, False, 0.0, conc.device, power.dtype)
    return (power * high).sum(dim=(-2, -1)).mean()
```

Log fields:

```text
rho_before=...
rho_after=...
delta_rho=...
attn_delta_mean=...
attn_delta_max=...
entropy_before=...
```

Use `spectral_diag="sampled"` by default for debug runs only, e.g. first batch item and first 4 heads, to avoid huge overhead.

### P1-B. Add target filters after logging actual block metadata

The paper applies AFM to encoder cross-attention in SD U-Net, but Anima/Cosmos is DiT-like. The current log lacks block metadata, so we cannot claim the edited call corresponds to the paper's encoder scope.

Add logging of:

```text
transformer_options_keys
block
block_index
patches_replace keys
attn_name if present
module path if available
```

Add filters:

```python
target_query_lens: str = ""       # comma-separated, e.g. "4096"
target_text_lens: str = ""        # e.g. "512"
target_blocks: str = ""           # optional pattern list
target_strategy: str = "all_eligible"  # all_eligible, query_text_shape, block_pattern
```

For the current log-based workflow, a useful controlled test is:

```text
target_strategy=query_text_shape
target_query_lens=4096
target_text_lens=512
```

### P1-C. Alpha safety

Entropy gating with negative strength can produce negative alphas. Add:

```python
min_alpha: float = 0.05
max_alpha: float = 4.0
alpha_policy: str = "warn_and_clamp"  # warn_and_clamp, fallback, allow
```

Log when clamp occurs.

### P1-D. Simple mask support

Current code falls back on any mask. Add support only if the mask is clearly additive/padding-compatible with logits:

- accepted shapes broadcastable to `[B, heads, Q, T]`
- values are additive logits mask or boolean mask

If unsupported, keep fallback with shape in verbose log.

---

## 4. Updated tests

Add or update these tests in `test_anima_afm.py`.

### 4.1 Progress indexing

```python
def test_progress_last_model_step_reaches_one():
    info = progress_from_sigmas({
        "sigmas": torch.tensor([0.5]),
        "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
    })
    assert info.index == 1
    assert info.num_steps == 2
    assert info.last_index == 1
    assert info.progress == 1.0


def test_progress_24_steps_last_index():
    sample_sigmas = torch.linspace(1.0, 0.0, 25)
    sigma = sample_sigmas[23:24]
    info = progress_from_sigmas({"sigmas": sigma, "sample_sigmas": sample_sigmas})
    assert info.index == 23
    assert info.num_steps == 24
    assert info.last_index == 23
    assert info.progress == 1.0
```

### 4.2 Branch preservation

```python
def test_positive_only_preserves_negative_branch_original_backend():
    # cond_or_uncond=[1,0], batch=4 => positive indices [2,3]
    # non-selected [0,1] must exactly equal original_func output
```

Repeat for `negative_only`.

### 4.3 Observe mode

```python
def test_observe_mode_returns_original_and_records_eligible():
    original = CountingAttention()
    override = AnimaAFMAttentionOverride(AFMConfig(mode="observe", strength=0.2))
    out = override(original, q, k, v, heads, skip_reshape=True, transformer_options=...)
    expected = original.reference_output
    assert torch.equal(out, expected)
    assert override.stats.observed_calls == 1
    assert override.stats.edited_calls == 0
```

### 4.4 GQA

```python
def test_gqa_repeats_kv_heads():
    q: [2, 8, 16, 4]
    k/v: [2, 2, 5, 4]
    kwargs={"enable_gqa": True, ...}
    assert no fallback
    assert output shape is valid
```

### 4.5 Spectral diagnostics smoke test

```python
def test_spectral_diag_reports_rho_delta():
    config = AFMConfig(strength=0.2, spectral_diag="sampled")
    out = override(...)
    assert last_step_stats.rho_before is not None
    assert last_step_stats.rho_after is not None
```

---

## 5. Updated README validation protocol

Replace the current initial validation section with this order:

```text
1. True baseline: no AFM node.
2. Observe baseline: AFM node mode=observe, strength=0.0, debug_level=summary.
   Expected: image should match true baseline; log should show eligible shapes but edited=0.
3. Manual no-op diagnostic: mode=edit, strength=0.0, zero_strength_mode=manual only for testing.
   Expected: may differ slightly from true baseline due to manual attention backend; do not use as image baseline.
4. Edit sweep: mode=edit, entropy_gate=false, strength=0.05, 0.1, 0.2.
5. Branch sweep after branch preservation fix: branch_mode=both, positive_only, negative_only.
6. Spectral debug: spectral_diag=sampled for one short run only.
```

Expected log for the current workflow after progress fix:

```text
step_index=23 num_steps=24 last_index=23 u=1.0000
q=(2,16,4096,128) k=(2,16,512,128) spatial=(64,64)
alpha_lf=1.0000 alpha_hf=1.2000
```

---

## 6. Codex prompt

```text
Refactor the Anima AFM ComfyUI custom node according to ANIMA_AFM_REFACTOR_SPEC_V2_LOG_BASED.md.

Prioritize in this exact order:
1. Fix progress_from_sigmas so a 24-step run logs final u=1.0, not 0.9583.
2. Add mode={edit, observe, off} and zero_strength_mode={observe, original, manual}; observe/original must return original_func output.
3. Preserve unselected CFG branches exactly for positive_only/negative_only by merging selected AFM output into original_func output.
4. Add per-step fallback reason histograms and branch-layout logging: cond_or_uncond, selected_indices, selected_count.
5. Add GQA/head mismatch support.
6. Add VRAM estimate/guard.
7. Add optional sampled post-softmax spectral diagnostics: rho_before, rho_after, delta_rho, attn_delta_mean/max.
8. Update tests and README.

Keep the current square-grid AFM edit path working for q=(B,16,4096,128), k/v=(B,16,512,128), spatial=(64,64).
Do not implement rectangular/video editing yet; only add discovery logging for it.
Do not change AFM FFT math except where needed for selected-branch computation, diagnostics, mask support, or safety checks.

After implementation, report:
- Changed files
- Test results
- A sample expected debug log for the uploaded pre-refactor workflow
- Any assumptions about Anima/Cosmos transformer_options metadata
```

---

## 7. Acceptance criteria

The refactor is accepted when:

1. A 24-step run logs final `u=1.0000` and `alpha_hf=1.2000` for `strength=0.2`, `schedule=curve`, `entropy_gate=false`.
2. `mode=observe`, `strength=0.0` returns original attention output and logs eligible calls without editing.
3. `positive_only` and `negative_only` preserve non-selected branches exactly relative to `original_func`.
4. Summary debug prints fallback reason histograms, not only cumulative fallback count.
5. The current square shape `q=(2,16,4096,128)`, `k=(2,16,512,128)` remains editable.
6. Optional spectral diagnostics can log `rho_before`, `rho_after`, and `delta_rho` for a sampled subset.
7. Tests cover progress indexing, observe mode, branch preservation, GQA, and spectral diagnostics smoke behavior.
