# ComfyUI Anima AFM

Experimental ComfyUI custom node for Attention Frequency Modulation (AFM) on Anima/Cosmos DiT cross-attention.

This node is Anima-only. It does not implement a Stable Diffusion 1.5 U-Net path.

## Node

`Anima AFM Model Patch`

Connect it after the Anima model loader and before sampling.

## Initial Validation

Use a fixed seed, fixed prompt, fixed sampler, and fixed latent.

Recommended protocol:

```text
1. True baseline: no AFM node.
2. Observe baseline: AFM node mode=observe, strength=0.0, debug_level=summary, debug_format=both.
   Expected: image should match true baseline; log should show eligible shapes but edited=0.
3. Manual no-op diagnostic: mode=edit, strength=0.0, zero_strength_mode=manual only for testing.
   Expected: may differ slightly from true baseline due to manual attention backend; do not use as image baseline.
4. Edit sweep: mode=edit, entropy_gate=false, strength=0.05, 0.1, 0.2.
5. Branch sweep after branch preservation fix: branch_mode=both, positive_only, negative_only.
6. Spectral debug: spectral_diag=sampled, diagnostic_branch=both_separate, diagnostic_call_indices=0,7,14,21,27 for one short run only.
```

Expected final-step snapshot for a 24-step square Anima text-to-image run with `strength=0.2`, `schedule=curve`, and `entropy_gate=false`:

```text
[AnimaAFM] step_snapshot step_index=23 num_steps=24 last_index=23 u=1.0000 ...
q=(2, 16, 4096, 128) k=(2, 16, 512, 128) ... spatial=(64, 64)
alpha_lf=1.0000 alpha_hf=1.2000
```

`step_snapshot` is emitted at eligible calls. `step_final_summary` is emitted when a later step is first observed, and the final denoising step is finalized automatically once it reaches the inferred per-step call count. Calling the override's `finalize()` method also emits `run_final_summary`. Summary logs include per-step fallback reason histograms, CFG branch layout (`cond_or_uncond`), selected indices, selected count, eligible call index, eligible call-index histograms, discovered transformer block metadata, target-scope skip counts, the estimated temporary logits size, and a conservative peak memory estimate. Final summaries keep scalar `rho_before`, `rho_after`, and `delta_rho` null; use `spectral_diag_count`, `spectral_delta_rho_*`, and `spectral_by_call_branch` for step-level spectral aggregates. Use `mode=observe` to confirm eligible Anima/Cosmos cross-attention shapes and branch-separated spectral baselines without replacing the original attention backend.

Set `target_call_indices=all`, a comma list such as `0,7,14`, or a range such as `7-13` to scope editing by stable per-step eligible call index. Calls outside the target scope return the original attention output and are counted as `target_skipped`, not fallbacks.

Set `debug_format=jsonl` or `both` for machine-readable records. Set `jsonl_path` to also append those records to a JSONL file while preserving logger output. JSONL spectral diagnostics include separate `negative` and `positive` records when `cond_or_uncond=[1, 0]` and `diagnostic_branch=both_separate`. Use `diagnostic_call_indices=all`, comma lists, or ranges plus `diagnostic_every_n_steps` to limit spectral work while preserving stable `eligible_call_index` identities. In `positive_only` or `negative_only`, set `diagnostic_include_unselected=true` to emit passthrough spectral diagnostics for the unedited CFG branch without changing the output.

Use `scripts/parse_anima_afm_log.py` to extract spectral diagnostic rows from JSONL, and `scripts/compare_afm_runs.py` to compare observe-vs-edit rho trajectories by `step_index`, `eligible_call_index`, and branch.

## Limits

- The MVP only edits square image query grids.
- Non-square/video query layouts fall back to the original attention.
- If another node already installs `optimized_attention_override`, this node raises a clear error.
- Full pre-softmax logits are materialized for edited calls. `max_logits_mib` guards logits allocation and `max_peak_mib` guards the conservative full-FFT peak estimate until a later chunked implementation exists.

