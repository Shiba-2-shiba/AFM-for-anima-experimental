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

`mode=discover` records all eligible cross-attention calls so scope maps can be built without hiding candidates. `mode=observe` respects `target_call_indices` and block/stage scope filters, so it can be used as a candidate-scope baseline before `mode=edit`.

The node defaults are tuned for the current Anima validation path: `mode=edit`, `strength=0.20`, `scope_mode=block_scope`, and `stage_scope=early`. This edits the 10 discovered early cross-attention blocks by default; switch `scope_mode=all` explicitly for all-call paper-like stress tests.

Set `debug_format=jsonl` or `both` for machine-readable records. Set `jsonl_path` to also append those records to a JSONL file while preserving logger output. JSONL records now use `schema_version=2`. Spectral diagnostics include separate `negative` and `positive` records when `cond_or_uncond=[1, 0]` and `diagnostic_branch=both_separate`. Use `diagnostic_call_indices=all`, comma lists, or ranges plus `diagnostic_every_n_steps` to limit spectral work while preserving stable `eligible_call_index` identities. `target_call_indices` and `diagnostic_call_indices` are independent: the former controls edited calls, the latter controls diagnostic work. In `positive_only` or `negative_only`, set `diagnostic_include_unselected=true` to emit passthrough spectral diagnostics for the unedited CFG branch without changing the output.

Schema v2 separates fields that were ambiguous in earlier logs:

- `call_mode` is the whole attention call behavior: `edit`, `observe`, `passthrough`, or `off`. Legacy `mode` remains as a compatibility alias.
- `diagnostic_mode` is the measured branch behavior: `edited`, `observe`, `passthrough`, or `target_skipped`.
- `edit_selected_indices` is the batch slice eligible for editing. Legacy `selected_indices` remains as its alias.
- `diagnostic_batch_indices` is the batch slice measured by that spectral row. Legacy `batch_indices` remains as its alias.
- `edit_applied=false` on an unselected branch means the branch was measured, not edited.
- `rho` is the high-frequency ratio of a post-softmax top-K concentration map. It is not a direct image-frequency measurement.
- `delta_rho_local = rho_after - rho_before` within one run. `delta_rho_vs_observe = rho_edit_after - rho_observe` is produced by compare against an observe baseline.

Use `scripts/parse_anima_afm_log.py` to extract spectral diagnostic rows from JSONL, and `scripts/compare_afm_runs.py` to compare observe-vs-edit rho trajectories by `step_index`, `eligible_call_index`, and branch. Both scripts keep legacy v1 JSONL/CSV compatibility.

```bash
python scripts/parse_anima_afm_log.py logs/observe.jsonl --format csv > out/observe.csv
python scripts/parse_anima_afm_log.py logs/edit.jsonl --format csv > out/edit.csv
python scripts/compare_afm_runs.py logs/observe.jsonl logs/edit.jsonl --format csv > out/compare.csv
python scripts/compare_afm_runs.py out/observe.csv out/edit.csv --input-format csv --summary --late-start-step 16 --format json
```

For multi-run analysis, compare each edit run against the same observe baseline, then aggregate:

```bash
python scripts/compare_afm_runs.py out/A_observe.parsed.csv out/B_call0.parsed.csv \
  --input-format csv --format csv > out/compare_B_call0_vs_A.csv
python scripts/compare_afm_runs.py out/A_observe.parsed.csv out/C_call7-13.parsed.csv \
  --input-format csv --format csv > out/compare_C_call7-13_vs_A.csv
python scripts/compare_afm_runs.py out/A_observe.parsed.csv out/D_positive-only-preservation.parsed.csv \
  --input-format csv --format csv > out/compare_D_positive-only-preservation_vs_A.csv

python scripts/summarize_afm_experiments.py \
  --compare B_call0_vs_A=out/compare_B_call0_vs_A.csv \
  --compare C_call7-13_vs_A=out/compare_C_call7-13_vs_A.csv \
  --compare D_positive-only-preservation_vs_A=out/compare_D_positive-only-preservation_vs_A.csv \
  --late-start-step 16 \
  --preservation-run D_positive-only-preservation_vs_A \
  --require-branch-preservation \
  --report-md \
  --out-dir out/summary
```

`compare_afm_runs.py` reports `duplicate_observe_pairs`, `duplicate_edit_pairs`, and `unmatched_observe_pairs` in summary mode. Use `--fail-on-duplicate-pairs` and `--fail-on-unmatched-observe` when validating a research dataset before aggregation.

In branch-scoped runs, unselected branch diagnostics may have nonzero `delta_rho_vs_observe` because the generation trajectory changed. That does not mean AFM was directly applied to that branch. Use `edit_applied`, `diagnostic_mode`, and tensor-level branch preservation tests to distinguish direct editing from trajectory effects.

## Next Logs

Collect these with the same seed, prompt, sampler, and latent:

```text
A. observe baseline
mode=observe
strength=0.0
branch_mode=both
spectral_diag=sampled
diagnostic_branch=both_separate
diagnostic_call_indices=0,7,14,21,27
debug_format=both

B. call0 edit
mode=edit
strength=0.1
branch_mode=both
target_call_indices=0
spectral_diag=sampled
diagnostic_branch=both_separate
diagnostic_call_indices=0,7,14,21,27
debug_format=both

C. call7-13 edit
mode=edit
strength=0.1
branch_mode=both
target_call_indices=7-13
spectral_diag=sampled
diagnostic_branch=both_separate
diagnostic_call_indices=0,7,14,21,27
debug_format=both

D. positive-only preservation diagnostic
mode=edit
strength=0.1
branch_mode=positive_only
target_call_indices=all
diagnostic_include_unselected=true
spectral_diag=sampled
diagnostic_branch=both_separate
diagnostic_call_indices=0,7,14,21,27
debug_format=both
```

## Limits

- Square image query grids are still the automatic compatibility path.
- Static non-square images are supported when the shape is explicit: set `spatial_shape_mode=explicit_pixels` with matching `image_width` / `image_height`, or `spatial_shape_mode=explicit_latent` with matching `latent_width` / `latent_height`.
- Without explicit dimensions or trusted runtime metadata, non-square layouts fall back to the original attention instead of guessing. Conflicting runtime shape candidates are rejected with `spatial_shape_ambiguous`.
- Current Anima runtime checks did not expose a reliable non-square shape candidate to `auto`; use `explicit_pixels` for 16:9, 9:16, and other non-square static images.
- Video or time-folded query layouts are not supported yet.
- If another node already installs `optimized_attention_override`, this node raises a clear error.
- Full pre-softmax logits are materialized for edited calls. `max_logits_mib` guards logits allocation and `max_peak_mib` guards the conservative full-FFT peak estimate until a later chunked implementation exists.

