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
6. Spectral debug: spectral_diag=sampled, diagnostic_branch=both_separate for one short run only.
```

Expected final-step snapshot for a 24-step square Anima text-to-image run with `strength=0.2`, `schedule=curve`, and `entropy_gate=false`:

```text
[AnimaAFM] step_snapshot step_index=23 num_steps=24 last_index=23 u=1.0000 ...
q=(2, 16, 4096, 128) k=(2, 16, 512, 128) ... spatial=(64, 64)
alpha_lf=1.0000 alpha_hf=1.2000
```

`step_snapshot` is emitted at eligible calls. `step_final_summary` is emitted when a later step is first observed, so it includes fallback calls that happened after the snapshot; the last step may need an explicit runtime flush to produce a final summary. Summary logs include per-step fallback reason histograms, CFG branch layout (`cond_or_uncond`), selected indices, selected count, eligible call index, discovered transformer block metadata, and the estimated temporary logits size. Use `mode=observe` to confirm eligible Anima/Cosmos cross-attention shapes without replacing the original attention backend.

Set `debug_format=jsonl` or `both` for machine-readable records. JSONL spectral diagnostics include separate `negative` and `positive` records when `cond_or_uncond=[1, 0]` and `diagnostic_branch=both_separate`.

## Limits

- The MVP only edits square image query grids.
- Non-square/video query layouts fall back to the original attention.
- If another node already installs `optimized_attention_override`, this node raises a clear error.
- Full pre-softmax logits are materialized for edited calls. `max_logits_mib` guards high-memory shapes until a later chunked implementation exists.

