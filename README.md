# ComfyUI Anima AFM

Experimental ComfyUI custom node for Attention Frequency Modulation (AFM) on Anima/Cosmos DiT cross-attention.

This node is Anima-only. It does not implement a Stable Diffusion 1.5 U-Net path.

## Node

`Anima AFM Model Patch`

Connect it after the Anima model loader and before sampling.

## Initial Validation

Use a fixed seed, fixed prompt, fixed sampler, and fixed latent.

Recommended sweep:

```text
strength: -0.2, 0.0, 0.1, 0.2, 0.4
schedule: curve
branch_mode: both, then positive_only
entropy_gate: false first
```

`strength=0.0` should match the baseline. Enable `debug_level=summary` to confirm that Anima/Cosmos cross-attention calls are edited and to inspect inferred tensor shapes.

## Limits

- The MVP only edits square image query grids.
- Non-square/video query layouts fall back to the original attention.
- If another node already installs `optimized_attention_override`, this node raises a clear error.
- Full pre-softmax logits are materialized, so high resolutions may need a later chunked implementation.

