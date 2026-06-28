from __future__ import annotations

import logging

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .anima_afm import (
    AFMConfig,
    AFM_MODES,
    AnimaAFMAttentionOverride,
    BRANCH_MODES,
    DEBUG_FORMATS,
    DEBUG_LEVELS,
    DIAGNOSTIC_BRANCHES,
    FAIL_MODES,
    SCHEDULES,
    SCOPE_MODES,
    SPATIAL_SHAPE_MODES,
    SPECTRAL_DIAG_MODES,
    ZERO_STRENGTH_MODES,
    install_anima_block_metadata_wrappers,
    is_anima_like_model,
)


LOGGER = logging.getLogger(__name__)


class AnimaAFMModelPatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnimaAFMModelPatch",
            display_name="Anima AFM Model Patch",
            category="model_patches/anima",
            description="Training-free AFM patch for Anima/Cosmos DiT cross-attention logits.",
            search_aliases=["anima afm", "attention frequency modulation", "cross attention fft"],
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("mode", options=AFM_MODES, default="edit"),
                io.Float.Input("strength", default=0.2, min=-1.0, max=2.0, step=0.01),
                io.Float.Input("cutoff", default=0.25, min=0.01, max=0.95, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("schedule", options=SCHEDULES, default="curve"),
                io.Combo.Input("branch_mode", options=BRANCH_MODES, default="both"),
                io.Combo.Input("zero_strength_mode", options=ZERO_STRENGTH_MODES, default="observe", advanced=True),
                io.Boolean.Input("entropy_gate", default=False),
                io.Float.Input("beta", default=20.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Float.Input("gamma", default=4.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Boolean.Input("preserve_dc", default=True, advanced=True),
                io.Boolean.Input("soft_mask", default=True, advanced=True),
                io.Float.Input("mask_width", default=0.05, min=0.0, max=0.5, step=0.01, advanced=True),
                io.Combo.Input("debug_level", options=DEBUG_LEVELS, default="off", advanced=True),
                io.Combo.Input("debug_format", options=DEBUG_FORMATS, default="text", advanced=True),
                io.Combo.Input("fail_mode", options=FAIL_MODES, default="fallback", advanced=True),
                io.Float.Input("max_logits_mib", default=1024.0, min=1.0, max=65536.0, step=16.0, advanced=True),
                io.Float.Input("max_peak_mib", default=4096.0, min=1.0, max=262144.0, step=16.0, advanced=True),
                io.String.Input("target_call_indices", default="all", advanced=True),
                io.Combo.Input("scope_mode", options=SCOPE_MODES, default="block_scope", optional=True, advanced=True),
                io.String.Input("scope_map_path", default="", optional=True, advanced=True),
                io.String.Input("block_scope", default="", optional=True, advanced=True),
                io.String.Input("stage_scope", default="early", optional=True, advanced=True),
                io.Combo.Input("spectral_diag", options=SPECTRAL_DIAG_MODES, default="off", advanced=True),
                io.String.Input("diagnostic_call_indices", default="0", advanced=True),
                io.Boolean.Input("diagnostic_include_unselected", default=False, advanced=True),
                io.Int.Input("diagnostic_every_n_steps", default=1, min=1, max=100, step=1, advanced=True),
                io.Combo.Input("diagnostic_branch", options=DIAGNOSTIC_BRANCHES, default="both_separate", advanced=True),
                io.String.Input("jsonl_path", default="", advanced=True),
                io.Int.Input("max_verbose_fallbacks_per_step_per_reason", default=3, min=0, max=100, step=1, advanced=True),
                io.Combo.Input("spatial_shape_mode", options=SPATIAL_SHAPE_MODES, default="auto", optional=True, advanced=True),
                io.Int.Input("image_width", default=0, min=0, max=65536, step=8, optional=True, advanced=True),
                io.Int.Input("image_height", default=0, min=0, max=65536, step=8, optional=True, advanced=True),
                io.Int.Input("latent_width", default=0, min=0, max=65536, step=1, optional=True, advanced=True),
                io.Int.Input("latent_height", default=0, min=0, max=65536, step=1, optional=True, advanced=True),
                io.Int.Input("latent_downscale", default=16, min=1, max=256, step=1, optional=True, advanced=True),
                io.Float.Input("aspect_tolerance", default=0.001, min=0.0, max=1.0, step=0.0001, optional=True, advanced=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        mode,
        strength,
        cutoff,
        start_percent,
        end_percent,
        schedule,
        branch_mode,
        zero_strength_mode,
        entropy_gate,
        beta,
        gamma,
        preserve_dc,
        soft_mask,
        mask_width,
        debug_level,
        debug_format,
        fail_mode,
        max_logits_mib,
        max_peak_mib,
        target_call_indices,
        scope_mode="block_scope",
        scope_map_path="",
        block_scope="",
        stage_scope="early",
        spectral_diag="off",
        diagnostic_call_indices="0",
        diagnostic_include_unselected=False,
        diagnostic_every_n_steps=1,
        diagnostic_branch="both_separate",
        jsonl_path="",
        max_verbose_fallbacks_per_step_per_reason=3,
        spatial_shape_mode="auto",
        image_width=0,
        image_height=0,
        latent_width=0,
        latent_height=0,
        latent_downscale=16,
        aspect_tolerance=0.001,
    ) -> io.NodeOutput:
        if not is_anima_like_model(model):
            raise ValueError("Anima AFM Model Patch requires an Anima MODEL. The connected model does not look Anima-like.")

        config = AFMConfig(
            mode=mode,
            strength=strength,
            cutoff=cutoff,
            start_percent=start_percent,
            end_percent=end_percent,
            schedule=schedule,
            branch_mode=branch_mode,
            zero_strength_mode=zero_strength_mode,
            entropy_gate=entropy_gate,
            beta=beta,
            gamma=gamma,
            preserve_dc=preserve_dc,
            soft_mask=soft_mask,
            mask_width=mask_width,
            debug_level=debug_level,
            debug_format=debug_format,
            jsonl_path=jsonl_path.strip() or None,
            fail_mode=fail_mode,
            max_logits_mib=max_logits_mib,
            max_peak_mib=max_peak_mib,
            target_call_indices=target_call_indices,
            scope_mode=scope_mode,
            scope_map_path=scope_map_path.strip() or None,
            block_scope=block_scope,
            stage_scope=stage_scope,
            spatial_shape_mode=spatial_shape_mode,
            image_width=image_width,
            image_height=image_height,
            latent_width=latent_width,
            latent_height=latent_height,
            latent_downscale=latent_downscale,
            aspect_tolerance=aspect_tolerance,
            spectral_diag=spectral_diag,
            diagnostic_call_indices=diagnostic_call_indices,
            diagnostic_include_unselected=diagnostic_include_unselected,
            diagnostic_every_n_steps=diagnostic_every_n_steps,
            diagnostic_branch=diagnostic_branch,
            max_verbose_fallbacks_per_step_per_reason=max_verbose_fallbacks_per_step_per_reason,
        )
        config.validate()

        patched = model.clone()
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        if "optimized_attention_override" in transformer_options:
            raise ValueError("Anima AFM cannot be combined with an existing optimized_attention_override patch yet.")

        transformer_options["optimized_attention_override"] = AnimaAFMAttentionOverride(config)
        metadata_wrappers = install_anima_block_metadata_wrappers(patched)
        LOGGER.info(
            "[AnimaAFM] installed model patch mode=%s strength=%s cutoff=%s schedule=%s branch_mode=%s zero_strength_mode=%s metadata_wrappers=%s",
            mode,
            strength,
            cutoff,
            schedule,
            branch_mode,
            zero_strength_mode,
            metadata_wrappers,
        )
        return io.NodeOutput(patched)


class AnimaAFMExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [AnimaAFMModelPatch]


async def comfy_entrypoint() -> AnimaAFMExtension:
    return AnimaAFMExtension()

