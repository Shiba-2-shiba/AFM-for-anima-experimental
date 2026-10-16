"""ComfyUI sampling boundaries for AFM's per-run state.

Supported contract: ModelPatcher.add_wrapper_with_key, WrappersMP.OUTER_SAMPLE
(CFGGuider with model_options), and WrappersMP.APPLY_MODEL (transformer_options
is argument 6). Verified against ComfyUI 15ef24d1c0333a3eba56c5cd153d8db65363ff8f.
The installed patch contains only configuration; each outer sampling call owns
its own attention override, statistics, and run identifier.
"""
from __future__ import annotations

import logging
import sys
from uuid import uuid4

from .anima_afm import AFMConfig, AnimaAFMAttentionOverride


LOGGER = logging.getLogger(__name__)
WRAPPER_KEY = "anima_afm_sampling_lifecycle"


class AnimaAFMSamplingLifecycle:
    def __init__(self, config: AFMConfig):
        self.config = config

    def attention_outside_sampling(self, *args, **kwargs):
        raise RuntimeError(
            "Unsupported Anima AFM sampling path: attention ran without the "
            "ComfyUI OUTER_SAMPLE lifecycle wrapper. Use a compatible ComfyUI sampler."
        )

    def outer_sample(self, executor, *args, **kwargs):
        guider = getattr(executor, "class_obj", None)
        previous_options = getattr(guider, "model_options", None)
        if not isinstance(previous_options, dict):
            raise RuntimeError("Unsupported Anima AFM OUTER_SAMPLE contract: missing guider.model_options.")
        transformer_options = previous_options.get("transformer_options")
        if not isinstance(transformer_options, dict):
            raise RuntimeError("Unsupported Anima AFM OUTER_SAMPLE contract: missing transformer_options.")
        patcher = getattr(guider, "model_patcher", None)
        get_extra_models = getattr(patcher, "get_additional_models_with_key", None)
        if previous_options.get("multigpu_clones") or (
            callable(get_extra_models) and get_extra_models("multigpu")
        ):
            raise RuntimeError("Unsupported Anima AFM sampling path: parallel multi-GPU evaluation.")

        override = AnimaAFMAttentionOverride(self.config)
        override.run_id = f"afm-{uuid4().hex}"
        override.managed_lifecycle = True
        override.sampling_status = "running"
        # CFGGuider already cloned these options before OUTER_SAMPLE. Copy only
        # the mappings we replace, leaving other wrappers and sampler arguments
        # untouched. inner_sample's subsequent clone propagates this override.
        run_options = dict(previous_options)
        run_options["transformer_options"] = dict(transformer_options)
        run_options["transformer_options"]["optimized_attention_override"] = override
        guider.model_options = run_options
        sampling_failed = False
        try:
            # Calling execute() here would re-enter this same wrapper.
            result = executor(*args, **kwargs)
            override.sampling_status = "completed"
            return result
        except BaseException as error:
            sampling_failed = True
            management = sys.modules.get("comfy.model_management")
            interrupt_type = getattr(management, "InterruptProcessingException", ())
            interrupted = not isinstance(error, Exception) or isinstance(error, interrupt_type)
            override.sampling_status = "interrupted" if interrupted else "failed"
            override.sampling_error_type = type(error).__name__
            raise
        finally:
            try:
                override.finalize()
            except Exception:
                if not sampling_failed:
                    raise
                LOGGER.exception("[AnimaAFM] finalization failed while preserving the sampler exception")
            finally:
                guider.model_options = previous_options

    def apply_model(self, executor, *args, **kwargs):
        transformer_options = kwargs.get("transformer_options")
        if transformer_options is None and len(args) > 5:
            transformer_options = args[5]
        override = transformer_options.get("optimized_attention_override") if isinstance(transformer_options, dict) else None
        if not isinstance(override, AnimaAFMAttentionOverride) or not override.managed_lifecycle:
            raise RuntimeError("Unsupported Anima AFM APPLY_MODEL path: no active outer sampling run.")
        override.begin_model_evaluation()
        return executor(*args, **kwargs)


def record_post_input_spatial_shape(data):
    """Record the actual Cosmos/Anima B,T,H,W,D patch grid, not latent pixels.

    The pinned post_input hook reads only the returned img. Its options mapping
    must therefore be updated in place for the subsequent blocks to see it.
    """
    options = data.get("transformer_options")
    if not isinstance(options, dict):
        raise RuntimeError("Unsupported Anima AFM post_input contract: missing transformer_options.")
    image = data.get("img")
    shape = getattr(image, "shape", ())
    valid = len(shape) == 5 and int(shape[1]) == 1 and int(shape[2]) > 0 and int(shape[3]) > 0
    actual = [int(shape[2]), int(shape[3])] if valid else [0, 0]
    previous = options.get("afm_spatial_shape")
    previous_is_ours = isinstance(previous, (tuple, list)) and tuple(previous) == options.get("_anima_afm_runtime_shape")
    if previous is not None and not previous_is_ours and previous != actual and previous != tuple(actual):
        candidates = list(options.get("afm_spatial_shape_candidates", []))
        candidates.append({"source": "preexisting_afm_spatial_shape", "afm_spatial_shape": previous})
        options["afm_spatial_shape_candidates"] = candidates
    # Only remove values this hook owned. Preserve user/other-plugin conflicts
    # so automatic shape inference fails closed rather than guessing.
    if options.pop("_anima_afm_invalid_shape_owned", False):
        options.pop("afm_spatial_shape_invalid", None)
    if not valid:
        options["_anima_afm_invalid_shape_owned"] = not options.get("afm_spatial_shape_invalid", False)
        options["afm_spatial_shape_invalid"] = True
    options["afm_spatial_shape"] = actual
    options["_anima_afm_runtime_shape"] = tuple(actual)
    return data


def install_sampling_lifecycle(model_patcher, config: AFMConfig) -> None:
    try:
        from comfy.patcher_extension import WrappersMP
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Unsupported ComfyUI version: Anima AFM requires OUTER_SAMPLE and APPLY_MODEL wrappers.") from error
    register = getattr(model_patcher, "add_wrapper_with_key", None)
    if not callable(register) or not all(hasattr(WrappersMP, name) for name in ("OUTER_SAMPLE", "APPLY_MODEL")):
        raise RuntimeError("Unsupported ComfyUI version: Anima AFM requires ModelPatcher.add_wrapper_with_key, OUTER_SAMPLE, and APPLY_MODEL.")
    register_patch = getattr(model_patcher, "set_model_patch", None)
    if not callable(register_patch):
        raise RuntimeError("Unsupported ComfyUI version: Anima AFM requires ModelPatcher.set_model_patch for the post_input spatial hook.")
    lifecycle = AnimaAFMSamplingLifecycle(config)
    model_patcher.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = lifecycle.attention_outside_sampling
    register(WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, lifecycle.outer_sample)
    register(WrappersMP.APPLY_MODEL, WRAPPER_KEY, lifecycle.apply_model)
    register_patch(record_post_input_spatial_shape, "post_input")
