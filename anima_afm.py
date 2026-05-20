from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Callable

import torch


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[AnimaAFM]"

SCHEDULES = ["curve", "lf_only", "hf_only", "constant"]
BRANCH_MODES = ["both", "positive_only", "negative_only"]
DEBUG_LEVELS = ["off", "summary", "verbose"]
FAIL_MODES = ["fallback", "raise"]


@dataclass(frozen=True)
class AFMConfig:
    strength: float = 0.2
    cutoff: float = 0.25
    start_percent: float = 0.0
    end_percent: float = 1.0
    schedule: str = "curve"
    branch_mode: str = "both"
    entropy_gate: bool = False
    beta: float = 20.0
    gamma: float = 4.0
    preserve_dc: bool = True
    soft_mask: bool = True
    mask_width: float = 0.05
    debug_level: str = "off"
    fail_mode: str = "fallback"

    def validate(self) -> None:
        if self.schedule not in SCHEDULES:
            raise ValueError(f"Unsupported AFM schedule: {self.schedule!r}")
        if self.branch_mode not in BRANCH_MODES:
            raise ValueError(f"Unsupported AFM branch_mode: {self.branch_mode!r}")
        if self.debug_level not in DEBUG_LEVELS:
            raise ValueError(f"Unsupported AFM debug_level: {self.debug_level!r}")
        if self.fail_mode not in FAIL_MODES:
            raise ValueError(f"Unsupported AFM fail_mode: {self.fail_mode!r}")
        if not 0.0 <= self.start_percent <= 1.0:
            raise ValueError("start_percent must be in [0, 1]")
        if not 0.0 <= self.end_percent <= 1.0:
            raise ValueError("end_percent must be in [0, 1]")
        if self.end_percent < self.start_percent:
            raise ValueError("end_percent must be greater than or equal to start_percent")
        if not 0.0 < self.cutoff < 1.0:
            raise ValueError("cutoff must be in (0, 1)")
        if self.mask_width < 0.0:
            raise ValueError("mask_width must be non-negative")


@dataclass
class AFMRuntimeStats:
    edited_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    summaries_by_step: set[int] = field(default_factory=set)
    verbose_counts_by_step: dict[int, int] = field(default_factory=dict)

    def record_fallback(self, reason: str) -> None:
        self.fallback_calls += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1


@dataclass(frozen=True)
class ProgressInfo:
    index: int
    total: int
    progress: float
    sigma: float


def infer_square_spatial_shape(query_len: int) -> tuple[int, int] | None:
    side = math.isqrt(int(query_len))
    if side * side != query_len:
        return None
    return side, side


def progress_from_sigmas(transformer_options: dict[str, Any]) -> ProgressInfo | None:
    sigmas = transformer_options.get("sigmas")
    sample_sigmas = transformer_options.get("sample_sigmas")
    if sigmas is None or sample_sigmas is None:
        return None
    if not torch.is_tensor(sigmas) or not torch.is_tensor(sample_sigmas):
        return None
    if sigmas.numel() == 0 or sample_sigmas.numel() < 2:
        return None

    sigma = sigmas.detach().float().flatten()[0].to(device=sample_sigmas.device)
    candidates = sample_sigmas.detach().float().flatten()
    matches = torch.where(torch.isclose(candidates, sigma, rtol=1e-4, atol=1e-5))[0]
    if matches.numel() > 0:
        index = int(matches[0].item())
    else:
        index = int(torch.argmin((candidates - sigma).abs()).item())
        if torch.abs(candidates[index] - sigma) > max(1e-5, float(torch.abs(sigma).item()) * 1e-4):
            return None

    total = max(int(candidates.numel()) - 1, 1)
    progress = min(max(index / total, 0.0), 1.0)
    return ProgressInfo(index=index, total=total, progress=progress, sigma=float(sigma.item()))


def schedule_alphas(config: AFMConfig, progress: float, entropy_value: float | None = None) -> tuple[float, float]:
    strength = float(config.strength)
    if config.schedule == "curve":
        lf_gain = 1.0 - progress
        hf_gain = progress
    elif config.schedule == "lf_only":
        lf_gain = 1.0
        hf_gain = 0.0
    elif config.schedule == "hf_only":
        lf_gain = 0.0
        hf_gain = 1.0
    elif config.schedule == "constant":
        lf_gain = 1.0
        hf_gain = 1.0
    else:
        raise ValueError(f"Unsupported AFM schedule: {config.schedule!r}")

    if config.entropy_gate and entropy_value is not None:
        entropy_value = min(max(float(entropy_value), 0.0), 1.0)
        return (
            1.0 + strength * lf_gain * (1.0 + config.beta * entropy_value),
            1.0 + strength * hf_gain * (1.0 + config.gamma * (1.0 - entropy_value)),
        )
    return 1.0 + strength * lf_gain, 1.0 + strength * hf_gain


def selected_branch_indices(batch: int, cond_or_uncond: list[int] | None, branch_mode: str, device: torch.device) -> torch.Tensor:
    if branch_mode == "both" or not cond_or_uncond:
        return torch.arange(batch, device=device)
    chunks = len(cond_or_uncond)
    if chunks <= 0 or batch % chunks != 0:
        return torch.arange(0, 0, device=device)
    per_chunk = batch // chunks
    selected: list[int] = []
    for chunk_index, branch in enumerate(cond_or_uncond):
        if branch_mode == "positive_only" and branch != 0:
            continue
        if branch_mode == "negative_only" and branch != 1:
            continue
        selected.extend(range(chunk_index * per_chunk, (chunk_index + 1) * per_chunk))
    return torch.tensor(selected, dtype=torch.long, device=device)


def radial_low_high_masks(
    height: int,
    width: int,
    cutoff: float,
    soft_mask: bool,
    mask_width: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    radius = radius / math.sqrt(0.5)
    if soft_mask and mask_width > 0.0:
        half_width = mask_width * 0.5
        lower = cutoff - half_width
        upper = cutoff + half_width
        low = torch.empty_like(radius)
        low[radius <= lower] = 1.0
        low[radius >= upper] = 0.0
        transition = (radius > lower) & (radius < upper)
        if transition.any():
            t = (radius[transition] - lower) / max(upper - lower, 1e-6)
            low[transition] = 0.5 * (1.0 + torch.cos(math.pi * t))
    else:
        low = (radius <= cutoff).to(dtype=torch.float32)
    low = low.to(dtype=dtype)
    high = (1.0 - low).to(dtype=dtype)
    return low, high


def normalized_token_entropy(logits: torch.Tensor) -> float:
    text_len = logits.shape[-1]
    if text_len <= 1:
        return 0.0
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
    return float((entropy / math.log(text_len)).mean().detach().cpu().item())


def edit_logits_fft(
    logits: torch.Tensor,
    spatial_shape: tuple[int, int],
    alpha_lf: float,
    alpha_hf: float,
    config: AFMConfig,
) -> torch.Tensor:
    batch, heads, query_len, text_len = logits.shape
    height, width = spatial_shape
    if height * width != query_len:
        raise ValueError(f"spatial_shape {spatial_shape} does not match query_len {query_len}")

    compute = logits.float()
    maps = compute.reshape(batch, heads, height, width, text_len).permute(0, 1, 4, 2, 3)
    spectrum = torch.fft.fft2(maps, dim=(-2, -1))
    low, high = radial_low_high_masks(
        height,
        width,
        config.cutoff,
        config.soft_mask,
        config.mask_width,
        device=logits.device,
        dtype=compute.dtype,
    )
    scale = alpha_lf * low + alpha_hf * high
    edited = spectrum * scale
    if config.preserve_dc:
        edited[..., 0, 0] = spectrum[..., 0, 0]
    restored = torch.fft.ifft2(edited, dim=(-2, -1)).real
    return restored.permute(0, 1, 3, 4, 2).reshape(batch, heads, query_len, text_len).to(dtype=logits.dtype)


def attention_from_edited_logits(
    logits: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    skip_output_reshape: bool,
) -> torch.Tensor:
    attn = torch.softmax(logits, dim=-1).to(value.dtype)
    out = torch.matmul(attn, value)
    if skip_output_reshape:
        return out
    batch, _, query_len, head_dim = out.shape
    return out.transpose(1, 2).reshape(batch, query_len, heads * head_dim)


class AnimaAFMAttentionOverride:
    def __init__(self, config: AFMConfig):
        config.validate()
        self.config = config
        self.stats = AFMRuntimeStats()

    def __call__(self, original_func: Callable, *args: Any, **kwargs: Any) -> torch.Tensor:
        try:
            return self._call(original_func, *args, **kwargs)
        except Exception as exc:
            if self.config.fail_mode == "raise":
                LOGGER.exception("%s attention override failed", LOG_PREFIX)
                raise
            self.stats.record_fallback("oom_or_runtime_error")
            LOGGER.warning("%s falling back after %s: %s", LOG_PREFIX, type(exc).__name__, exc)
            return original_func(*args, **kwargs)

    def _call(self, original_func: Callable, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) < 4:
            return self._fallback(original_func, "bad_rank", *args, **kwargs)

        q, k, v, heads = args[:4]
        if len(args) > 4 and args[4] is not None:
            return self._fallback(original_func, "mask_shape_unsupported", *args, **kwargs)
        if kwargs.get("mask") is not None:
            return self._fallback(original_func, "mask_shape_unsupported", *args, **kwargs)
        if not kwargs.get("skip_reshape", False):
            return self._fallback(original_func, "not_skip_reshape", *args, **kwargs)
        if not all(torch.is_tensor(t) and t.ndim == 4 for t in (q, k, v)):
            return self._fallback(original_func, "bad_rank", *args, **kwargs)

        query_len = int(q.shape[-2])
        text_len = int(k.shape[-2])
        if query_len == text_len:
            return self._fallback(original_func, "not_cross_attention", *args, **kwargs)

        spatial_shape = infer_square_spatial_shape(query_len)
        if spatial_shape is None:
            return self._fallback(original_func, "cannot_infer_spatial_shape", *args, **kwargs)

        transformer_options = kwargs.get("transformer_options") or {}
        progress = progress_from_sigmas(transformer_options)
        if progress is None:
            return self._fallback(original_func, "missing_sigmas", *args, **kwargs)
        if progress.progress < self.config.start_percent or progress.progress > self.config.end_percent:
            return self._fallback(original_func, "progress_outside_window", *args, **kwargs)

        batch = int(q.shape[0])
        selected = selected_branch_indices(batch, transformer_options.get("cond_or_uncond"), self.config.branch_mode, q.device)
        if selected.numel() == 0:
            return self._fallback(original_func, "branch_not_selected", *args, **kwargs)

        scale = kwargs.get("scale", q.shape[-1] ** -0.5)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(scale)
        selected_logits = logits.index_select(0, selected)
        entropy_value = normalized_token_entropy(selected_logits) if self.config.entropy_gate else None
        alpha_lf, alpha_hf = schedule_alphas(self.config, progress.progress, entropy_value)

        edited_logits = logits.clone()
        edited_selected = edit_logits_fft(selected_logits, spatial_shape, alpha_lf, alpha_hf, self.config)
        edited_logits.index_copy_(0, selected, edited_selected)

        out = attention_from_edited_logits(
            edited_logits,
            v,
            int(heads),
            bool(kwargs.get("skip_output_reshape", False)),
        )
        self.stats.edited_calls += 1
        self._log_success(progress, q, k, v, spatial_shape, alpha_lf, alpha_hf, entropy_value, int(selected.numel()), logits, edited_logits, out)
        return out

    def _fallback(self, original_func: Callable, reason: str, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.stats.record_fallback(reason)
        if self.config.debug_level == "verbose":
            LOGGER.info("%s fallback reason=%s", LOG_PREFIX, reason)
        return original_func(*args, **kwargs)

    def _log_success(
        self,
        progress: ProgressInfo,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spatial_shape: tuple[int, int],
        alpha_lf: float,
        alpha_hf: float,
        entropy_value: float | None,
        edited_slices: int,
        logits_before: torch.Tensor,
        logits_after: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if self.config.debug_level == "off":
            return
        if self.config.debug_level == "summary" and progress.index in self.stats.summaries_by_step:
            return
        verbose_count = self.stats.verbose_counts_by_step.get(progress.index, 0)
        if self.config.debug_level == "verbose" and verbose_count >= 3:
            return

        delta = (logits_after - logits_before).abs().max().detach().cpu().item()
        message = (
            "%s step=%s/%s u=%.4f sigma=%.6g q=%s k=%s v=%s spatial=%s "
            "strength=%.4f cutoff=%.4f alpha_lf=%.4f alpha_hf=%.4f entropy=%s "
            "edited_slices=%s fallbacks=%s max_logit_delta=%.6g"
        )
        LOGGER.info(
            message,
            LOG_PREFIX,
            progress.index,
            progress.total,
            progress.progress,
            progress.sigma,
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            spatial_shape,
            self.config.strength,
            self.config.cutoff,
            alpha_lf,
            alpha_hf,
            "off" if entropy_value is None else f"{entropy_value:.4f}",
            edited_slices,
            self.stats.fallback_calls,
            delta,
        )
        if self.config.debug_level == "verbose":
            LOGGER.info(
                "%s diagnostics logits_before=(mean=%.6g,std=%.6g) logits_after=(mean=%.6g,std=%.6g) out=(mean=%.6g,std=%.6g)",
                LOG_PREFIX,
                float(logits_before.mean().detach().cpu().item()),
                float(logits_before.std().detach().cpu().item()),
                float(logits_after.mean().detach().cpu().item()),
                float(logits_after.std().detach().cpu().item()),
                float(out.float().mean().detach().cpu().item()),
                float(out.float().std().detach().cpu().item()),
            )
            self.stats.verbose_counts_by_step[progress.index] = verbose_count + 1
        self.stats.summaries_by_step.add(progress.index)


def is_anima_like_model(model: Any) -> bool:
    inner = getattr(model, "model", model)
    diffusion_model = getattr(inner, "diffusion_model", inner)
    if diffusion_model.__class__.__name__ == "Anima":
        return True
    return hasattr(diffusion_model, "llm_adapter") and hasattr(diffusion_model, "blocks")
