from __future__ import annotations

from collections import Counter
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
AFM_MODES = ["edit", "observe", "off"]
ZERO_STRENGTH_MODES = ["original", "observe", "manual"]
SPECTRAL_DIAG_MODES = ["off", "sampled", "full"]


@dataclass(frozen=True)
class AFMConfig:
    mode: str = "edit"
    strength: float = 0.2
    cutoff: float = 0.25
    start_percent: float = 0.0
    end_percent: float = 1.0
    schedule: str = "curve"
    branch_mode: str = "both"
    zero_strength_mode: str = "observe"
    entropy_gate: bool = False
    beta: float = 20.0
    gamma: float = 4.0
    preserve_dc: bool = True
    soft_mask: bool = True
    mask_width: float = 0.05
    debug_level: str = "off"
    fail_mode: str = "fallback"
    max_logits_mib: float = 1024.0
    spectral_diag: str = "off"
    diagnostic_top_k: int = 8
    diagnostic_max_batches: int = 1
    diagnostic_max_heads: int = 4

    def validate(self) -> None:
        if self.mode not in AFM_MODES:
            raise ValueError(f"Unsupported AFM mode: {self.mode!r}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"Unsupported AFM schedule: {self.schedule!r}")
        if self.branch_mode not in BRANCH_MODES:
            raise ValueError(f"Unsupported AFM branch_mode: {self.branch_mode!r}")
        if self.zero_strength_mode not in ZERO_STRENGTH_MODES:
            raise ValueError(f"Unsupported AFM zero_strength_mode: {self.zero_strength_mode!r}")
        if self.debug_level not in DEBUG_LEVELS:
            raise ValueError(f"Unsupported AFM debug_level: {self.debug_level!r}")
        if self.fail_mode not in FAIL_MODES:
            raise ValueError(f"Unsupported AFM fail_mode: {self.fail_mode!r}")
        if self.spectral_diag not in SPECTRAL_DIAG_MODES:
            raise ValueError(f"Unsupported AFM spectral_diag: {self.spectral_diag!r}")
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
        if self.max_logits_mib <= 0.0:
            raise ValueError("max_logits_mib must be positive")
        if self.diagnostic_top_k <= 0:
            raise ValueError("diagnostic_top_k must be positive")
        if self.diagnostic_max_batches <= 0:
            raise ValueError("diagnostic_max_batches must be positive")
        if self.diagnostic_max_heads <= 0:
            raise ValueError("diagnostic_max_heads must be positive")


@dataclass
class StepStats:
    index: int
    num_steps: int
    last_index: int
    progress: float
    sigma: float
    total_calls: int = 0
    eligible_calls: int = 0
    edited_calls: int = 0
    observed_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    shape_counts: Counter[str] = field(default_factory=Counter)
    selected_counts: Counter[str] = field(default_factory=Counter)
    max_logit_delta: float = 0.0
    max_attn_delta: float | None = None
    rho_before: float | None = None
    rho_after: float | None = None
    delta_rho: float | None = None
    max_estimated_logits_mib: float = 0.0


@dataclass(frozen=True)
class GQAInfo:
    enabled: bool
    repeats: int
    q_heads: int
    kv_heads_before: int
    kv_heads_after: int


@dataclass
class AFMRuntimeStats:
    edited_calls: int = 0
    observed_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    steps: dict[int, StepStats] = field(default_factory=dict)
    summaries_by_step: set[int] = field(default_factory=set)
    verbose_counts_by_step: dict[int, int] = field(default_factory=dict)

    def step_for(self, progress: "ProgressInfo") -> StepStats:
        stats = self.steps.get(progress.index)
        if stats is None:
            stats = StepStats(
                index=progress.index,
                num_steps=progress.num_steps,
                last_index=progress.last_index,
                progress=progress.progress,
                sigma=progress.sigma,
            )
            self.steps[progress.index] = stats
        return stats

    def record_fallback(self, reason: str, progress: "ProgressInfo | None" = None) -> None:
        self.fallback_calls += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        if progress is not None:
            step = self.step_for(progress)
            step.fallback_calls += 1
            step.fallback_reasons[reason] += 1


@dataclass(frozen=True)
class ProgressInfo:
    index: int
    num_steps: int
    last_index: int
    progress: float
    sigma: float

    @property
    def total(self) -> int:
        return self.num_steps


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

    num_steps = max(int(candidates.numel()) - 1, 1)
    last_index = max(num_steps - 1, 1)
    index = min(index, num_steps - 1)
    progress = min(max(index / last_index, 0.0), 1.0)
    return ProgressInfo(
        index=index,
        num_steps=num_steps,
        last_index=last_index,
        progress=progress,
        sigma=float(sigma.item()),
    )


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
    if branch_mode == "both":
        return torch.arange(batch, device=device)
    if not cond_or_uncond:
        return torch.arange(0, 0, device=device)
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


def estimate_logits_mib(batch: int, heads: int, query_len: int, text_len: int, bytes_per: int = 4) -> float:
    return batch * heads * query_len * text_len * bytes_per / (1024**2)


class AFMFallback(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def maybe_repeat_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, GQAInfo]:
    q_heads = int(q.shape[1])
    k_heads = int(k.shape[1])
    v_heads = int(v.shape[1])
    if q_heads == k_heads == v_heads:
        return q, k, v, GQAInfo(False, 1, q_heads, k_heads, k_heads)

    if not bool(kwargs.get("enable_gqa", False)):
        raise AFMFallback("head_mismatch")
    if k_heads != v_heads or k_heads <= 0 or q_heads % k_heads != 0:
        raise AFMFallback("gqa_head_mismatch")

    repeats = q_heads // k_heads
    return (
        q,
        k.repeat_interleave(repeats, dim=1),
        v.repeat_interleave(repeats, dim=1),
        GQAInfo(True, repeats, q_heads, k_heads, q_heads),
    )


def shape_key(q: torch.Tensor, k: torch.Tensor) -> str:
    return f"q{int(q.shape[-2])},k{int(k.shape[-2])},h{int(q.shape[1])},d{int(q.shape[-1])}"


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


def topk_concentration_map_from_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    k = min(int(top_k), int(probs.shape[-1]))
    return torch.topk(probs, k=k, dim=-1).values.mean(dim=-1)


def hf_ratio_from_concentration(concentration: torch.Tensor, spatial_shape: tuple[int, int], cutoff: float) -> float:
    height, width = spatial_shape
    maps = concentration.float().reshape(concentration.shape[0], concentration.shape[1], height, width)
    maps = maps - maps.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.fft.fft2(maps, dim=(-2, -1))
    power = spectrum.abs().square()
    power = power / power.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    _, high = radial_low_high_masks(height, width, cutoff, False, 0.0, concentration.device, power.dtype)
    return float((power * high).sum(dim=(-2, -1)).mean().detach().cpu().item())


def sampled_spectral_diagnostics(
    before: torch.Tensor,
    after: torch.Tensor,
    spatial_shape: tuple[int, int],
    config: AFMConfig,
) -> tuple[float, float, float, float, float]:
    if config.spectral_diag == "sampled":
        batch = min(int(before.shape[0]), config.diagnostic_max_batches)
        heads = min(int(before.shape[1]), config.diagnostic_max_heads)
        before = before[:batch, :heads]
        after = after[:batch, :heads]
    conc_before = topk_concentration_map_from_logits(before, config.diagnostic_top_k)
    conc_after = topk_concentration_map_from_logits(after, config.diagnostic_top_k)
    rho_before = hf_ratio_from_concentration(conc_before, spatial_shape, config.cutoff)
    rho_after = hf_ratio_from_concentration(conc_after, spatial_shape, config.cutoff)
    probs_before = torch.softmax(before.float(), dim=-1)
    probs_after = torch.softmax(after.float(), dim=-1)
    attn_delta = (probs_after - probs_before).abs()
    return (
        rho_before,
        rho_after,
        rho_after - rho_before,
        float(attn_delta.mean().detach().cpu().item()),
        float(attn_delta.max().detach().cpu().item()),
    )


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
            self.stats.record_fallback("oom_or_runtime_error", self._progress_from_kwargs(kwargs))
            LOGGER.warning("%s falling back after %s: %s", LOG_PREFIX, type(exc).__name__, exc)
            return original_func(*args, **kwargs)

    def _call(self, original_func: Callable, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.config.mode == "off":
            return original_func(*args, **kwargs)
        if (
            self.config.mode == "edit"
            and abs(float(self.config.strength)) == 0.0
            and self.config.zero_strength_mode == "original"
        ):
            return original_func(*args, **kwargs)

        if len(args) < 4:
            return self._fallback(original_func, "bad_rank", *args, **kwargs)

        q, k, v, heads = args[:4]
        transformer_options = kwargs.get("transformer_options") or {}
        progress = progress_from_sigmas(transformer_options)
        if progress is not None:
            self.stats.step_for(progress).total_calls += 1

        if len(args) > 4 and args[4] is not None:
            return self._fallback(original_func, "mask_shape_unsupported", *args, progress=progress, **kwargs)
        if kwargs.get("mask") is not None:
            return self._fallback(original_func, "mask_shape_unsupported", *args, progress=progress, **kwargs)
        if not kwargs.get("skip_reshape", False):
            return self._fallback(original_func, "not_skip_reshape", *args, progress=progress, **kwargs)
        if not all(torch.is_tensor(t) and t.ndim == 4 for t in (q, k, v)):
            return self._fallback(original_func, "bad_rank", *args, progress=progress, **kwargs)

        query_len = int(q.shape[-2])
        text_len = int(k.shape[-2])
        if query_len == text_len:
            return self._fallback(original_func, "not_cross_attention", *args, progress=progress, **kwargs)

        spatial_shape = infer_square_spatial_shape(query_len)
        if spatial_shape is None:
            return self._fallback(original_func, "cannot_infer_spatial_shape", *args, progress=progress, **kwargs)

        if progress is None:
            return self._fallback(original_func, "missing_sigmas", *args, progress=None, **kwargs)
        if progress.progress < self.config.start_percent or progress.progress > self.config.end_percent:
            return self._fallback(original_func, "progress_outside_window", *args, progress=progress, **kwargs)

        batch = int(q.shape[0])
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        selected = selected_branch_indices(batch, cond_or_uncond, self.config.branch_mode, q.device)
        if selected.numel() == 0:
            reason = "branch_layout_unknown" if self.config.branch_mode != "both" and not cond_or_uncond else "branch_not_selected"
            return self._fallback(original_func, reason, *args, progress=progress, **kwargs)

        active_mode = self.config.mode
        if active_mode == "edit" and abs(float(self.config.strength)) == 0.0:
            if self.config.zero_strength_mode == "observe":
                active_mode = "observe"

        step = self.stats.step_for(progress)
        step.eligible_calls += 1
        step.shape_counts[shape_key(q, k)] += 1
        step.selected_counts[f"{int(selected.numel())}/{batch}"] += 1

        if active_mode == "observe":
            self.stats.observed_calls += 1
            step.observed_calls += 1
            estimated = estimate_logits_mib(int(selected.numel()), int(q.shape[1]), query_len, text_len)
            step.max_estimated_logits_mib = max(step.max_estimated_logits_mib, estimated)
            out = original_func(*args, **kwargs)
            self._log_eligible(
                mode="observe",
                progress=progress,
                q=q,
                k=k,
                v=v,
                spatial_shape=spatial_shape,
                alpha_lf=None,
                alpha_hf=None,
                entropy_value=None,
                selected=selected,
                cond_or_uncond=cond_or_uncond,
                gqa_info=None,
                estimated_logits_mib=estimated,
                logits_delta=None,
                attn_delta=None,
            )
            return out

        scale = kwargs.get("scale", q.shape[-1] ** -0.5)
        all_selected = int(selected.numel()) == batch
        q_sel = q if all_selected else q.index_select(0, selected)
        k_sel = k if all_selected else k.index_select(0, selected)
        v_sel = v if all_selected else v.index_select(0, selected)

        try:
            q_sel, k_sel, v_sel, gqa_info = maybe_repeat_gqa(q_sel, k_sel, v_sel, kwargs)
        except AFMFallback as exc:
            return self._fallback(original_func, exc.reason, *args, progress=progress, **kwargs)

        estimated = estimate_logits_mib(int(q_sel.shape[0]), int(q_sel.shape[1]), int(q_sel.shape[-2]), int(k_sel.shape[-2]))
        step.max_estimated_logits_mib = max(step.max_estimated_logits_mib, estimated)
        if estimated > self.config.max_logits_mib:
            reason = "vram_guard_exceeded"
            if self.config.fail_mode == "raise":
                raise RuntimeError(
                    f"{LOG_PREFIX} {reason}: q={tuple(q_sel.shape)} k={tuple(k_sel.shape)} "
                    f"estimated_logits_mib={estimated:.1f} guard={self.config.max_logits_mib:.1f}"
                )
            return self._fallback(original_func, reason, *args, progress=progress, **kwargs)

        logits = torch.matmul(q_sel.float(), k_sel.float().transpose(-2, -1)) * float(scale)
        entropy_value = normalized_token_entropy(logits) if self.config.entropy_gate else None
        alpha_lf, alpha_hf = schedule_alphas(self.config, progress.progress, entropy_value)

        edited_logits = edit_logits_fft(logits, spatial_shape, alpha_lf, alpha_hf, self.config)

        out_sel = attention_from_edited_logits(
            edited_logits,
            v_sel,
            int(q_sel.shape[1]),
            bool(kwargs.get("skip_output_reshape", False)),
        )

        out = out_sel
        if not all_selected:
            base_out = original_func(*args, **kwargs)
            out = base_out.index_copy(0, selected.to(device=base_out.device), out_sel.to(device=base_out.device, dtype=base_out.dtype))

        self.stats.edited_calls += 1
        step.edited_calls += 1
        logits_delta = float((edited_logits - logits).abs().max().detach().cpu().item())
        step.max_logit_delta = max(step.max_logit_delta, logits_delta)
        attn_delta: tuple[float, float, float, float, float] | None = None
        if self.config.spectral_diag != "off":
            attn_delta = sampled_spectral_diagnostics(logits, edited_logits, spatial_shape, self.config)
            step.rho_before = attn_delta[0]
            step.rho_after = attn_delta[1]
            step.delta_rho = attn_delta[2]
            step.max_attn_delta = max(step.max_attn_delta or 0.0, attn_delta[4])
        self._log_eligible(
            mode="edit",
            progress=progress,
            q=q,
            k=k,
            v=v,
            spatial_shape=spatial_shape,
            alpha_lf=alpha_lf,
            alpha_hf=alpha_hf,
            entropy_value=entropy_value,
            selected=selected,
            cond_or_uncond=cond_or_uncond,
            gqa_info=gqa_info,
            estimated_logits_mib=estimated,
            logits_delta=logits_delta,
            attn_delta=attn_delta,
        )
        return out

    def _progress_from_kwargs(self, kwargs: dict[str, Any]) -> ProgressInfo | None:
        transformer_options = kwargs.get("transformer_options") or {}
        return progress_from_sigmas(transformer_options)

    def _fallback(self, original_func: Callable, reason: str, *args: Any, progress: ProgressInfo | None = None, **kwargs: Any) -> torch.Tensor:
        self.stats.record_fallback(reason, progress)
        if self.config.debug_level == "verbose":
            q, k, v = (args + (None, None, None))[:3]
            shapes = ""
            if torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v):
                shapes = f" q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
            LOGGER.info("%s fallback reason=%s%s", LOG_PREFIX, reason, shapes)
        return original_func(*args, **kwargs)

    def _log_eligible(
        self,
        mode: str,
        progress: ProgressInfo,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spatial_shape: tuple[int, int],
        alpha_lf: float | None,
        alpha_hf: float | None,
        entropy_value: float | None,
        selected: torch.Tensor,
        cond_or_uncond: list[int] | None,
        gqa_info: GQAInfo | None,
        estimated_logits_mib: float,
        logits_delta: float | None,
        attn_delta: tuple[float, float, float, float, float] | None,
    ) -> None:
        if self.config.debug_level == "off":
            return
        if self.config.debug_level == "summary" and progress.index in self.stats.summaries_by_step:
            return
        verbose_count = self.stats.verbose_counts_by_step.get(progress.index, 0)
        if self.config.debug_level == "verbose" and verbose_count >= 3:
            return

        step = self.stats.step_for(progress)
        fallback_summary = "{" + ", ".join(f"{key}:{value}" for key, value in step.fallback_reasons.items()) + "}"
        shape_summary = "{" + ", ".join(f"{key}:{value}" for key, value in step.shape_counts.items()) + "}"
        selected_indices = [int(i) for i in selected.detach().cpu().tolist()]
        message = (
            "%s step_summary step_index=%s num_steps=%s last_index=%s u=%.4f sigma=%.6g "
            "mode=%s calls=%s eligible=%s edited=%s observed=%s fallbacks=%s fallback_reasons=%s "
            "q=%s k=%s v=%s spatial=%s shapes=%s cond_or_uncond=%s branch_mode=%s "
            "selected_indices=%s selected_count=%s batch=%s strength=%.4f cutoff=%.4f "
            "alpha_lf=%s alpha_hf=%s entropy=%s gqa=%s estimated_logits_mib=%.1f max_logit_delta=%s"
        )
        LOGGER.info(
            message,
            LOG_PREFIX,
            progress.index,
            progress.num_steps,
            progress.last_index,
            progress.progress,
            progress.sigma,
            mode,
            step.total_calls,
            step.eligible_calls,
            step.edited_calls,
            step.observed_calls,
            step.fallback_calls,
            fallback_summary,
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            spatial_shape,
            shape_summary,
            cond_or_uncond,
            self.config.branch_mode,
            selected_indices,
            len(selected_indices),
            int(q.shape[0]),
            self.config.strength,
            self.config.cutoff,
            "n/a" if alpha_lf is None else f"{alpha_lf:.4f}",
            "n/a" if alpha_hf is None else f"{alpha_hf:.4f}",
            "off" if entropy_value is None else f"{entropy_value:.4f}",
            "none" if gqa_info is None else (
                f"enabled={gqa_info.enabled},repeats={gqa_info.repeats},"
                f"q_heads={gqa_info.q_heads},kv_heads_before={gqa_info.kv_heads_before},"
                f"kv_heads_after={gqa_info.kv_heads_after}"
            ),
            estimated_logits_mib,
            "n/a" if logits_delta is None else f"{logits_delta:.6g}",
        )
        if attn_delta is not None:
            LOGGER.info(
                "%s spectral_diag step_index=%s rho_before=%.6g rho_after=%.6g delta_rho=%.6g attn_delta_mean=%.6g attn_delta_max=%.6g",
                LOG_PREFIX,
                progress.index,
                attn_delta[0],
                attn_delta[1],
                attn_delta[2],
                attn_delta[3],
                attn_delta[4],
            )
        if self.config.debug_level == "verbose":
            self.stats.verbose_counts_by_step[progress.index] = verbose_count + 1
        self.stats.summaries_by_step.add(progress.index)


def is_anima_like_model(model: Any) -> bool:
    inner = getattr(model, "model", model)
    diffusion_model = getattr(inner, "diffusion_model", inner)
    if diffusion_model.__class__.__name__ == "Anima":
        return True
    return hasattr(diffusion_model, "llm_adapter") and hasattr(diffusion_model, "blocks")
