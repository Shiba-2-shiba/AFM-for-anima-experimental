from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import torch


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[AnimaAFM]"

SCHEDULES = ["curve", "lf_only", "hf_only", "constant"]
BRANCH_MODES = ["both", "positive_only", "negative_only"]
DEBUG_LEVELS = ["off", "summary", "verbose"]
DEBUG_FORMATS = ["text", "jsonl", "both"]
FAIL_MODES = ["fallback", "raise"]
AFM_MODES = ["edit", "observe", "off"]
ZERO_STRENGTH_MODES = ["original", "observe", "manual"]
SPECTRAL_DIAG_MODES = ["off", "sampled", "full"]
DIAGNOSTIC_BRANCHES = ["selected_mean", "positive", "negative", "both_separate"]
SUMMARY_ONLY_FALLBACK_REASONS = ("not_cross_attention",)
TRANSFORMER_METADATA_KEYS = (
    "block",
    "block_index",
    "transformer_index",
    "module_path",
    "patches_replace",
)


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
    debug_format: str = "text"
    jsonl_path: str | None = None
    fail_mode: str = "fallback"
    max_logits_mib: float = 1024.0
    max_peak_mib: float = 4096.0
    target_call_indices: str = "all"
    spectral_diag: str = "off"
    diagnostic_call_indices: str = "0"
    diagnostic_include_unselected: bool = False
    diagnostic_every_n_steps: int = 1
    diagnostic_branch: str = "both_separate"
    diagnostic_top_k: int = 8
    diagnostic_max_batches: int = 1
    diagnostic_max_heads: int = 4
    max_verbose_fallbacks_per_step_per_reason: int = 3
    fallback_summary_only_reasons: tuple[str, ...] = SUMMARY_ONLY_FALLBACK_REASONS

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
        if self.debug_format not in DEBUG_FORMATS:
            raise ValueError(f"Unsupported AFM debug_format: {self.debug_format!r}")
        if self.fail_mode not in FAIL_MODES:
            raise ValueError(f"Unsupported AFM fail_mode: {self.fail_mode!r}")
        if self.spectral_diag not in SPECTRAL_DIAG_MODES:
            raise ValueError(f"Unsupported AFM spectral_diag: {self.spectral_diag!r}")
        if self.diagnostic_branch not in DIAGNOSTIC_BRANCHES:
            raise ValueError(f"Unsupported AFM diagnostic_branch: {self.diagnostic_branch!r}")
        parse_call_index_scope(self.diagnostic_call_indices)
        parse_call_index_scope(self.target_call_indices)
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
        if self.max_peak_mib <= 0.0:
            raise ValueError("max_peak_mib must be positive")
        if self.diagnostic_top_k <= 0:
            raise ValueError("diagnostic_top_k must be positive")
        if self.diagnostic_max_batches <= 0:
            raise ValueError("diagnostic_max_batches must be positive")
        if self.diagnostic_max_heads <= 0:
            raise ValueError("diagnostic_max_heads must be positive")
        if self.diagnostic_every_n_steps <= 0:
            raise ValueError("diagnostic_every_n_steps must be positive")
        if self.max_verbose_fallbacks_per_step_per_reason < 0:
            raise ValueError("max_verbose_fallbacks_per_step_per_reason must be non-negative")


@dataclass(frozen=True)
class SpectralDiagnostic:
    branch: str
    batch_indices: list[int]
    rho_before: float
    rho_after: float
    delta_rho: float
    attn_delta_mean: float
    attn_delta_max: float
    edit_applied: bool = True


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
    target_skipped_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    shape_counts: Counter[str] = field(default_factory=Counter)
    eligible_call_indices: Counter[int] = field(default_factory=Counter)
    selected_counts: Counter[str] = field(default_factory=Counter)
    max_logit_delta: float = 0.0
    max_attn_delta: float | None = None
    rho_before: float | None = None
    rho_after: float | None = None
    delta_rho: float | None = None
    max_estimated_logits_mib: float = 0.0
    max_estimated_peak_mib: float = 0.0
    eligible_call_metadata: dict[int, dict[str, Any]] = field(default_factory=dict)
    spectral_diagnostics: dict[str, SpectralDiagnostic] = field(default_factory=dict)
    spectral_diagnostic_records: list[tuple[int, SpectralDiagnostic]] = field(default_factory=list)
    fallback_suppressed_reasons: Counter[str] = field(default_factory=Counter)


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
    target_skipped_calls: int = 0
    fallback_calls: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    steps: dict[int, StepStats] = field(default_factory=dict)
    summaries_by_step: set[int] = field(default_factory=set)
    verbose_counts_by_step: dict[int, int] = field(default_factory=dict)
    finalized_steps: set[int] = field(default_factory=set)
    active_step_index: int | None = None
    expected_total_calls_per_step: int | None = None
    run_final_summary_emitted: bool = False
    verbose_fallback_counts_by_step_reason: dict[tuple[int, str], int] = field(default_factory=dict)

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


def _branch_indices(batch: int, cond_or_uncond: list[int] | None, branch_name: str) -> list[int]:
    if not cond_or_uncond:
        return []
    branch_value = 0 if branch_name == "positive" else 1
    chunks = len(cond_or_uncond)
    if chunks <= 0 or batch % chunks != 0:
        return []
    per_chunk = batch // chunks
    indices: list[int] = []
    for chunk_index, branch in enumerate(cond_or_uncond):
        if branch != branch_value:
            continue
        indices.extend(range(chunk_index * per_chunk, (chunk_index + 1) * per_chunk))
    return indices


def diagnostic_batch_positions(
    selected: torch.Tensor,
    batch: int,
    cond_or_uncond: list[int] | None,
    diagnostic_branch: str,
) -> list[tuple[str, list[int], list[int]]]:
    selected_indices = [int(i) for i in selected.detach().cpu().tolist()]
    selected_position_by_batch = {batch_index: pos for pos, batch_index in enumerate(selected_indices)}
    if diagnostic_branch == "selected_mean":
        return [("selected_mean", list(range(len(selected_indices))), selected_indices)]

    requested = ["negative", "positive"] if diagnostic_branch == "both_separate" else [diagnostic_branch]
    positions: list[tuple[str, list[int], list[int]]] = []
    for branch in requested:
        batch_indices = _branch_indices(batch, cond_or_uncond, branch)
        selected_batch_indices = [idx for idx in batch_indices if idx in selected_position_by_batch]
        if selected_batch_indices:
            positions.append((branch, [selected_position_by_batch[idx] for idx in selected_batch_indices], selected_batch_indices))
    if not positions and diagnostic_branch == "both_separate":
        return [("selected_mean", list(range(len(selected_indices))), selected_indices)]
    return positions


def _safe_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if torch.is_tensor(value):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, tuple | list):
        if all(item is None or isinstance(item, str | int | float | bool) for item in value):
            return list(value)
        return {"type": type(value).__name__, "len": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "count": len(value), "keys": [str(key) for key in list(value.keys())[:16]]}
    return str(value)


def discover_transformer_metadata(transformer_options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    metadata = {
        key: _safe_metadata_value(transformer_options[key])
        for key in TRANSFORMER_METADATA_KEYS
        if key in transformer_options
    }
    block = metadata.get("block")
    if isinstance(block, list) and block:
        block_id = ":".join(str(part) for part in block)
    elif "block_index" in metadata:
        block_id = str(metadata["block_index"])
    elif "transformer_index" in metadata:
        block_id = str(metadata["transformer_index"])
    else:
        block_id = "unknown"
    return block_id, metadata


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in counter.items()}


def _counter_key_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: item[0])}


def parse_call_index_scope(spec: str) -> set[int] | None:
    normalized = str(spec).strip().lower()
    if normalized == "all":
        return None
    if not normalized:
        raise ValueError("call index scope must be 'all' or a comma-separated list of non-negative integers/ranges")

    indices: set[int] = set()
    for raw_part in normalized.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"Invalid call index scope entry in {spec!r}")
        if "-" in part:
            start_text, sep, end_text = part.partition("-")
            if not sep or not start_text or not end_text:
                raise ValueError(f"Invalid call index range: {part!r}")
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(f"Invalid call index range: {part!r}") from exc
            if start < 0 or end < 0:
                raise ValueError("call index scope entries must be non-negative")
            if end < start:
                raise ValueError(f"Invalid descending call index range: {part!r}")
            indices.update(range(start, end + 1))
        else:
            try:
                index = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid call index scope entry: {part!r}") from exc
            if index < 0:
                raise ValueError("call index scope entries must be non-negative")
            indices.add(index)
    return indices


def estimate_logits_mib(batch: int, heads: int, query_len: int, text_len: int, bytes_per: int = 4) -> float:
    return batch * heads * query_len * text_len * bytes_per / (1024**2)


def estimate_peak_mib(logits_mib: float, peak_multiplier: float = 4.0) -> float:
    return float(logits_mib) * peak_multiplier


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
    selected: torch.Tensor,
    full_batch: int,
    cond_or_uncond: list[int] | None,
    edit_applied: bool = True,
) -> list[SpectralDiagnostic]:
    diagnostics: list[SpectralDiagnostic] = []
    for branch, positions, batch_indices in diagnostic_batch_positions(
        selected,
        full_batch,
        cond_or_uncond,
        config.diagnostic_branch,
    ):
        if not positions:
            continue
        if config.spectral_diag == "sampled":
            positions = positions[: config.diagnostic_max_batches]
            batch_indices = batch_indices[: config.diagnostic_max_batches]
        index = torch.tensor(positions, dtype=torch.long, device=before.device)
        before_branch = before.index_select(0, index)
        after_branch = after.index_select(0, index)
        if config.spectral_diag == "sampled":
            heads = min(int(before_branch.shape[1]), config.diagnostic_max_heads)
            before_branch = before_branch[:, :heads]
            after_branch = after_branch[:, :heads]
        with torch.no_grad():
            conc_before = topk_concentration_map_from_logits(before_branch, config.diagnostic_top_k)
            conc_after = topk_concentration_map_from_logits(after_branch, config.diagnostic_top_k)
            rho_before = hf_ratio_from_concentration(conc_before, spatial_shape, config.cutoff)
            rho_after = hf_ratio_from_concentration(conc_after, spatial_shape, config.cutoff)
            probs_before = torch.softmax(before_branch.float(), dim=-1)
            probs_after = torch.softmax(after_branch.float(), dim=-1)
            attn_delta = (probs_after - probs_before).abs()
            diagnostics.append(SpectralDiagnostic(
                branch=branch,
                batch_indices=batch_indices,
                rho_before=rho_before,
                rho_after=rho_after,
                delta_rho=rho_after - rho_before,
                attn_delta_mean=float(attn_delta.mean().detach().cpu().item()),
                attn_delta_max=float(attn_delta.max().detach().cpu().item()),
                edit_applied=edit_applied,
            ))
    return diagnostics


class AnimaAFMAttentionOverride:
    def __init__(self, config: AFMConfig):
        config.validate()
        self.config = config
        self.stats = AFMRuntimeStats()
        self.run_id = f"afm-{id(self):x}"
        self._diagnostic_call_indices = parse_call_index_scope(config.diagnostic_call_indices)
        self._target_call_indices = parse_call_index_scope(config.target_call_indices)

    def finalize(self) -> None:
        if self.stats.active_step_index is not None:
            self._finalize_step(self.stats.active_step_index, "finalize_called")
        self._log_run_final_summary()

    def _step_for(self, progress: ProgressInfo) -> StepStats:
        active = self.stats.active_step_index
        if active is not None and active != progress.index:
            self._finalize_step(active, "step_change")
        self.stats.active_step_index = progress.index
        return self.stats.step_for(progress)

    def _finalize_step(self, step_index: int, final_reason: str) -> None:
        if step_index in self.stats.finalized_steps:
            return
        step = self.stats.steps.get(step_index)
        if step is None:
            return
        progress = ProgressInfo(
            index=step.index,
            num_steps=step.num_steps,
            last_index=step.last_index,
            progress=step.progress,
            sigma=step.sigma,
        )
        if self.config.debug_level != "off":
            self._log_step_summary(step, "step_final_summary", progress, final_reason, final_reason=final_reason)
        self.stats.finalized_steps.add(step_index)
        if step.total_calls > 0 and step_index != step.last_index and self.stats.expected_total_calls_per_step is None:
            self.stats.expected_total_calls_per_step = step.total_calls

    def _maybe_finalize_current_step(self, progress: ProgressInfo | None) -> None:
        if progress is None or progress.index != progress.last_index:
            return
        expected = self.stats.expected_total_calls_per_step
        if expected is None:
            return
        step = self.stats.steps.get(progress.index)
        if step is None or step.total_calls < expected:
            return
        self._finalize_step(progress.index, "expected_call_count_reached")

    def _target_call_selected(self, eligible_call_index: int) -> bool:
        return self._target_call_indices is None or eligible_call_index in self._target_call_indices

    def _update_peak_estimate(self, step: StepStats, estimated_logits_mib: float) -> float:
        estimated_peak_mib = estimate_peak_mib(estimated_logits_mib)
        step.max_estimated_logits_mib = max(step.max_estimated_logits_mib, estimated_logits_mib)
        step.max_estimated_peak_mib = max(step.max_estimated_peak_mib, estimated_peak_mib)
        return estimated_peak_mib

    def _record_spectral_diagnostics(
        self,
        step: StepStats,
        eligible_call_index: int,
        diagnostics: list[SpectralDiagnostic],
    ) -> None:
        for diagnostic in diagnostics:
            step.spectral_diagnostics[diagnostic.branch] = diagnostic
            step.spectral_diagnostic_records.append((eligible_call_index, diagnostic))
            step.rho_before = diagnostic.rho_before
            step.rho_after = diagnostic.rho_after
            step.delta_rho = diagnostic.delta_rho
            step.max_attn_delta = max(step.max_attn_delta or 0.0, diagnostic.attn_delta_max)

    def _unselected_indices(self, batch: int, selected: torch.Tensor) -> torch.Tensor:
        selected_set = set(int(index) for index in selected.detach().cpu().tolist())
        unselected = [index for index in range(batch) if index not in selected_set]
        return torch.tensor(unselected, dtype=torch.long, device=selected.device)

    def _emit_text(self, message: str, *args: Any) -> None:
        if self.config.debug_format in ("text", "both"):
            LOGGER.info(message, *args)

    def _emit_jsonl(self, record: dict[str, Any]) -> None:
        if self.config.debug_format in ("jsonl", "both"):
            line = json.dumps(record, sort_keys=True, separators=(",", ":"))
            LOGGER.info(line)
            if self.config.jsonl_path:
                path = Path(self.config.jsonl_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.write("\n")

    def _record_base(self, record_type: str, progress: ProgressInfo | None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_type": record_type,
            "run_id": self.run_id,
        }
        if progress is not None:
            record.update({
                "step_index": progress.index,
                "num_steps": progress.num_steps,
                "last_index": progress.last_index,
                "u": progress.progress,
                "sigma": progress.sigma,
            })
        return record

    def _spectral_summary(self, step: StepStats) -> dict[str, Any]:
        records = step.spectral_diagnostic_records
        if not records:
            return {
                "spectral_diag_count": 0,
                "spectral_delta_rho_mean": None,
                "spectral_delta_rho_min": None,
                "spectral_delta_rho_max": None,
                "spectral_by_call_branch": {},
            }

        deltas = [diagnostic.delta_rho for _, diagnostic in records]
        by_key: dict[str, list[SpectralDiagnostic]] = {}
        for eligible_call_index, diagnostic in records:
            by_key.setdefault(f"{eligible_call_index}/{diagnostic.branch}", []).append(diagnostic)

        return {
            "spectral_diag_count": len(records),
            "spectral_delta_rho_mean": sum(deltas) / len(deltas),
            "spectral_delta_rho_min": min(deltas),
            "spectral_delta_rho_max": max(deltas),
            "spectral_by_call_branch": {
                key: {
                    "n": len(items),
                    "delta_rho_mean": sum(item.delta_rho for item in items) / len(items),
                    "rho_before_mean": sum(item.rho_before for item in items) / len(items),
                    "rho_after_mean": sum(item.rho_after for item in items) / len(items),
                    "edit_applied": any(item.edit_applied for item in items),
                }
                for key, items in sorted(by_key.items())
            },
        }

    def _step_totals_record(self, step: StepStats) -> dict[str, Any]:
        spectral_summary = self._spectral_summary(step)
        return {
            "calls": step.total_calls,
            "eligible": step.eligible_calls,
            "edited": step.edited_calls,
            "observed": step.observed_calls,
            "target_skipped": step.target_skipped_calls,
            "fallbacks": step.fallback_calls,
            "fallback_reasons": _counter_dict(step.fallback_reasons),
            "fallback_suppressed_reasons": _counter_dict(step.fallback_suppressed_reasons),
            "eligible_call_indices": _counter_key_dict(step.eligible_call_indices),
            "shape_counts": _counter_dict(step.shape_counts),
            "max_estimated_logits_mib": step.max_estimated_logits_mib,
            "max_peak_mib": step.max_estimated_peak_mib,
            "max_logit_delta": step.max_logit_delta,
            "rho_before": None,
            "rho_after": None,
            "delta_rho": None,
            **spectral_summary,
        }

    def _log_run_final_summary(self) -> None:
        if self.stats.run_final_summary_emitted or self.config.debug_level == "off":
            return
        if not self.stats.steps:
            return
        last_step_index = max(self.stats.steps)
        record = self._record_base("run_final_summary", None)
        record.update({
            "last_step_index": last_step_index,
            "expected_total_calls_per_step": self.stats.expected_total_calls_per_step,
            "edited": self.stats.edited_calls,
            "observed": self.stats.observed_calls,
            "target_skipped": self.stats.target_skipped_calls,
            "fallbacks": self.stats.fallback_calls,
            "fallback_reasons": dict(self.stats.fallback_reasons),
            "steps": {
                str(step_index): self._step_totals_record(step)
                for step_index, step in sorted(self.stats.steps.items())
            },
        })
        self._emit_text(
            "%s run_final_summary last_step_index=%s expected_total_calls_per_step=%s edited=%s observed=%s fallbacks=%s",
            LOG_PREFIX,
            last_step_index,
            self.stats.expected_total_calls_per_step,
            self.stats.edited_calls,
            self.stats.observed_calls,
            self.stats.fallback_calls,
        )
        self._emit_jsonl(record)
        self.stats.run_final_summary_emitted = True

    def __call__(self, original_func: Callable, *args: Any, **kwargs: Any) -> torch.Tensor:
        progress = self._progress_from_kwargs(kwargs)
        try:
            result = self._call(original_func, *args, **kwargs)
        except Exception as exc:
            if self.config.fail_mode == "raise":
                LOGGER.exception("%s attention override failed", LOG_PREFIX)
                raise
            if progress is not None:
                self._step_for(progress)
            self.stats.record_fallback("oom_or_runtime_error", progress)
            LOGGER.warning("%s falling back after %s: %s", LOG_PREFIX, type(exc).__name__, exc)
            result = original_func(*args, **kwargs)
        self._maybe_finalize_current_step(progress)
        return result

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
            self._step_for(progress).total_calls += 1

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

        step = self._step_for(progress)
        eligible_call_index = step.eligible_calls
        block_id, metadata = discover_transformer_metadata(transformer_options)
        step.eligible_calls += 1
        step.eligible_call_indices[eligible_call_index] += 1
        step.shape_counts[shape_key(q, k)] += 1
        step.selected_counts[f"{int(selected.numel())}/{batch}"] += 1
        step.eligible_call_metadata[eligible_call_index] = {"block_id": block_id, **metadata}

        if active_mode == "observe":
            self.stats.observed_calls += 1
            step.observed_calls += 1
            estimated = estimate_logits_mib(int(selected.numel()), int(q.shape[1]), query_len, text_len)
            estimated_peak = self._update_peak_estimate(step, estimated)
            diagnostics: list[SpectralDiagnostic] = []
            if self._should_run_spectral_diag(progress, eligible_call_index):
                scale = kwargs.get("scale", q.shape[-1] ** -0.5)
                q_diag = q if int(selected.numel()) == batch else q.index_select(0, selected)
                k_diag = k if int(selected.numel()) == batch else k.index_select(0, selected)
                v_diag = v if int(selected.numel()) == batch else v.index_select(0, selected)
                try:
                    q_diag, k_diag, v_diag, _ = maybe_repeat_gqa(q_diag, k_diag, v_diag, kwargs)
                    logits = torch.matmul(q_diag.float(), k_diag.float().transpose(-2, -1)) * float(scale)
                    diagnostics = sampled_spectral_diagnostics(
                        logits,
                        logits,
                        spatial_shape,
                        self.config,
                        selected,
                        batch,
                        cond_or_uncond,
                        edit_applied=False,
                    )
                    self._record_spectral_diagnostics(step, eligible_call_index, diagnostics)
                except AFMFallback:
                    diagnostics = []
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
                estimated_peak_mib=estimated_peak,
                logits_delta=None,
                diagnostics=diagnostics,
                eligible_call_index=eligible_call_index,
                block_id=block_id,
                metadata=metadata,
            )
            return out

        if not self._target_call_selected(eligible_call_index):
            self.stats.target_skipped_calls += 1
            step.target_skipped_calls += 1
            estimated = estimate_logits_mib(int(selected.numel()), int(q.shape[1]), query_len, text_len)
            estimated_peak = self._update_peak_estimate(step, estimated)
            diagnostics: list[SpectralDiagnostic] = []
            if self._should_run_spectral_diag(progress, eligible_call_index):
                scale = kwargs.get("scale", q.shape[-1] ** -0.5)
                q_diag = q if int(selected.numel()) == batch else q.index_select(0, selected)
                k_diag = k if int(selected.numel()) == batch else k.index_select(0, selected)
                v_diag = v if int(selected.numel()) == batch else v.index_select(0, selected)
                try:
                    q_diag, k_diag, v_diag, _ = maybe_repeat_gqa(q_diag, k_diag, v_diag, kwargs)
                    logits = torch.matmul(q_diag.float(), k_diag.float().transpose(-2, -1)) * float(scale)
                    diagnostics = sampled_spectral_diagnostics(
                        logits,
                        logits,
                        spatial_shape,
                        self.config,
                        selected,
                        batch,
                        cond_or_uncond,
                        edit_applied=False,
                    )
                    self._record_spectral_diagnostics(step, eligible_call_index, diagnostics)
                except AFMFallback:
                    diagnostics = []
            out = original_func(*args, **kwargs)
            self._log_eligible(
                mode="passthrough",
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
                estimated_peak_mib=estimated_peak,
                logits_delta=None,
                diagnostics=diagnostics,
                eligible_call_index=eligible_call_index,
                block_id=block_id,
                metadata=metadata,
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
        estimated_peak = self._update_peak_estimate(step, estimated)
        if estimated > self.config.max_logits_mib:
            reason = "vram_guard_exceeded"
            if self.config.fail_mode == "raise":
                raise RuntimeError(
                    f"{LOG_PREFIX} {reason}: q={tuple(q_sel.shape)} k={tuple(k_sel.shape)} "
                    f"estimated_logits_mib={estimated:.1f} guard={self.config.max_logits_mib:.1f}"
                )
            return self._fallback(original_func, reason, *args, progress=progress, **kwargs)
        if estimated_peak > self.config.max_peak_mib:
            reason = "peak_vram_guard_exceeded"
            if self.config.fail_mode == "raise":
                raise RuntimeError(
                    f"{LOG_PREFIX} {reason}: q={tuple(q_sel.shape)} k={tuple(k_sel.shape)} "
                    f"estimated_peak_mib={estimated_peak:.1f} guard={self.config.max_peak_mib:.1f}"
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
        diagnostics: list[SpectralDiagnostic] = []
        if self._should_run_spectral_diag(progress, eligible_call_index):
            diagnostics = sampled_spectral_diagnostics(
                logits,
                edited_logits,
                spatial_shape,
                self.config,
                selected,
                batch,
                cond_or_uncond,
                edit_applied=True,
            )
            if self.config.diagnostic_include_unselected and int(selected.numel()) < batch:
                unselected = self._unselected_indices(batch, selected)
                if unselected.numel() > 0:
                    q_unselected = q.index_select(0, unselected)
                    k_unselected = k.index_select(0, unselected)
                    v_unselected = v.index_select(0, unselected)
                    try:
                        q_unselected, k_unselected, v_unselected, _ = maybe_repeat_gqa(q_unselected, k_unselected, v_unselected, kwargs)
                        passthrough_logits = torch.matmul(
                            q_unselected.float(),
                            k_unselected.float().transpose(-2, -1),
                        ) * float(scale)
                        diagnostics.extend(sampled_spectral_diagnostics(
                            passthrough_logits,
                            passthrough_logits,
                            spatial_shape,
                            self.config,
                            unselected,
                            batch,
                            cond_or_uncond,
                            edit_applied=False,
                        ))
                    except AFMFallback:
                        pass
            self._record_spectral_diagnostics(step, eligible_call_index, diagnostics)
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
            estimated_peak_mib=estimated_peak,
            logits_delta=logits_delta,
            diagnostics=diagnostics,
            eligible_call_index=eligible_call_index,
            block_id=block_id,
            metadata=metadata,
        )
        return out

    def _progress_from_kwargs(self, kwargs: dict[str, Any]) -> ProgressInfo | None:
        transformer_options = kwargs.get("transformer_options") or {}
        return progress_from_sigmas(transformer_options)

    def _should_run_spectral_diag(self, progress: ProgressInfo, eligible_call_index: int) -> bool:
        if self.config.spectral_diag == "off":
            return False
        if progress.index % self.config.diagnostic_every_n_steps != 0:
            return False
        return self._diagnostic_call_indices is None or eligible_call_index in self._diagnostic_call_indices

    def _fallback(self, original_func: Callable, reason: str, *args: Any, progress: ProgressInfo | None = None, **kwargs: Any) -> torch.Tensor:
        if progress is not None:
            self._step_for(progress)
        self.stats.record_fallback(reason, progress)
        if self.config.debug_level == "verbose" and self._should_log_verbose_fallback(reason, progress):
            q, k, v = (args + (None, None, None))[:3]
            shapes = ""
            q_shape = k_shape = v_shape = None
            if torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v):
                shapes = f" q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
                q_shape = list(q.shape)
                k_shape = list(k.shape)
                v_shape = list(v.shape)
            self._emit_text("%s fallback reason=%s%s", LOG_PREFIX, reason, shapes)
            transformer_options = kwargs.get("transformer_options") or {}
            spatial_shape = None
            if torch.is_tensor(q):
                spatial = infer_square_spatial_shape(int(q.shape[-2]))
                spatial_shape = None if spatial is None else list(spatial)
            record = self._record_base("fallback", progress)
            record.update({
                "reason": reason,
                "eligible_call_index": None,
                "q_shape": q_shape,
                "k_shape": k_shape,
                "v_shape": v_shape,
                "spatial_shape": spatial_shape,
                "cond_or_uncond": transformer_options.get("cond_or_uncond"),
                "branch_mode": self.config.branch_mode,
                "selected_indices": [],
                "diagnostic_branch": self.config.diagnostic_branch,
                "alpha_lf": None,
                "alpha_hf": None,
                "rho_before": None,
                "rho_after": None,
                "delta_rho": None,
                "estimated_logits_mib": None,
                "estimated_peak_mib": None,
            })
            self._emit_jsonl(record)
        return original_func(*args, **kwargs)

    def _should_log_verbose_fallback(self, reason: str, progress: ProgressInfo | None) -> bool:
        if progress is None:
            return True
        summary_only_reasons = set(self.config.fallback_summary_only_reasons)
        if reason not in summary_only_reasons:
            return True
        key = (progress.index, reason)
        count = self.stats.verbose_fallback_counts_by_step_reason.get(key, 0)
        self.stats.verbose_fallback_counts_by_step_reason[key] = count + 1
        if count < self.config.max_verbose_fallbacks_per_step_per_reason:
            return True
        step = self.stats.step_for(progress)
        step.fallback_suppressed_reasons[reason] += 1
        return False

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
        estimated_peak_mib: float,
        logits_delta: float | None,
        diagnostics: list[SpectralDiagnostic],
        eligible_call_index: int,
        block_id: str,
        metadata: dict[str, Any],
    ) -> None:
        if self.config.debug_level == "off":
            return
        should_log_snapshot = True
        if self.config.debug_level == "summary" and progress.index in self.stats.summaries_by_step:
            should_log_snapshot = False
        verbose_count = self.stats.verbose_counts_by_step.get(progress.index, 0)
        if self.config.debug_level == "verbose" and verbose_count >= 3:
            should_log_snapshot = False

        step = self.stats.step_for(progress)
        if should_log_snapshot:
            self._log_step_summary(
                step=step,
                record_type="step_snapshot",
                progress=progress,
                snapshot_reason="eligible_call",
                mode=mode,
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
                estimated_logits_mib=estimated_logits_mib,
                estimated_peak_mib=estimated_peak_mib,
                logits_delta=logits_delta,
                eligible_call_index=eligible_call_index,
                block_id=block_id,
                metadata=metadata,
            )
        for diagnostic in diagnostics:
            self._log_spectral_diag(
                mode=mode,
                progress=progress,
                diagnostic=diagnostic,
                eligible_call_index=eligible_call_index,
                q=q,
                k=k,
                v=v,
                spatial_shape=spatial_shape,
                selected=selected,
                cond_or_uncond=cond_or_uncond,
                alpha_lf=alpha_lf,
                alpha_hf=alpha_hf,
                estimated_logits_mib=estimated_logits_mib,
                estimated_peak_mib=estimated_peak_mib,
            )
        if should_log_snapshot and self.config.debug_level == "verbose":
            self.stats.verbose_counts_by_step[progress.index] = verbose_count + 1
        if should_log_snapshot:
            self.stats.summaries_by_step.add(progress.index)

    def _log_step_summary(
        self,
        step: StepStats,
        record_type: str,
        progress: ProgressInfo | None,
        snapshot_reason: str,
        mode: str | None = None,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        spatial_shape: tuple[int, int] | None = None,
        alpha_lf: float | None = None,
        alpha_hf: float | None = None,
        entropy_value: float | None = None,
        selected: torch.Tensor | None = None,
        cond_or_uncond: list[int] | None = None,
        gqa_info: GQAInfo | None = None,
        estimated_logits_mib: float | None = None,
        estimated_peak_mib: float | None = None,
        logits_delta: float | None = None,
        eligible_call_index: int | None = None,
        block_id: str = "unknown",
        metadata: dict[str, Any] | None = None,
        final_reason: str | None = None,
    ) -> None:
        fallback_summary = "{" + ", ".join(f"{key}:{value}" for key, value in step.fallback_reasons.items()) + "}"
        shape_summary = "{" + ", ".join(f"{key}:{value}" for key, value in step.shape_counts.items()) + "}"
        eligible_index_summary = "{" + ", ".join(f"{key}:{value}" for key, value in sorted(step.eligible_call_indices.items())) + "}"
        suppressed_summary = "{" + ", ".join(f"{key}:{value}" for key, value in step.fallback_suppressed_reasons.items()) + "}"
        selected_indices = [] if selected is None else [int(i) for i in selected.detach().cpu().tolist()]
        q_shape = None if q is None else tuple(q.shape)
        k_shape = None if k is None else tuple(k.shape)
        v_shape = None if v is None else tuple(v.shape)
        summary_rho_before = None if record_type in ("step_final_summary", "run_final_summary") else step.rho_before
        summary_rho_after = None if record_type in ("step_final_summary", "run_final_summary") else step.rho_after
        summary_delta_rho = None if record_type in ("step_final_summary", "run_final_summary") else step.delta_rho
        effective_peak_mib = step.max_estimated_peak_mib if estimated_peak_mib is None else estimated_peak_mib
        spectral_summary = self._spectral_summary(step)
        message = (
            "%s %s step_index=%s num_steps=%s last_index=%s u=%.4f sigma=%.6g "
            "snapshot_reason=%s snapshot_call_index=%s mode=%s calls=%s eligible=%s edited=%s observed=%s "
            "target_skipped=%s fallbacks=%s fallback_reasons=%s fallback_suppressed_reasons=%s eligible_call_indices=%s "
            "eligible_call_index=%s block_id=%s metadata=%s q=%s k=%s v=%s spatial=%s shapes=%s "
            "cond_or_uncond=%s branch_mode=%s selected_indices=%s selected_count=%s batch=%s "
            "strength=%.4f cutoff=%.4f alpha_lf=%s alpha_hf=%s entropy=%s gqa=%s "
            "estimated_logits_mib=%s estimated_peak_mib=%s max_estimated_logits_mib=%.1f max_peak_mib=%.1f "
            "spectral_diag_count=%s max_logit_delta=%s final_reason=%s"
        )
        self._emit_text(
            message,
            LOG_PREFIX,
            record_type,
            progress.index,
            progress.num_steps,
            progress.last_index,
            progress.progress,
            progress.sigma,
            snapshot_reason,
            eligible_call_index,
            mode,
            step.total_calls,
            step.eligible_calls,
            step.edited_calls,
            step.observed_calls,
            step.target_skipped_calls,
            step.fallback_calls,
            fallback_summary,
            suppressed_summary,
            eligible_index_summary,
            eligible_call_index,
            block_id,
            metadata or {},
            q_shape,
            k_shape,
            v_shape,
            spatial_shape,
            shape_summary,
            cond_or_uncond,
            self.config.branch_mode,
            selected_indices,
            len(selected_indices),
            "n/a" if q is None else int(q.shape[0]),
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
            "n/a" if estimated_logits_mib is None else f"{estimated_logits_mib:.1f}",
            f"{effective_peak_mib:.1f}",
            step.max_estimated_logits_mib,
            step.max_estimated_peak_mib,
            spectral_summary["spectral_diag_count"],
            "n/a" if logits_delta is None else f"{logits_delta:.6g}",
            final_reason,
        )
        record = self._record_base(record_type, progress)
        record.update({
            "snapshot_reason": snapshot_reason,
            "snapshot_call_index": eligible_call_index,
            "final_reason": final_reason,
            "mode": mode,
            "calls": step.total_calls,
            "eligible": step.eligible_calls,
            "edited": step.edited_calls,
            "observed": step.observed_calls,
            "target_skipped": step.target_skipped_calls,
            "fallbacks": step.fallback_calls,
            "fallback_reasons": _counter_dict(step.fallback_reasons),
            "fallback_suppressed_reasons": _counter_dict(step.fallback_suppressed_reasons),
            "eligible_call_indices": _counter_key_dict(step.eligible_call_indices),
            "eligible_call_index": eligible_call_index,
            "block_id": block_id,
            "metadata": metadata or {},
            "q_shape": None if q is None else list(q.shape),
            "k_shape": None if k is None else list(k.shape),
            "v_shape": None if v is None else list(v.shape),
            "spatial_shape": None if spatial_shape is None else list(spatial_shape),
            "shape_counts": _counter_dict(step.shape_counts),
            "cond_or_uncond": cond_or_uncond,
            "branch_mode": self.config.branch_mode,
            "selected_indices": selected_indices,
            "selected_count": len(selected_indices),
            "diagnostic_branch": self.config.diagnostic_branch,
            "batch": None if q is None else int(q.shape[0]),
            "strength": float(self.config.strength),
            "cutoff": float(self.config.cutoff),
            "alpha_lf": alpha_lf,
            "alpha_hf": alpha_hf,
            "rho_before": summary_rho_before,
            "rho_after": summary_rho_after,
            "delta_rho": summary_delta_rho,
            "entropy": entropy_value,
            "gqa": None if gqa_info is None else {
                "enabled": gqa_info.enabled,
                "repeats": gqa_info.repeats,
                "q_heads": gqa_info.q_heads,
                "kv_heads_before": gqa_info.kv_heads_before,
                "kv_heads_after": gqa_info.kv_heads_after,
            },
            "estimated_logits_mib": estimated_logits_mib,
            "estimated_peak_mib": effective_peak_mib,
            "max_estimated_logits_mib": step.max_estimated_logits_mib,
            "max_peak_mib": step.max_estimated_peak_mib,
            "target_call_indices": self.config.target_call_indices,
            "diagnostic_include_unselected": self.config.diagnostic_include_unselected,
            "max_logit_delta": logits_delta,
        })
        record.update(spectral_summary)
        self._emit_jsonl(record)

    def _log_spectral_diag(
        self,
        mode: str,
        progress: ProgressInfo,
        diagnostic: SpectralDiagnostic,
        eligible_call_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spatial_shape: tuple[int, int],
        selected: torch.Tensor,
        cond_or_uncond: list[int] | None,
        alpha_lf: float | None,
        alpha_hf: float | None,
        estimated_logits_mib: float,
        estimated_peak_mib: float,
    ) -> None:
        self._emit_text(
            "%s spectral_diag step_index=%s eligible_call_index=%s mode=%s branch=%s rho_before=%.6g "
            "rho_after=%.6g delta_rho=%.6g edit_applied=%s attn_delta_mean=%.6g attn_delta_max=%.6g batch_indices=%s",
            LOG_PREFIX,
            progress.index,
            eligible_call_index,
            mode,
            diagnostic.branch,
            diagnostic.rho_before,
            diagnostic.rho_after,
            diagnostic.delta_rho,
            diagnostic.edit_applied,
            diagnostic.attn_delta_mean,
            diagnostic.attn_delta_max,
            diagnostic.batch_indices,
        )
        record = self._record_base("spectral_diag", progress)
        record.update({
            "mode": mode,
            "eligible_call_index": eligible_call_index,
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "spatial_shape": list(spatial_shape),
            "cond_or_uncond": cond_or_uncond,
            "branch_mode": self.config.branch_mode,
            "selected_indices": [int(i) for i in selected.detach().cpu().tolist()],
            "diagnostic_branch": diagnostic.branch,
            "batch_indices": diagnostic.batch_indices,
            "alpha_lf": alpha_lf,
            "alpha_hf": alpha_hf,
            "rho_before": diagnostic.rho_before,
            "rho_after": diagnostic.rho_after,
            "delta_rho": diagnostic.delta_rho,
            "delta_rho_local": diagnostic.delta_rho,
            "edit_applied": diagnostic.edit_applied,
            "attn_delta_mean": diagnostic.attn_delta_mean,
            "attn_delta_max": diagnostic.attn_delta_max,
            "estimated_logits_mib": estimated_logits_mib,
            "estimated_peak_mib": estimated_peak_mib,
        })
        self._emit_jsonl(record)


def is_anima_like_model(model: Any) -> bool:
    inner = getattr(model, "model", model)
    diffusion_model = getattr(inner, "diffusion_model", inner)
    if diffusion_model.__class__.__name__ == "Anima":
        return True
    return hasattr(diffusion_model, "llm_adapter") and hasattr(diffusion_model, "blocks")
