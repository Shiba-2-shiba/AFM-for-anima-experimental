import json
from pathlib import Path
import tempfile
import unittest

import torch

from anima_afm import (
    AFMConfig,
    AnimaAFMAttentionOverride,
    edit_logits_fft,
    estimate_peak_mib,
    infer_square_spatial_shape,
    normalized_token_entropy,
    parse_call_index_scope,
    progress_from_sigmas,
    radial_low_high_masks,
    schedule_alphas,
    selected_branch_indices,
)
from scripts.compare_afm_runs import compare_rows
from scripts.parse_anima_afm_log import parse_records


def reference_attention(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    del mask, kwargs
    assert skip_reshape
    logits = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    out = torch.matmul(torch.softmax(logits, dim=-1), v)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], heads * q.shape[-1])


class CountingAttention:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return reference_attention(*args, **kwargs)


class SentinelAttention:
    def __init__(self):
        self.calls = 0
        self.reference_output = None

    def __call__(self, *args, **kwargs):
        self.calls += 1
        out = reference_attention(*args, **kwargs)
        offsets = torch.arange(out.shape[0], device=out.device, dtype=out.dtype).reshape(
            out.shape[0],
            *([1] * (out.ndim - 1)),
        )
        self.reference_output = out + offsets * 1000.0
        return self.reference_output


class AnimaAFMTests(unittest.TestCase):
    def _cfg_kwargs(self, sigma):
        return {
            "skip_reshape": True,
            "transformer_options": {
                "sigmas": torch.tensor([sigma]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                "cond_or_uncond": [1, 0],
            },
        }

    def _json_log_records(self, records):
        parsed = []
        for record in records:
            try:
                parsed.append(json.loads(record.getMessage()))
            except json.JSONDecodeError:
                pass
        return parsed

    def test_infer_square_spatial_shape(self):
        self.assertEqual(infer_square_spatial_shape(64), (8, 8))
        self.assertIsNone(infer_square_spatial_shape(65))

    def test_progress_last_model_step_reaches_one(self):
        info = progress_from_sigmas({
            "sigmas": torch.tensor([0.5]),
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
        })
        self.assertIsNotNone(info)
        self.assertEqual(info.index, 1)
        self.assertEqual(info.num_steps, 2)
        self.assertEqual(info.last_index, 1)
        self.assertAlmostEqual(info.progress, 1.0)

    def test_progress_24_steps_last_index(self):
        sample_sigmas = torch.linspace(1.0, 0.0, 25)
        info = progress_from_sigmas({
            "sigmas": sample_sigmas[23:24],
            "sample_sigmas": sample_sigmas,
        })
        self.assertIsNotNone(info)
        self.assertEqual(info.index, 23)
        self.assertEqual(info.num_steps, 24)
        self.assertEqual(info.last_index, 23)
        self.assertAlmostEqual(info.progress, 1.0)

    def test_schedule_alphas(self):
        config = AFMConfig(strength=0.2, schedule="curve")
        self.assertEqual(schedule_alphas(config, 0.0), (1.2, 1.0))
        self.assertEqual(schedule_alphas(config, 1.0), (1.0, 1.2))

    def test_branch_selection(self):
        idx = selected_branch_indices(4, [1, 0], "positive_only", torch.device("cpu"))
        self.assertEqual(idx.tolist(), [2, 3])
        idx = selected_branch_indices(4, [1, 0], "negative_only", torch.device("cpu"))
        self.assertEqual(idx.tolist(), [0, 1])

    def test_parse_call_index_scope_supports_all_lists_and_ranges(self):
        self.assertIsNone(parse_call_index_scope("all"))
        self.assertEqual(parse_call_index_scope("0,7,14"), {0, 7, 14})
        self.assertEqual(parse_call_index_scope("0-2,7,10-11"), {0, 1, 2, 7, 10, 11})
        with self.assertRaises(ValueError):
            parse_call_index_scope("3-1")

    def test_masks_have_expected_shape(self):
        low, high = radial_low_high_masks(8, 8, 0.25, True, 0.05, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(low.shape), (8, 8))
        self.assertTrue(torch.allclose(low + high, torch.ones_like(low)))
        self.assertGreater(float(low.sum()), 0.0)
        self.assertGreater(float(high.sum()), 0.0)

    def test_entropy_range(self):
        logits = torch.zeros(1, 1, 4, 8)
        entropy = normalized_token_entropy(logits)
        self.assertGreaterEqual(entropy, 0.99)
        self.assertLessEqual(entropy, 1.01)

    def test_edit_logits_shape_and_dc(self):
        logits = torch.randn(2, 3, 16, 5)
        edited = edit_logits_fft(logits, (4, 4), 1.2, 1.1, AFMConfig(preserve_dc=True))
        self.assertEqual(tuple(edited.shape), tuple(logits.shape))
        before_mean = logits.reshape(2, 3, 4, 4, 5).mean(dim=(2, 3))
        after_mean = edited.reshape(2, 3, 4, 4, 5).mean(dim=(2, 3))
        self.assertTrue(torch.allclose(before_mean, after_mean, atol=1e-5, rtol=1e-5))

    def test_override_strength_zero_matches_reference(self):
        torch.manual_seed(1)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        kwargs = {
            "skip_reshape": True,
            "transformer_options": {
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                "cond_or_uncond": [1, 0],
            },
        }
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.0, zero_strength_mode="manual"))
        out = override(reference_attention, q, k, v, 2, **kwargs)
        expected = reference_attention(q, k, v, 2, **kwargs)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5))

    def test_entropy_gate_strength_zero_matches_reference(self):
        torch.manual_seed(11)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        kwargs = {
            "skip_reshape": True,
            "transformer_options": {
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            },
        }
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.0, entropy_gate=True, zero_strength_mode="manual"))
        out = override(reference_attention, q, k, v, 2, **kwargs)
        expected = reference_attention(q, k, v, 2, **kwargs)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5, rtol=1e-5))

    def test_override_preserves_shape(self):
        torch.manual_seed(2)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2))
        out = override(
            reference_attention,
            q,
            k,
            v,
            2,
            skip_reshape=True,
            transformer_options={
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                "cond_or_uncond": [1, 0],
            },
        )
        self.assertEqual(tuple(out.shape), (2, 16, 8))

    def test_observe_mode_returns_original_and_records_eligible(self):
        torch.manual_seed(22)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        original = SentinelAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(mode="observe", strength=0.2))
        kwargs = {
            "skip_reshape": True,
            "transformer_options": {
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                "cond_or_uncond": [1, 0],
            },
        }
        out = override(original, q, k, v, 2, **kwargs)
        self.assertTrue(torch.equal(out, original.reference_output))
        self.assertEqual(original.calls, 1)
        self.assertEqual(override.stats.observed_calls, 1)
        self.assertEqual(override.stats.edited_calls, 0)
        self.assertEqual(override.stats.steps[1].observed_calls, 1)

    def test_positive_only_preserves_negative_branch_original_backend(self):
        torch.manual_seed(23)
        q = torch.randn(4, 2, 16, 4)
        k = torch.randn(4, 2, 5, 4)
        v = torch.randn(4, 2, 5, 4)
        original = SentinelAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2, branch_mode="positive_only"))
        kwargs = {
            "skip_reshape": True,
            "transformer_options": {
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                "cond_or_uncond": [1, 0],
            },
        }
        out = override(original, q, k, v, 2, **kwargs)
        self.assertEqual(original.calls, 1)
        self.assertTrue(torch.equal(out[:2], original.reference_output[:2]))
        self.assertFalse(torch.equal(out[2:], original.reference_output[2:]))

    def test_negative_only_preserves_positive_branch_original_backend(self):
        torch.manual_seed(24)
        q = torch.randn(4, 2, 16, 4)
        k = torch.randn(4, 2, 5, 4)
        v = torch.randn(4, 2, 5, 4)
        original = SentinelAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2, branch_mode="negative_only"))
        kwargs = {
            "skip_reshape": True,
            "transformer_options": {
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                "cond_or_uncond": [1, 0],
            },
        }
        out = override(original, q, k, v, 2, **kwargs)
        self.assertEqual(original.calls, 1)
        self.assertFalse(torch.equal(out[:2], original.reference_output[:2]))
        self.assertTrue(torch.equal(out[2:], original.reference_output[2:]))

    def test_branch_layout_unknown_falls_back_for_selected_modes(self):
        q = torch.randn(4, 2, 16, 4)
        k = torch.randn(4, 2, 5, 4)
        v = torch.randn(4, 2, 5, 4)
        original = CountingAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2, branch_mode="positive_only"))
        override(
            original,
            q,
            k,
            v,
            2,
            skip_reshape=True,
            transformer_options={
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            },
        )
        self.assertEqual(original.calls, 1)
        self.assertEqual(override.stats.fallback_reasons["branch_layout_unknown"], 1)

    def test_gqa_repeats_kv_heads(self):
        torch.manual_seed(25)
        q = torch.randn(2, 8, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2))
        out = override(
            reference_attention,
            q,
            k,
            v,
            8,
            skip_reshape=True,
            skip_output_reshape=True,
            enable_gqa=True,
            transformer_options={
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            },
        )
        self.assertEqual(tuple(out.shape), (2, 8, 16, 4))
        self.assertEqual(override.stats.edited_calls, 1)

    def test_vram_guard_falls_back(self):
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        original = CountingAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2, max_logits_mib=0.0001))
        override(
            original,
            q,
            k,
            v,
            2,
            skip_reshape=True,
            transformer_options={
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            },
        )
        self.assertEqual(original.calls, 1)
        self.assertEqual(override.stats.fallback_reasons["vram_guard_exceeded"], 1)

    def test_peak_memory_estimate_and_guard_are_separate_from_logits_guard(self):
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(debug_level="summary", debug_format="jsonl"))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
        records = self._json_log_records(logs.records)
        snapshot = [record for record in records if record["record_type"] == "step_snapshot"][0]
        self.assertIsInstance(snapshot["estimated_peak_mib"], float)
        self.assertAlmostEqual(snapshot["estimated_peak_mib"], estimate_peak_mib(snapshot["estimated_logits_mib"]))
        self.assertEqual(snapshot["max_peak_mib"], snapshot["estimated_peak_mib"])

        original = CountingAttention()
        guarded = AnimaAFMAttentionOverride(AFMConfig(max_logits_mib=1024.0, max_peak_mib=0.0001))
        guarded(original, q, k, v, 2, **self._cfg_kwargs(0.5))
        self.assertEqual(original.calls, 1)
        self.assertEqual(guarded.stats.fallback_reasons["peak_vram_guard_exceeded"], 1)

    def test_target_call_indices_scope_edits_only_selected_calls(self):
        torch.manual_seed(260)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        original = CountingAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(
            target_call_indices="0",
            debug_level="verbose",
            debug_format="jsonl",
        ))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            for _ in range(3):
                override(original, q, k, v, 2, **self._cfg_kwargs(0.5))
            override.finalize()

        step = override.stats.steps[1]
        self.assertEqual(step.eligible_calls, 3)
        self.assertEqual(step.edited_calls, 1)
        self.assertEqual(step.target_skipped_calls, 2)
        self.assertEqual(original.calls, 2)
        records = self._json_log_records(logs.records)
        final_summary = [
            record for record in records
            if record["record_type"] == "step_final_summary" and record["step_index"] == 1
        ][0]
        self.assertEqual(final_summary["edited"], 1)
        self.assertEqual(final_summary["target_skipped"], 2)
        self.assertEqual(final_summary["target_call_indices"], "0")

    def test_target_call_indices_range_scope(self):
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(target_call_indices="0-1"))
        for _ in range(3):
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
        step = override.stats.steps[1]
        self.assertEqual(step.edited_calls, 2)
        self.assertEqual(step.target_skipped_calls, 1)

    def test_spectral_diag_reports_rho_delta(self):
        torch.manual_seed(26)
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2, spectral_diag="sampled"))
        override(
            reference_attention,
            q,
            k,
            v,
            2,
            skip_reshape=True,
            transformer_options={
                "sigmas": torch.tensor([0.5]),
                "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            },
        )
        step = override.stats.steps[1]
        self.assertIsNotNone(step.rho_before)
        self.assertIsNotNone(step.rho_after)
        self.assertIsNotNone(step.delta_rho)

    def test_step_final_summary_uses_explicit_spectral_aggregates_not_last_scalar(self):
        torch.manual_seed(261)
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(
            spectral_diag="sampled",
            diagnostic_call_indices="all",
            diagnostic_branch="selected_mean",
            debug_level="verbose",
            debug_format="jsonl",
        ))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
            override.finalize()

        records = self._json_log_records(logs.records)
        final_summary = [
            record for record in records
            if record["record_type"] == "step_final_summary" and record["step_index"] == 1
        ][0]
        self.assertIsNone(final_summary["rho_before"])
        self.assertIsNone(final_summary["rho_after"])
        self.assertIsNone(final_summary["delta_rho"])
        self.assertEqual(final_summary["spectral_diag_count"], 2)
        self.assertIsInstance(final_summary["spectral_delta_rho_mean"], float)
        self.assertEqual(set(final_summary["spectral_by_call_branch"]), {"0/selected_mean", "1/selected_mean"})
        self.assertIsInstance(final_summary["estimated_peak_mib"], float)
        self.assertEqual(final_summary["estimated_peak_mib"], final_summary["max_peak_mib"])

    def test_branch_aware_spectral_diag_uses_cfg_batch_indices(self):
        torch.manual_seed(27)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(
            strength=0.2,
            spectral_diag="sampled",
            diagnostic_branch="both_separate",
            debug_level="summary",
            debug_format="jsonl",
        ))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
        records = self._json_log_records(logs.records)
        spectral = [record for record in records if record["record_type"] == "spectral_diag"]
        by_branch = {record["diagnostic_branch"]: record for record in spectral}
        self.assertEqual(by_branch["negative"]["batch_indices"], [0])
        self.assertEqual(by_branch["positive"]["batch_indices"], [1])
        self.assertEqual(override.stats.steps[1].spectral_diagnostics["negative"].batch_indices, [0])
        self.assertEqual(override.stats.steps[1].spectral_diagnostics["positive"].batch_indices, [1])

    def test_diagnostic_include_unselected_emits_passthrough_branch_without_output_change(self):
        torch.manual_seed(270)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        original = SentinelAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(
            strength=0.2,
            branch_mode="positive_only",
            spectral_diag="sampled",
            diagnostic_branch="both_separate",
            diagnostic_include_unselected=True,
            debug_level="summary",
            debug_format="jsonl",
        ))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            out = override(original, q, k, v, 2, **self._cfg_kwargs(0.5))

        self.assertTrue(torch.equal(out[:1], original.reference_output[:1]))
        self.assertFalse(torch.equal(out[1:], original.reference_output[1:]))
        records = self._json_log_records(logs.records)
        spectral = [record for record in records if record["record_type"] == "spectral_diag"]
        by_branch = {record["diagnostic_branch"]: record for record in spectral}
        self.assertEqual(set(by_branch), {"negative", "positive"})
        self.assertFalse(by_branch["negative"]["edit_applied"])
        self.assertTrue(by_branch["positive"]["edit_applied"])
        self.assertEqual(by_branch["negative"]["rho_before"], by_branch["negative"]["rho_after"])
        self.assertEqual(by_branch["negative"]["delta_rho_local"], 0.0)

    def test_observe_mode_emits_branch_spectral_baseline_and_returns_original(self):
        torch.manual_seed(271)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        original = SentinelAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(
            mode="observe",
            strength=0.0,
            spectral_diag="sampled",
            diagnostic_branch="both_separate",
            debug_level="summary",
            debug_format="jsonl",
        ))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            out = override(original, q, k, v, 2, **self._cfg_kwargs(0.5))
        self.assertTrue(torch.equal(out, original.reference_output))
        self.assertEqual(original.calls, 1)

        records = self._json_log_records(logs.records)
        spectral = [record for record in records if record["record_type"] == "spectral_diag"]
        by_branch = {record["diagnostic_branch"]: record for record in spectral}
        self.assertEqual(set(by_branch), {"negative", "positive"})
        self.assertEqual({record["mode"] for record in spectral}, {"observe"})
        self.assertTrue(all(record["rho_before"] == record["rho_after"] for record in spectral))
        self.assertTrue(all(record["delta_rho_local"] == 0.0 for record in spectral))
        self.assertEqual(override.stats.edited_calls, 0)
        self.assertEqual(override.stats.observed_calls, 1)

    def test_diagnostic_call_indices_and_every_n_steps_filter_only_spectral_work(self):
        torch.manual_seed(272)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(
            spectral_diag="sampled",
            diagnostic_branch="selected_mean",
            diagnostic_call_indices="0,2",
            diagnostic_every_n_steps=2,
            debug_level="verbose",
            debug_format="jsonl",
        ))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            for _ in range(3):
                override(reference_attention, q, k, v, 2, **self._cfg_kwargs(1.0))
            for _ in range(3):
                override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
            override.finalize()

        records = self._json_log_records(logs.records)
        spectral = [record for record in records if record["record_type"] == "spectral_diag"]
        self.assertEqual([(record["step_index"], record["eligible_call_index"]) for record in spectral], [(0, 0), (0, 2)])
        self.assertEqual(override.stats.steps[0].eligible_call_indices, {0: 1, 1: 1, 2: 1})
        self.assertEqual(override.stats.steps[1].eligible_call_indices, {0: 1, 1: 1, 2: 1})

    def test_final_summary_exposes_all_eligible_call_indices(self):
        torch.manual_seed(273)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(debug_level="verbose", debug_format="jsonl"))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            for _ in range(4):
                override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
            override.finalize()
        records = self._json_log_records(logs.records)
        final_summary = [
            record for record in records
            if record["record_type"] == "step_final_summary" and record["step_index"] == 1
        ][0]
        self.assertEqual(final_summary["eligible"], 4)
        self.assertEqual(final_summary["eligible_call_indices"], {"0": 1, "1": 1, "2": 1, "3": 1})

    def test_step_final_summary_counts_fallback_after_snapshot_once(self):
        torch.manual_seed(28)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        q_self = torch.randn(2, 2, 16, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(debug_level="summary", debug_format="jsonl"))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(1.0))
            override(reference_attention, q_self, q_self, q_self, 2, **self._cfg_kwargs(1.0))
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
        records = self._json_log_records(logs.records)
        final_step_zero = [
            record for record in records
            if record["record_type"] == "step_final_summary" and record["step_index"] == 0
        ]
        self.assertEqual(len(final_step_zero), 1)
        self.assertEqual(final_step_zero[0]["fallbacks"], 1)
        self.assertEqual(final_step_zero[0]["fallback_reasons"], {"not_cross_attention": 1})

    def test_last_step_finalizes_when_expected_call_count_reached(self):
        torch.manual_seed(281)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        q_self = torch.randn(2, 2, 16, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(debug_level="summary", debug_format="jsonl"))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(1.0))
            override(reference_attention, q_self, q_self, q_self, 2, **self._cfg_kwargs(1.0))
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
            override(reference_attention, q_self, q_self, q_self, 2, **self._cfg_kwargs(0.5))

        records = self._json_log_records(logs.records)
        final_step_one = [
            record for record in records
            if record["record_type"] == "step_final_summary" and record["step_index"] == 1
        ]
        self.assertEqual(len(final_step_one), 1)
        self.assertEqual(final_step_one[0]["final_reason"], "expected_call_count_reached")
        self.assertEqual(final_step_one[0]["calls"], 2)
        self.assertEqual(final_step_one[0]["fallback_reasons"], {"not_cross_attention": 1})

    def test_finalize_emits_run_final_summary_for_last_step(self):
        torch.manual_seed(282)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(debug_level="summary", debug_format="jsonl"))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
            override.finalize()
        records = self._json_log_records(logs.records)
        run_summary = [record for record in records if record["record_type"] == "run_final_summary"][0]
        self.assertEqual(run_summary["last_step_index"], 1)
        self.assertEqual(run_summary["steps"]["1"]["eligible"], 1)
        self.assertEqual(run_summary["steps"]["1"]["eligible_call_indices"], {"0": 1})

    def test_eligible_call_index_and_block_metadata_are_logged(self):
        torch.manual_seed(29)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(debug_level="verbose", debug_format="jsonl"))
        kwargs = self._cfg_kwargs(0.5)
        kwargs["transformer_options"] = {
            **kwargs["transformer_options"],
            "block": ("input", 7),
            "module_path": "diffusion_model.blocks.7.attn",
        }
        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(reference_attention, q, k, v, 2, **kwargs)
            override(reference_attention, q, k, v, 2, **kwargs)
        records = self._json_log_records(logs.records)
        snapshots = [record for record in records if record["record_type"] == "step_snapshot"]
        self.assertEqual([record["eligible_call_index"] for record in snapshots], [0, 1])
        self.assertEqual(snapshots[0]["block_id"], "input:7")
        self.assertEqual(snapshots[0]["metadata"]["module_path"], "diffusion_model.blocks.7.attn")

    def test_absent_block_metadata_logs_unknown(self):
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(debug_level="summary", debug_format="jsonl"))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
        records = self._json_log_records(logs.records)
        snapshot = [record for record in records if record["record_type"] == "step_snapshot"][0]
        self.assertEqual(snapshot["block_id"], "unknown")
        self.assertEqual(snapshot["metadata"], {})

    def test_verbose_fallback_spam_is_throttled_and_summarized(self):
        q_self = torch.randn(2, 2, 16, 4)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(
            debug_level="verbose",
            debug_format="jsonl",
            max_verbose_fallbacks_per_step_per_reason=2,
        ))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            for _ in range(5):
                override(reference_attention, q_self, q_self, q_self, 2, **self._cfg_kwargs(1.0))
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
        records = self._json_log_records(logs.records)
        fallback_records = [
            record for record in records
            if record["record_type"] == "fallback" and record["step_index"] == 0
        ]
        final_summary = [
            record for record in records
            if record["record_type"] == "step_final_summary" and record["step_index"] == 0
        ][0]
        self.assertEqual(len(fallback_records), 2)
        self.assertEqual(final_summary["fallback_suppressed_reasons"], {"not_cross_attention": 3})

    def test_debug_format_both_preserves_text_and_emits_valid_jsonl(self):
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(debug_level="summary", debug_format="both"))
        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
        messages = [record.getMessage() for record in logs.records]
        self.assertTrue(any("step_snapshot" in message and message.startswith("[AnimaAFM]") for message in messages))
        json_records = self._json_log_records(logs.records)
        self.assertTrue(any(record["record_type"] == "step_snapshot" for record in json_records))

    def test_jsonl_path_writes_machine_readable_records(self):
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            jsonl_path = str(Path(temp_dir) / "afm-debug.jsonl")
            override = AnimaAFMAttentionOverride(AFMConfig(
                spectral_diag="sampled",
                debug_level="summary",
                debug_format="jsonl",
                jsonl_path=jsonl_path,
            ))
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
            lines = Path(jsonl_path).read_text(encoding="utf-8").splitlines()

        records = [json.loads(line) for line in lines]
        self.assertTrue(any(record["record_type"] == "step_snapshot" for record in records))
        spectral = [record for record in records if record["record_type"] == "spectral_diag"]
        self.assertTrue(spectral)
        self.assertIn("delta_rho_local", spectral[0])
        self.assertIn("eligible_call_index", spectral[0])

    def test_parser_and_compare_rows_emit_observe_vs_edit_trajectory_fields(self):
        observe_records = [
            {
                "record_type": "spectral_diag",
                "step_index": 1,
                "eligible_call_index": 7,
                "diagnostic_branch": "positive",
                "mode": "observe",
                "rho_before": 0.2,
                "rho_after": 0.2,
                "delta_rho": 0.0,
                "delta_rho_local": 0.0,
                "alpha_lf": None,
                "alpha_hf": None,
                "selected_indices": [1],
                "edit_applied": False,
            }
        ]
        edit_records = [
            {
                "record_type": "spectral_diag",
                "step_index": 1,
                "eligible_call_index": 7,
                "diagnostic_branch": "positive",
                "mode": "edit",
                "rho_before": 0.21,
                "rho_after": 0.27,
                "delta_rho": 0.06,
                "delta_rho_local": 0.06,
                "alpha_lf": 1.0,
                "alpha_hf": 1.1,
                "selected_indices": [1],
                "edit_applied": True,
            }
        ]
        observe_rows = parse_records(observe_records)
        edit_rows = parse_records(edit_records)
        self.assertEqual(observe_rows[0]["branch"], "positive")
        self.assertEqual(observe_rows[0]["selected_indices"], "[1]")

        comparison = compare_rows(observe_rows, edit_rows)
        self.assertEqual(comparison[0]["step_index"], 1)
        self.assertEqual(comparison[0]["eligible_call_index"], 7)
        self.assertAlmostEqual(comparison[0]["delta_rho_local"], 0.06)
        self.assertAlmostEqual(comparison[0]["delta_rho_vs_observe"], 0.07)

    def test_self_attention_falls_back(self):
        torch.manual_seed(3)
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 16, 4)
        v = torch.randn(1, 2, 16, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2))
        out = override(reference_attention, q, k, v, 2, skip_reshape=True)
        expected = reference_attention(q, k, v, 2, skip_reshape=True)
        self.assertTrue(torch.allclose(out, expected))
        self.assertEqual(override.stats.fallback_reasons["not_cross_attention"], 1)

    def test_fallback_calls_original_once(self):
        q = torch.randn(1, 2, 15, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        original = CountingAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2))
        override(original, q, k, v, 2, skip_reshape=True)
        self.assertEqual(original.calls, 1)

    def test_positional_mask_falls_back(self):
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        original = CountingAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2))
        override(original, q, k, v, 2, torch.ones(1, 1, 16, 5), skip_reshape=True)
        self.assertEqual(original.calls, 1)
        self.assertEqual(override.stats.fallback_reasons["mask_shape_unsupported"], 1)

    def test_non_square_falls_back(self):
        q = torch.randn(1, 2, 15, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2))
        override(reference_attention, q, k, v, 2, skip_reshape=True)
        self.assertEqual(override.stats.fallback_reasons["cannot_infer_spatial_shape"], 1)


if __name__ == "__main__":
    unittest.main()
