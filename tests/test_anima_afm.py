import json
from pathlib import Path
import tempfile
import unittest
import io

import torch
import torch.nn as nn

from anima_afm import (
    AFMConfig,
    AnimaAFMBlockMetadataWrapper,
    AnimaAFMAttentionOverride,
    edit_logits_fft,
    estimate_peak_mib,
    hf_ratio_from_concentration,
    infer_spatial_shape,
    infer_square_spatial_shape,
    normalized_token_entropy,
    parse_call_index_scope,
    progress_from_sigmas,
    radial_low_high_masks,
    sampled_spectral_diagnostics,
    schedule_alphas,
    selected_branch_indices,
    install_anima_block_metadata_wrappers,
)
from scripts.compare_afm_runs import compare_rows, main as compare_main, read_rows, summarize_rows
from scripts.build_afm_scope_map import build_scope_map
from scripts.parse_anima_afm_log import iter_json_records, parse_records, write_csv


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


class MetadataEchoCrossAttention(nn.Module):
    context_dim = 8

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        self.seen.append(dict(transformer_options or {}))
        return x + 1


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = MetadataEchoCrossAttention()


class FakeDiffusionModel(nn.Module):
    def __init__(self, block_count=3):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(block_count)])


class FakeModelPatcher:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model
        self.patches = {}

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self.diffusion_model
        raise KeyError(name)

    def add_object_patch(self, path, value):
        self.patches[path] = value
        current = self
        parts = path.split(".")
        for part in parts[:-1]:
            if part.isdigit():
                current = current[int(part)]
            else:
                current = getattr(current, part)
        setattr(current, parts[-1], value)


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

    def test_infer_spatial_shape_explicit_latent_rectangular(self):
        config = AFMConfig(
            spatial_shape_mode="explicit_latent",
            latent_width=84,
            latent_height=48,
        )
        info = infer_spatial_shape(4032, config, {}, {})
        self.assertEqual(info.shape, (48, 84))
        self.assertEqual(info.source, "explicit_latent")
        self.assertIsNone(info.aspect_error)

    def test_infer_spatial_shape_explicit_pixels_rectangular_by_downscale(self):
        config = AFMConfig(
            spatial_shape_mode="explicit_pixels",
            image_width=1344,
            image_height=768,
            latent_downscale=16,
        )
        info = infer_spatial_shape(4032, config, {}, {})
        self.assertEqual(info.shape, (48, 84))
        self.assertEqual(info.source, "explicit_pixels_downscale")

    def test_infer_spatial_shape_explicit_pixels_uses_aspect_when_square_product_is_possible(self):
        config = AFMConfig(
            spatial_shape_mode="explicit_pixels",
            image_width=1600,
            image_height=900,
            latent_downscale=16,
        )
        info = infer_spatial_shape(2304, config, {}, {})
        self.assertEqual(info.shape, (36, 64))
        self.assertEqual(info.source, "explicit_pixels_aspect")
        self.assertLess(info.aspect_error, 1e-9)

    def test_infer_spatial_shape_explicit_mismatch_is_not_silent_square(self):
        config = AFMConfig(
            spatial_shape_mode="explicit_latent",
            latent_width=64,
            latent_height=36,
        )
        info = infer_spatial_shape(4096, config, {}, {})
        self.assertIsNone(info.shape)
        self.assertEqual(info.reason, "spatial_shape_mismatch")

    def test_infer_spatial_shape_auto_reads_runtime_metadata(self):
        config = AFMConfig(spatial_shape_mode="auto")
        info = infer_spatial_shape(4032, config, {"latent_height": 48, "latent_width": 84}, {})
        self.assertEqual(info.shape, (48, 84))
        self.assertEqual(info.source, "runtime_metadata")

    def test_infer_spatial_shape_auto_rejects_ambiguous_runtime_metadata(self):
        config = AFMConfig(spatial_shape_mode="auto")
        info = infer_spatial_shape(
            2304,
            config,
            {
                "spatial_shape": [36, 64],
                "latent_shape": [48, 48],
            },
            {},
        )
        self.assertIsNone(info.shape)
        self.assertEqual(info.source, "runtime_metadata")
        self.assertEqual(info.reason, "spatial_shape_ambiguous")

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

    def test_rectangular_frequency_masks_and_diagnostics(self):
        low, high = radial_low_high_masks(3, 4, 0.25, True, 0.05, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(low.shape), (3, 4))
        self.assertEqual(tuple(high.shape), (3, 4))
        self.assertTrue(torch.allclose(low + high, torch.ones_like(low)))

        concentration = torch.rand(1, 2, 12)
        rho = hf_ratio_from_concentration(concentration, (3, 4), 0.25)
        self.assertIsInstance(rho, float)

        before = torch.randn(1, 2, 12, 5)
        after = before + 0.05 * torch.randn_like(before)
        diagnostics = sampled_spectral_diagnostics(
            before,
            after,
            (3, 4),
            AFMConfig(spectral_diag="sampled", diagnostic_branch="selected_mean"),
            torch.tensor([0]),
            1,
            None,
        )
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].branch, "selected_mean")

    def test_branch_selection(self):
        idx = selected_branch_indices(4, [1, 0], "positive_only", torch.device("cpu"))
        self.assertEqual(idx.tolist(), [2, 3])
        idx = selected_branch_indices(4, [1, 0], "negative_only", torch.device("cpu"))
        self.assertEqual(idx.tolist(), [0, 1])

    def test_parse_call_index_scope_supports_all_lists_and_ranges(self):
        self.assertIsNone(parse_call_index_scope("all"))
        self.assertEqual(parse_call_index_scope("0"), {0})
        self.assertEqual(parse_call_index_scope("0,7,14"), {0, 7, 14})
        self.assertEqual(parse_call_index_scope("7-13"), {7, 8, 9, 10, 11, 12, 13})
        self.assertEqual(parse_call_index_scope("0-0"), {0})
        self.assertEqual(parse_call_index_scope("0-2,7,10-11"), {0, 1, 2, 7, 10, 11})
        for invalid in ("", "-1", "7-3", "a", "1,,2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_call_index_scope(invalid)

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

    def test_observe_mode_respects_scope_filters(self):
        torch.manual_seed(221)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        original = CountingAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(
            mode="observe",
            scope_mode="block_scope",
            stage_scope="early",
            debug_level="verbose",
            debug_format="jsonl",
        ))
        selected_kwargs = self._cfg_kwargs(0.5)
        selected_kwargs["transformer_options"].update({"module_path": "model.blocks.1.attn2", "stage_tag": "early"})
        rejected_kwargs = self._cfg_kwargs(0.5)
        rejected_kwargs["transformer_options"].update({"module_path": "model.blocks.20.attn2", "stage_tag": "late"})

        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(original, q, k, v, 2, **selected_kwargs)
            override(original, q, k, v, 2, **rejected_kwargs)
            override.finalize()

        step = override.stats.steps[1]
        self.assertEqual(step.eligible_calls, 2)
        self.assertEqual(step.observed_calls, 1)
        self.assertEqual(step.scope_skipped_calls, 1)
        self.assertEqual(override.stats.edited_calls, 0)
        self.assertEqual(original.calls, 2)
        records = self._json_log_records(logs.records)
        snapshots = [record for record in records if record["record_type"] == "step_snapshot"]
        self.assertEqual([record["call_mode"] for record in snapshots], ["observe", "scope_skipped"])
        self.assertEqual(snapshots[1]["scope_reject_reason"], "block_scope_not_selected")

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
        self.assertEqual(snapshot["schema_version"], 2)
        self.assertIsInstance(snapshot["estimated_peak_mib"], float)
        self.assertAlmostEqual(snapshot["estimated_peak_mib"], estimate_peak_mib(snapshot["estimated_logits_mib"]))
        self.assertEqual(snapshot["max_peak_mib"], snapshot["estimated_peak_mib"])
        self.assertEqual(snapshot["memory_estimate"]["method"], "conservative_full_fft_multiplier")
        self.assertAlmostEqual(snapshot["memory_estimate"]["logits_mib"], snapshot["estimated_logits_mib"])
        self.assertEqual(snapshot["memory_estimate"]["selected_batch"], 1)
        self.assertEqual(snapshot["memory_estimate"]["heads"], 2)
        self.assertEqual(snapshot["memory_estimate"]["query_len"], 16)
        self.assertEqual(snapshot["memory_estimate"]["text_len"], 5)
        self.assertIn("fft_complex_mib", snapshot["memory_estimate"])

        original = CountingAttention()
        guarded = AnimaAFMAttentionOverride(AFMConfig(max_logits_mib=1024.0, max_peak_mib=0.0001))
        guarded(original, q, k, v, 2, **self._cfg_kwargs(0.5))
        self.assertEqual(original.calls, 1)
        self.assertEqual(guarded.stats.fallback_reasons["peak_vram_guard_exceeded"], 1)

        raising = AnimaAFMAttentionOverride(AFMConfig(max_logits_mib=0.0001, fail_mode="raise"))
        with self.assertRaises(RuntimeError):
            raising(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))

        raising_peak = AnimaAFMAttentionOverride(AFMConfig(max_logits_mib=1024.0, max_peak_mib=0.0001, fail_mode="raise"))
        with self.assertRaises(RuntimeError):
            raising_peak(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))

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

    def test_target_call_indices_range_scope_edits_seven_eligible_calls(self):
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(target_call_indices="0-6"))
        for _ in range(8):
            override(reference_attention, q, k, v, 2, **self._cfg_kwargs(0.5))
        step = override.stats.steps[1]
        self.assertEqual(step.edited_calls, 7)
        self.assertEqual(step.target_skipped_calls, 1)

    def test_discover_mode_returns_original_and_emits_scope_record(self):
        torch.manual_seed(262)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        original = SentinelAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(
            mode="discover",
            debug_level="summary",
            debug_format="jsonl",
        ))
        kwargs = self._cfg_kwargs(0.5)
        kwargs["transformer_options"].update({
            "module_path": "model.blocks.1.attn2",
            "module_class": "CrossAttention",
            "stage_tag": "encoder_candidate",
        })
        with self.assertLogs("anima_afm", level="INFO") as logs:
            out = override(original, q, k, v, 2, **kwargs)

        self.assertTrue(torch.equal(out, original.reference_output))
        self.assertEqual(original.calls, 1)
        self.assertEqual(override.stats.observed_calls, 1)
        self.assertEqual(override.stats.edited_calls, 0)
        records = self._json_log_records(logs.records)
        discovery = [record for record in records if record["record_type"] == "afm_scope_discovery_call"][0]
        self.assertEqual(discovery["block_id"], "model.blocks.1.attn2")
        self.assertEqual(discovery["block_path"], "model.blocks.1.attn2")
        self.assertEqual(discovery["module_class"], "CrossAttention")
        self.assertEqual(discovery["stage_tag"], "encoder_candidate")
        self.assertEqual(discovery["metadata_source"], "transformer_options")
        self.assertTrue(discovery["scope_selected"])

    def test_block_scope_edits_only_selected_block(self):
        torch.manual_seed(263)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        original = CountingAttention()
        override = AnimaAFMAttentionOverride(AFMConfig(
            scope_mode="block_scope",
            block_scope="model.blocks.1.attn2",
            debug_level="verbose",
            debug_format="jsonl",
        ))
        selected_kwargs = self._cfg_kwargs(0.5)
        selected_kwargs["transformer_options"].update({"module_path": "model.blocks.1.attn2"})
        rejected_kwargs = self._cfg_kwargs(0.5)
        rejected_kwargs["transformer_options"].update({"module_path": "model.blocks.2.attn2"})

        with self.assertLogs("anima_afm", level="INFO") as logs:
            override(original, q, k, v, 2, **selected_kwargs)
            override(original, q, k, v, 2, **rejected_kwargs)
            override.finalize()

        step = override.stats.steps[1]
        self.assertEqual(step.edited_calls, 1)
        self.assertEqual(step.scope_skipped_calls, 1)
        self.assertEqual(original.calls, 1)
        records = self._json_log_records(logs.records)
        snapshots = [record for record in records if record["record_type"] == "step_snapshot"]
        self.assertTrue(any(record["scope_selected"] is True for record in snapshots))
        self.assertTrue(any(record["scope_reject_reason"] == "block_scope_not_selected" for record in snapshots))
        final_summary = [
            record for record in records
            if record["record_type"] == "step_final_summary" and record["step_index"] == 1
        ][0]
        self.assertEqual(final_summary["scope_skipped"], 1)

    def test_encoder_equivalent_scope_map_edits_only_candidate(self):
        torch.manual_seed(264)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        original = CountingAttention()
        with tempfile.TemporaryDirectory() as tmp:
            scope_map = Path(tmp) / "scope_map.json"
            scope_map.write_text(json.dumps({
                "schema_version": 1,
                "entries": [
                    {
                        "scope_id": "block:model.blocks.3.attn2",
                        "block_path": "model.blocks.3.attn2",
                        "stage_tag": "encoder_candidate",
                    }
                ],
            }), encoding="utf-8")
            override = AnimaAFMAttentionOverride(AFMConfig(
                scope_mode="encoder_equivalent",
                scope_map_path=str(scope_map),
                debug_level="verbose",
                debug_format="jsonl",
            ))
            candidate_kwargs = self._cfg_kwargs(0.5)
            candidate_kwargs["transformer_options"].update({"module_path": "model.blocks.3.attn2"})
            rejected_kwargs = self._cfg_kwargs(0.5)
            rejected_kwargs["transformer_options"].update({"module_path": "model.blocks.4.attn2"})

            with self.assertLogs("anima_afm", level="INFO") as logs:
                override(original, q, k, v, 2, **candidate_kwargs)
                override(original, q, k, v, 2, **rejected_kwargs)
                override.finalize()

        step = override.stats.steps[1]
        self.assertEqual(step.edited_calls, 1)
        self.assertEqual(step.scope_skipped_calls, 1)
        self.assertEqual(original.calls, 1)
        records = self._json_log_records(logs.records)
        snapshots = [record for record in records if record["record_type"] == "step_snapshot"]
        self.assertTrue(any(record["scope_selected"] is True for record in snapshots))
        self.assertTrue(any(record["scope_reject_reason"] == "encoder_equivalent_not_selected" for record in snapshots))

    def test_scope_map_builder_promotes_explicit_encoder_candidates_only(self):
        scope_map = build_scope_map([
            {
                "record_type": "afm_scope_discovery_call",
                "run_id": "run-a",
                "step_index": 0,
                "eligible_call_index": 1,
                "block_id": "model.blocks.1.attn2",
                "block_path": "model.blocks.1.attn2",
                "module_class": "CrossAttention",
                "stage_tag": "encoder_candidate",
                "spatial_shape": [64, 64],
                "cond_or_uncond": [1, 0],
                "metadata_source": "transformer_options",
            },
            {
                "record_type": "afm_scope_discovery_call",
                "run_id": "run-a",
                "step_index": 0,
                "eligible_call_index": 2,
                "block_id": "unknown",
                "stage_tag": "unclassified",
                "fallback_reason": "missing_transformer_metadata",
            },
        ], model_fingerprint="model-x", workflow_fingerprint="workflow-y")

        self.assertEqual(scope_map["model_fingerprint"], "model-x")
        self.assertEqual(scope_map["workflow_fingerprint"], "workflow-y")
        entries = {entry["block_id"]: entry for entry in scope_map["entries"]}
        self.assertTrue(entries["model.blocks.1.attn2"]["encoder_equivalent"])
        self.assertFalse(entries["unknown"]["encoder_equivalent"])
        self.assertEqual(entries["model.blocks.1.attn2"]["eligible_call_indices_seen"], [1])
        self.assertIn("stage_tag=encoder_candidate", entries["model.blocks.1.attn2"]["candidate_reasons"])

    def test_scope_map_path_is_ignored_outside_encoder_equivalent_mode(self):
        override = AnimaAFMAttentionOverride(AFMConfig(
            scope_mode="all",
            scope_map_path="does-not-exist.json",
            fail_mode="raise",
        ))
        self.assertIsNotNone(override)

    def test_block_metadata_wrapper_injects_and_restores_transformer_options(self):
        original = MetadataEchoCrossAttention()
        wrapper = AnimaAFMBlockMetadataWrapper(
            original=original,
            block_index=1,
            block_count=3,
            block_path="diffusion_model.blocks.1.cross_attn",
        )
        options = {"sigmas": "keep"}
        x = torch.zeros(1)
        out = wrapper(x, transformer_options=options)

        self.assertTrue(torch.equal(out, torch.ones(1)))
        self.assertEqual(options, {"sigmas": "keep"})
        seen = original.seen[0]
        self.assertEqual(seen["block_id"], "blocks.1.cross_attn")
        self.assertEqual(seen["block_index"], 1)
        self.assertEqual(seen["block_path"], "diffusion_model.blocks.1.cross_attn")
        self.assertEqual(seen["stage_tag"], "middle")

    def test_block_metadata_wrapper_exposes_4d_input_shape_candidate(self):
        original = MetadataEchoCrossAttention()
        wrapper = AnimaAFMBlockMetadataWrapper(
            original=original,
            block_index=0,
            block_count=3,
            block_path="diffusion_model.blocks.0.cross_attn",
        )
        options = {}
        x = torch.zeros(2, 36, 64, 8)
        wrapper(x, transformer_options=options)

        seen = original.seen[0]
        info = infer_spatial_shape(2304, AFMConfig(spatial_shape_mode="auto"), seen, {})
        self.assertEqual(info.shape, (36, 64))
        self.assertEqual(info.source, "runtime_metadata")
        self.assertNotIn("afm_spatial_shape_candidates", options)

    def test_install_anima_block_metadata_wrappers_uses_object_patches(self):
        diffusion_model = FakeDiffusionModel(block_count=3)
        patcher = FakeModelPatcher(diffusion_model)
        installed = install_anima_block_metadata_wrappers(patcher)

        self.assertEqual(installed, 3)
        self.assertEqual(sorted(patcher.patches), [
            "diffusion_model.blocks.0.cross_attn",
            "diffusion_model.blocks.1.cross_attn",
            "diffusion_model.blocks.2.cross_attn",
        ])
        self.assertIsInstance(diffusion_model.blocks[0].cross_attn, AnimaAFMBlockMetadataWrapper)

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
        self.assertEqual(by_branch["positive"]["schema_version"], 2)
        self.assertEqual(by_branch["positive"]["call_mode"], "edit")
        self.assertEqual(by_branch["positive"]["diagnostic_mode"], "edited")
        self.assertEqual(by_branch["positive"]["edit_selected_indices"], [1])
        self.assertEqual(by_branch["positive"]["diagnostic_batch_indices"], [1])
        self.assertEqual(by_branch["positive"]["selected_indices"], [1])
        self.assertEqual(by_branch["positive"]["batch_indices"], [1])
        self.assertEqual(by_branch["negative"]["call_mode"], "edit")
        self.assertEqual(by_branch["negative"]["diagnostic_mode"], "passthrough")
        self.assertEqual(by_branch["negative"]["edit_selected_indices"], [1])
        self.assertEqual(by_branch["negative"]["diagnostic_batch_indices"], [0])
        self.assertFalse(by_branch["negative"]["edit_applied"])
        self.assertTrue(by_branch["positive"]["edit_applied"])
        self.assertEqual(by_branch["negative"]["rho_before"], by_branch["negative"]["rho_after"])
        self.assertEqual(by_branch["negative"]["delta_rho_local"], 0.0)

    def test_diagnostic_include_unselected_does_not_change_tensor_output(self):
        torch.manual_seed(274)
        q = torch.randn(2, 2, 16, 4)
        k = torch.randn(2, 2, 5, 4)
        v = torch.randn(2, 2, 5, 4)
        kwargs = self._cfg_kwargs(0.5)
        base_config = {
            "strength": 0.2,
            "branch_mode": "positive_only",
            "spectral_diag": "sampled",
            "diagnostic_branch": "both_separate",
            "debug_level": "summary",
            "debug_format": "jsonl",
        }
        without_unselected = AnimaAFMAttentionOverride(AFMConfig(
            **base_config,
            diagnostic_include_unselected=False,
        ))
        with_unselected = AnimaAFMAttentionOverride(AFMConfig(
            **base_config,
            diagnostic_include_unselected=True,
        ))

        out_without = without_unselected(reference_attention, q, k, v, 2, **kwargs)
        with self.assertLogs("anima_afm", level="INFO") as logs:
            out_with = with_unselected(reference_attention, q, k, v, 2, **kwargs)
        torch.testing.assert_close(out_with, out_without, rtol=0, atol=0)

        records = self._json_log_records(logs.records)
        by_branch = {
            record["diagnostic_branch"]: record
            for record in records
            if record["record_type"] == "spectral_diag"
        }
        self.assertEqual(by_branch["negative"]["schema_version"], 2)
        self.assertEqual(by_branch["negative"]["diagnostic_mode"], "passthrough")
        self.assertFalse(by_branch["negative"]["edit_applied"])

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
        self.assertEqual({record["call_mode"] for record in spectral}, {"observe"})
        self.assertEqual({record["diagnostic_mode"] for record in spectral}, {"observe"})
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
        self.assertEqual(run_summary["schema_version"], 2)
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
        discovery = [record for record in records if record["record_type"] == "metadata_discovery"]
        self.assertEqual(len(discovery), 1)
        self.assertEqual(discovery[0]["schema_version"], 2)
        self.assertTrue(discovery[0]["metadata_available"])
        self.assertEqual(discovery[0]["metadata_status"], "found")
        self.assertIn("transformer_option_keys", discovery[0])
        self.assertIn("block", discovery[0]["safe_transformer_options"])
        self.assertEqual(discovery[0]["eligible_call_to_block"], {"0": "input:7"})
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
        discovery = [record for record in records if record["record_type"] == "metadata_discovery"][0]
        self.assertFalse(discovery["metadata_available"])
        self.assertEqual(discovery["metadata_status"], "not_found")
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
        with tempfile.TemporaryDirectory() as temp_dir:
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
                "schema_version": 1,
                "record_type": "spectral_diag",
                "run_id": "observe",
                "step_index": 1,
                "num_steps": 2,
                "last_index": 1,
                "u": 1.0,
                "sigma": 0.5,
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
                "schema_version": 2,
                "record_type": "spectral_diag",
                "run_id": "edit",
                "step_index": 1,
                "num_steps": 2,
                "last_index": 1,
                "u": 1.0,
                "sigma": 0.5,
                "eligible_call_index": 7,
                "target_call_indices": "7-13",
                "diagnostic_call_indices": "0,7",
                "target_call_selected": True,
                "diagnostic_call_selected": True,
                "diagnostic_branch": "positive",
                "call_mode": "edit",
                "diagnostic_mode": "edited",
                "mode": "edit",
                "rho_before": 0.21,
                "rho_after": 0.27,
                "delta_rho": 0.06,
                "delta_rho_local": 0.06,
                "alpha_lf": 1.0,
                "alpha_hf": 1.1,
                "edit_selected_indices": [1],
                "diagnostic_batch_indices": [1],
                "edit_applied": True,
            }
        ]
        observe_rows = parse_records(observe_records)
        edit_rows = parse_records(edit_records)
        self.assertEqual(observe_rows[0]["schema_version"], 1)
        self.assertEqual(observe_rows[0]["branch"], "positive")
        self.assertEqual(observe_rows[0]["call_mode"], "observe")
        self.assertEqual(observe_rows[0]["diagnostic_mode"], "observe")
        self.assertEqual(observe_rows[0]["selected_indices"], "[1]")
        self.assertEqual(edit_rows[0]["edit_selected_indices"], "[1]")
        self.assertEqual(edit_rows[0]["diagnostic_batch_indices"], "[1]")

        csv_out = io.StringIO()
        write_csv(edit_rows, csv_out)
        self.assertIn("schema_version,record_type,run_id", csv_out.getvalue().splitlines()[0])
        self.assertIn("diagnostic_batch_indices", csv_out.getvalue().splitlines()[0])

        comparison = compare_rows(observe_rows, edit_rows)
        self.assertEqual(comparison[0]["pair_status"], "matched")
        self.assertEqual(comparison[0]["step_index"], 1)
        self.assertEqual(comparison[0]["eligible_call_index"], 7)
        self.assertAlmostEqual(comparison[0]["rho_edit_before"], 0.21)
        self.assertAlmostEqual(comparison[0]["delta_rho_local"], 0.06)
        self.assertAlmostEqual(comparison[0]["delta_rho_vs_observe"], 0.07)
        self.assertEqual(comparison[0]["diagnostic_mode"], "edited")

    def test_parser_skips_malformed_json_unless_strict(self):
        lines = [
            '{"record_type":"spectral_diag","step_index":0,"eligible_call_index":0,"diagnostic_branch":"positive"}\n',
            '{bad json}\n',
            '["not", "a", "record"]\n',
            '\n',
        ]
        self.assertEqual(len(list(iter_json_records(lines))), 1)
        with self.assertRaises(json.JSONDecodeError):
            list(iter_json_records(lines, strict=True))

    def test_compare_detects_missing_observe_and_summarizes(self):
        edit_rows = parse_records([
            {
                "schema_version": 2,
                "record_type": "spectral_diag",
                "step_index": 16,
                "eligible_call_index": 7,
                "diagnostic_branch": "positive",
                "call_mode": "edit",
                "diagnostic_mode": "edited",
                "rho_before": 0.21,
                "rho_after": 0.27,
                "delta_rho_local": 0.06,
                "edit_applied": True,
            }
        ])
        rows = compare_rows([], edit_rows)
        self.assertEqual(rows[0]["pair_status"], "missing_observe")
        summary = summarize_rows(rows, late_start_step=16)
        self.assertEqual(summary["rows"], 1)
        self.assertEqual(summary["missing_observe"], 1)
        self.assertEqual(summary["late_window"]["start_step"], 16)

        with tempfile.TemporaryDirectory() as temp_dir:
            observe_path = Path(temp_dir) / "observe.jsonl"
            edit_path = Path(temp_dir) / "edit.jsonl"
            observe_path.write_text("", encoding="utf-8")
            edit_path.write_text(json.dumps({
                "record_type": "spectral_diag",
                "step_index": 16,
                "eligible_call_index": 7,
                "diagnostic_branch": "positive",
                "rho_after": 0.27,
            }) + "\n", encoding="utf-8")
            self.assertEqual(compare_main([str(observe_path), str(edit_path), "--fail-on-missing-observe"]), 3)

    def test_compare_reads_csv_and_jsonl_inputs(self):
        observe_record = {
            "record_type": "spectral_diag",
            "step_index": 0,
            "eligible_call_index": 0,
            "diagnostic_branch": "positive",
            "mode": "observe",
            "rho_after": 0.2,
        }
        edit_record = {
            "record_type": "spectral_diag",
            "step_index": 0,
            "eligible_call_index": 0,
            "diagnostic_branch": "positive",
            "mode": "edit",
            "rho_before": 0.2,
            "rho_after": 0.25,
            "delta_rho_local": 0.05,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "observe.jsonl"
            csv_path = Path(temp_dir) / "edit.csv"
            jsonl_path.write_text(json.dumps(observe_record) + "\n", encoding="utf-8")
            edit_rows = parse_records([edit_record])
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                write_csv(edit_rows, handle)

            self.assertEqual(len(read_rows(jsonl_path, input_format="jsonl")), 1)
            self.assertEqual(len(read_rows(csv_path, input_format="csv")), 1)

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

    def test_rectangular_explicit_latent_edits_attention(self):
        torch.manual_seed(4)
        q = torch.randn(1, 2, 12, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        config = AFMConfig(
            strength=0.2,
            preserve_dc=False,
            spatial_shape_mode="explicit_latent",
            latent_width=4,
            latent_height=3,
        )
        override = AnimaAFMAttentionOverride(config)
        out = override(reference_attention, q, k, v, 2, **self._cfg_kwargs(1.0))
        baseline = reference_attention(q, k, v, 2, **self._cfg_kwargs(1.0))
        self.assertEqual(override.stats.edited_calls, 1)
        self.assertEqual(override.stats.steps[0].shape_counts["q12,k5,h2,d4"], 1)
        self.assertFalse(torch.allclose(out, baseline))

    def test_rectangular_shape_mismatch_falls_back(self):
        q = torch.randn(1, 2, 16, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        original = CountingAttention()
        config = AFMConfig(
            strength=0.2,
            spatial_shape_mode="explicit_latent",
            latent_width=5,
            latent_height=3,
        )
        override = AnimaAFMAttentionOverride(config)
        override(original, q, k, v, 2, **self._cfg_kwargs(1.0))
        self.assertEqual(original.calls, 1)
        self.assertEqual(override.stats.fallback_reasons["spatial_shape_mismatch"], 1)

    def test_non_square_falls_back(self):
        q = torch.randn(1, 2, 15, 4)
        k = torch.randn(1, 2, 5, 4)
        v = torch.randn(1, 2, 5, 4)
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.2))
        override(reference_attention, q, k, v, 2, **self._cfg_kwargs(1.0))
        self.assertEqual(override.stats.fallback_reasons["spatial_shape_missing"], 1)


if __name__ == "__main__":
    unittest.main()
