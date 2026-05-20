import math
import unittest

import torch

from anima_afm import (
    AFMConfig,
    AnimaAFMAttentionOverride,
    edit_logits_fft,
    infer_square_spatial_shape,
    normalized_token_entropy,
    progress_from_sigmas,
    radial_low_high_masks,
    schedule_alphas,
    selected_branch_indices,
)


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


class AnimaAFMTests(unittest.TestCase):
    def test_infer_square_spatial_shape(self):
        self.assertEqual(infer_square_spatial_shape(64), (8, 8))
        self.assertIsNone(infer_square_spatial_shape(65))

    def test_progress_from_sigmas(self):
        info = progress_from_sigmas({
            "sigmas": torch.tensor([0.5]),
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
        })
        self.assertIsNotNone(info)
        self.assertEqual(info.index, 1)
        self.assertEqual(info.total, 2)
        self.assertAlmostEqual(info.progress, 0.5)

    def test_schedule_alphas(self):
        config = AFMConfig(strength=0.2, schedule="curve")
        self.assertEqual(schedule_alphas(config, 0.0), (1.2, 1.0))
        self.assertEqual(schedule_alphas(config, 1.0), (1.0, 1.2))

    def test_branch_selection(self):
        idx = selected_branch_indices(4, [1, 0], "positive_only", torch.device("cpu"))
        self.assertEqual(idx.tolist(), [2, 3])
        idx = selected_branch_indices(4, [1, 0], "negative_only", torch.device("cpu"))
        self.assertEqual(idx.tolist(), [0, 1])

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
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.0))
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
        override = AnimaAFMAttentionOverride(AFMConfig(strength=0.0, entropy_gate=True))
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
