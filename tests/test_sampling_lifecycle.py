"""CPU contract tests for the ComfyUI outer-sampling/apply-model hooks.

The small executor reproduces the public wrapper contract at ComfyUI commit
15ef24d1c0333a3eba56c5cd153d8db65363ff8f. No ComfyUI/GPU import is needed.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_anima_afm_lifecycle_tests"


def copy_options(value):
    if isinstance(value, dict):
        return {key: copy_options(item) for key, item in value.items()}
    if isinstance(value, list):
        return value.copy()
    return value


class WrapperExecutor:
    def __init__(self, original, class_obj, wrappers, index=0):
        self.original = original
        self.class_obj = class_obj
        self.wrappers = wrappers
        self.index = index

    @classmethod
    def new_class_executor(cls, original, class_obj, wrappers, idx=0):
        return cls(original, class_obj, wrappers, idx)

    def __call__(self, *args, **kwargs):
        return WrapperExecutor(self.original, self.class_obj, self.wrappers, self.index + 1).execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if self.index == len(self.wrappers):
            return self.original(*args, **kwargs)
        return self.wrappers[self.index](self, *args, **kwargs)


class FakeModelPatcher:
    def __init__(self):
        self.model = SimpleNamespace(diffusion_model=type("Anima", (), {})())
        self.model_options = {"transformer_options": {"existing_option": "preserved"}}
        self.wrappers = {}

    def clone(self):
        cloned = FakeModelPatcher()
        cloned.model = self.model
        cloned.model_options = copy_options(self.model_options)
        cloned.wrappers = copy_options(self.wrappers)
        return cloned

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def set_model_patch(self, patch, name):
        self.model_options["transformer_options"].setdefault("patches", {}).setdefault(name, []).append(patch)

    def get_additional_models_with_key(self, key):
        return []


class FakeGuider:
    def __init__(self, patcher):
        self.model_patcher = patcher
        self.model_options = copy_options(patcher.model_options)
        # prepare_model_patcher merges patcher.wrappers into the cloned options.
        self.model_options["transformer_options"]["wrappers"] = copy_options(patcher.wrappers)

    def run(self, body, *args, **kwargs):
        wrappers = self.model_options["transformer_options"]["wrappers"].get("outer_sample", {})
        return WrapperExecutor.new_class_executor(body, self, [item for group in wrappers.values() for item in group]).execute(*args, **kwargs)


def reference_attention(q, k, v, heads, **kwargs):
    result = torch.softmax(q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5, dim=-1) @ v
    return result.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


class SamplingLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        sys.modules[PACKAGE] = package
        comfy_api = ModuleType("comfy_api")
        latest = ModuleType("comfy_api.latest")
        latest.ComfyExtension = type("ComfyExtension", (), {})
        latest.io = SimpleNamespace(ComfyNode=type("ComfyNode", (), {}), NodeOutput=lambda *values: values)
        comfy = ModuleType("comfy")
        extension = ModuleType("comfy.patcher_extension")
        extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", APPLY_MODEL="apply_model")
        comfy.patcher_extension = extension
        cls.modules = {"comfy_api": comfy_api, "comfy_api.latest": latest, "comfy": comfy, "comfy.patcher_extension": extension}
        with patch.dict(sys.modules, cls.modules):
            cls.nodes = importlib.import_module(f"{PACKAGE}.nodes")

    def setUp(self):
        self.comfy_modules = patch.dict(sys.modules, self.modules)
        self.comfy_modules.start()
        self.addCleanup(self.comfy_modules.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log_path = Path(directory.name) / "run.jsonl"
        generator = torch.Generator().manual_seed(11)
        self.q = torch.randn(1, 1, 4, 2, generator=generator)
        self.k = torch.randn(1, 1, 3, 2, generator=generator)
        self.v = torch.randn(1, 1, 3, 2, generator=generator)

    def make_patch(self, model=None, **changes):
        settings = dict(mode="observe", strength=0.2, cutoff=0.25, start_percent=0.0, end_percent=1.0,
                        schedule="constant", branch_mode="both", zero_strength_mode="observe", entropy_gate=False,
                        beta=20.0, gamma=4.0, preserve_dc=True, soft_mask=True, mask_width=0.05,
                        debug_level="summary", debug_format="jsonl", fail_mode="raise", max_logits_mib=1024.0,
                        max_peak_mib=4096.0, target_call_indices="all", scope_mode="all", stage_scope="",
                        jsonl_path=str(self.log_path))
        settings.update(changes)
        return self.nodes.AnimaAFMModelPatch.execute(model or FakeModelPatcher(), **settings)[0]

    def records(self):
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines() if line.strip()]

    def evaluate(self, guider, *, branch=0, calls=1):
        # inner_sample and BaseModel copy options while preserving callables.
        options = copy_options(guider.model_options)["transformer_options"]
        options.update(sample_sigmas=torch.tensor([1.0, 0.0]), sigmas=torch.tensor([1.0]), cond_or_uncond=[branch], afm_spatial_shape=(2, 2))
        override = options["optimized_attention_override"]
        wrappers = options.get("wrappers", {}).get("apply_model", {})

        def body(x, t, c_concat, c_crossattn, control, transformer_options):
            output = None
            for _ in range(calls):
                output = override(reference_attention, self.q, self.k, self.v, 1,
                                  skip_reshape=True, transformer_options=transformer_options.copy())
            return output

        output = WrapperExecutor.new_class_executor(body, object(), [item for group in wrappers.values() for item in group]).execute(
            self.q, torch.tensor([1.0]), None, self.k, None, options)
        return override, output

    def test_node_registers_supported_outer_and_apply_wrappers(self):
        original = FakeModelPatcher()
        patched = self.make_patch(original)
        self.assertIn("outer_sample", patched.wrappers)
        self.assertIn("apply_model", patched.wrappers)
        self.assertEqual(original.wrappers, {})
        self.assertNotIn("optimized_attention_override", original.model_options["transformer_options"])

    def test_reused_patch_gets_fresh_state_and_complete_single_step_logs(self):
        patched = self.make_patch()
        runs = []
        guider = FakeGuider(patched)
        for count in (1, 2):
            original_options = guider.model_options
            def sample():
                override, output = self.evaluate(guider, calls=count)
                runs.append(override)
                return output
            output = guider.run(sample)
            self.assertIs(guider.model_options, original_options)
            torch.testing.assert_close(output, reference_attention(self.q, self.k, self.v, 1))
        summaries = [row for row in self.records() if row["record_type"] == "run_final_summary"]
        self.assertEqual(len(summaries), 2)
        self.assertIsNot(runs[0], runs[1])
        self.assertIsNot(runs[0].stats, runs[1].stats)
        self.assertNotEqual(runs[0].run_id, runs[1].run_id)
        self.assertEqual([row["observed"] for row in summaries], [1, 2])
        self.assertEqual([row.get("sampling_status") for row in summaries], ["completed", "completed"])
        finals = [row for row in self.records() if row["record_type"] == "step_final_summary"]
        self.assertEqual(len(finals), 2)

    def test_sampling_exception_still_finalizes_and_restores_options(self):
        guider = FakeGuider(self.make_patch())
        original_options = guider.model_options
        error = RuntimeError("sampler failed")
        def sample():
            self.evaluate(guider)
            raise error
        with self.assertRaises(RuntimeError) as caught:
            guider.run(sample)
        self.assertIs(caught.exception, error)
        self.assertIs(guider.model_options, original_options)
        summaries = [row for row in self.records() if row["record_type"] == "run_final_summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].get("sampling_status"), "failed")
        self.assertEqual(summaries[0].get("sampling_error_type"), "RuntimeError")

    def test_sampling_interrupt_is_logged_and_propagated(self):
        guider = FakeGuider(self.make_patch())
        original_options = guider.model_options
        interruption = KeyboardInterrupt()
        def sample():
            self.evaluate(guider)
            raise interruption
        with self.assertRaises(KeyboardInterrupt) as caught:
            guider.run(sample)
        self.assertIs(caught.exception, interruption)
        self.assertIs(guider.model_options, original_options)
        summaries = [row for row in self.records() if row["record_type"] == "run_final_summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].get("sampling_status"), "interrupted")

    def test_comfy_interrupt_exception_is_classified_as_interrupted(self):
        class InterruptProcessingException(Exception):
            pass
        management = ModuleType("comfy.model_management")
        management.InterruptProcessingException = InterruptProcessingException
        guider = FakeGuider(self.make_patch())
        def sample():
            self.evaluate(guider)
            raise InterruptProcessingException()
        with patch.dict(sys.modules, {"comfy.model_management": management}):
            with self.assertRaises(InterruptProcessingException):
                guider.run(sample)
        summaries = [row for row in self.records() if row["record_type"] == "run_final_summary"]
        self.assertEqual(summaries[0].get("sampling_status"), "interrupted")

    def test_failure_before_attention_emits_failed_empty_run(self):
        guider = FakeGuider(self.make_patch())
        def sample():
            raise RuntimeError("before attention")
        with self.assertRaises(RuntimeError):
            guider.run(sample)
        summaries = [row for row in self.records() if row["record_type"] == "run_final_summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].get("sampling_status"), "failed")
        self.assertEqual(summaries[0]["steps"], {})

    def test_sampler_error_is_preserved_if_finalization_also_fails(self):
        guider = FakeGuider(self.make_patch())
        original_options = guider.model_options
        sampling_error = RuntimeError("original sampler failure")
        def sample():
            override, _ = self.evaluate(guider)
            def broken_finalize():
                raise OSError("log write failed")
            override.finalize = broken_finalize
            raise sampling_error
        with self.assertLogs(f"{PACKAGE}.sampling_lifecycle", level="ERROR"):
            with self.assertRaises(RuntimeError) as caught:
                guider.run(sample)
        self.assertIs(caught.exception, sampling_error)
        self.assertIs(guider.model_options, original_options)

    def test_split_cfg_evaluations_restart_call_selection_ordinal(self):
        guider = FakeGuider(self.make_patch(target_call_indices="0"))
        runs = []
        def sample():
            for branch in (0, 1):
                override, _ = self.evaluate(guider, branch=branch, calls=2)
                runs.append(override)
        guider.run(sample)
        self.assertEqual(runs[-1].stats.observed_calls, 2)
        self.assertEqual(runs[-1].stats.target_skipped_calls, 2)
        self.assertEqual(dict(runs[-1].stats.steps[0].eligible_call_indices), {0: 2, 1: 2})

    def test_post_input_hook_uses_runtime_patch_grid_and_preserves_existing_hooks(self):
        model = FakeModelPatcher()
        existing = lambda data: data
        model.set_model_patch(existing, "post_input")
        patched = self.make_patch(model)
        hooks = patched.model_options["transformer_options"].get("patches", {}).get("post_input", [])
        self.assertEqual(len(hooks), 2)
        self.assertIs(hooks[0], existing)
        self.assertEqual(model.model_options["transformer_options"]["patches"]["post_input"], [existing])
        options = {"unrelated": "keep"}
        img = torch.empty(1, 1, 2, 6, 8)
        data = {"img": img, "x": torch.empty(1, 4, 1, 4, 12), "transformer_options": options}
        result = hooks[-1](data)
        self.assertIs(result, data)
        self.assertIs(result["img"], img)
        self.assertEqual(options["afm_spatial_shape"], [2, 6])
        self.assertEqual(options["unrelated"], "keep")

    def test_post_input_hook_marks_temporal_or_unknown_layout_unsupported(self):
        patched = self.make_patch()
        hooks = patched.model_options["transformer_options"].get("patches", {}).get("post_input", [])
        self.assertTrue(hooks, "the patch must register the runtime post_input grid hook")
        for shape in ((1, 2, 2, 6, 8), (1, 12, 8)):
            options = {}
            hooks[-1]({"img": torch.empty(shape), "transformer_options": options})
            self.assertTrue(options.get("afm_spatial_shape_invalid"))
            self.assertEqual(options["afm_spatial_shape"], [0, 0])

    def test_post_input_hook_replaces_its_own_grid_between_resolutions(self):
        patched = self.make_patch()
        hook = patched.model_options["transformer_options"]["patches"]["post_input"][-1]
        options = {}
        for height, width in ((36, 64), (64, 36)):
            hook({"img": torch.empty(1, 1, height, width, 1), "transformer_options": options})
            core = importlib.import_module(f"{PACKAGE}.anima_afm")
            shape = core.infer_spatial_shape(height * width, core.AFMConfig(), options)
            self.assertEqual(shape.shape, (height, width))

    def test_post_input_hook_clears_its_own_invalid_flag_on_next_valid_image(self):
        patched = self.make_patch()
        hook = patched.model_options["transformer_options"]["patches"]["post_input"][-1]
        options = {}
        hook({"img": torch.empty(1, 2, 2, 6, 1), "transformer_options": options})
        hook({"img": torch.empty(1, 1, 2, 6, 1), "transformer_options": options})
        core = importlib.import_module(f"{PACKAGE}.anima_afm")
        self.assertEqual(core.infer_spatial_shape(12, core.AFMConfig(), options).shape, (2, 6))

    def test_post_input_hook_keeps_conflicting_external_grid_fail_closed(self):
        patched = self.make_patch()
        hook = patched.model_options["transformer_options"]["patches"]["post_input"][-1]
        options = {"afm_spatial_shape": [6, 2]}
        hook({"img": torch.empty(1, 1, 2, 6, 1), "transformer_options": options})
        core = importlib.import_module(f"{PACKAGE}.anima_afm")
        self.assertIsNone(core.infer_spatial_shape(12, core.AFMConfig(), options).shape)

    def test_pinned_comfy_executor_preserves_wrapper_chain_and_arguments(self):
        source = os.environ.get("AFM_COMFYUI_SOURCE_ROOT")
        if not source:
            self.skipTest("set AFM_COMFYUI_SOURCE_ROOT to the pinned ComfyUI checkout for upstream executor integration")
        path = Path(source) / "comfy" / "patcher_extension.py"
        spec = importlib.util.spec_from_file_location("_afm_pinned_patcher_extension", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        patched = self.make_patch()
        events = []
        def another_wrapper(executor, *args, **kwargs):
            events.append("before")
            result = executor(*args, **kwargs)
            events.append("after")
            return result
        patched.add_wrapper_with_key("outer_sample", "another_plugin", another_wrapper)
        guider = FakeGuider(patched)
        sentinel = object()
        def sample(first, *, second):
            self.assertIs(first, sentinel)
            self.assertEqual(second, "unchanged")
            override, output = self.evaluate(guider)
            self.assertEqual(override.model_evaluation_index, 0)
            events.append("sample")
            return output
        with patch.object(sys.modules[__name__], "WrapperExecutor", module.WrapperExecutor):
            output = guider.run(sample, sentinel, second="unchanged")
        torch.testing.assert_close(output, reference_attention(self.q, self.k, self.v, 1))
        self.assertEqual(events, ["before", "sample", "after"])
        self.assertEqual(len([row for row in self.records() if row["record_type"] == "run_final_summary"]), 1)

    def test_pinned_apply_model_contract_supplies_options_at_argument_six(self):
        source = os.environ.get("AFM_COMFYUI_SOURCE_ROOT")
        if not source:
            self.skipTest("set AFM_COMFYUI_SOURCE_ROOT for the pinned upstream signature check")
        tree = ast.parse((Path(source) / "comfy" / "model_base.py").read_text())
        base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseModel")
        apply = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "apply_model")
        self.assertEqual([arg.arg for arg in apply.args.args[:7]],
                         ["self", "x", "t", "c_concat", "c_crossattn", "control", "transformer_options"])
        calls = [node for node in ast.walk(apply) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "execute"]
        self.assertEqual(len(calls), 1)
        self.assertEqual([arg.id for arg in calls[0].args],
                         ["x", "t", "c_concat", "c_crossattn", "control", "transformer_options"])

    def test_parallel_multigpu_path_fails_before_attention(self):
        guider = FakeGuider(self.make_patch())
        guider.model_options["multigpu_clones"] = {"cpu": object()}
        with self.assertRaisesRegex(RuntimeError, "multi-GPU"):
            guider.run(lambda: self.fail("unsupported multi-GPU sampler must not execute"))

    def test_apply_model_outside_outer_sampling_is_explicitly_unsupported(self):
        patched = self.make_patch()
        guider = FakeGuider(patched)
        with self.assertRaisesRegex(RuntimeError, "no active outer sampling"):
            self.evaluate(guider)

    def test_missing_post_input_registration_api_is_explicitly_unsupported(self):
        model = FakeModelPatcher()
        cloned = model.clone()
        cloned.set_model_patch = None
        model.clone = lambda: cloned
        with self.assertRaisesRegex(RuntimeError, "unsupported|Unsupported"):
            self.make_patch(model)

    def test_missing_wrapper_api_is_explicitly_unsupported(self):
        model = FakeModelPatcher()
        model.clone = lambda: SimpleNamespace(model=model.model, model_options=copy_options(model.model_options))
        with self.assertRaisesRegex(RuntimeError, "unsupported|Unsupported"):
            self.make_patch(model)

    def test_direct_attention_without_sampling_lifecycle_is_explicitly_unsupported(self):
        patched = self.make_patch()
        override = patched.model_options["transformer_options"]["optimized_attention_override"]
        with self.assertRaisesRegex(RuntimeError, "OUTER_SAMPLE|outer.*sampl"):
            override(reference_attention, self.q, self.k, self.v, 1, skip_reshape=True)


if __name__ == "__main__":
    unittest.main()
