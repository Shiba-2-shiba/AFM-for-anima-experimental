"""Regression tests for evidence-safe log comparison and preservation checks."""
import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from collections import Counter

from scripts.compare_afm_runs import compare_rows, main as compare_main, summarize_rows, write_csv
from scripts.parse_anima_afm_log import iter_json_records, parse_records, write_csv as write_parsed_csv
from scripts.summarize_afm_experiments import main as summary_main, summarize_comparisons


def record(run_id="observe", **changes):
    row = dict(record_type="spectral_diag", schema_version=2, run_id=run_id,
               step_index=0, model_evaluation_index=0, eligible_call_index=7, diagnostic_branch="negative",
               block_id="blocks.7.cross_attn", block_path="blocks.7.cross_attn",
               num_steps=20, sigma=1.0, spatial_shape=[8, 8],
               q_shape=[2, 4, 64, 8], k_shape=[2, 4, 32, 8], v_shape=[2, 4, 32, 8],
               diagnostic_batch_indices=[0], cutoff=0.25, diagnostic_top_k=8,
               spectral_diag="sampled", diagnostic_max_batches=1, diagnostic_max_heads=4,
               rho_definition="centered_concentration_hf_energy_ratio",
               rho_centering="spatial_mean_subtracted", metric_defined=True, metric_undefined_reason=None, rho_before=0.2, rho_after=0.2,
               delta_rho_local=0.0, edit_applied=False, call_mode="observe",
               diagnostic_mode="observe", attn_delta_mean=0.0, attn_delta_max=0.0)
    row.update(changes)
    return row


def manifest(run_id):
    return dict(schema_version=1, run_id=run_id, source_revision="a" * 40,
                workflow_sha256="b" * 64,
                generation_settings=dict(seed=7, model="anima.safetensors", model_sha256="c" * 64,
                    prompt="landscape", negative_prompt="", sampler="euler", scheduler="normal",
                    num_steps=20, cfg=4.0, width=128, height=128, batch_size=1))


def terminal(records, **changes):
    counts = Counter(str(row["step_index"]) for row in records)
    result = dict(record_type="run_final_summary", run_id=records[0]["run_id"],
        sampling_status="completed", sampling_error_type=None, telemetry_errors=0,
        fallback_reasons={"not_cross_attention": 20}, fallbacks=20, last_step_index=records[0]["num_steps"] - 1,
        steps={str(i): {"spectral_diag_count": counts[str(i)]} for i in range(records[0]["num_steps"])})
    result.update(changes)
    return result


def with_completion(records, **changes):
    return parse_records([*records, terminal(records, **changes)])


def compared_records():
    return with_completion([record()]), with_completion([record("edit", call_mode="edit", diagnostic_mode="passthrough")])


class ComparisonValidationTests(unittest.TestCase):
    def test_duplicate_observe_requires_explicit_averaging(self):
        obs, edit = compared_records()
        obs.append(dict(obs[0], rho_after=0.6))
        rows = compare_rows(obs, edit)
        self.assertIsNone(rows[0]["rho_observe"])
        self.assertIsNone(rows[0]["delta_rho_vs_observe"])
        allowed = compare_rows(obs, edit, allow_duplicate_average=True)
        self.assertAlmostEqual(allowed[0]["rho_observe"], 0.4)
        self.assertEqual(allowed[0]["pair_status"], "duplicate_averaged")

    def test_block_identity_participates_in_pairing(self):
        obs, edit = compared_records()
        edit[0]["block_id"] = "blocks.8.cross_attn"
        rows = compare_rows(obs, edit)
        self.assertNotEqual(rows[0]["pair_status"], "matched")
        self.assertIsNone(rows[0]["delta_rho_vs_observe"])

    def test_known_incompatible_row_settings_are_never_subtracted(self):
        for field, value in (("sigma", 0.8), ("num_steps", 30), ("spatial_shape", "[4,16]"),
                             ("diagnostic_batch_indices", "[1]"), ("cutoff", 0.5),
                             ("rho_centering", "none"), ("rho_definition", "raw_concentration_hf_energy_ratio")):
            with self.subTest(field=field):
                obs, edit = compared_records()
                edit[0][field] = value
                row = compare_rows(obs, edit)[0]
                self.assertEqual(row["pair_status"], "incompatible")
                self.assertIsNone(row["delta_rho_vs_observe"])

    def test_unknown_historical_provenance_stays_unverified(self):
        obs, edit = compared_records()
        for row in obs + edit:
            for key in ("run_id", "rho_definition", "rho_centering"):
                row.pop(key)
        result = compare_rows(obs, edit)
        self.assertEqual(result[0]["provenance_status"], "unverified")
        self.assertEqual(result[0]["rho_definition"], "legacy_unspecified")
        self.assertFalse(summarize_rows(result)["validation_passed"])

    def test_mixed_runs_do_not_average_even_with_duplicate_opt_in(self):
        obs, edit = compared_records()
        obs.append(dict(obs[0], run_id="another", rho_after=0.8))
        result = compare_rows(obs, edit, allow_duplicate_average=True)
        self.assertIsNone(result[0]["delta_rho_vs_observe"])
        self.assertFalse(summarize_rows(result)["validation_passed"])

    def test_nonfinite_rho_never_becomes_numeric_evidence(self):
        obs, edit = compared_records()
        for invalid in (float("nan"), float("inf"), "-Infinity", True):
            with self.subTest(invalid=invalid):
                edit[0]["rho_after"] = invalid
                rows = compare_rows(obs, edit)
                self.assertIsNone(rows[0]["delta_rho_vs_observe"])
                self.assertFalse(summarize_rows(rows)["validation_passed"])

    def test_missing_or_fractional_pair_key_is_invalid(self):
        for value in (None, 0.5, True):
            obs, edit = compared_records()
            obs[0]["step_index"] = edit[0]["step_index"] = value
            rows = compare_rows(obs, edit)
            self.assertNotEqual(rows[0]["pair_status"], "matched")
            self.assertIsNone(rows[0]["delta_rho_vs_observe"])

    def test_tensor_and_metric_fields_survive_csv_pipeline(self):
        obs, edit = compared_records()
        edit[0].update(attn_delta_mean=0.01, attn_delta_max=0.03)
        output = io.StringIO()
        write_csv(compare_rows(obs, edit), output)
        headers = next(csv.reader(io.StringIO(output.getvalue())))
        self.assertEqual(len(headers), len(set(headers)))
        row = next(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(float(row["attn_delta_max"]), 0.03)
        self.assertEqual(row["rho_definition"], "centered_concentration_hf_energy_ratio")
        self.assertEqual(row["observe_run_id"], "observe")
        self.assertEqual(row["edit_run_id"], "edit")

    def test_strict_json_parser_rejects_nonobjects_and_nonfinite(self):
        for line in ("[]", "null", '{"rho_after":NaN}'):
            with self.subTest(line=line), self.assertRaises(ValueError):
                list(iter_json_records([line], strict=True))

    def test_cli_strict_manifests_and_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            args = []
            for run in ("observe", "edit"):
                records = [record(run)]
                if run == "edit":
                    records[0].update(call_mode="edit", diagnostic_mode="passthrough")
                (path / f"{run}.jsonl").write_text("\n".join(json.dumps(row) for row in [*records, terminal(records)]) + "\n")
                with (path / f"{run}.csv").open("w") as handle:
                    write_parsed_csv(with_completion(records), handle)
                (path / f"{run}.manifest.json").write_text(json.dumps(manifest(run)))
            manifest_args = ["--observe-manifest", str(path / "observe.manifest.json"),
                             "--edit-manifest", str(path / "edit.manifest.json")]
            for extension in ("jsonl", "csv"):
                args = [str(path / f"observe.{extension}"), str(path / f"edit.{extension}"), "--strict-validation", "--summary"]
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(compare_main(args + manifest_args), 0)
                summary = json.loads(output.getvalue())
                self.assertTrue(summary["validation_passed"])
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertNotEqual(compare_main(args), 0)
            wrong = manifest("edit")
            wrong["generation_settings"]["seed"] = 8
            (path / "edit.manifest.json").write_text(json.dumps(wrong))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertNotEqual(compare_main(args + manifest_args), 0)

    def test_repeated_model_evaluations_are_distinct_pairs(self):
        obs, edit = compared_records()
        obs.append(dict(obs[0], model_evaluation_index=1))
        edit.append(dict(edit[0], model_evaluation_index=1))
        rows = compare_rows(with_completion(obs), with_completion(edit), observe_manifest=manifest("observe"), edit_manifest=manifest("edit"))
        self.assertEqual([row["pair_status"] for row in rows], ["matched", "matched"])
        self.assertTrue(summarize_rows(rows)["validation_passed"])

    def test_invalid_sampling_and_shape_metadata_fail_validation(self):
        for field, value in (("diagnostic_batch_indices", "[]"), ("diagnostic_max_heads", 0),
                             ("diagnostic_top_k", -1), ("spatial_shape", "[0,8]"),
                             ("q_shape", "[2,4,32,8]"), ("model_evaluation_index", None),
                             ("cutoff", 1.2), ("step_index", 20), ("k_shape", "[1,4,32,8]"),
                             ("rho_definition", "invented"), ("rho_centering", "invented"),
                             ("metric_defined", False), ("metric_defined", None)):
            with self.subTest(field=field):
                obs, edit = compared_records()
                obs[0][field] = edit[0][field] = value
                rows = compare_rows(obs, edit, observe_manifest=manifest("observe"), edit_manifest=manifest("edit"))
                self.assertFalse(summarize_rows(rows)["validation_passed"])

    def test_invalid_edit_flag_is_not_coerced_to_unedited(self):
        from scripts.compare_afm_runs import read_rows
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.csv"
            with path.open("w") as handle:
                write_parsed_csv(parse_records([record(edit_applied="garbage")]), handle)
            row = read_rows(path)[0]
            self.assertEqual(row["edit_applied"], "garbage")
            result = compare_rows([row], [dict(row, run_id="edit")],
                                  observe_manifest=manifest("observe"), edit_manifest=manifest("edit"))
            self.assertFalse(summarize_rows(result)["validation_passed"])

    def test_metric_definition_changes_within_run_fail_validation(self):
        obs, edit = compared_records()
        for rows in (obs, edit):
            rows.append(dict(rows[0], step_index=1, rho_definition="raw_concentration_hf_energy_ratio", rho_centering="none"))
        result = compare_rows(obs, edit, observe_manifest=manifest("observe"), edit_manifest=manifest("edit"))
        self.assertFalse(summarize_rows(result)["validation_passed"])

    def test_observe_input_must_be_an_unedited_baseline(self):
        obs, edit = compared_records()
        obs[0].update(call_mode="edit", diagnostic_mode="edited", edit_applied=True)
        result = compare_rows(obs, edit, observe_manifest=manifest("observe"), edit_manifest=manifest("edit"))
        self.assertFalse(summarize_rows(result)["validation_passed"])

    def test_known_manifest_mismatch_never_produces_cross_run_delta(self):
        obs, edit = compared_records()
        changed = manifest("edit")
        changed["generation_settings"]["seed"] = 8
        rows = compare_rows(obs, edit, observe_manifest=manifest("observe"), edit_manifest=changed)
        self.assertIsNone(rows[0]["delta_rho_vs_observe"])
        self.assertEqual(rows[0]["pair_status"], "invalid_provenance")

    def test_strict_compare_output_passes_strict_summary(self):
        obs, edit = compared_records()
        rows = compare_rows(obs, edit, observe_manifest=manifest("observe"), edit_manifest=manifest("edit"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with (path / "compare.csv").open("w") as handle:
                write_csv(rows, handle)
            self.assertEqual(summary_main(["--compare", f"D={path / 'compare.csv'}", "--out-dir", str(path / "out"),
                "--strict-validation", "--preservation-run", "D", "--require-branch-preservation"]), 0)
            summary = json.loads((path / "out" / "summary.json").read_text())
            self.assertTrue(summary["validation_passed"])
            self.assertEqual(summary["overview"][0]["provenance_status"], "manifest_verified")
            provenance = summary["provenance_by_comparison"]["D"]
            self.assertEqual(provenance["observe_manifest"]["generation_settings"]["seed"], 7)
            self.assertEqual(provenance["edit_run_id"], "edit")

    def test_unmatched_observe_count_survives_flat_rows(self):
        obs, edit = compared_records()
        obs.append(dict(obs[0], step_index=1))
        self.assertEqual(summarize_rows(list(compare_rows(obs, edit)))["unmatched_observe_pairs"], 1)

    def test_matching_failed_truncated_or_dirty_runs_fail_strict(self):
        cases = [dict(sampling_status="failed"), dict(sampling_status="interrupted"),
                 dict(telemetry_errors=1), dict(fallbacks=21), dict(fallback_reasons={"unexpected_error": 1}),
                 dict(last_step_index=0), dict(steps={"0": {"spectral_diag_count": 1}}),
                 dict(steps={str(i): {"spectral_diag_count": 0} for i in range(20)})]
        for changes in cases:
            with self.subTest(changes=changes):
                obs = with_completion([record()], **changes)
                edit = with_completion([record("edit", call_mode="edit")], **changes)
                rows = compare_rows(obs, edit, observe_manifest=manifest("observe"), edit_manifest=manifest("edit"))
                self.assertFalse(summarize_rows(rows)["validation_passed"])

    def test_missing_terminal_or_rows_removed_from_parsed_csv_fail_strict(self):
        obs, edit = compared_records()
        missing = parse_records([record()])
        self.assertFalse(summarize_rows(compare_rows(missing, edit, observe_manifest=manifest("observe"), edit_manifest=manifest("edit")))["validation_passed"])
        obs = with_completion([record(), record(step_index=1)])[:1]
        edit = with_completion([record("edit"), record("edit", step_index=1)])[:1]
        self.assertFalse(summarize_rows(compare_rows(obs, edit, observe_manifest=manifest("observe"), edit_manifest=manifest("edit")))["validation_passed"])

    def test_strict_summary_rechecks_terminal_coverage_after_rows_removed(self):
        obs = with_completion([record(), record(step_index=1)])
        edit = with_completion([record("edit", call_mode="edit"), record("edit", step_index=1, call_mode="edit")])
        rows = compare_rows(obs, edit, observe_manifest=manifest("observe"), edit_manifest=manifest("edit"))
        self.assertTrue(summarize_rows(rows)["validation_passed"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "compare.json").write_text(json.dumps(rows[:1]))
            self.assertNotEqual(summary_main(["--compare", f"D={path / 'compare.json'}", "--out-dir", str(path / "out"), "--strict-validation"]), 0)

    def test_empty_data_cannot_pass_strict_validation(self):
        self.assertFalse(summarize_rows(compare_rows([], []))["validation_passed"])


class PreservationValidationTests(unittest.TestCase):
    def run_summary(self, rows, extra=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "compare.json").write_text(json.dumps(rows))
            with contextlib.redirect_stdout(io.StringIO()):
                return summary_main(["--compare", f"D={path / 'compare.json'}", "--out-dir", str(path / "out"),
                    "--require-branch-preservation"] + (extra or []))

    def preservation_row(self, **changes):
        obs, edit = compared_records()
        row = dict(compare_rows(obs, edit)[0])
        row.update(changes)
        return row

    def test_requires_named_nonempty_preservation_evidence(self):
        row = self.preservation_row()
        for rows, args in (([row], []), ([], ["--preservation-run", "D"]),
                           ([row], ["--preservation-run", "absent"])):
            with self.subTest(rows=rows, args=args):
                self.assertNotEqual(self.run_summary(rows, args), 0)

    def test_tensor_changes_fail_even_if_rho_is_unchanged(self):
        row = self.preservation_row(attn_delta_max=0.1, attn_delta_mean=0.0, delta_rho_local=0.0)
        self.assertNotEqual(self.run_summary([row], ["--preservation-run", "D"]), 0)

    def test_signed_rho_cancellation_does_not_prove_preservation(self):
        rows = [self.preservation_row(delta_rho_local=value) for value in (-0.1, 0.1)]
        self.assertNotEqual(self.run_summary(rows, ["--preservation-run", "D"]), 0)

    def test_missing_nonfinite_or_negative_tensor_evidence_fails(self):
        for value in (None, float("nan"), float("inf"), -0.1):
            row = self.preservation_row(attn_delta_max=value)
            self.assertNotEqual(self.run_summary([row], ["--preservation-run", "D"]), 0)

    def test_trajectory_difference_does_not_disprove_same_call_preservation(self):
        row = self.preservation_row(delta_rho_vs_observe=0.2)
        self.assertEqual(self.run_summary([row], ["--preservation-run", "D"]), 0)

    def test_all_edited_rows_are_not_preservation_evidence(self):
        row = self.preservation_row(edit_applied=True, diagnostic_mode="edited")
        self.assertNotEqual(self.run_summary([row], ["--preservation-run", "D"]), 0)

    def test_diagnostics_without_edit_applied_are_not_evidence(self):
        row = self.preservation_row()
        row.pop("edit_applied")
        self.assertNotEqual(self.run_summary([row], ["--preservation-run", "D"]), 0)


if __name__ == "__main__":
    unittest.main()
