import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_afm_runs import compare_rows, main as compare_main, summarize_rows, write_csv
from scripts.parse_anima_afm_log import parse_records, write_csv as write_parsed_csv


class CompareAFMRunsTests(unittest.TestCase):
    def test_duplicate_and_unmatched_pair_detection(self):
        observe_rows = [
            {"step_index": 0, "eligible_call_index": 7, "branch": "positive", "rho_after": 0.2},
            {"step_index": 0, "eligible_call_index": 7, "branch": "positive", "rho_after": 0.4},
            {"step_index": 1, "eligible_call_index": 7, "branch": "positive", "rho_after": 0.5},
        ]
        edit_rows = [
            {"step_index": 0, "eligible_call_index": 7, "branch": "positive", "rho_before": 0.2, "rho_after": 0.3},
            {"step_index": 0, "eligible_call_index": 7, "branch": "positive", "rho_before": 0.2, "rho_after": 0.35},
            {"step_index": 2, "eligible_call_index": 7, "branch": "positive", "rho_before": 0.2, "rho_after": 0.25},
        ]

        rows = compare_rows(observe_rows, edit_rows)
        self.assertEqual(rows[0]["observe_pair_count"], 2)
        self.assertEqual(rows[0]["edit_pair_count"], 2)
        self.assertTrue(rows[0]["observe_duplicate"])
        self.assertTrue(rows[0]["edit_duplicate"])
        self.assertEqual(rows[2]["pair_status"], "missing_observe")

        summary = summarize_rows(rows)
        self.assertEqual(summary["duplicate_observe_pairs"], 1)
        self.assertEqual(summary["duplicate_edit_pairs"], 1)
        self.assertEqual(summary["unmatched_observe_pairs"], 1)
        self.assertEqual(summary["pair_key_columns"], ["step_index", "eligible_call_index", "branch"])

    def test_fail_flags_for_duplicates_and_unmatched_observe(self):
        observe_records = [
            {"record_type": "spectral_diag", "step_index": 0, "eligible_call_index": 7, "diagnostic_branch": "positive", "rho_after": 0.2},
            {"record_type": "spectral_diag", "step_index": 0, "eligible_call_index": 7, "diagnostic_branch": "positive", "rho_after": 0.4},
            {"record_type": "spectral_diag", "step_index": 1, "eligible_call_index": 7, "diagnostic_branch": "positive", "rho_after": 0.5},
        ]
        edit_records = [
            {"record_type": "spectral_diag", "step_index": 0, "eligible_call_index": 7, "diagnostic_branch": "positive", "rho_after": 0.3},
            {"record_type": "spectral_diag", "step_index": 0, "eligible_call_index": 7, "diagnostic_branch": "positive", "rho_after": 0.35},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            observe_path = temp / "observe.jsonl"
            edit_path = temp / "edit.jsonl"
            observe_path.write_text("\n".join(json.dumps(record) for record in observe_records) + "\n", encoding="utf-8")
            edit_path.write_text("\n".join(json.dumps(record) for record in edit_records) + "\n", encoding="utf-8")

            self.assertEqual(compare_main([str(observe_path), str(edit_path), "--fail-on-duplicate-pairs"]), 4)
            self.assertEqual(compare_main([str(observe_path), str(edit_path), "--fail-on-unmatched-observe"]), 5)

    def test_csv_and_jsonl_inputs_produce_same_pair_counts(self):
        observe_records = [
            {"record_type": "spectral_diag", "step_index": 0, "eligible_call_index": 0, "diagnostic_branch": "positive", "rho_after": 0.2},
            {"record_type": "spectral_diag", "step_index": 0, "eligible_call_index": 0, "diagnostic_branch": "positive", "rho_after": 0.3},
        ]
        edit_records = [
            {"record_type": "spectral_diag", "step_index": 0, "eligible_call_index": 0, "diagnostic_branch": "positive", "rho_before": 0.2, "rho_after": 0.4},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            observe_jsonl = temp / "observe.jsonl"
            edit_jsonl = temp / "edit.jsonl"
            observe_csv = temp / "observe.csv"
            edit_csv = temp / "edit.csv"
            observe_jsonl.write_text("\n".join(json.dumps(record) for record in observe_records) + "\n", encoding="utf-8")
            edit_jsonl.write_text("\n".join(json.dumps(record) for record in edit_records) + "\n", encoding="utf-8")
            with observe_csv.open("w", encoding="utf-8", newline="") as handle:
                write_parsed_csv(parse_records(observe_records), handle)
            with edit_csv.open("w", encoding="utf-8", newline="") as handle:
                write_parsed_csv(parse_records(edit_records), handle)

            json_rows = compare_rows(parse_records(observe_records), parse_records(edit_records))
            with tempfile.TemporaryFile("w+", encoding="utf-8", newline="") as handle:
                write_csv(json_rows, handle)
                handle.seek(0)
                self.assertIn("observe_pair_count", csv.DictReader(handle).fieldnames)

            self.assertEqual(compare_main([str(observe_jsonl), str(edit_jsonl), "--summary"]), 0)
            self.assertEqual(compare_main([str(observe_csv), str(edit_csv), "--input-format", "csv", "--summary"]), 0)


if __name__ == "__main__":
    unittest.main()
