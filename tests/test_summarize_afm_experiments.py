import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_afm_experiments import summarize_comparisons, write_outputs


CALLS = [0, 7, 14, 21, 27]
BRANCHES = ["negative", "positive"]


def _bool_text(value):
    return "True" if value else "False"


def _edited_value(overall_mean, late_mean, edited_rows, late_edited_rows):
    early_rows = edited_rows - late_edited_rows
    return ((overall_mean * edited_rows) - (late_mean * late_edited_rows)) / early_rows


def _rows_for_expected(expected, edit_selector):
    late_steps = {16, 17, 18, 19}
    edited_rows = expected["edited_rows"]
    late_edited_rows = sum(
        1
        for step in range(20)
        for call in CALLS
        for branch in BRANCHES
        if step in late_steps and edit_selector(call, branch)
    )
    early_value = _edited_value(
        expected["edited_delta_rho_vs_observe_mean"],
        expected["late_edited_delta_rho_vs_observe_mean"],
        edited_rows,
        late_edited_rows,
    )
    rows = []
    for step in range(20):
        for call in CALLS:
            for branch in BRANCHES:
                edited = edit_selector(call, branch)
                if edited and step in late_steps:
                    delta = expected["late_edited_delta_rho_vs_observe_mean"]
                elif edited:
                    delta = early_value
                else:
                    delta = 0.0
                rows.append({
                    "pair_status": "matched",
                    "step_index": step,
                    "eligible_call_index": call,
                    "branch": branch,
                    "rho_observe": 0.2,
                    "rho_edit_before": 0.2,
                    "rho_edit_after": 0.2 + delta,
                    "delta_rho_local": delta,
                    "delta_rho_vs_observe": delta,
                    "call_mode": "edit",
                    "diagnostic_mode": "edited" if edited else "passthrough",
                    "mode": "edit",
                    "alpha_lf": 1.0,
                    "alpha_hf": 1.1,
                    "edit_applied": _bool_text(edited),
                    "observe_pair_count": 1,
                    "edit_pair_count": 1,
                    "observe_duplicate": "False",
                    "edit_duplicate": "False",
                })
    return rows


def _write_csv(path, rows):
    fieldnames = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class SummarizeAFMExperimentsTests(unittest.TestCase):
    def test_reproduces_v7_expected_attached_log_values(self):
        expected = json.loads(Path("tests/fixtures/v7_expected_summary.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            temp = Path(temp_dir)
            paths = {
                "B_call0_vs_A": temp / "B.csv",
                "C_call7-13_vs_A": temp / "C.csv",
                "D_positive-only-preservation_vs_A": temp / "D.csv",
            }
            _write_csv(paths["B_call0_vs_A"], _rows_for_expected(expected["B_call0_vs_A"], lambda call, branch: call == 0))
            _write_csv(paths["C_call7-13_vs_A"], _rows_for_expected(expected["C_call7-13_vs_A"], lambda call, branch: call == 7))
            _write_csv(
                paths["D_positive-only-preservation_vs_A"],
                _rows_for_expected(expected["D_positive-only-preservation_vs_A"], lambda call, branch: branch == "positive"),
            )

            result = summarize_comparisons(
                [(name, path) for name, path in paths.items()],
                late_start_step=16,
                preservation_runs=["D_positive-only-preservation_vs_A"],
            )

        for name in paths:
            overview = result["overview_by_name"][name]
            self.assertEqual(overview["rows"], expected[name]["rows"])
            self.assertEqual(overview["matched"], expected[name]["matched"])
            self.assertEqual(overview["missing_observe"], expected[name]["missing_observe"])
            self.assertEqual(overview["edited_rows"], expected[name]["edited_rows"])
            self.assertAlmostEqual(
                overview["edited_delta_rho_vs_observe_mean"],
                expected[name]["edited_delta_rho_vs_observe_mean"],
            )
            self.assertAlmostEqual(
                overview["late_edited_delta_rho_vs_observe_mean"],
                expected[name]["late_edited_delta_rho_vs_observe_mean"],
            )

        counts = {
            f"{row['diagnostic_branch']}/{row['call_mode']}/{row['diagnostic_mode']}/{row['edit_applied']}": row["rows"]
            for row in result["branch_preservation_counts"]
        }
        self.assertEqual(counts, expected["D_branch_preservation_counts"])

    def test_writes_expected_summary_outputs(self):
        rows = _rows_for_expected(
            {
                "edited_rows": 40,
                "edited_delta_rho_vs_observe_mean": 0.1,
                "late_edited_delta_rho_vs_observe_mean": 0.2,
            },
            lambda call, branch: call == 0,
        )
        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            temp = Path(temp_dir)
            compare_path = temp / "compare.csv"
            out_dir = temp / "summary"
            _write_csv(compare_path, rows)
            result = summarize_comparisons([("B", compare_path)], late_start_step=16)
            write_outputs(result, out_dir, output_format="both", report_md=True)
            for name in (
                "summary_overview.csv",
                "summary_by_call_branch.csv",
                "summary_by_step.csv",
                "summary_by_diagnostic_mode.csv",
                "branch_preservation_counts.csv",
                "summary_report.md",
                "summary.json",
            ):
                self.assertTrue((out_dir / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
