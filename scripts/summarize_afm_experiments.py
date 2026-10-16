from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .compare_afm_runs import FIELDNAMES as COMPARE_FIELDNAMES
    from .compare_afm_runs import PAIR_KEY_COLUMNS
    from .compare_afm_runs import _boolish, _intish, _mean, _numeric, _key, _boolean, _json_object, _completion_issues, summarize_rows
except ImportError:  # pragma: no cover - direct script execution
    from compare_afm_runs import FIELDNAMES as COMPARE_FIELDNAMES
    from compare_afm_runs import PAIR_KEY_COLUMNS
    from compare_afm_runs import _boolish, _intish, _mean, _numeric, _key, _boolean, _json_object, _completion_issues, summarize_rows


OVERVIEW_FIELDNAMES = [
    "comparison",
    "rows",
    "matched",
    "missing_observe",
    "unmatched_observe_pairs",
    "duplicate_observe_pairs",
    "duplicate_edit_pairs",
    "overall_delta_rho_local_mean",
    "overall_delta_rho_vs_observe_mean",
    "edited_rows",
    "edited_delta_rho_local_mean",
    "edited_delta_rho_vs_observe_mean",
    "late_start_step",
    "late_rows",
    "late_edited_delta_rho_vs_observe_mean",
    "max_abs_delta_rho_vs_observe",
    "validation_passed",
    "provenance_status",
    "rho_definition",
    "rho_centering",
]

BY_CALL_BRANCH_FIELDNAMES = [
    "comparison",
    "eligible_call_index",
    "branch",
    "call_branch",
    "rows",
    "matched",
    "missing_observe",
    "edited_rows",
    "diagnostic_mode",
    "edit_applied_count",
    "delta_rho_local_mean",
    "delta_rho_vs_observe_mean",
    "late_delta_rho_vs_observe_mean",
]

BY_STEP_FIELDNAMES = [
    "comparison",
    "step_index",
    "rows",
    "matched",
    "missing_observe",
    "edited_rows",
    "delta_rho_local_mean",
    "delta_rho_vs_observe_mean",
    "edited_delta_rho_vs_observe_mean",
]

BY_DIAGNOSTIC_MODE_FIELDNAMES = [
    "comparison",
    "call_mode",
    "diagnostic_mode",
    "edit_applied",
    "branch",
    "rows",
    "delta_rho_local_mean",
    "delta_rho_vs_observe_mean",
]

BY_CALL_FIELDNAMES = [
    "comparison",
    "eligible_call_index",
    "rows",
    "matched",
    "missing_observe",
    "edited_rows",
    "delta_rho_local_mean",
    "delta_rho_vs_observe_mean",
    "late_delta_rho_vs_observe_mean",
]

PRESERVATION_FIELDNAMES = [
    "comparison",
    "diagnostic_branch",
    "call_mode",
    "diagnostic_mode",
    "edit_applied",
    "rows",
    "delta_rho_local_mean",
    "attn_delta_mean_mean",
    "attn_delta_max_max",
    "delta_rho_local_max_abs",
    "complete_tensor_rows",
    "same_call_preservation_passed",
]


def _read_compare_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            data = data["rows"]
        if not isinstance(data, list):
            raise ValueError(f"Expected compare JSON rows in {path}")
        if any(not isinstance(row, dict) for row in data):
            raise ValueError(f"Every compare JSON row must be an object in {path}")
        return [_normalize_row(row) for row in data]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_normalize_row(row) for row in csv.DictReader(handle)]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key in ("step_index", "eligible_call_index", "observe_pair_count", "edit_pair_count"):
        parsed = _intish(normalized.get(key))
        if parsed is not None:
            normalized[key] = parsed
    for key in (
        "rho_observe",
        "rho_edit_before",
        "rho_edit_after",
        "delta_rho_local",
        "delta_rho_vs_observe",
        "alpha_lf",
        "alpha_hf",
        "attn_delta_mean",
        "attn_delta_max",
    ):
        parsed = _numeric(normalized.get(key))
        if parsed is not None:
            normalized[key] = parsed
    for key in ("edit_applied", "observe_duplicate", "edit_duplicate"):
        if normalized.get(key) not in (None, ""):
            parsed = _boolean(normalized.get(key))
            if parsed is not None:
                normalized[key] = parsed
    if "diagnostic_branch" not in normalized:
        normalized["diagnostic_branch"] = normalized.get("branch")
    return normalized


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_numeric(row.get(field)) for row in rows]
    present = [value for value in values if value is not None]
    return _mean(present)


def _max_abs_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_numeric(row.get(field)) for row in rows]
    present = [abs(value) for value in values if value is not None]
    return max(present) if present else None


def _max_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_numeric(row.get(field)) for row in rows]
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _is_matched(row: dict[str, Any]) -> bool:
    return row.get("pair_status") == "matched"


def _is_missing(row: dict[str, Any]) -> bool:
    return row.get("pair_status") == "missing_observe"


def _is_edited(row: dict[str, Any]) -> bool:
    return _boolish(row.get("edit_applied"))


def _edit_label(row: dict[str, Any]) -> str:
    value = _boolean(row.get("edit_applied"))
    return str(value) if value is not None else "Unknown"


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return _key(row)


def _duplicate_pairs(rows: list[dict[str, Any]], duplicate_field: str, count_field: str) -> int:
    keys = set()
    for row in rows:
        count = _intish(row.get(count_field)) or 0
        if _boolish(row.get(duplicate_field)) or count > 1:
            keys.add(_pair_key(row))
    return len(keys)


def _overview_row(comparison: str, rows: list[dict[str, Any]], late_start_step: int | None) -> dict[str, Any]:
    evidence = summarize_rows(rows)
    edited = [row for row in rows if _is_edited(row)]
    late_rows = [
        row for row in rows
        if late_start_step is not None and _intish(row.get("step_index")) is not None and int(row["step_index"]) >= late_start_step
    ]
    late_edited = [row for row in late_rows if _is_edited(row)]
    return {
        "comparison": comparison,
        "rows": len(rows),
        "matched": sum(1 for row in rows if _is_matched(row)),
        "missing_observe": sum(1 for row in rows if _is_missing(row)),
        "unmatched_observe_pairs": evidence["unmatched_observe_pairs"],
        "duplicate_observe_pairs": _duplicate_pairs(rows, "observe_duplicate", "observe_pair_count"),
        "duplicate_edit_pairs": _duplicate_pairs(rows, "edit_duplicate", "edit_pair_count"),
        "overall_delta_rho_local_mean": _mean_field(rows, "delta_rho_local"),
        "overall_delta_rho_vs_observe_mean": _mean_field(rows, "delta_rho_vs_observe"),
        "edited_rows": len(edited),
        "edited_delta_rho_local_mean": _mean_field(edited, "delta_rho_local"),
        "edited_delta_rho_vs_observe_mean": _mean_field(edited, "delta_rho_vs_observe"),
        "late_start_step": late_start_step,
        "late_rows": len(late_rows),
        "late_edited_delta_rho_vs_observe_mean": _mean_field(late_edited, "delta_rho_vs_observe"),
        "max_abs_delta_rho_vs_observe": _max_abs_field(rows, "delta_rho_vs_observe"),
        "validation_passed": _valid_comparison(rows, evidence),
        "provenance_status": evidence["provenance_status"],
        "rho_definition": _mode_label(rows, "rho_definition") or "legacy_unspecified",
        "rho_centering": _mode_label(rows, "rho_centering") or "unknown",

    }


def _mode_label(rows: list[dict[str, Any]], field: str) -> Any:
    values = {row.get(field) for row in rows if row.get(field) not in (None, "")}
    if len(values) == 1:
        return next(iter(values))
    if not values:
        return None
    return "mixed"


def _by_call_branch_rows(comparison: str, rows: list[dict[str, Any]], late_start_step: int | None) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("eligible_call_index"), row.get("branch"))].append(row)
    output = []
    for (eligible_call_index, branch), items in sorted(grouped.items(), key=lambda item: repr(item[0])):
        edited = [row for row in items if _is_edited(row)]
        late = [
            row for row in items
            if late_start_step is not None and _intish(row.get("step_index")) is not None and int(row["step_index"]) >= late_start_step
        ]
        output.append({
            "comparison": comparison,
            "eligible_call_index": eligible_call_index,
            "branch": branch,
            "call_branch": f"{eligible_call_index}/{branch}",
            "rows": len(items),
            "matched": sum(1 for row in items if _is_matched(row)),
            "missing_observe": sum(1 for row in items if _is_missing(row)),
            "edited_rows": len(edited),
            "diagnostic_mode": _mode_label(items, "diagnostic_mode"),
            "edit_applied_count": len(edited),
            "delta_rho_local_mean": _mean_field(items, "delta_rho_local"),
            "delta_rho_vs_observe_mean": _mean_field(items, "delta_rho_vs_observe"),
            "late_delta_rho_vs_observe_mean": _mean_field(late, "delta_rho_vs_observe"),
        })
    return output


def _by_call_rows(comparison: str, rows: list[dict[str, Any]], late_start_step: int | None) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("eligible_call_index")].append(row)
    output = []
    for eligible_call_index, items in sorted(grouped.items(), key=lambda item: repr(item[0])):
        edited = [row for row in items if _is_edited(row)]
        late = [
            row for row in items
            if late_start_step is not None and _intish(row.get("step_index")) is not None and int(row["step_index"]) >= late_start_step
        ]
        output.append({
            "comparison": comparison,
            "eligible_call_index": eligible_call_index,
            "rows": len(items),
            "matched": sum(1 for row in items if _is_matched(row)),
            "missing_observe": sum(1 for row in items if _is_missing(row)),
            "edited_rows": len(edited),
            "delta_rho_local_mean": _mean_field(items, "delta_rho_local"),
            "delta_rho_vs_observe_mean": _mean_field(items, "delta_rho_vs_observe"),
            "late_delta_rho_vs_observe_mean": _mean_field(late, "delta_rho_vs_observe"),
        })
    return output


def _by_step_rows(comparison: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("step_index")].append(row)
    output = []
    for step_index, items in sorted(grouped.items(), key=lambda item: repr(item[0])):
        edited = [row for row in items if _is_edited(row)]
        output.append({
            "comparison": comparison,
            "step_index": step_index,
            "rows": len(items),
            "matched": sum(1 for row in items if _is_matched(row)),
            "missing_observe": sum(1 for row in items if _is_missing(row)),
            "edited_rows": len(edited),
            "delta_rho_local_mean": _mean_field(items, "delta_rho_local"),
            "delta_rho_vs_observe_mean": _mean_field(items, "delta_rho_vs_observe"),
            "edited_delta_rho_vs_observe_mean": _mean_field(edited, "delta_rho_vs_observe"),
        })
    return output


def _by_diagnostic_mode_rows(comparison: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row.get("call_mode"),
            row.get("diagnostic_mode"),
            _edit_label(row),
            row.get("branch"),
        )].append(row)
    output = []
    for (call_mode, diagnostic_mode, edit_applied, branch), items in sorted(grouped.items(), key=lambda item: repr(item[0])):
        output.append({
            "comparison": comparison,
            "call_mode": call_mode,
            "diagnostic_mode": diagnostic_mode,
            "edit_applied": edit_applied,
            "branch": branch,
            "rows": len(items),
            "delta_rho_local_mean": _mean_field(items, "delta_rho_local"),
            "delta_rho_vs_observe_mean": _mean_field(items, "delta_rho_vs_observe"),
        })
    return output


def _branch_preservation_rows(
    comparison: str,
    rows: list[dict[str, Any]],
    include_all: bool,
) -> list[dict[str, Any]]:
    if not include_all:
        return []
    grouped: dict[tuple[Any, Any, Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row.get("diagnostic_branch") or row.get("branch"),
            row.get("call_mode"),
            row.get("diagnostic_mode"),
            _edit_label(row),
        )].append(row)
    output = []
    for (branch, call_mode, diagnostic_mode, edit_applied), items in sorted(grouped.items(), key=lambda item: repr(item[0])):
        output.append({
            "comparison": comparison,
            "diagnostic_branch": branch,
            "call_mode": call_mode,
            "diagnostic_mode": diagnostic_mode,
            "edit_applied": edit_applied,
            "rows": len(items),
            "delta_rho_local_mean": _mean_field(items, "delta_rho_local"),
            "attn_delta_mean_mean": _mean_field(items, "attn_delta_mean"),
            "attn_delta_max_max": _max_field(items, "attn_delta_max"),
            "delta_rho_local_max_abs": _max_abs_field(items, "delta_rho_local"),
            "complete_tensor_rows": sum(_has_tensor_evidence(row) for row in items),
            "same_call_preservation_passed": all(_preserved(row) for row in items) if edit_applied == "False" else None,
        })
    return output


def _valid_comparison(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> bool:
    if not rows or not evidence["validation_passed"] or evidence["provenance_status"] != "manifest_verified":
        return False
    for side in ("observe", "edit"):
        reconstructed = [dict(row, run_id=row.get(f"{side}_run_id"), run_completion_json=row.get(f"{side}_completion_json")) for row in rows]
        if _completion_issues(reconstructed, side):
            return False
    if any(count != 1 for count in Counter(_pair_key(row) for row in rows).values()):
        return False
    for row in rows:
        if row.get("pair_status") != "matched" or _intish(row.get("unmatched_observe_pairs")) != 0:
            return False
        if _intish(row.get("observe_pair_count")) != 1 or _intish(row.get("edit_pair_count")) != 1:
            return False
        if any(_numeric(row.get(field)) is None for field in ("rho_observe", "rho_edit_before", "rho_edit_after", "delta_rho_local", "delta_rho_vs_observe")):
            return False
    return True


def _has_tensor_evidence(row: dict[str, Any]) -> bool:
    mean = _numeric(row.get("attn_delta_mean"))
    maximum = _numeric(row.get("attn_delta_max"))
    return mean is not None and maximum is not None and 0 <= mean <= maximum


def _preserved(row: dict[str, Any]) -> bool:
    local_delta = _numeric(row.get("delta_rho_local"))
    return (_boolean(row.get("edit_applied")) is False and _has_tensor_evidence(row)
            and local_delta is not None and abs(local_delta) <= 1e-12
            and _numeric(row.get("attn_delta_max")) <= 1e-12)


def _preservation_check(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Cross-run rho is intentionally excluded: trajectory drift cannot prove or
    # disprove preservation of an unselected branch in the same attention call.
    eligible = [row for row in rows if _boolean(row.get("edit_applied")) is False
                and row.get("call_mode") == "edit" and row.get("diagnostic_mode") == "passthrough"]
    errors = []
    if not rows:
        errors.append("No rows for requested preservation run")
    if any(_boolean(row.get("edit_applied")) is None for row in rows):
        errors.append("Missing or invalid edit_applied")
    if not eligible:
        errors.append("No unedited same-call branch diagnostics")
    if any(not _preserved(row) for row in eligible):
        errors.append("Missing, invalid, or nonzero same-call tensor/local-rho evidence")
    return {"comparison": name, "checked_rows": len(eligible), "passed": not errors, "errors": errors,
            "evidence_scope": "sampled same-call attention diagnostics; not cross-run rho or final output identity"}


def summarize_comparisons(
    comparisons: list[tuple[str, str | Path]],
    late_start_step: int | None = None,
    preservation_runs: list[str] | None = None,
) -> dict[str, Any]:
    preservation_runs = preservation_runs or []
    overview = []
    by_call_branch = []
    by_step = []
    by_diagnostic_mode = []
    by_call = []
    branch_preservation = []
    preservation_checks = []
    rows_by_name = {}
    for name, path in comparisons:
        if name in rows_by_name:
            raise ValueError(f"Duplicate comparison name: {name}")
        rows = _read_compare_rows(path)
        rows_by_name[name] = rows
        overview_row = _overview_row(name, rows, late_start_step)
        overview.append(overview_row)
        by_call_branch.extend(_by_call_branch_rows(name, rows, late_start_step))
        by_step.extend(_by_step_rows(name, rows))
        by_diagnostic_mode.extend(_by_diagnostic_mode_rows(name, rows))
        by_call.extend(_by_call_rows(name, rows, late_start_step))
        branch_preservation.extend(_branch_preservation_rows(name, rows, name in preservation_runs))
    preservation_checks = [_preservation_check(name, rows_by_name.get(name, [])) for name in preservation_runs]
    return {
        "pair_key_columns": PAIR_KEY_COLUMNS,
        "preservation_checks": preservation_checks,
        "provenance_by_comparison": {
            name: {
                "status": _mode_label(rows, "provenance_status") or "unverified",
                "observe_run_id": _mode_label(rows, "observe_run_id"),
                "edit_run_id": _mode_label(rows, "edit_run_id"),
                "observe_manifest": _json_object(_mode_label(rows, "observe_manifest_json")),
                "edit_manifest": _json_object(_mode_label(rows, "edit_manifest_json")),
                "observe_completion": _json_object(_mode_label(rows, "observe_completion_json")),
                "edit_completion": _json_object(_mode_label(rows, "edit_completion_json")),
                "verification_scope": "supplied-manifest consistency; content hashes are not independently recomputed",
            } for name, rows in rows_by_name.items()
        },
        "validation_passed": bool(overview) and all(row["validation_passed"] for row in overview),
        "overview": overview,
        "overview_by_name": {row["comparison"]: row for row in overview},
        "by_call_branch": by_call_branch,
        "by_step": by_step,
        "by_diagnostic_mode": by_diagnostic_mode,
        "by_call": by_call,
        "branch_preservation_counts": branch_preservation,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# AFM Experiment Summary",
        "",
        "Validation: " + ("passed" if result["validation_passed"] else "UNVERIFIED / FAILED; exploratory values only"),
        "",
        "| comparison | rows | matched | missing_observe | edited_rows | edited_delta_rho_vs_observe_mean | late_edited_delta_rho_vs_observe_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["overview"]:
        lines.append(
            f"| {row['comparison']} | {row['rows']} | {row['matched']} | {row['missing_observe']} | "
            f"{row['edited_rows']} | {row['edited_delta_rho_vs_observe_mean']} | "
            f"{row['late_edited_delta_rho_vs_observe_mean']} |"
        )
    if result["branch_preservation_counts"]:
        lines.extend([
            "",
            "## Branch Preservation Counts",
            "",
            "| comparison | branch | call_mode | diagnostic_mode | edit_applied | rows |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ])
        for row in result["branch_preservation_counts"]:
            lines.append(
                f"| {row['comparison']} | {row['diagnostic_branch']} | {row['call_mode']} | "
                f"{row['diagnostic_mode']} | {row['edit_applied']} | {row['rows']} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    result: dict[str, Any],
    out_dir: str | Path,
    output_format: str = "both",
    report_md: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_format in ("csv", "both"):
        _write_csv(out_dir / "summary_overview.csv", OVERVIEW_FIELDNAMES, result["overview"])
        _write_csv(out_dir / "summary_by_call_branch.csv", BY_CALL_BRANCH_FIELDNAMES, result["by_call_branch"])
        _write_csv(out_dir / "summary_by_step.csv", BY_STEP_FIELDNAMES, result["by_step"])
        _write_csv(out_dir / "summary_by_diagnostic_mode.csv", BY_DIAGNOSTIC_MODE_FIELDNAMES, result["by_diagnostic_mode"])
        _write_csv(out_dir / "summary_by_call.csv", BY_CALL_FIELDNAMES, result["by_call"])
        _write_csv(out_dir / "branch_preservation_counts.csv", PRESERVATION_FIELDNAMES, result["branch_preservation_counts"])
    if output_format in ("json", "both"):
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if report_md:
        (out_dir / "summary_report.md").write_text(_report_markdown(result), encoding="utf-8")


def _parse_compare_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--compare must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--compare must be NAME=PATH")
    return name, path


def _has_preservation_violation(result: dict[str, Any]) -> bool:
    checks = result.get("preservation_checks", [])
    return not checks or any(not check["passed"] for check in checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize multiple Anima AFM compare outputs.")
    parser.add_argument("--compare", action="append", type=_parse_compare_arg, required=True, help="Comparison input as NAME=PATH.")
    parser.add_argument("--late-start-step", type=int, default=None, help="Start step for late-window summaries.")
    parser.add_argument("--out-dir", required=True, help="Directory for generated summary files.")
    parser.add_argument("--format", choices=("csv", "json", "both"), default="both", help="Output format.")
    parser.add_argument("--fail-on-missing-observe", action="store_true")
    parser.add_argument("--fail-on-duplicate-pairs", action="store_true")
    parser.add_argument("--require-branch-preservation", action="store_true")
    parser.add_argument("--preservation-run", action="append", default=[], help="Comparison name to include in preservation counts.")
    parser.add_argument("--report-md", action="store_true", help="Write summary_report.md.")
    parser.add_argument("--strict-validation", action="store_true", help="Require nonempty fully validated, manifest-backed comparisons.")
    args = parser.parse_args(argv)

    try:
        result = summarize_comparisons(
            args.compare, late_start_step=args.late_start_step, preservation_runs=args.preservation_run,
        )
    except (ValueError, OSError) as exc:
        print(f"Invalid summary input: {exc}", file=sys.stderr)
        return 2
    write_outputs(result, args.out_dir, output_format=args.format, report_md=args.report_md)
    if args.fail_on_missing_observe and any(row["missing_observe"] > 0 for row in result["overview"]):
        return 3
    if args.fail_on_duplicate_pairs and any(
        row["duplicate_observe_pairs"] > 0 or row["duplicate_edit_pairs"] > 0
        for row in result["overview"]
    ):
        return 4
    if args.require_branch_preservation and _has_preservation_violation(result):
        return 6
    if args.strict_validation and not result["validation_passed"]:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
