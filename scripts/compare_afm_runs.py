from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .parse_anima_afm_log import FIELDNAMES as PARSE_FIELDNAMES
    from .parse_anima_afm_log import parse_log_file
except ImportError:  # pragma: no cover - direct script execution
    from parse_anima_afm_log import FIELDNAMES as PARSE_FIELDNAMES
    from parse_anima_afm_log import parse_log_file


FIELDNAMES = [
    "pair_status",
    "step_index",
    "eligible_call_index",
    "branch",
    "rho_observe",
    "rho_edit_before",
    "rho_edit_after",
    "delta_rho_local",
    "delta_rho_vs_observe",
    "call_mode",
    "diagnostic_mode",
    "mode",
    "alpha_lf",
    "alpha_hf",
    "edit_applied",
    "edit_selected_indices",
    "diagnostic_batch_indices",
    "selected_indices",
]


def _key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return _cell(row.get("step_index")), _cell(row.get("eligible_call_index")), row.get("branch")


def _cell(value: Any) -> Any:
    if value == "":
        return None
    return value


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _intish(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {name: row.get(name) for name in PARSE_FIELDNAMES}
    for key in ("schema_version", "step_index", "num_steps", "last_index", "eligible_call_index"):
        parsed = _intish(normalized.get(key))
        if parsed is not None:
            normalized[key] = parsed
    for key in (
        "u",
        "sigma",
        "rho_before",
        "rho_after",
        "delta_rho",
        "delta_rho_local",
        "delta_rho_vs_observe",
        "alpha_lf",
        "alpha_hf",
        "attn_delta_mean",
        "attn_delta_max",
        "estimated_logits_mib",
        "estimated_peak_mib",
    ):
        parsed = _numeric(normalized.get(key))
        if parsed is not None:
            normalized[key] = parsed
    for key in ("edit_applied", "target_call_selected", "diagnostic_call_selected"):
        if normalized.get(key) not in (None, ""):
            normalized[key] = _boolish(normalized.get(key))
    return normalized


def read_rows(path: str | Path, input_format: str = "auto") -> list[dict[str, Any]]:
    path = Path(path)
    resolved = input_format
    if resolved == "auto":
        resolved = "csv" if path.suffix.lower() == ".csv" else "jsonl"
    if resolved == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [_normalize_csv_row(row) for row in csv.DictReader(handle)]
    return parse_log_file(path)


def compare_rows(observe_rows: list[dict[str, Any]], edit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observe_by_key: dict[tuple[Any, Any, Any], list[float]] = defaultdict(list)
    for row in observe_rows:
        rho = _numeric(row.get("rho_after"))
        if rho is not None:
            observe_by_key[_key(row)].append(rho)

    rows: list[dict[str, Any]] = []
    for row in edit_rows:
        key = _key(row)
        rho_observe = _mean(observe_by_key.get(key, []))
        rho_edit_after = _numeric(row.get("rho_after"))
        delta_vs_observe = None
        if rho_observe is not None and rho_edit_after is not None:
            delta_vs_observe = rho_edit_after - rho_observe
        rows.append({
            "pair_status": "matched" if rho_observe is not None else "missing_observe",
            "step_index": row.get("step_index"),
            "eligible_call_index": row.get("eligible_call_index"),
            "branch": row.get("branch"),
            "rho_observe": rho_observe,
            "rho_edit_before": _numeric(row.get("rho_before")),
            "rho_edit_after": rho_edit_after,
            "delta_rho_local": _numeric(row.get("delta_rho_local")),
            "delta_rho_vs_observe": delta_vs_observe,
            "call_mode": row.get("call_mode"),
            "diagnostic_mode": row.get("diagnostic_mode"),
            "mode": row.get("mode"),
            "alpha_lf": _numeric(row.get("alpha_lf")),
            "alpha_hf": _numeric(row.get("alpha_hf")),
            "edit_applied": row.get("edit_applied"),
            "edit_selected_indices": row.get("edit_selected_indices"),
            "diagnostic_batch_indices": row.get("diagnostic_batch_indices"),
            "selected_indices": row.get("selected_indices"),
        })
    return rows


def compare_files(
    observe_path: str | Path,
    edit_path: str | Path,
    input_format: str = "auto",
) -> list[dict[str, Any]]:
    return compare_rows(read_rows(observe_path, input_format), read_rows(edit_path, input_format))


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_numeric(row.get(field)) for row in rows]
    present = [value for value in values if value is not None]
    return _mean(present)


def summarize_rows(rows: list[dict[str, Any]], late_start_step: int | None = None) -> dict[str, Any]:
    matched = [row for row in rows if row.get("pair_status") == "matched"]
    missing = [row for row in rows if row.get("pair_status") == "missing_observe"]
    edited = [row for row in rows if _boolish(row.get("edit_applied"))]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "matched": len(matched),
        "missing_observe": len(missing),
        "overall": {
            "delta_rho_local_mean": _mean_field(rows, "delta_rho_local"),
            "delta_rho_vs_observe_mean": _mean_field(rows, "delta_rho_vs_observe"),
        },
        "edited_only": {
            "rows": len(edited),
            "delta_rho_local_mean": _mean_field(edited, "delta_rho_local"),
            "delta_rho_vs_observe_mean": _mean_field(edited, "delta_rho_vs_observe"),
        },
        "by_call_branch": {},
    }
    if late_start_step is not None:
        late_rows = [
            row for row in rows
            if _intish(row.get("step_index")) is not None and int(row["step_index"]) >= late_start_step
        ]
        late_edited = [row for row in late_rows if _boolish(row.get("edit_applied"))]
        summary["late_window"] = {
            "start_step": late_start_step,
            "rows": len(late_rows),
            "edited_delta_rho_vs_observe_mean": _mean_field(late_edited, "delta_rho_vs_observe"),
        }

    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[f"{row.get('eligible_call_index')}/{row.get('branch')}"].append(row)
    summary["by_call_branch"] = {
        key: {
            "rows": len(items),
            "delta_rho_vs_observe_mean": _mean_field(items, "delta_rho_vs_observe"),
        }
        for key, items in sorted(by_key.items())
    }
    return summary


def write_csv(rows: list[dict[str, Any]], output: Any) -> None:
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare observe and edit Anima AFM rho trajectories.")
    parser.add_argument("observe_path", help="Observe-mode JSONL or parsed CSV.")
    parser.add_argument("edit_path", help="Edit-mode JSONL or parsed CSV.")
    parser.add_argument("--input-format", choices=("auto", "jsonl", "csv"), default="auto", help="Input format.")
    parser.add_argument("--format", choices=("csv", "json"), default="csv", help="Output format.")
    parser.add_argument("--summary", action="store_true", help="Emit JSON summary instead of rows.")
    parser.add_argument("--late-start-step", type=int, default=None, help="Start step for late-window summary.")
    parser.add_argument("--fail-on-missing-observe", action="store_true", help="Exit non-zero if any edit row lacks observe pair.")
    args = parser.parse_args(argv)

    rows = compare_files(args.observe_path, args.edit_path, input_format=args.input_format)
    missing = any(row.get("pair_status") == "missing_observe" for row in rows)
    if args.summary:
        json.dump(summarize_rows(rows, late_start_step=args.late_start_step), sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.format == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        write_csv(rows, sys.stdout)
    return 3 if missing and args.fail_on_missing_observe else 0


if __name__ == "__main__":
    raise SystemExit(main())
