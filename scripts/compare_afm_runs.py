from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .parse_anima_afm_log import parse_log_file
except ImportError:  # pragma: no cover - direct script execution
    from parse_anima_afm_log import parse_log_file


FIELDNAMES = [
    "step_index",
    "eligible_call_index",
    "branch",
    "mode",
    "rho_observe",
    "rho_edit_after",
    "delta_rho_local",
    "delta_rho_vs_observe",
    "alpha_lf",
    "alpha_hf",
    "selected_indices",
    "edit_applied",
]


def _key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("step_index"), row.get("eligible_call_index"), row.get("branch")


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
            "step_index": row.get("step_index"),
            "eligible_call_index": row.get("eligible_call_index"),
            "branch": row.get("branch"),
            "mode": row.get("mode"),
            "rho_observe": rho_observe,
            "rho_edit_after": rho_edit_after,
            "delta_rho_local": row.get("delta_rho_local"),
            "delta_rho_vs_observe": delta_vs_observe,
            "alpha_lf": row.get("alpha_lf"),
            "alpha_hf": row.get("alpha_hf"),
            "selected_indices": row.get("selected_indices"),
            "edit_applied": row.get("edit_applied"),
        })
    return rows


def compare_files(observe_path: str | Path, edit_path: str | Path) -> list[dict[str, Any]]:
    return compare_rows(parse_log_file(observe_path), parse_log_file(edit_path))


def write_csv(rows: list[dict[str, Any]], output: Any) -> None:
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare observe and edit Anima AFM rho trajectories.")
    parser.add_argument("observe_jsonl", help="Observe-mode JSONL log.")
    parser.add_argument("edit_jsonl", help="Edit-mode JSONL log.")
    parser.add_argument("--format", choices=("csv", "json"), default="csv", help="Output format.")
    args = parser.parse_args(argv)

    rows = compare_files(args.observe_jsonl, args.edit_jsonl)
    if args.format == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        write_csv(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
