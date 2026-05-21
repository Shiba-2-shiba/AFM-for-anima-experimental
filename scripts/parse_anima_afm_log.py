from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


FIELDNAMES = [
    "step_index",
    "eligible_call_index",
    "branch",
    "mode",
    "rho_before",
    "rho_after",
    "delta_rho",
    "delta_rho_local",
    "alpha_lf",
    "alpha_hf",
    "selected_indices",
    "edit_applied",
]


def iter_json_records(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def spectral_record_to_row(record: dict[str, Any]) -> dict[str, Any]:
    delta_rho = record.get("delta_rho")
    return {
        "step_index": record.get("step_index"),
        "eligible_call_index": record.get("eligible_call_index"),
        "branch": record.get("diagnostic_branch") or record.get("branch"),
        "mode": record.get("mode"),
        "rho_before": record.get("rho_before"),
        "rho_after": record.get("rho_after"),
        "delta_rho": delta_rho,
        "delta_rho_local": record.get("delta_rho_local", delta_rho),
        "alpha_lf": record.get("alpha_lf"),
        "alpha_hf": record.get("alpha_hf"),
        "selected_indices": json.dumps(record.get("selected_indices", []), separators=(",", ":")),
        "edit_applied": record.get("edit_applied"),
    }


def parse_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        spectral_record_to_row(record)
        for record in records
        if record.get("record_type") == "spectral_diag"
    ]


def parse_log_file(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return parse_records(iter_json_records(handle))


def write_csv(rows: list[dict[str, Any]], output: Any) -> None:
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse Anima AFM JSONL spectral diagnostics.")
    parser.add_argument("jsonl_path", help="Path to an Anima AFM JSONL log.")
    parser.add_argument("--format", choices=("csv", "json"), default="csv", help="Output format.")
    args = parser.parse_args(argv)

    rows = parse_log_file(args.jsonl_path)
    if args.format == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        write_csv(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
