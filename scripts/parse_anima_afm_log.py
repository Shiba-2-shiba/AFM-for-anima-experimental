from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FIELDNAMES = [
    "schema_version",
    "record_type",
    "run_id",
    "run_completion_json",
    "step_index",
    "model_evaluation_index",
    "num_steps",
    "last_index",
    "u",
    "sigma",
    "eligible_call_index",
    "block_id",
    "block_path",
    "module_class",
    "block_index",
    "stage_tag",
    "branch",
    "call_mode",
    "diagnostic_mode",
    "mode",
    "rho_before",
    "rho_after",
    "delta_rho",
    "delta_rho_local",
    "delta_rho_vs_observe",
    "alpha_lf",
    "alpha_hf",
    "edit_applied",
    "edit_selected_indices",
    "diagnostic_batch_indices",
    "selected_indices",
    "batch_indices",
    "target_call_indices",
    "scope_mode",
    "scope_map_path",
    "block_scope",
    "stage_scope",
    "diagnostic_call_indices",
    "target_call_selected",
    "scope_selected",
    "scope_id",
    "scope_reject_reason",
    "diagnostic_call_selected",
    "branch_mode",
    "diagnostic_branch",
    "attn_delta_mean",
    "attn_delta_max",
    "estimated_logits_mib",
    "estimated_peak_mib",
    "memory_estimate_json",
    "metric_defined",
    "metric_undefined_reason",
    "rho_definition",
    "rho_centering",
    "rho_raw_before",
    "rho_raw_after",
    "rho_centered_before",
    "rho_centered_after",
    "spatial_shape",
    "q_shape",
    "k_shape",
    "v_shape",
    "cutoff",
    "spectral_diag",
    "diagnostic_top_k",
    "diagnostic_max_batches",
    "diagnostic_max_heads",
]


def _json_cell(value: Any) -> Any:
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, separators=(",", ":"))
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def iter_json_records(lines: Iterable[str], strict: bool = False) -> Iterable[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line, parse_constant=_reject_constant) if strict else json.loads(line)
        except json.JSONDecodeError:
            if strict:
                raise
            continue
        if isinstance(record, dict):
            yield record
        elif strict:
            raise ValueError("JSONL records must be objects")


def infer_diagnostic_mode(record: dict[str, Any], call_mode: str) -> str:
    explicit = record.get("diagnostic_mode")
    if explicit:
        return str(explicit)
    if record.get("edit_applied") is True:
        return "edited"
    if call_mode == "observe":
        return "observe"
    if call_mode == "passthrough":
        return "target_skipped"
    return "passthrough"


def record_to_row(record: dict[str, Any]) -> dict[str, Any]:
    schema_version = record.get("schema_version", 1)
    branch = record.get("diagnostic_branch") or record.get("branch")
    call_mode = record.get("call_mode") or record.get("mode")
    diagnostic_mode = infer_diagnostic_mode(record, str(call_mode))
    edit_selected = record.get("edit_selected_indices", record.get("selected_indices", []))
    diagnostic_batches = record.get("diagnostic_batch_indices", record.get("batch_indices"))
    selected_indices = record.get("selected_indices", edit_selected)
    batch_indices = record.get("batch_indices", diagnostic_batches)
    delta_rho = record.get("delta_rho")
    return {
        "schema_version": schema_version,
        "record_type": record.get("record_type"),
        "run_id": record.get("run_id"),
        "step_index": record.get("step_index"),
        "model_evaluation_index": record.get("model_evaluation_index"),
        "num_steps": record.get("num_steps"),
        "last_index": record.get("last_index"),
        "u": record.get("u"),
        "sigma": record.get("sigma"),
        "eligible_call_index": record.get("eligible_call_index"),
        "block_id": record.get("block_id"),
        "block_path": record.get("block_path"),
        "module_class": record.get("module_class"),
        "block_index": record.get("block_index"),
        "stage_tag": record.get("stage_tag"),
        "branch": branch,
        "call_mode": call_mode,
        "diagnostic_mode": diagnostic_mode,
        "mode": record.get("mode", call_mode),
        "rho_before": record.get("rho_before"),
        "rho_after": record.get("rho_after"),
        "delta_rho": delta_rho,
        "delta_rho_local": record.get("delta_rho_local", delta_rho),
        "delta_rho_vs_observe": record.get("delta_rho_vs_observe"),
        "alpha_lf": record.get("alpha_lf"),
        "alpha_hf": record.get("alpha_hf"),
        "edit_applied": record.get("edit_applied"),
        "edit_selected_indices": _json_cell(edit_selected),
        "diagnostic_batch_indices": _json_cell(diagnostic_batches),
        "selected_indices": _json_cell(selected_indices),
        "batch_indices": _json_cell(batch_indices),
        "target_call_indices": record.get("target_call_indices"),
        "scope_mode": record.get("scope_mode"),
        "scope_map_path": record.get("scope_map_path"),
        "block_scope": record.get("block_scope"),
        "stage_scope": record.get("stage_scope"),
        "diagnostic_call_indices": record.get("diagnostic_call_indices"),
        "target_call_selected": record.get("target_call_selected"),
        "scope_selected": record.get("scope_selected"),
        "scope_id": record.get("scope_id"),
        "scope_reject_reason": record.get("scope_reject_reason"),
        "diagnostic_call_selected": record.get("diagnostic_call_selected"),
        "branch_mode": record.get("branch_mode"),
        "diagnostic_branch": record.get("diagnostic_branch"),
        "attn_delta_mean": record.get("attn_delta_mean"),
        "attn_delta_max": record.get("attn_delta_max"),
        "estimated_logits_mib": record.get("estimated_logits_mib"),
        "estimated_peak_mib": record.get("estimated_peak_mib"),
        "memory_estimate_json": _json_cell(record.get("memory_estimate")),
        "metric_defined": record.get("metric_defined"),
        "metric_undefined_reason": record.get("metric_undefined_reason"),
        "rho_definition": record.get("rho_definition") or "legacy_unspecified",
        "rho_centering": record.get("rho_centering") or "unknown",
        **{key: _json_cell(record.get(key)) for key in (
            "rho_raw_before", "rho_raw_after", "rho_centered_before", "rho_centered_after",
            "spatial_shape", "q_shape", "k_shape", "v_shape", "cutoff", "spectral_diag",
            "diagnostic_top_k", "diagnostic_max_batches", "diagnostic_max_heads",
        )},
    }


def parse_records(
    records: Iterable[dict[str, Any]],
    record_type: str = "spectral_diag",
    include_summaries: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    terminals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    after_terminal: set[str] = set()
    for record in records:
        current_type = record.get("record_type")
        run_id = str(record.get("run_id"))
        if current_type == "run_final_summary":
            terminals[run_id].append(record)
        elif current_type == "spectral_diag" and terminals.get(run_id):
            after_terminal.add(run_id)
        if current_type == record_type or (include_summaries and current_type in {"step_snapshot", "step_final_summary", "run_final_summary", "fallback"}):
            rows.append(record_to_row(record))
    for row in rows:
        run_id = str(row.get("run_id"))
        completed = terminals.get(run_id, [])
        if not completed:
            row["run_completion_json"] = None
            continue
        final = completed[-1]
        steps = final.get("steps")
        step_counts = None
        if isinstance(steps, dict):
            step_counts = {}
            for step, info in steps.items():
                if not isinstance(info, dict):
                    step_counts[str(step)] = None
                elif "spectral_emitted_count" in info:
                    step_counts[str(step)] = info["spectral_emitted_count"]
                else:
                    defined, undefined = info.get("spectral_diag_count"), info.get("spectral_undefined_count", 0)
                    step_counts[str(step)] = defined + undefined if type(defined) is int and type(undefined) is int else None
        row["run_completion_json"] = _json_cell({
            "run_id": final.get("run_id"), "terminal_record_count": len(completed),
            "sampling_status": final.get("sampling_status"),
            "sampling_error_type": final.get("sampling_error_type"),
            "telemetry_errors": final.get("telemetry_errors"),
            "fallback_reasons": final.get("fallback_reasons"),
            "fallbacks": final.get("fallbacks"),
            "last_step_index": final.get("last_step_index"),
            "spectral_emitted_counts_by_step": step_counts,
            "diagnostics_after_terminal": run_id in after_terminal,
        })
    return rows


def parse_log_file(
    path: str | Path,
    record_type: str = "spectral_diag",
    include_summaries: bool = False,
    strict: bool = False,
) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return parse_records(iter_json_records(handle, strict=strict), record_type=record_type, include_summaries=include_summaries)


def write_csv(rows: list[dict[str, Any]], output: Any) -> None:
    writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse Anima AFM JSONL diagnostics.")
    parser.add_argument("jsonl_path", help="Path to an Anima AFM JSONL log.")
    parser.add_argument("--format", choices=("csv", "json"), default="csv", help="Output format.")
    parser.add_argument("--record-type", default="spectral_diag", help="Record type to extract.")
    parser.add_argument("--include-summaries", action="store_true", help="Also include summary/fallback records.")
    parser.add_argument("--strict", action="store_true", help="Fail on malformed JSON lines.")
    args = parser.parse_args(argv)

    try:
        rows = parse_log_file(
            args.jsonl_path,
            record_type=args.record_type,
            include_summaries=args.include_summaries,
            strict=args.strict,
        )
    except ValueError as exc:
        print(f"Malformed JSONL: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        write_csv(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
