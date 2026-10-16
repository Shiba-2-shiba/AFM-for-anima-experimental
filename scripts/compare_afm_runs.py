from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
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
    "model_evaluation_index",
    "eligible_call_index",
    "block_id",
    "block_path",
    "stage_tag",
    "branch",
    "observe_pair_count",
    "edit_pair_count",
    "observe_duplicate",
    "edit_duplicate",
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
    "scope_mode",
    "scope_selected",
    "scope_id",
    "scope_reject_reason",
]

# Run IDs identify each input independently; observe/edit intentionally have different IDs.
PAIR_KEY_COLUMNS = ["step_index", "model_evaluation_index", "eligible_call_index", "block_id", "block_path", "branch"]
ALIGNMENT_FIELDS = [
    "num_steps", "sigma", "spatial_shape", "q_shape", "k_shape", "v_shape",
    "diagnostic_batch_indices", "cutoff", "spectral_diag", "diagnostic_top_k",
    "diagnostic_max_batches", "diagnostic_max_heads", "rho_definition", "rho_centering",
]
PROVENANCE_FIELDS = [
    "observe_run_id", "edit_run_id", "observe_completion_json", "edit_completion_json", "provenance_status", "provenance_issues",
    "observe_manifest_json", "edit_manifest_json", "comparison_validation_passed",
    "comparison_validation_errors", "unmatched_observe_pairs",
]
CARRIED_FIELDS = [
    "attn_delta_mean", "attn_delta_max", "rho_definition", "rho_centering",
    "metric_defined", "metric_undefined_reason",
    "rho_raw_before", "rho_raw_after", "rho_centered_before", "rho_centered_after",
    *[field for field in ALIGNMENT_FIELDS if field not in {"rho_definition", "rho_centering"}],
]
FIELDNAMES = list(dict.fromkeys(FIELDNAMES + PROVENANCE_FIELDS + CARRIED_FIELDS))


class PairRows(list):
    def __init__(self, rows: list[dict[str, Any]], pair_analysis: dict[str, Any]):
        super().__init__(rows)
        self.pair_analysis = pair_analysis


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_canonical(row.get(field)) for field in PAIR_KEY_COLUMNS)


def _key_record(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(PAIR_KEY_COLUMNS, key))


def _canonical(value: Any) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _cell(value)


def _cell(value: Any) -> Any:
    if value == "":
        return None
    return value


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _numeric(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return None


def _intish(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value) if re.fullmatch(r"[+-]?[0-9]+", value.strip()) else None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _normalize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {name: row.get(name) for name in PARSE_FIELDNAMES}
    for key in ("schema_version", "step_index", "model_evaluation_index", "num_steps", "last_index", "eligible_call_index"):
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
    for key in ("edit_applied", "metric_defined", "target_call_selected", "scope_selected", "diagnostic_call_selected"):
        if normalized.get(key) not in (None, ""):
            parsed = _boolean(normalized.get(key))
            if parsed is not None:
                normalized[key] = parsed
    return normalized


def read_rows(path: str | Path, input_format: str = "auto", strict: bool = False) -> list[dict[str, Any]]:
    path = Path(path)
    resolved = input_format
    if resolved == "auto":
        resolved = "csv" if path.suffix.lower() == ".csv" else "jsonl"
    if resolved == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [_normalize_csv_row(row) for row in csv.DictReader(handle)]
    return parse_log_file(path, strict=strict)


def _pair_analysis(
    observe_counts: Counter[tuple[Any, ...]],
    edit_counts: Counter[tuple[Any, ...]],
) -> dict[str, Any]:
    duplicate_observe = [key for key, count in observe_counts.items() if count > 1]
    duplicate_edit = [key for key, count in edit_counts.items() if count > 1]
    unmatched_observe = [key for key in observe_counts if key not in edit_counts]
    return {
        "pair_key_columns": PAIR_KEY_COLUMNS,
        "duplicate_observe_pairs": len(duplicate_observe),
        "duplicate_edit_pairs": len(duplicate_edit),
        "unmatched_observe_pairs": len(unmatched_observe),
        "duplicate_observe_keys": [_key_record(key) for key in sorted(duplicate_observe, key=repr)],
        "duplicate_edit_keys": [_key_record(key) for key in sorted(duplicate_edit, key=repr)],
        "unmatched_observe_keys": [_key_record(key) for key in sorted(unmatched_observe, key=repr)],
    }


def _manifest_issues(manifest: Any, rows: list[dict[str, Any]], label: str) -> list[str]:
    if not isinstance(manifest, dict):
        return [f"{label}: missing provenance manifest"]
    issues = []
    run_ids = {_cell(row.get("run_id")) for row in rows}
    if not manifest.get("run_id") or run_ids != {manifest.get("run_id")}:
        issues.append(f"{label}: manifest run_id does not identify every row")
    if manifest.get("schema_version") != 1:
        issues.append(f"{label}: unsupported manifest schema_version")
    for key, length in (("source_revision", 40), ("workflow_sha256", 64)):
        if not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", str(manifest.get(key, ""))):
            issues.append(f"{label}: missing or invalid {key}")
    settings = manifest.get("generation_settings")
    if not isinstance(settings, dict):
        return issues + [f"{label}: missing generation_settings"]
    for key in ("model", "prompt", "sampler", "scheduler"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            issues.append(f"{label}: missing or invalid {key}")
    if not isinstance(settings.get("negative_prompt"), str):
        issues.append(f"{label}: missing negative_prompt")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(settings.get("model_sha256", ""))):
        issues.append(f"{label}: missing or invalid model_sha256")
    for key in ("seed", "num_steps", "width", "height", "batch_size"):
        value = _intish(settings.get(key))
        if value is None or value < (0 if key == "seed" else 1):
            issues.append(f"{label}: missing or invalid {key}")
    cfg = _numeric(settings.get("cfg"))
    if cfg is None or cfg < 0:
        issues.append(f"{label}: missing or invalid cfg")
    for row in rows:
        if _intish(row.get("num_steps")) != _intish(settings.get("num_steps")):
            issues.append(f"{label}: manifest num_steps disagrees with log")
            break
    return issues


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _completion_issues(rows: list[dict[str, Any]], label: str) -> list[str]:
    if not rows:
        return []
    values = {_canonical(row.get("run_completion_json")) for row in rows}
    completion = _json_object(rows[0].get("run_completion_json"))
    if len(values) != 1 or completion is None:
        return [f"{label}: missing or inconsistent run completion evidence"]
    issues = []
    if completion.get("sampling_status") != "completed" or completion.get("sampling_error_type") not in (None, ""):
        issues.append(f"{label}: sampling did not complete successfully")
    if _intish(completion.get("terminal_record_count")) != 1 or completion.get("diagnostics_after_terminal") is not False:
        issues.append(f"{label}: missing, duplicate, or premature terminal record")
    if _intish(completion.get("telemetry_errors")) != 0:
        issues.append(f"{label}: missing or nonzero telemetry_errors")
    if any(row.get("run_id") != completion.get("run_id") for row in rows):
        issues.append(f"{label}: completion run_id mismatch")
    fallbacks = completion.get("fallback_reasons")
    if not isinstance(fallbacks, dict) or any(
        _intish(count) is None or _intish(count) < 0
        or (_intish(count) > 0 and reason not in {"not_cross_attention", "progress_outside_window"})
        for reason, count in (fallbacks.items() if isinstance(fallbacks, dict) else [])
    ):
        issues.append(f"{label}: missing or unexpected fallback evidence")
    fallback_total = _intish(completion.get("fallbacks"))
    if fallback_total is None or fallback_total < 0 or (isinstance(fallbacks, dict) and fallback_total != sum(_intish(count) or 0 for count in fallbacks.values())):
        issues.append(f"{label}: fallback count does not match reasons")
    steps = _intish(rows[0].get("num_steps"))
    expected = completion.get("spectral_emitted_counts_by_step")
    if steps is None or steps <= 0 or not isinstance(expected, dict) or set(expected) != {str(i) for i in range(steps)}:
        issues.append(f"{label}: terminal step coverage is incomplete")
    else:
        counts = Counter(str(row.get("step_index")) for row in rows)
        if any(_intish(expected[step]) != counts.get(step, 0) for step in expected) or any(step not in expected for step in counts):
            issues.append(f"{label}: diagnostic rows do not match terminal coverage")
    if steps is None or _intish(completion.get("last_step_index")) != steps - 1:
        issues.append(f"{label}: terminal last step is incomplete")
    return issues


def _json_array(value: Any) -> list[Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, list) else None


def _row_issues(row: dict[str, Any], label: str) -> list[str]:
    issues = []
    for field in ("step_index", "model_evaluation_index", "eligible_call_index"):
        value = _intish(row.get(field))
        if value is None or value < 0:
            issues.append(f"{label}: missing or invalid {field}")
    if row.get("branch") not in {"positive", "negative"}:
        issues.append(f"{label}: missing or invalid branch")
    if not any(_cell(row.get(field)) not in (None, "unknown") for field in ("block_id", "block_path")):
        issues.append(f"{label}: missing block identity")
    for field in ALIGNMENT_FIELDS:
        value = _cell(row.get(field))
        if value is None or value in ("unknown", "legacy_unspecified"):
            issues.append(f"{label}: missing {field}")
    for field in ("num_steps", "diagnostic_top_k", "diagnostic_max_batches", "diagnostic_max_heads"):
        value = _intish(row.get(field))
        if value is None or value < 1:
            issues.append(f"{label}: missing or invalid {field}")
    shapes = {}
    for field, length in (("spatial_shape", 2), ("q_shape", 4), ("k_shape", 4), ("v_shape", 4)):
        value = _json_array(row.get(field))
        if value is None or len(value) != length or any(_intish(n) is None or _intish(n) <= 0 for n in value):
            issues.append(f"{label}: invalid {field}")
        else:
            shapes[field] = [_intish(n) for n in value]
    if "spatial_shape" in shapes and "q_shape" in shapes:
        if math.prod(shapes["spatial_shape"]) != shapes["q_shape"][2]:
            issues.append(f"{label}: spatial shape does not match query tokens")
    indices = _json_array(row.get("diagnostic_batch_indices"))
    if not indices or any(_intish(n) is None or _intish(n) < 0 for n in indices):
        issues.append(f"{label}: missing or invalid diagnostic_batch_indices")
    elif len(set(_intish(n) for n in indices)) != len(indices) or ("q_shape" in shapes and max(_intish(n) for n in indices) >= shapes["q_shape"][0]):
        issues.append(f"{label}: invalid diagnostic_batch_indices")
    for field in ("rho_before", "rho_after", "delta_rho_local", "sigma", "cutoff"):
        if _numeric(row.get(field)) is None:
            issues.append(f"{label}: missing or non-finite {field}")
    if "q_shape" in shapes and "k_shape" in shapes and "v_shape" in shapes:
        q, k, v = (shapes[field] for field in ("q_shape", "k_shape", "v_shape"))
        if q[0] != k[0] or k[0] != v[0] or q[3] != k[3] or k[:3] != v[:3] or q[1] % k[1] != 0:
            issues.append(f"{label}: incompatible q/k/v shapes")
    if _boolean(row.get("edit_applied")) is None:
        issues.append(f"{label}: missing or invalid edit_applied")
    if _boolean(row.get("metric_defined")) is not True:
        issues.append(f"{label}: undefined or unverified spectral metric")
    if (row.get("rho_definition"), row.get("rho_centering")) != ("centered_concentration_hf_energy_ratio", "spatial_mean_subtracted"):
        issues.append(f"{label}: unsupported or unverified rho semantics")
    if row.get("spectral_diag") not in {"sampled", "full"}:
        issues.append(f"{label}: invalid spectral_diag")
    cutoff, sigma = _numeric(row.get("cutoff")), _numeric(row.get("sigma"))
    if cutoff is not None and not 0 < cutoff < 1:
        issues.append(f"{label}: cutoff outside (0, 1)")
    if sigma is not None and sigma < 0:
        issues.append(f"{label}: negative sigma")
    step, steps = _intish(row.get("step_index")), _intish(row.get("num_steps"))
    if step is not None and steps is not None and step >= steps:
        issues.append(f"{label}: step_index outside run")
    for field in ("attn_delta_mean", "attn_delta_max", "rho_raw_before", "rho_raw_after", "rho_centered_before", "rho_centered_after"):
        if _cell(row.get(field)) is not None and _numeric(row.get(field)) is None:
            issues.append(f"{label}: non-finite {field}")
    before, after, delta = (_numeric(row.get(field)) for field in ("rho_before", "rho_after", "delta_rho_local"))
    if before is not None and after is not None and delta is not None:
        if not (0 <= before <= 1 and 0 <= after <= 1):
            issues.append(f"{label}: rho outside [0, 1]")
        if not math.isclose(after - before, delta, abs_tol=1e-7, rel_tol=1e-5):
            issues.append(f"{label}: inconsistent local rho delta")
    return issues


def compare_rows(
    observe_rows: list[dict[str, Any]], edit_rows: list[dict[str, Any]],
    allow_duplicate_average: bool = False,
    observe_manifest: dict[str, Any] | None = None,
    edit_manifest: dict[str, Any] | None = None,
) -> PairRows:
    observe_counts = Counter(_key(row) for row in observe_rows)
    edit_counts = Counter(_key(row) for row in edit_rows)
    analysis = _pair_analysis(observe_counts, edit_counts)
    issues = []
    if not observe_rows or not edit_rows:
        issues.append("Both inputs must contain spectral diagnostic rows")
    run_ids = [{_cell(row.get("run_id")) for row in items} for items in (observe_rows, edit_rows)]
    mixed_runs = any(len(ids) > 1 for ids in run_ids)
    if mixed_runs:
        issues.append("Each input must contain exactly one run_id")
    if any(None in ids for ids in run_ids):
        issues.append("Missing run_id")
    provenance_issues = _manifest_issues(observe_manifest, observe_rows, "observe")
    provenance_issues += _manifest_issues(edit_manifest, edit_rows, "edit")
    if isinstance(observe_manifest, dict) and isinstance(edit_manifest, dict):
        for field in ("source_revision", "generation_settings"):
            if _canonical(observe_manifest.get(field)) != _canonical(edit_manifest.get(field)):
                provenance_issues.append(f"Manifest {field} differs between runs")
    provenance_status = "manifest_verified" if not provenance_issues else "unverified"
    issues.extend(provenance_issues)
    for label, items in (("observe", observe_rows), ("edit", edit_rows)):
        issues.extend(_completion_issues(items, label))
        for field in ("num_steps", "cutoff", "spectral_diag", "diagnostic_top_k", "diagnostic_max_batches", "diagnostic_max_heads", "rho_definition", "rho_centering"):
            if len({_canonical(row.get(field)) for row in items}) > 1:
                issues.append(f"{label}: {field} changes within run")
        for row in items:
            issues.extend(_row_issues(row, label))
    if any(row.get("call_mode") != "observe" or _boolean(row.get("edit_applied")) is not False for row in observe_rows):
        issues.append("Observe input is not an unedited observe-mode baseline")
    for field in ("duplicate_observe_pairs", "duplicate_edit_pairs", "unmatched_observe_pairs"):
        if analysis[field]:
            issues.append(f"{field}: {analysis[field]}")
    observe_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in observe_rows:
        observe_by_key[_key(row)].append(row)
    output = []
    for row in edit_rows:
        key = _key(row)
        baseline = observe_by_key.get(key, [])
        observe_count, edit_count = observe_counts.get(key, 0), edit_counts[key]
        status = "matched" if baseline else "missing_observe"
        incompatible = any(
            _cell(other.get(field)) is not None and _cell(row.get(field)) is not None
            and _canonical(other.get(field)) != _canonical(row.get(field))
            for other in baseline for field in ALIGNMENT_FIELDS
        )
        if any(_intish(row.get(field)) is None or _intish(row.get(field)) < 0
               for field in ("step_index", "eligible_call_index")) or row.get("branch") not in {"positive", "negative"}:
            status = "invalid_pair_key"
        elif _cell(row.get("model_evaluation_index")) is not None and (
            _intish(row.get("model_evaluation_index")) is None or _intish(row.get("model_evaluation_index")) < 0
        ):
            status = "invalid_pair_key"
        elif mixed_runs:
            status = "invalid_run"
        elif provenance_issues and (observe_manifest is not None or edit_manifest is not None):
            status = "invalid_provenance"
        elif _boolean(row.get("metric_defined")) is False or any(_boolean(other.get("metric_defined")) is False for other in baseline):
            status = "undefined_metric"
        elif incompatible:
            status = "incompatible"
        elif observe_count > 1 or edit_count > 1:
            status = "duplicate_averaged" if allow_duplicate_average and observe_count > 1 and edit_count == 1 else "ambiguous_duplicate"
        values = [_numeric(other.get("rho_after")) for other in baseline]
        if status in {"matched", "duplicate_averaged"} and (any(value is None for value in values) or _numeric(row.get("rho_after")) is None):
            status = "invalid_numeric"
        if status != "matched":
            issues.append(f"{status}: {key}")
        usable = status in {"matched", "duplicate_averaged"}
        rho_observe = _mean(values) if usable else None
        rho_after = _numeric(row.get("rho_after"))
        out = {field: row.get(field) for field in FIELDNAMES}
        out.update({
            "pair_status": status,
            "observe_pair_count": observe_count, "edit_pair_count": edit_count,
            "observe_duplicate": observe_count > 1, "edit_duplicate": edit_count > 1,
            "rho_observe": rho_observe, "rho_edit_before": _numeric(row.get("rho_before")),
            "rho_edit_after": rho_after, "delta_rho_local": _numeric(row.get("delta_rho_local")),
            "delta_rho_vs_observe": rho_after - rho_observe if rho_after is not None and rho_observe is not None else None,
            "observe_run_id": baseline[0].get("run_id") if baseline else None,
            "edit_run_id": row.get("run_id"), "provenance_status": provenance_status,
            "observe_completion_json": baseline[0].get("run_completion_json") if baseline else None,
            "edit_completion_json": row.get("run_completion_json"),
            "provenance_issues": json.dumps(provenance_issues),
            "observe_manifest_json": json.dumps(observe_manifest, sort_keys=True) if observe_manifest else None,
            "edit_manifest_json": json.dumps(edit_manifest, sort_keys=True) if edit_manifest else None,
            "rho_definition": row.get("rho_definition") or "legacy_unspecified",
            "rho_centering": row.get("rho_centering") or "unknown",
            "unmatched_observe_pairs": analysis["unmatched_observe_pairs"],
        })
        output.append(out)
    issues = list(dict.fromkeys(issues))
    analysis.update(validation_passed=not issues, validation_errors=issues, provenance_status=provenance_status)
    for row in output:
        row.update(comparison_validation_passed=not issues, comparison_validation_errors=json.dumps(issues))
    return PairRows(output, analysis)


def compare_files(
    observe_path: str | Path, edit_path: str | Path, input_format: str = "auto",
    *, strict: bool = False, allow_duplicate_average: bool = False,
    observe_manifest: dict[str, Any] | None = None, edit_manifest: dict[str, Any] | None = None,
) -> PairRows:
    return compare_rows(read_rows(observe_path, input_format, strict), read_rows(edit_path, input_format, strict),
                        allow_duplicate_average, observe_manifest, edit_manifest)


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [_numeric(row.get(field)) for row in rows]
    present = [value for value in values if value is not None]
    return _mean(present)


def _error_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return [value]
    return value if isinstance(value, list) else []


def summarize_rows(rows: list[dict[str, Any]], late_start_step: int | None = None) -> dict[str, Any]:
    matched = [row for row in rows if row.get("pair_status") == "matched"]
    missing = [row for row in rows if row.get("pair_status") == "missing_observe"]
    edited = [row for row in rows if _boolish(row.get("edit_applied"))]
    pair_analysis = getattr(rows, "pair_analysis", None)
    if pair_analysis is None:
        observe_duplicate_keys = {_key(row) for row in rows if _boolish(row.get("observe_duplicate"))}
        edit_duplicate_keys = {_key(row) for row in rows if _boolish(row.get("edit_duplicate"))}
        pair_analysis = {
            "pair_key_columns": PAIR_KEY_COLUMNS,
            "duplicate_observe_pairs": len(observe_duplicate_keys),
            "duplicate_edit_pairs": len(edit_duplicate_keys),
            "unmatched_observe_pairs": max((_intish(row.get("unmatched_observe_pairs")) or 0 for row in rows), default=0),
            "duplicate_observe_keys": [_key_record(key) for key in sorted(observe_duplicate_keys, key=repr)],
            "duplicate_edit_keys": [_key_record(key) for key in sorted(edit_duplicate_keys, key=repr)],
            "unmatched_observe_keys": [],
            "validation_passed": bool(rows) and all(_boolish(row.get("comparison_validation_passed")) for row in rows),
            "validation_errors": sorted({error for row in rows for error in _error_list(row.get("comparison_validation_errors"))}),
            "provenance_status": "manifest_verified" if rows and all(row.get("provenance_status") == "manifest_verified" for row in rows) else "unverified",
        }
    summary: dict[str, Any] = {
        "rows": len(rows),
        "matched": len(matched),
        "missing_observe": len(missing),
        **pair_analysis,
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
    parser.add_argument("--fail-on-duplicate-pairs", action="store_true", help="Exit non-zero if observe or edit duplicate pair keys exist.")
    parser.add_argument("--fail-on-unmatched-observe", action="store_true", help="Exit non-zero if observe pair keys have no edit row.")
    parser.add_argument("--allow-duplicate-average", action="store_true", help="Retain backward-compatible averaging of duplicate observe rows.")
    parser.add_argument("--strict-validation", action="store_true", help="Fail closed on incomplete, incompatible, ambiguous or unverified evidence.")
    parser.add_argument("--observe-manifest", help="JSON provenance manifest bound to observe run_id.")
    parser.add_argument("--edit-manifest", help="JSON provenance manifest bound to edit run_id.")
    args = parser.parse_args(argv)

    try:
        manifests = [json.loads(Path(path).read_text(encoding="utf-8")) if path else None
                     for path in (args.observe_manifest, args.edit_manifest)]
        rows = compare_files(args.observe_path, args.edit_path, input_format=args.input_format,
                             strict=args.strict_validation, allow_duplicate_average=args.allow_duplicate_average,
                             observe_manifest=manifests[0], edit_manifest=manifests[1])
    except (ValueError, OSError) as exc:
        print(f"Invalid comparison input: {exc}", file=sys.stderr)
        return 2
    summary = summarize_rows(rows, late_start_step=args.late_start_step)
    if args.summary:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif args.format == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        write_csv(rows, sys.stdout)
    if args.fail_on_missing_observe and summary["missing_observe"] > 0:
        return 3
    if args.fail_on_duplicate_pairs and (
        summary["duplicate_observe_pairs"] > 0 or summary["duplicate_edit_pairs"] > 0
    ):
        return 4
    if args.fail_on_unmatched_observe and summary["unmatched_observe_pairs"] > 0:
        return 5
    if args.strict_validation and not summary["validation_passed"]:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
