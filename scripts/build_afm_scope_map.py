from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .parse_anima_afm_log import iter_json_records
except ImportError:  # pragma: no cover - direct script execution
    from parse_anima_afm_log import iter_json_records


CANDIDATE_STAGE_TAGS = {"encoder_candidate", "encoder_equivalent"}

CANDIDATE_FIELDNAMES = [
    "scope_id",
    "block_id",
    "block_path",
    "module_class",
    "stage_tag",
    "candidate_score",
    "eligible_call_indices_seen",
    "steps_seen",
    "spatial_shapes_seen",
    "branches_seen",
    "candidate_reasons",
]


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _record_scope_key(record: dict[str, Any]) -> str:
    for key in ("block_path", "block_id", "scope_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _safe_list_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _sorted_jsonable(values: Iterable[Any]) -> list[Any]:
    converted = [_safe_list_value(value) for value in values if value not in (None, "")]
    return [
        list(value) if isinstance(value, tuple) else value
        for value in sorted(set(converted), key=lambda item: str(item))
    ]


def _candidate_reasons(stage_tag: str, block_id: str, block_path: str | None, observations: int) -> list[str]:
    reasons: list[str] = []
    if stage_tag in CANDIDATE_STAGE_TAGS:
        reasons.append(f"stage_tag={stage_tag}")
    if block_path:
        reasons.append("stable_block_path")
    elif block_id != "unknown":
        reasons.append("stable_block_id")
    if observations > 1:
        reasons.append("repeated_observations")
    return reasons


def _candidate_score(stage_tag: str, block_id: str, block_path: str | None, observations: int) -> float:
    if stage_tag not in CANDIDATE_STAGE_TAGS:
        return 0.0
    score = 0.6
    if block_path:
        score += 0.2
    elif block_id != "unknown":
        score += 0.1
    if observations > 1:
        score += 0.2
    return min(score, 1.0)


def build_scope_map(
    records: Iterable[dict[str, Any]],
    *,
    model_fingerprint: str = "",
    workflow_fingerprint: str = "",
    source_runs: list[str] | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("record_type") != "afm_scope_discovery_call":
            continue
        groups[_record_scope_key(record)].append(record)

    entries: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        first = items[0]
        block_id = str(first.get("block_id") or key)
        block_path = first.get("block_path")
        module_class = first.get("module_class")
        stage_tags = [str(item.get("stage_tag") or "unclassified") for item in items]
        stage_tag = max(set(stage_tags), key=stage_tags.count)
        observations = len(items)
        score = _candidate_score(stage_tag, block_id, block_path, observations)
        reasons = _candidate_reasons(stage_tag, block_id, block_path, observations)
        entries.append({
            "scope_id": f"block:{key}",
            "block_id": block_id,
            "block_path": block_path,
            "module_class": module_class,
            "stage_tag": stage_tag,
            "encoder_equivalent": score > 0.0,
            "eligible_call_indices_seen": _sorted_jsonable(item.get("eligible_call_index") for item in items),
            "steps_seen": len({item.get("step_index") for item in items if item.get("step_index") is not None}),
            "spatial_shapes_seen": _sorted_jsonable(item.get("spatial_shape") for item in items),
            "branches_seen": _sorted_jsonable(item.get("cond_or_uncond") for item in items),
            "metadata_sources_seen": _sorted_jsonable(item.get("metadata_source") for item in items),
            "fallback_reasons_seen": _sorted_jsonable(item.get("fallback_reason") for item in items),
            "candidate_score": score,
            "candidate_reasons": reasons,
            "observation_count": observations,
        })

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_fingerprint": model_fingerprint,
        "workflow_fingerprint": workflow_fingerprint,
        "source_runs": source_runs or _sorted_jsonable(record.get("run_id") for records in groups.values() for record in records),
        "classifier": {
            "name": "encoder_equivalent_v1",
            "rules": [
                "group by block_path/block_id from afm_scope_discovery_call records",
                "candidate only when stage_tag explicitly marks encoder_candidate or encoder_equivalent",
                "unknown/fallback records are retained but not promoted",
            ],
        },
        "entries": entries,
    }


def write_scope_outputs(scope_map: dict[str, Any], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "scope_map.json").write_text(
        json.dumps(scope_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    candidates = [entry for entry in scope_map["entries"] if entry.get("encoder_equivalent")]
    with (out_path / "encoder_equivalent_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDNAMES, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for entry in candidates:
            writer.writerow({
                **entry,
                "eligible_call_indices_seen": _json_cell(entry.get("eligible_call_indices_seen")),
                "spatial_shapes_seen": _json_cell(entry.get("spatial_shapes_seen")),
                "branches_seen": _json_cell(entry.get("branches_seen")),
                "candidate_reasons": _json_cell(entry.get("candidate_reasons")),
            })
    report_lines = [
        "# AFM Scope Map Report",
        "",
        f"Entries: {len(scope_map['entries'])}",
        f"Encoder-equivalent candidates: {len(candidates)}",
        "",
        "Candidate promotion is conservative: only records explicitly tagged as encoder candidates are promoted.",
    ]
    (out_path / "scope_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an Anima AFM block scope map from discovery JSONL records.")
    parser.add_argument("jsonl_path", help="Path to AFM discovery JSONL.")
    parser.add_argument("--out-dir", required=True, help="Output directory for scope_map.json and reports.")
    parser.add_argument("--model-fingerprint", default="", help="Optional model/checkpoint fingerprint.")
    parser.add_argument("--workflow-fingerprint", default="", help="Optional workflow fingerprint.")
    args = parser.parse_args(argv)

    with Path(args.jsonl_path).open("r", encoding="utf-8") as handle:
        records = list(iter_json_records(handle))
    scope_map = build_scope_map(
        records,
        model_fingerprint=args.model_fingerprint,
        workflow_fingerprint=args.workflow_fingerprint,
    )
    write_scope_outputs(scope_map, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
