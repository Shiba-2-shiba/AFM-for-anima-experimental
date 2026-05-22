# Log Analysis Example

This directory stores small expected summary artifacts for the V7 analysis workflow.
It intentionally does not include full JSONL or large compare CSV logs.

Workflow shape:

```bash
python scripts/parse_anima_afm_log.py logs/A_observe.jsonl --format csv > out/A_observe.parsed.csv
python scripts/parse_anima_afm_log.py logs/B_call0.jsonl --format csv > out/B_call0.parsed.csv
python scripts/compare_afm_runs.py out/A_observe.parsed.csv out/B_call0.parsed.csv \
  --input-format csv --format csv > out/compare_B_call0_vs_A.csv
python scripts/summarize_afm_experiments.py \
  --compare B_call0_vs_A=out/compare_B_call0_vs_A.csv \
  --late-start-step 16 \
  --out-dir out/summary \
  --report-md
```

Reference files:

- `expected_summary_overview.csv`
- `expected_branch_preservation_counts.csv`

