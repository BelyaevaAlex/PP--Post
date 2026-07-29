#!/usr/bin/env python3
"""Section 53: claim-package jobs for the AAAI PPtheta-Post paper.

These jobs do not introduce another model variant.  They collect the evidence
needed to make the paper's bounded claim auditable: within-source utility,
counterfactual controls, source compatibility, compact trace sufficiency, and
patient-level trace examples.  Each stage writes ordinary CSV/Markdown files so
the outputs can be aggregated into the paper after cluster completion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"
EVIDENCE_V2_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_evidence_v2_mortality_aaai_evidence_v2"
TRACE_ROOT = ROOT / "output/mortality_paper_jobs/audit_faithfulness_extended_v1"

DATASETS = {
    "eicu": "eICU",
    "mimic3": "MIMIC-III",
    "mimic4": "MIMIC-IV",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def num(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def dataset_from_args(args: list[str]) -> str:
    joined = " ".join(args).lower()
    for key in DATASETS:
        if key in joined:
            return key
    return "eicu"


def option_value(args: list[str], option: str, default: str) -> str:
    for idx, value in enumerate(args[:-1]):
        if value == option:
            return args[idx + 1]
    return default


def latest_json(pattern: str) -> Path | None:
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def rows_for_dataset(path: Path, dataset: str) -> list[dict[str, str]]:
    label = DATASETS[dataset]
    out = []
    for row in read_csv(path):
        dataset_value = row.get("dataset", "")
        if dataset_value in {dataset, label} or row.get("dataset_label") == label:
            out.append(row)
    return out


def claim_contract(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    paired = rows_for_dataset(GENERATED / "aaai_evidence_v2_paired_ci.csv", dataset)
    controls = rows_for_dataset(GENERATED / "aaai_evidence_v2_controls.csv", dataset)
    source = rows_for_dataset(GENERATED / "aaai_evidence_v2_source_compatibility.csv", dataset)
    trace = rows_for_dataset(GENERATED / "ppost_final_trace_sufficiency.csv", dataset)

    paired_row = paired[0] if paired else {}
    observed = next((r for r in controls if r.get("control") == "observed"), {})
    patient_perm = next((r for r in controls if r.get("control") == "patient_permuted"), {})
    best_source = max(source, key=lambda r: num(r.get("delta_mcc")), default={})
    compact = min(
        (r for r in trace if num(r.get("budget_fraction")) < 1.0),
        key=lambda r: abs(num(r.get("mcc_retained_vs_full"), 0.0) - 1.0),
        default={},
    )

    rows = [
        {
            "dataset": label,
            "claim": "Within-source utility",
            "test": "native source vs same source plus PPtheta",
            "evidence": f"Delta MCC {num(paired_row.get('delta_mcc')):+.3f}; Delta sensitivity {num(paired_row.get('delta_sensitivity')):+.3f}",
            "boundary": "source- and operating-point-specific",
        },
        {
            "dataset": label,
            "claim": "Non-random evidence",
            "test": "observed trace vs patient-permuted trace",
            "evidence": f"MCC drop {num(patient_perm.get('delta_vs_observed_mcc')):+.3f} from observed MCC {num(observed.get('mcc')):.3f}",
            "boundary": "control gap is not a calibrated-risk claim",
        },
        {
            "dataset": label,
            "claim": "Source compatibility",
            "test": "same posterior layer across native rule sources",
            "evidence": f"best source {best_source.get('rule_source', 'n/a')} / {best_source.get('variant', 'n/a')}: Delta MCC {num(best_source.get('delta_mcc')):+.3f}",
            "boundary": "strong sources can be negative boundaries",
        },
        {
            "dataset": label,
            "claim": "Compact replayable trace",
            "test": "top family trace vs full trace",
            "evidence": f"trace fraction {num(compact.get('trace_fraction')):.3f}; MCC retained {num(compact.get('mcc_retained_vs_full')):.3f}",
            "boundary": "compact trace summarizes, full trace remains faithful object",
        },
    ]
    write_csv(out_dir / "claim_contract.csv", rows)
    write_md(out_dir / "claim_contract.md", rows, ["dataset", "claim", "test", "evidence", "boundary"])


def source_boundary_map(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    rows = rows_for_dataset(GENERATED / "aaai_evidence_v2_source_compatibility.csv", dataset)
    rows = [r for r in rows if r.get("rule_source")]
    best_mcc = sorted(rows, key=lambda r: num(r.get("delta_mcc")), reverse=True)[:5]
    best_sens = sorted(rows, key=lambda r: num(r.get("delta_sensitivity")), reverse=True)[:5]
    worst_mcc = sorted(rows, key=lambda r: num(r.get("delta_mcc")))[:5]
    out = []
    for kind, part in (("best_mcc", best_mcc), ("best_sensitivity", best_sens), ("negative_boundary", worst_mcc)):
        for rank, row in enumerate(part, start=1):
            out.append(
                {
                    "dataset": label,
                    "kind": kind,
                    "rank": rank,
                    "source": row.get("rule_source", ""),
                    "variant": row.get("variant", ""),
                    "delta_mcc": row.get("delta_mcc", ""),
                    "delta_sensitivity": row.get("delta_sensitivity", ""),
                    "delta_brier": row.get("delta_brier_score", ""),
                    "trace_fraction": row.get("trace_fraction", ""),
                }
            )
    write_csv(out_dir / "source_boundary_map.csv", out)
    write_md(out_dir / "source_boundary_map.md", out, ["dataset", "kind", "rank", "source", "variant", "delta_mcc", "delta_sensitivity", "delta_brier"])


def control_gap_audit(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    controls = rows_for_dataset(GENERATED / "aaai_evidence_v2_controls.csv", dataset)
    keep = {"observed", "patient_permuted", "class_prior_only", "column_shuffled_class_scores", "temperature_flattened_t4"}
    out = []
    for row in controls:
        if row.get("control") not in keep:
            continue
        out.append(
            {
                "dataset": label,
                "control": row.get("control", ""),
                "mcc": row.get("mcc", ""),
                "delta_mcc": row.get("delta_vs_observed_mcc", ""),
                "sensitivity": row.get("sensitivity", ""),
                "delta_sensitivity": row.get("delta_vs_observed_sensitivity", ""),
                "log_loss": row.get("log_loss", ""),
                "delta_log_loss": row.get("delta_vs_observed_log_loss", ""),
                "ece_10": row.get("ece_10", ""),
                "delta_ece": row.get("delta_vs_observed_ece_10", ""),
            }
        )
    write_csv(out_dir / "control_gap_audit.csv", out)
    write_md(out_dir / "control_gap_audit.md", out, ["dataset", "control", "mcc", "delta_mcc", "sensitivity", "delta_sensitivity", "delta_log_loss", "delta_ece"])


def trace_sufficiency_refresh(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    rows = rows_for_dataset(GENERATED / "ppost_final_trace_sufficiency.csv", dataset)
    out = []
    for row in rows:
        out.append(
            {
                "dataset": label,
                "source": row.get("source", ""),
                "variant": row.get("variant", ""),
                "budget_fraction": row.get("budget_fraction", ""),
                "trace_fraction": row.get("trace_fraction", ""),
                "delta_mcc": row.get("delta_mcc", ""),
                "delta_sensitivity": row.get("delta_sensitivity", ""),
                "mcc_retained_vs_full": row.get("mcc_retained_vs_full", ""),
                "claim": row.get("claim", ""),
            }
        )
    write_csv(out_dir / "trace_sufficiency_refresh.csv", out)
    write_md(out_dir / "trace_sufficiency_refresh.md", out, ["dataset", "source", "variant", "budget_fraction", "trace_fraction", "delta_mcc", "delta_sensitivity", "mcc_retained_vs_full"])


def _short_rule(rule: Any, limit: int = 130) -> str:
    text = str(rule or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def reviewer_trace_examples(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    rows: list[dict[str, Any]] = []

    cand_path = EVIDENCE_V2_ROOT / dataset / "rahmatullaev_v2_case_trace_candidates" / "case_trace_candidates.csv"
    for row in read_csv(cand_path)[:3]:
        rows.append(
            {
                "dataset": label,
                "example_type": "tabular correction candidate",
                "case_id": row.get("patient_index_in_fold", ""),
                "true_label": row.get("y_true", ""),
                "native_probability": row.get("native_p_mortality", ""),
                "ppost_probability": row.get("ppost_p_mortality", ""),
                "probability_shift": row.get("ppost_minus_native", ""),
                "trace_detail": f"{row.get('source', '')} / {row.get('variant', '')}",
            }
        )

    trace_path = latest_json(f"output/mortality_paper_jobs/audit_faithfulness_extended_v1/{dataset}/case_studies/case_studies_*.json")
    if trace_path is None:
        trace_path = latest_json(f"output/mortality_paper_jobs/full_tabpfn_mortality_full_tabpfn_v3/{dataset}/case_studies/case_studies_*.json")
    if trace_path is not None:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        for sample in data.get("samples", [])[:2]:
            top = sample.get("top_branches", [])[:3]
            rules = "; ".join(_short_rule(branch.get("rule", "")) for branch in top)
            rows.append(
                {
                    "dataset": label,
                    "example_type": "replayable posterior trace",
                    "case_id": sample.get("x_id", ""),
                    "true_label": sample.get("true_class", ""),
                    "native_probability": "",
                    "ppost_probability": sample.get("class_proba", ""),
                    "probability_shift": "",
                    "trace_detail": rules,
                    "trace_file": str(trace_path.relative_to(ROOT)),
                }
            )

    write_csv(out_dir / "reviewer_trace_examples.csv", rows)
    write_md(out_dir / "reviewer_trace_examples.md", rows, ["dataset", "example_type", "case_id", "true_label", "native_probability", "ppost_probability", "probability_shift", "trace_detail"])


def claim_package_summary(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    paired = rows_for_dataset(GENERATED / "aaai_evidence_v2_paired_ci.csv", dataset)
    controls = rows_for_dataset(GENERATED / "aaai_evidence_v2_controls.csv", dataset)
    source = rows_for_dataset(GENERATED / "aaai_evidence_v2_source_compatibility.csv", dataset)
    trace = rows_for_dataset(GENERATED / "ppost_final_trace_sufficiency.csv", dataset)
    paired_row = paired[0] if paired else {}
    observed = next((r for r in controls if r.get("control") == "observed"), {})
    patient_perm = next((r for r in controls if r.get("control") == "patient_permuted"), {})
    class_prior = next((r for r in controls if r.get("control") == "class_prior_only"), {})
    best_source = max(source, key=lambda r: num(r.get("delta_mcc")), default={})
    compact = max(trace, key=lambda r: num(r.get("mcc_retained_vs_full")), default={})
    row = {
        "dataset": label,
        "selected_delta_mcc": paired_row.get("delta_mcc", ""),
        "selected_delta_sensitivity": paired_row.get("delta_sensitivity", ""),
        "selected_delta_brier": paired_row.get("delta_brier_score", ""),
        "observed_mcc": observed.get("mcc", ""),
        "patient_permuted_delta_mcc": patient_perm.get("delta_vs_observed_mcc", ""),
        "class_prior_delta_mcc": class_prior.get("delta_vs_observed_mcc", ""),
        "best_source": best_source.get("rule_source", ""),
        "best_source_variant": best_source.get("variant", ""),
        "best_source_delta_mcc": best_source.get("delta_mcc", ""),
        "best_source_delta_sensitivity": best_source.get("delta_sensitivity", ""),
        "compact_trace_fraction": compact.get("trace_fraction", ""),
        "compact_mcc_retained_vs_full": compact.get("mcc_retained_vs_full", ""),
    }
    write_csv(out_dir / "claim_package_summary.csv", [row])
    write_md(out_dir / "claim_package_summary.md", [row], list(row))


EXPERIMENTS = {
    "claim_contract": claim_contract,
    "source_boundary_map": source_boundary_map,
    "control_gap_audit": control_gap_audit,
    "trace_sufficiency_refresh": trace_sufficiency_refresh,
    "reviewer_trace_examples": reviewer_trace_examples,
    "claim_package_summary": claim_package_summary,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    args, rest = parser.parse_known_args(argv)
    dataset = dataset_from_args(rest)
    out_dir = Path(option_value(rest, "--output-dir", str(ROOT / "output/paper/53_aaai_claim_package" / dataset)))
    out_dir.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS[args.experiment](dataset, out_dir)
    print(f"section_53 {args.experiment} wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
