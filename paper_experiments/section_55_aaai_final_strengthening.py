#!/usr/bin/env python3
"""Section 55: final reviewer-facing strengthening checks for PPtheta-Post.

The stages here are validation and presentation jobs, not new model variants.
They package existing fold outputs and exported traces into tables that answer
common reviewer questions: is PPtheta useful, replayable, clinically readable,
compact under deletion/sufficiency, and honest about failure modes?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"
TRACE_ROOT = ROOT / "output/mortality_paper_jobs/audit_faithfulness_extended_v1"
CLAIM_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_claim_package_mortality_aaai_claim_package_v1"
EVIDENCE_V2_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_aaai_evidence_v2_mortality_aaai_evidence_v2"
DATASETS = {"eicu": "eICU", "mimic3": "MIMIC-III", "mimic4": "MIMIC-IV"}


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


def rows_for_dataset(path: Path, dataset: str) -> list[dict[str, str]]:
    label = DATASETS[dataset]
    out = []
    for row in read_csv(path):
        if row.get("dataset") in {dataset, label} or row.get("dataset_label") == label:
            out.append(row)
    return out


def trace_path(dataset: str) -> Path | None:
    candidates = sorted((TRACE_ROOT / dataset / "case_studies").glob("case_studies_*.json"))
    if not candidates:
        candidates = sorted((ROOT / f"output/mortality_paper_jobs/full_tabpfn_mortality_full_tabpfn_v3/{dataset}/case_studies").glob("case_studies_*.json"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def load_trace(dataset: str) -> dict[str, Any]:
    p = trace_path(dataset)
    if p is None:
        return {"samples": [], "trace_file": ""}
    data = json.loads(p.read_text(encoding="utf-8"))
    data["trace_file"] = str(p.relative_to(ROOT))
    return data


FEATURE_NAMES = {
    "spo2": "oxygen saturation",
    "systolic_bp": "systolic blood pressure",
    "diastolic_bp": "diastolic blood pressure",
    "mean_bp": "mean blood pressure",
    "heart_rate": "heart rate",
    "respiratory_rate": "respiratory rate",
    "bun": "blood urea nitrogen",
    "wbc": "white blood cell count",
    "bilirubin_total": "total bilirubin",
    "bicarbonate": "bicarbonate",
    "platelet": "platelet count",
    "hematocrit": "hematocrit",
    "hemoglobin": "hemoglobin",
    "glucose": "glucose",
    "lactate": "lactate",
    "potassium": "potassium",
    "sodium": "sodium",
    "chloride": "chloride",
    "temperature": "temperature",
}


def pretty_feature(raw: str) -> str:
    return FEATURE_NAMES.get(raw, raw.replace("_", " "))


def pretty_rule(rule: str, max_terms: int = 4) -> str:
    clauses = []
    for raw in rule.replace("≤", "<=").replace("≥", ">=").split(" AND "):
        raw = raw.strip()
        if not raw:
            continue
        value_match = re.search(r"value\(([^)]+)\)\s*(<=|>=|>|<)", raw)
        mask_match = re.search(r"mask\(([^)]+)\)", raw)
        if value_match:
            feature = pretty_feature(value_match.group(1))
            direction = "lower-range" if value_match.group(2) in {"<=", "<"} else "higher-range"
            clauses.append(f"{direction} condition on {feature}")
        elif mask_match:
            clauses.append("measurement pattern for " + pretty_feature(mask_match.group(1)))
        else:
            clauses.append(raw)
    text = "; ".join(clauses[:max_terms])
    if len(clauses) > max_terms:
        text += "; ..."
    return text


def mortality_probability(value: Any) -> float:
    if isinstance(value, list) and len(value) > 1:
        return num(value[1])
    return num(value)


def slim_usefulness(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    summary = rows_for_dataset(GENERATED / "ppost_claim_package_summary.csv", dataset)
    usefulness = rows_for_dataset(GENERATED / "ppost_final_usefulness.csv", dataset)
    row = summary[0] if summary else {}
    use = usefulness[0] if usefulness else {}
    out = [{
        "dataset": label,
        "native_source": use.get("source", row.get("best_source", "")),
        "ppost_variant": use.get("variant", row.get("best_source_variant", "")),
        "delta_mcc": row.get("selected_delta_mcc", use.get("delta_mcc", "")),
        "delta_sensitivity": row.get("selected_delta_sensitivity", use.get("delta_sensitivity", "")),
        "patient_permuted_delta_mcc": row.get("patient_permuted_delta_mcc", ""),
        "compact_trace_fraction": row.get("compact_trace_fraction", ""),
        "claim": use.get("claim", "bounded source-specific claim"),
    }]
    write_csv(out_dir / "slim_usefulness.csv", out)
    write_md(out_dir / "slim_usefulness.md", out, list(out[0]))


def replay_integrity(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    data = load_trace(dataset)
    samples = data.get("samples", [])
    n = len(samples)
    max_sum_err = 0.0
    decision_match = 0
    complete_top = 0
    complete_counterfactual = 0
    finite_prob = 0
    for sample in samples:
        proba = sample.get("class_proba", [])
        if isinstance(proba, list) and len(proba) >= 2 and all(math.isfinite(num(x)) for x in proba):
            finite_prob += 1
            max_sum_err = max(max_sum_err, abs(sum(float(x) for x in proba) - 1.0))
            threshold = num(sample.get("decision_threshold"), 0.5)
            if len(proba) == 2:
                pred = int(float(proba[1]) >= threshold)
            else:
                pred = int(max(range(len(proba)), key=lambda i: proba[i]))
            decision_match += int(pred == int(sample.get("predicted_class", -1)))
        top = sample.get("top_branches", [])
        if top and all("rule" in b and "theta_k" in b and "p_z_aggregated" in b for b in top[:3]):
            complete_top += 1
        if sample.get("proba_top_rules_only") and sample.get("proba_without_top_rules") and sample.get("proba_without_rule_families"):
            complete_counterfactual += 1
    out = [{
        "dataset": label,
        "samples": n,
        "finite_probability_vectors": finite_prob,
        "decision_matches_argmax": decision_match,
        "complete_top_branch_records": complete_top,
        "complete_counterfactual_records": complete_counterfactual,
        "max_probability_sum_error": max_sum_err,
        "trace_file": data.get("trace_file", ""),
        "status": "pass" if n and finite_prob == n and decision_match == n and complete_top == n else "check",
    }]
    write_csv(out_dir / "replay_integrity.csv", out)
    write_md(out_dir / "replay_integrity.md", out, list(out[0]))


def deletion_sufficiency(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    data = load_trace(dataset)
    rows = []
    for sample in data.get("samples", []):
        full_p = mortality_probability(sample.get("class_proba"))
        for k in ("1", "3", "5", "10", "25", "50", "100"):
            top_p = mortality_probability(sample.get("proba_top_rules_only", {}).get(k))
            without_p = mortality_probability(sample.get("proba_without_top_rules", {}).get(k))
            family_without_p = mortality_probability(sample.get("proba_without_rule_families", {}).get(k))
            if not math.isfinite(top_p):
                continue
            rows.append({
                "dataset": label,
                "case_type": sample.get("case_type", ""),
                "k": k,
                "full_mortality_probability": full_p,
                "topk_only_mortality_probability": top_p,
                "topk_sufficiency_gap": abs(top_p - full_p),
                "without_topk_mortality_probability": without_p,
                "topk_deletion_shift": without_p - full_p if math.isfinite(without_p) else float("nan"),
                "without_top_families_mortality_probability": family_without_p,
                "family_deletion_shift": family_without_p - full_p if math.isfinite(family_without_p) else float("nan"),
            })
    # summarize by k for the paper table
    summary = []
    for k in ("1", "3", "5", "10", "25", "50", "100"):
        part = [r for r in rows if r["k"] == k]
        if not part:
            continue
        def avg(field: str) -> float:
            vals = [num(r[field]) for r in part if math.isfinite(num(r[field]))]
            return sum(vals) / len(vals) if vals else float("nan")
        summary.append({
            "dataset": label,
            "k": k,
            "cases": len(part),
            "mean_topk_sufficiency_gap": avg("topk_sufficiency_gap"),
            "mean_topk_deletion_shift": avg("topk_deletion_shift"),
            "mean_family_deletion_shift": avg("family_deletion_shift"),
        })
    write_csv(out_dir / "deletion_sufficiency_cases.csv", rows)
    write_csv(out_dir / "deletion_sufficiency_summary.csv", summary)
    write_md(out_dir / "deletion_sufficiency_summary.md", summary, ["dataset", "k", "cases", "mean_topk_sufficiency_gap", "mean_topk_deletion_shift", "mean_family_deletion_shift"])


def clinical_trace(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    rows = []
    cand = read_csv(EVIDENCE_V2_ROOT / dataset / "rahmatullaev_v2_case_trace_candidates" / "case_trace_candidates.csv")
    if cand:
        best = max(cand, key=lambda r: num(r.get("ppost_minus_native")))
        rows.append({
            "dataset": label,
            "example": "corrected mortality-positive case",
            "case_id": best.get("patient_index_in_fold", ""),
            "true_label": best.get("y_true", ""),
            "native_mortality_probability": best.get("native_p_mortality", ""),
            "ppost_mortality_probability": best.get("ppost_p_mortality", ""),
            "probability_shift": best.get("ppost_minus_native", ""),
            "evidence": "native rule source corrected by posterior evidence",
        })
    data = load_trace(dataset)
    samples = data.get("samples", [])
    preferred = [s for s in samples if s.get("case_type") in {"tp", "fn"}] or samples
    if preferred:
        sample = max(preferred, key=lambda s: mortality_probability(s.get("class_proba")))
        for role, branches in (("supporting", sample.get("top_branches", [])), ("opposing", sample.get("opposing_branches", []))):
            for rank, branch in enumerate(branches[:3], start=1):
                rows.append({
                    "dataset": label,
                    "example": f"{role} posterior rule",
                    "case_id": sample.get("x_id", ""),
                    "true_label": sample.get("true_class", ""),
                    "predicted_label": sample.get("predicted_class", ""),
                    "ppost_mortality_probability": mortality_probability(sample.get("class_proba")),
                    "rank": rank,
                    "posterior_activation": branch.get("p_z_aggregated", ""),
                    "class_support": branch.get("theta_k", ""),
                    "evidence": pretty_rule(branch.get("rule", "")),
                })
    write_csv(out_dir / "clinical_trace.csv", rows)
    write_md(out_dir / "clinical_trace.md", rows, ["dataset", "example", "case_id", "true_label", "ppost_mortality_probability", "rank", "posterior_activation", "class_support", "evidence"])


def failure_modes(dataset: str, out_dir: Path) -> None:
    label = DATASETS[dataset]
    source = rows_for_dataset(GENERATED / "ppost_claim_source_boundary_map.csv", dataset)
    neg = [r for r in source if r.get("kind") == "negative_boundary"][:5]
    rows = []
    for row in neg:
        reason = "calibration cost" if num(row.get("delta_brier")) > 0 else "lost discrimination/sensitivity"
        rows.append({
            "dataset": label,
            "source": row.get("source", ""),
            "variant": row.get("variant", ""),
            "delta_mcc": row.get("delta_mcc", ""),
            "delta_sensitivity": row.get("delta_sensitivity", ""),
            "delta_brier": row.get("delta_brier", ""),
            "interpretation": reason,
            "paper_claim": "negative boundary, not recommended operating point",
        })
    write_csv(out_dir / "failure_modes.csv", rows)
    write_md(out_dir / "failure_modes.md", rows, ["dataset", "source", "variant", "delta_mcc", "delta_sensitivity", "delta_brier", "interpretation"])


EXPERIMENTS = {
    "slim_usefulness": slim_usefulness,
    "replay_integrity": replay_integrity,
    "clinical_trace": clinical_trace,
    "deletion_sufficiency": deletion_sufficiency,
    "failure_modes": failure_modes,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    args, rest = parser.parse_known_args(argv)
    dataset = dataset_from_args(rest)
    out_dir = Path(option_value(rest, "--output-dir", str(ROOT / "output/paper/55_final_strengthening" / dataset)))
    out_dir.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS[args.experiment](dataset, out_dir)
    print(f"section_55 {args.experiment} wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
