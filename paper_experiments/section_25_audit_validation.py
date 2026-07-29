#!/usr/bin/env python3
"""Paper Section 25: systematic audit-evidence validation.

This section turns existing case-study JSON files into quantitative audit
readiness tables. It is intentionally conservative: current case-study exports
contain top supporting rules and, for newer exports, counterfactual probability
fields. The script reports which audit surfaces are present and keeps a checklist
of fields required for stronger sufficiency, comprehensiveness, and stability
claims.

Example:
    python paper_experiments/section_25_audit_validation.py \
        --case-root output/mortality_paper_jobs/full_tabpfn_mortality_full_tabpfn_v3
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_ROOT = ROOT / "output" / "mortality_paper_jobs" / "full_tabpfn_mortality_full_tabpfn_v3"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "paper" / "25_audit_validation"


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rule_len(rule: str) -> int:
    rule = str(rule or "").strip()
    if not rule:
        return 0
    return rule.count(" AND ") + 1


def _find_case_files(case_root: Path, pattern: str) -> list[Path]:
    return sorted(case_root.glob(pattern), key=lambda p: (str(p.parent), p.name))


def _sample_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    rows: list[dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        top = list(sample.get("top_branches", []) or [])
        support = [_safe_float(b.get("support_score")) for b in top]
        support = [x for x in support if x == x]
        support_sum = float(sum(support)) if support else float("nan")
        top1_share = float(max(support) / support_sum) if support and support_sum > 0 else float("nan")
        rule_lens = [_rule_len(b.get("rule", "")) for b in top]
        branch_ids = [str(b.get("branch_id", "")) for b in top]
        probs = list(sample.get("class_proba", []) or [])
        pred = int(sample.get("predicted_class", -1))
        true = int(sample.get("true_class", -1))
        pred_conf = _safe_float(probs[pred]) if 0 <= pred < len(probs) else float("nan")
        rows.append({
            "case_file": str(path),
            "dataset": payload.get("dataset", ""),
            "stage": path.parent.name,
            "level": payload.get("level", ""),
            "aggregation": payload.get("aggregation", ""),
            "n_branches_total": int(payload.get("n_branches", 0) or 0),
            "sample_idx": sample_idx,
            "x_id": sample.get("x_id", ""),
            "case_type": sample.get("case_type", ""),
            "true_class": true,
            "predicted_class": pred,
            "is_correct": int(true == pred),
            "predicted_confidence": pred_conf,
            "n_top_rules": len(top),
            "top_support_sum": support_sum,
            "top1_support_share": top1_share,
            "mean_top_rule_len": float(stats.mean(rule_lens)) if rule_lens else float("nan"),
            "max_top_rule_len": max(rule_lens) if rule_lens else 0,
            "unique_top_branches": len(set(branch_ids)),
            "has_opposing_rules": int(bool(sample.get("opposing_branches")) or any("opposing" in b for b in top)),
            "has_total_support_mass": int("total_support_mass" in sample or "total_support_mass_by_class" in sample),
            "has_deletion_predictions": int("proba_without_top_rules" in sample or "proba_without_top_rules_by_k" in sample),
            "has_random_deletion_predictions": int("proba_without_random_rules" in sample),
            "has_family_deletion_predictions": int("proba_without_rule_families" in sample),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    values = [v for v in values if isinstance(v, (int, float)) and v == v]
    return float(stats.mean(values)) if values else float("nan")


def _file_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_file"])].append(row)
    out: list[dict[str, Any]] = []
    for case_file, sub in sorted(grouped.items()):
        classes = Counter(str(r["true_class"]) for r in sub)
        case_types = Counter(str(r.get("case_type", "")) for r in sub if str(r.get("case_type", "")))
        out.append({
            "case_file": case_file,
            "dataset": sub[0].get("dataset", ""),
            "stage": sub[0].get("stage", ""),
            "n_samples": len(sub),
            "class_counts": json.dumps(dict(classes), sort_keys=True),
            "case_type_counts": json.dumps(dict(case_types), sort_keys=True),
            "accuracy_on_exported_cases": _mean([float(r["is_correct"]) for r in sub]),
            "mean_predicted_confidence": _mean([_safe_float(r["predicted_confidence"]) for r in sub]),
            "mean_top_support_sum": _mean([_safe_float(r["top_support_sum"]) for r in sub]),
            "mean_top1_support_share": _mean([_safe_float(r["top1_support_share"]) for r in sub]),
            "mean_top_rule_len": _mean([_safe_float(r["mean_top_rule_len"]) for r in sub]),
            "has_opposing_rules": min(int(r["has_opposing_rules"]) for r in sub),
            "has_total_support_mass": min(int(r["has_total_support_mass"]) for r in sub),
            "has_deletion_predictions": min(int(r["has_deletion_predictions"]) for r in sub),
            "has_random_deletion_predictions": min(int(r["has_random_deletion_predictions"]) for r in sub),
            "has_family_deletion_predictions": min(int(r["has_family_deletion_predictions"]) for r in sub),
        })
    return out


def _write_md(path: Path, summaries: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Audit Validation Summary",
        "",
        "This report summarizes exported patient-level rule traces. Values marked as missing are not failures of the posterior method; they identify fields that must be exported for stronger auditability claims.",
        "",
        "## Case-Study Files",
        "",
        "| Dataset | Stage | Samples | Class counts | Case types | Acc. exported | Mean conf. | Mean top-1 share | Opposing? | Total mass? | Top deletion? | Random? | Family? |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {row['stage']} | {row['n_samples']} | `{row['class_counts']}` | `{row['case_type_counts']}` | "
            f"{row['accuracy_on_exported_cases']:.3f} | {row['mean_predicted_confidence']:.3f} | "
            f"{row['mean_top1_support_share']:.3f} | {row['has_opposing_rules']} | "
            f"{row['has_total_support_mass']} | {row['has_deletion_predictions']} | "
            f"{row['has_random_deletion_predictions']} | {row['has_family_deletion_predictions']} |"
        )
    lines.extend([
        "",
        "## Required Fields For Full Audit Claim",
        "",
        "| Claim | Required artifact field | Current status |",
        "| --- | --- | --- |",
        "| coverage@K | total posterior support mass per predicted class | checked via `has_total_support_mass` |",
        "| sufficiency@K | prediction using only top-K rules | export `proba_top_rules_only` |",
        "| comprehensiveness/deletion@K | prediction after removing top-K rules | checked via `has_deletion_predictions` |",
        "| random deletion baseline | prediction after removing random K rules | checked via `has_random_deletion_predictions` |",
        "| family-level deletion | prediction after removing grouped similar rule families | checked via `has_family_deletion_predictions` |",
        "| opposing evidence | top opposing branch list and support | checked via `has_opposing_rules` |",
        "| stability@K | fold/bootstrap replicate id and branch sets | requires repeated exports |",
        "| SHAP/TreeSHAP comparison | matched feature/rule deletion curves | requires explanation-baseline artifacts |",
        "",
        "## Sample-Level File",
        "",
        f"Sample-level rows: `{(path.parent / 'audit_sample_rows.csv').name}` ({len(sample_rows)} rows).",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-root", default=str(DEFAULT_CASE_ROOT))
    p.add_argument("--pattern", default="**/case_studies_*.json")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    case_root = Path(args.case_root)
    output_dir = Path(args.output_dir)
    files = _find_case_files(case_root, args.pattern)
    rows: list[dict[str, Any]] = []
    for path in files:
        try:
            rows.extend(_sample_rows(path))
        except Exception as exc:
            print(f"[warn] failed to parse {path}: {exc}")
    summaries = _file_summary(rows)
    _write_csv(output_dir / "audit_sample_rows.csv", rows)
    _write_csv(output_dir / "audit_file_summary.csv", summaries)
    _write_md(output_dir / "AUDIT_VALIDATION_SUMMARY.md", summaries, rows)
    print(f"case_files={len(files)} sample_rows={len(rows)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
