#!/usr/bin/env python3
"""Paper Section 31: audit faithfulness metrics.

Post-processes case-study or audit JSON/JSONL files into quantitative audit
faithfulness tables. The script is intentionally conservative: if a case export
lacks the fields needed for coverage, sufficiency, deletion, or stability, the
metric is left blank and the missing field is recorded.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_ROOT = ROOT / "output" / "mortality_paper_jobs"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "paper" / "31_audit_faithfulness"

BRANCH_LIST_KEYS = (
    "top_branches",
    "top_rules",
    "supporting_rules",
    "supporting_branches",
    "branch_support",
    "branches",
)
PROBA_KEYS = ("class_probabilities", "class_proba", "probabilities", "proba", "y_proba")
WITHOUT_TOP_KEYS = ("proba_without_top_rules", "proba_without_top_rules_by_k", "proba_without_top_k", "without_top_k_proba")
TOP_ONLY_KEYS = ("proba_top_rules_only", "proba_top_rules_only_by_k", "proba_top_k_only", "top_k_proba", "top_rules_proba")
RANDOM_WITHOUT_KEYS = ("proba_without_random_rules", "proba_without_random_rules_by_k", "random_without_top_k_proba")
FAMILY_WITHOUT_KEYS = ("proba_without_rule_families", "proba_without_rule_families_by_k", "family_without_top_k_proba")
FAMILY_DELETE_COUNT_KEYS = ("n_deleted_rule_family_branches", "family_deleted_branch_count", "n_family_deleted_branches")
TOTAL_SUPPORT_KEYS = ("total_support_mass", "total_support_mass_by_class", "total_support", "support_denominator", "predicted_class_support")


def _iter_json_records(path: Path):
    try:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                for key in ("cases", "records", "examples", "audit_traces", "samples"):
                    if isinstance(data.get(key), list):
                        for item in data[key]:
                            yield item
                        return
                yield data
    except Exception as exc:
        yield {"_load_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _branch_list(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = _first_present(record, BRANCH_LIST_KEYS)
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def _branch_id(branch: dict[str, Any]) -> str:
    for key in ("branch_id", "rule_id", "id", "name"):
        if key in branch:
            return str(branch[key])
    conds = branch.get("conditions")
    if isinstance(conds, list):
        return "|".join(str(c) for c in conds[:4])
    return json.dumps(branch, sort_keys=True)[:80]


def _support(branch: dict[str, Any]) -> float:
    for key in ("support_score", "support", "posterior_support", "a_jk", "score", "mass"):
        val = branch.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    theta = branch.get("theta_k")
    pz = branch.get("p_z_posterior", branch.get("posterior_activation"))
    if isinstance(theta, (int, float)) and isinstance(pz, (int, float)):
        return float(theta) * float(pz)
    return 0.0


def _as_float_list(value: Any) -> list[float] | None:
    if isinstance(value, list):
        out = []
        for x in value:
            if not isinstance(x, (int, float)):
                return None
            out.append(float(x))
        return out
    return None


def _pred_class(record: dict[str, Any], proba: list[float] | None) -> int | None:
    for key in ("predicted_class", "prediction", "y_pred"):
        val = record.get(key)
        if isinstance(val, int):
            return int(val)
    if proba:
        return int(max(range(len(proba)), key=lambda i: proba[i]))
    return None


def _proba_for_class(value: Any, cls: int | None) -> float | None:
    if cls is None:
        return None
    vals = _as_float_list(value)
    if vals is not None and 0 <= cls < len(vals):
        return vals[cls]
    if isinstance(value, dict):
        for key in (str(cls), cls, f"class_{cls}"):
            val = value.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _value_for_k(value: Any, k: int) -> Any:
    if isinstance(value, dict):
        for key in (str(k), k, f"k{k}", f"top_{k}", f"at_{k}", f"@{k}"):
            if key in value:
                return value[key]
    return value


def _support_mass_for_class(value: Any, cls: int | None) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return _proba_for_class(value, cls)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def summarize_stability(rows: list[dict[str, Any]], k: int) -> dict[str, float]:
    groups: dict[str, list[set[str]]] = {}
    for row in rows:
        key = row.get("stability_key")
        ids = row.get(f"top{k}_ids")
        if key and isinstance(ids, set):
            groups.setdefault(key, []).append(ids)
    vals = []
    for sets in groups.values():
        if len(sets) < 2:
            continue
        vals.extend(_jaccard(a, b) for a, b in combinations(sets, 2))
    if not vals:
        return {"stability_count": 0, "stability_mean": float("nan")}
    return {"stability_count": len(vals), "stability_mean": sum(vals) / len(vals)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-root", default=str(DEFAULT_CASE_ROOT))
    p.add_argument("--glob", default="**/*.json*")
    p.add_argument("--k-values", default="1,3,5")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    case_root = Path(args.case_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    for path in sorted(case_root.glob(args.glob)) if case_root.exists() else []:
        if path.is_dir():
            continue
        for idx, record in enumerate(_iter_json_records(path)):
            if not isinstance(record, dict):
                continue
            branches = sorted(_branch_list(record), key=_support, reverse=True)
            proba = _as_float_list(_first_present(record, PROBA_KEYS))
            cls = _pred_class(record, proba)
            total_support_raw = _first_present(record, TOTAL_SUPPORT_KEYS)
            total_support = _support_mass_for_class(total_support_raw, cls)
            top_only = _first_present(record, TOP_ONLY_KEYS)
            without_top = _first_present(record, WITHOUT_TOP_KEYS)
            random_without = _first_present(record, RANDOM_WITHOUT_KEYS)
            family_without = _first_present(record, FAMILY_WITHOUT_KEYS)
            family_delete_counts = _first_present(record, FAMILY_DELETE_COUNT_KEYS)
            missing = []
            if total_support is None:
                missing.append("total_support_mass")
            if top_only is None:
                missing.append("top_only_proba")
            if without_top is None:
                missing.append("without_top_proba")
            if random_without is None:
                missing.append("random_without_proba")
            if family_without is None:
                missing.append("family_without_proba")
            if not branches:
                missing.append("top_branches")

            stability_key = str(record.get("sample_id", record.get("x_id", record.get("patient_id", f"{path}:{idx}"))))
            base = {
                "source_file": str(path),
                "record_index": idx,
                "dataset": record.get("dataset", ""),
                "fold": record.get("fold", ""),
                "case_type": record.get("case_type", record.get("label", "")),
                "true_class": record.get("true_class", record.get("label", "")),
                "predicted_class": cls if cls is not None else "",
                "n_branches_exported": len(branches),
                "missing_fields": ";".join(sorted(set(missing))),
                "stability_key": stability_key,
            }
            internal = dict(base)
            for k in k_values:
                top = branches[:k]
                top_mass = sum(_support(b) for b in top)
                coverage = top_mass / total_support if total_support and total_support > 0 else ""
                full_cls_proba = proba[cls] if proba is not None and cls is not None and 0 <= cls < len(proba) else None
                top_only_cls = _proba_for_class(_value_for_k(top_only, k), cls)
                without_cls = _proba_for_class(_value_for_k(without_top, k), cls)
                random_without_cls = _proba_for_class(_value_for_k(random_without, k), cls)
                family_without_cls = _proba_for_class(_value_for_k(family_without, k), cls)
                deletion_drop = full_cls_proba - without_cls if full_cls_proba is not None and without_cls is not None else ""
                random_deletion_drop = full_cls_proba - random_without_cls if full_cls_proba is not None and random_without_cls is not None else ""
                family_deletion_drop = full_cls_proba - family_without_cls if full_cls_proba is not None and family_without_cls is not None else ""
                deletion_lift = deletion_drop - random_deletion_drop if isinstance(deletion_drop, float) and isinstance(random_deletion_drop, float) else ""
                sufficiency_gap = full_cls_proba - top_only_cls if full_cls_proba is not None and top_only_cls is not None else ""
                ids = {_branch_id(b) for b in top}
                base[f"coverage_at_{k}"] = coverage
                base[f"sufficiency_gap_at_{k}"] = sufficiency_gap
                base[f"deletion_drop_at_{k}"] = deletion_drop
                base[f"random_deletion_drop_at_{k}"] = random_deletion_drop
                base[f"deletion_lift_over_random_at_{k}"] = deletion_lift
                base[f"family_deletion_drop_at_{k}"] = family_deletion_drop
                base[f"family_deleted_branch_count_at_{k}"] = _value_for_k(family_delete_counts, k) or ""
                base[f"top{k}_rule_ids"] = ";".join(sorted(ids))
                internal[f"top{k}_ids"] = ids
            rows.append(base)
            internal_rows.append(internal)

    row_csv = out_dir / "audit_faithfulness_rows.csv"
    summary_csv = out_dir / "audit_faithfulness_summary.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with row_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    summary_rows = []
    for k in k_values:
        numeric = {}
        metric_map = {
            "coverage": f"coverage_at_{k}",
            "sufficiency_gap": f"sufficiency_gap_at_{k}",
            "deletion_drop": f"deletion_drop_at_{k}",
            "random_deletion_drop": f"random_deletion_drop_at_{k}",
            "deletion_lift_over_random": f"deletion_lift_over_random_at_{k}",
            "family_deletion_drop": f"family_deletion_drop_at_{k}",
        }
        for label, metric in metric_map.items():
            vals = [float(r[metric]) for r in rows if r.get(metric) not in ("", None)]
            numeric[f"{label}_count"] = len(vals)
            numeric[f"{label}_mean"] = sum(vals) / len(vals) if vals else ""
        st = summarize_stability(internal_rows, k)
        summary_rows.append({"k": k, "records": len(rows), **numeric, **st})
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = []
        for row in summary_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["k"])
        writer.writeheader()
        writer.writerows(summary_rows)

    missing_counts: dict[str, int] = {}
    for row in rows:
        for item in str(row.get("missing_fields", "")).split(";"):
            if item:
                missing_counts[item] = missing_counts.get(item, 0) + 1
    def _fmt(value: Any) -> str:
        if value in ("", None):
            return ""
        try:
            return f"{float(value):.6f}"
        except Exception:
            return str(value)

    metric_lines = [
        "## Metric Means",
        "",
        "| K | Coverage | Suff. gap | Top deletion | Random deletion | Top-random lift | Family deletion |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        metric_lines.append(
            f"| {row['k']} | {_fmt(row.get('coverage_mean'))} | {_fmt(row.get('sufficiency_gap_mean'))} | "
            f"{_fmt(row.get('deletion_drop_mean'))} | {_fmt(row.get('random_deletion_drop_mean'))} | "
            f"{_fmt(row.get('deletion_lift_over_random_mean'))} | {_fmt(row.get('family_deletion_drop_mean'))} |"
        )

    report = out_dir / "AUDIT_FAITHFULNESS_SUMMARY.md"
    report.write_text(
        "\n".join(
            [
                "# Audit Faithfulness Summary",
                "",
                f"Records processed: {len(rows)}",
                f"Rows CSV: `{row_csv}`",
                f"Summary CSV: `{summary_csv}`",
                "",
            ]
            + metric_lines
            + [
                "",
                "## Missing Field Counts",
                "",
            ]
            + [f"- {k}: {v}" for k, v in sorted(missing_counts.items())]
            + ["", "Blank metric cells mean the case export did not contain the required counterfactual probability or support mass."]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"records={len(rows)}")
    print(f"rows_csv={row_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
