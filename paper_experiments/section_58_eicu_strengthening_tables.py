#!/usr/bin/env python3
"""Aggregate eICU strengthening jobs into paper-ready tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_eicu_strengthening_mortality_eicu_strengthening_v1/eicu"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"

STAGES = {
    "rulefit_official": ("rahmatullaev_eicu_rulefit_official", "rulefit_official_summary.csv"),
    "operating_points": ("rahmatullaev_eicu_operating_points", "operating_points_summary.csv"),
    "measurement_pattern": ("rahmatullaev_eicu_measurement_pattern_families", "measurement_pattern_families_summary.csv"),
    "measurement_policy_calibration": ("rahmatullaev_eicu_measurement_policy_calibration", "measurement_policy_calibration_summary.csv"),
    "family_pruning": ("rahmatullaev_eicu_family_pruning_sweep", "family_pruning_sweep_summary.csv"),
}

SOURCE_DISPLAY = {
    "rulefit": "RuleFit",
    "xgb": "XGBoost",
    "extratrees": "ExtraTrees",
}
VARIANT_DISPLAY = {
    "pp_theta_post_ebm_residual_mcc": "bounded residual evidence",
    "pp_theta_post_ebm_bounded_residual_gate": "bounded residual gate",
    "pp_theta_post_rule_family_calibrated": "rule-family calibrated evidence",
    "pp_theta_post_family_utility_pruned_topk": "utility-pruned top-k evidence",
    "pp_theta_post_operating_calibrated": "calibrated-risk operating point",
    "pp_theta_post_operating_mcc": "MCC operating point",
    "pp_theta_post_operating_sens90": "sensitivity@90 operating point",
    "pp_theta_post_operating_sens92": "sensitivity@92 operating point",
    "pp_theta_post_operating_sens95": "sensitivity@95 operating point",
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
        writer.writeheader(); writer.writerows(rows)


def num(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def f(value: Any, digits: int = 3) -> str:
    x = num(value)
    return "--" if not math.isfinite(x) else f"{x:.{digits}f}"


def d(value: Any) -> str:
    x = num(value)
    return "--" if not math.isfinite(x) else f"{x:+.3f}"


def tex_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def display_source(value: str) -> str:
    return SOURCE_DISPLAY.get(value, value.replace("_", " ").title())


def display_variant(value: str) -> str:
    return VARIANT_DISPLAY.get(value, value.replace("pp_theta_post_", "").replace("_", " "))


def write_tex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]]) -> None:
    br = " " + "\\" * 2
    lines = ["\\begin{table*}[t]", "\\centering", "\\small", f"\\begin{{tabular}}{{{'l'*len(headers)}}}", "\\toprule", " & ".join(headers) + br, "\\midrule"]
    for row in rows:
        lines.append(" & ".join(row) + br)
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    GENERATED.mkdir(parents=True, exist_ok=True)
    manifest = []
    combined = []
    for stage_key, (stage_dir, filename) in STAGES.items():
        path = RUN_ROOT / stage_dir / filename
        rows = read_csv(path)
        manifest.append({"stage": stage_key, "path": str(path), "rows": len(rows), "done": int(bool(rows))})
        for row in rows:
            row = dict(row)
            row["stage"] = stage_key
            combined.append(row)
    write_csv(GENERATED / "eicu_strengthening_manifest.csv", manifest)
    write_csv(GENERATED / "eicu_strengthening_all_results.csv", combined)

    def pick(stage: str, source: str, variant: str) -> dict[str, str] | None:
        rows = [r for r in combined if r.get("stage") == stage and r.get("rule_source") == source and r.get("variant") == variant]
        if not rows:
            return None
        return max(rows, key=lambda r: (num(r.get("delta_mcc")), num(r.get("delta_sensitivity"))))

    curated_specs = [
        ("source-selected", "rulefit_official", "rulefit", "pp_theta_post_ebm_bounded_residual_gate", "best balanced eICU candidate"),
        ("source-selected", "rulefit_official", "rulefit", "pp_theta_post_ebm_residual_mcc", "sensitivity-oriented candidate"),
        ("measurement-pattern only", "measurement_pattern", "rulefit", "pp_theta_post_ebm_residual_mcc", "stress-test; calibration cost"),
        ("measurement-pattern only", "measurement_pattern", "rulefit", "pp_theta_post_rule_family_calibrated", "compact measurement evidence"),
        ("clinical operating point", "operating_points", "rulefit", "pp_theta_post_operating_sens92", "sensitivity mode"),
    ]
    candidates = []
    tex_rows = []
    for role, stage, source, variant, note in curated_specs:
        row = pick(stage, source, variant)
        if row is None:
            continue
        row = dict(row)
        row["role"] = role
        row["interpretation"] = note
        candidates.append(row)
        tex_rows.append([
            tex_escape(role),
            tex_escape(display_source(row.get("rule_source", row.get("source", "")))),
            tex_escape(display_variant(row.get("variant", ""))),
            d(row.get("delta_mcc")),
            d(row.get("delta_sensitivity")),
            d(row.get("delta_brier_score")),
            tex_escape(note),
        ])
    write_csv(GENERATED / "eicu_strengthening_selected_candidates.csv", candidates)
    write_tex_table(
        GENERATED / "eicu_strengthening_best_candidates_table.tex",
        "eICU-focused PP$\\theta$-Post strengthening candidates. The initially selected eICU row is a source-specific boundary; with a RuleFit substrate, bounded residual evidence gives a stronger eICU operating point.",
        "tab:eicu-strengthening-best",
        ["Role", "Source", "Variant", "$\\Delta$MCC", "$\\Delta$Sens.", "$\\Delta$Brier", "Interpretation"],
        tex_rows,
    )

    cal_rows = [r for r in combined if r.get("stage") == "measurement_policy_calibration"]
    cal_tex = [[tex_escape(display_variant(r.get("variant", ""))), d(r.get("delta_mcc")), d(r.get("delta_sensitivity")), d(r.get("delta_brier_score")), d(r.get("delta_ece_10"))] for r in cal_rows]
    write_tex_table(
        GENERATED / "eicu_measurement_policy_calibration_table.tex",
        "eICU measurement-policy calibration proxy. Calibration offsets are fit on training folds using measurement-density strata and then applied to held-out fold probabilities.",
        "tab:eicu-measurement-policy-calibration",
        ["Variant", "$\\Delta$MCC", "$\\Delta$Sens.", "$\\Delta$Brier", "$\\Delta$ECE"],
        cal_tex,
    )

    print(f"Wrote eICU strengthening tables to {GENERATED}")
    print(f"manifest={len(manifest)} combined={len(combined)} candidates={len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
