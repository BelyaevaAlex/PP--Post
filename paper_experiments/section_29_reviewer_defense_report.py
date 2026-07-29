#!/usr/bin/env python3
"""Paper Section 29: reviewer-defense report bundle.

Collects outputs from Sections 25-33 and writes a compact markdown checklist
that maps reviewer concerns to artifacts. This script does not invent results;
it reports which evidence files are present and which remain missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "paper" / "29_reviewer_defense_report"

CHECKS = [
    ("Novelty", "main.tex contribution framing and Result Strata table"),
    ("Technical soundness", "Supplementary theory + hyperparameter/implementation notes + posterior parity checks"),
    ("Empirical strength", "clinical metrics CSV with AUPRC/Brier/ECE/net benefit"),
    ("Uncertainty", "paired or patient bootstrap CI CSV"),
    ("Auditability", "audit validation and audit faithfulness summaries with deletion/sufficiency/stability checks"),
    ("Explanation faithfulness", "protocol comparing PPtheta-Post audits with SHAP/TreeSHAP/feature/surrogate baselines"),
    ("Generalization", "OpenML/general-tabular command plan and outputs"),
    ("Clarity", "fully interpretable vs performance-oriented neuro-symbolic split"),
]


def _exists(path: Path) -> str:
    return "yes" if path.exists() and path.stat().st_size > 0 else "missing"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-summary", default=str(ROOT / "output" / "paper" / "25_audit_validation" / "AUDIT_VALIDATION_SUMMARY.md"))
    p.add_argument("--clinical-csv", default="")
    p.add_argument("--uncertainty-csv", default=str(ROOT / "output" / "paper" / "27_uncertainty_noninferiority" / "method_metric_ci.csv"))
    p.add_argument("--patient-bootstrap-csv", default=str(ROOT / "output" / "paper" / "28_prediction_artifact_metrics" / "patient_bootstrap_metrics.csv"))
    p.add_argument("--parity-summary", default=str(ROOT / "output" / "paper" / "30_posterior_parity_complexity" / "POSTERIOR_PARITY_COMPLEXITY.md"))
    p.add_argument("--audit-faithfulness-summary", default=str(ROOT / "output" / "paper" / "31_audit_faithfulness" / "AUDIT_FAITHFULNESS_SUMMARY.md"))
    p.add_argument("--explanation-protocol", default=str(ROOT / "output" / "paper" / "32_explanation_baselines" / "EXPLANATION_BASELINE_PROTOCOL.md"))
    p.add_argument("--openml-plan", default=str(ROOT / "output" / "paper" / "33_openml_generalization" / "OPENML_GENERALIZATION_PLAN.md"))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit_summary": Path(args.audit_summary),
        "clinical_csv": Path(args.clinical_csv) if args.clinical_csv else None,
        "uncertainty_csv": Path(args.uncertainty_csv),
        "patient_bootstrap_csv": Path(args.patient_bootstrap_csv),
        "parity_summary": Path(args.parity_summary),
        "audit_faithfulness_summary": Path(args.audit_faithfulness_summary),
        "explanation_protocol": Path(args.explanation_protocol),
        "openml_plan": Path(args.openml_plan),
        "main_tex": ROOT / "paper" / "aaai_pppost_mortality" / "main.tex",
    }
    lines = [
        "# Reviewer-Defense Evidence Bundle",
        "",
        "## Concern Checklist",
        "",
        "| Reviewer concern | Evidence target | Status |",
        "| --- | --- | --- |",
    ]
    status_map = {
        "Novelty": _exists(paths["main_tex"]),
        "Technical soundness": "yes" if _exists(paths["main_tex"]) == "yes" and _exists(paths["parity_summary"]) == "yes" else "partial",
        "Empirical strength": _exists(paths["clinical_csv"]) if paths["clinical_csv"] else "provide --clinical-csv",
        "Uncertainty": _exists(paths["uncertainty_csv"]),
        "Auditability": "yes" if _exists(paths["audit_summary"]) == "yes" and _exists(paths["audit_faithfulness_summary"]) == "yes" else "partial",
        "Explanation faithfulness": _exists(paths["explanation_protocol"]),
        "Generalization": _exists(paths["openml_plan"]),
        "Clarity": _exists(paths["main_tex"]),
    }
    for concern, target in CHECKS:
        lines.append(f"| {concern} | {target} | {status_map[concern]} |")
    lines.extend([
        "",
        "## Artifact Paths",
        "",
    ])
    for name, path in paths.items():
        if path is None:
            continue
        lines.append(f"- {name}: `{path}` ({_exists(path)})")
    lines.extend([
        "",
        "## Recommended Paper Updates After Running",
        "",
        "1. Move audit summary table into the Audit Evidence subsection.",
        "2. Move AUPRC/Brier/ECE/net-benefit results into the main or supplement metrics table.",
        "3. Use paired/patient bootstrap CIs for the non-inferiority paragraph.",
        "4. Keep Teacher-anchored PPtheta-Post as a secondary performance-oriented operating point unless calibration improves.",
        "5. Move posterior parity, audit-faithfulness, and explanation-baseline protocol tables into the supplement.",
        "6. Use OpenML/general-tabular results only as external robustness evidence, not as the primary clinical claim.",
    ])
    report = out_dir / "REVIEWER_DEFENSE_BUNDLE.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
