#!/usr/bin/env python3
"""Paper Section 32: explanation-baseline protocol.

Creates a reproducible comparison protocol for PPtheta-Post audit traces versus
post-hoc explanation baselines such as SHAP/TreeSHAP, feature importance, rule
lists, and surrogate trees. Optional baseline CSVs can be supplied later; the
script always writes the protocol table needed for the paper supplement.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "paper" / "32_explanation_baselines"

PROTOCOL_ROWS = [
    {
        "method_family": "EBM",
        "prediction_mechanism": "additive glass-box terms",
        "explanation_mechanism": "same additive terms",
        "posterior_semantics": "no branch posterior",
        "same_mechanism": "yes",
        "required_artifact": "term scores and prediction probabilities",
    },
    {
        "method_family": "RuleFit / rule list",
        "prediction_mechanism": "sparse weighted rules",
        "explanation_mechanism": "rule weights",
        "posterior_semantics": "no posterior update",
        "same_mechanism": "partly",
        "required_artifact": "rule weights plus deletion/sufficiency curves",
    },
    {
        "method_family": "TreeSHAP / SHAP",
        "prediction_mechanism": "original fitted model",
        "explanation_mechanism": "post-hoc feature attribution",
        "posterior_semantics": "no",
        "same_mechanism": "no",
        "required_artifact": "attributions plus matched deletion/sufficiency curves",
    },
    {
        "method_family": "feature importance",
        "prediction_mechanism": "original fitted model",
        "explanation_mechanism": "global feature ranking",
        "posterior_semantics": "no",
        "same_mechanism": "no",
        "required_artifact": "feature ranking plus matched deletion/sufficiency curves",
    },
    {
        "method_family": "surrogate tree",
        "prediction_mechanism": "original fitted model",
        "explanation_mechanism": "separately fitted surrogate",
        "posterior_semantics": "no",
        "same_mechanism": "no",
        "required_artifact": "surrogate fidelity plus deletion/sufficiency curves",
    },
    {
        "method_family": "PPtheta-Post",
        "prediction_mechanism": "posterior branch evidence",
        "explanation_mechanism": "same posterior branch evidence",
        "posterior_semantics": "yes",
        "same_mechanism": "yes",
        "required_artifact": "audit faithfulness CSV from Section 31",
    },
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _read_optional_csv(path: str) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return [{"source": str(p), "status": "missing"}]
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-faithfulness-csv", default=str(ROOT / "output" / "paper" / "31_audit_faithfulness" / "audit_faithfulness_summary.csv"))
    p.add_argument("--baseline-csv", action="append", default=[], help="Optional SHAP/TreeSHAP/feature baseline metric CSV. Can be repeated.")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol_csv = out_dir / "explanation_baseline_protocol.csv"
    _write_csv(protocol_csv, PROTOCOL_ROWS)

    baseline_rows = []
    for csv_path in args.baseline_csv:
        rows = _read_optional_csv(csv_path)
        for row in rows:
            row.setdefault("source", csv_path)
        baseline_rows.extend(rows)
    comparison_csv = out_dir / "explanation_baseline_comparison_inputs.csv"
    _write_csv(comparison_csv, baseline_rows)

    shap_available = importlib.util.find_spec("shap") is not None
    audit_path = Path(args.audit_faithfulness_csv)
    report = out_dir / "EXPLANATION_BASELINE_PROTOCOL.md"
    report.write_text(
        "\n".join(
            [
                "# Explanation-Baseline Protocol",
                "",
                f"Protocol CSV: `{protocol_csv}`",
                f"Optional comparison inputs CSV: `{comparison_csv}`",
                f"Audit faithfulness CSV: `{audit_path}` ({'present' if audit_path.exists() else 'missing'})",
                f"SHAP package available in this environment: {str(shap_available).lower()}",
                "",
                "## Required comparison rule",
                "",
                "All explanation baselines should be evaluated with the same deletion and sufficiency protocol used for PPtheta-Post top-K rules. This keeps the paper's auditability claim about faithfulness rather than visual plausibility.",
                "",
                "## Paper claim boundary",
                "",
                "SHAP, feature importance, and surrogate trees are post-hoc unless their explanations are the same computation that produces the prediction. PPtheta-Post is compared against them because its audit trace is prediction-time posterior evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"protocol_csv={protocol_csv}")
    print(f"shap_available={str(shap_available).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
