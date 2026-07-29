#!/usr/bin/env python3
"""Section 62: blinded clinician/user validation protocol package.

This script prepares a real-review packet from existing clinician-facing PPtheta
artifacts. It does not simulate clinician scores and does not create a completed
user study claim. Instead it creates the files needed for a blinded review:
randomized explanation order, a private key, blank Likert forms, preference
forms, and a short protocol.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASET_ORDER = ("eicu", "mimic3", "mimic4")
DATASET_LABELS = {"eicu": "eICU", "mimic3": "MIMIC-III", "mimic4": "MIMIC-IV"}
DEFAULT_INPUT_ROOT = (
    ROOT
    / "output/mortality_paper_jobs"
    / "rahmatullaev_aaai_acceptance_clinician_symbolic_mortality_accept_clinician_symbolic_v1"
)
DEFAULT_OUTPUT_DIR = ROOT / "output/mortality_paper_jobs/local_clinician_validation_protocol_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def dataset_packet_path(input_root: Path, dataset: str) -> Path:
    return input_root / dataset / "rahmatullaev_accept_clinician_audit_packet" / "clinician_audit_packet.csv"


def normalize_label(text: str) -> str:
    return (
        str(text)
        .replace("replayable posterior trace", "replayable posterior audit record")
        .replace("posterior trace", "posterior audit record")
        .replace("PPtheta posterior trace", "PPtheta posterior audit record")
        .replace("trace-sufficiency", "compact-sufficiency")
        .replace("source trace artifact", "source posterior-audit artifact")
    )


def build_protocol(input_root: Path, output_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    blinded_cases: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    likert_rows: list[dict[str, object]] = []
    preference_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []

    for dataset in DATASET_ORDER:
        rows = read_csv(dataset_packet_path(input_root, dataset))
        label = DATASET_LABELS[dataset]
        buckets: dict[str, int] = {}
        for row in rows:
            bucket = normalize_label(row.get("case_bucket", "unspecified"))
            buckets[bucket] = buckets.get(bucket, 0) + 1
            deid = row.get("deidentified_case_id", "")
            formats = [
                ("PPtheta posterior audit record", normalize_label(row.get("format_a_ppost_posterior_trace", ""))),
                ("EBM additive terms", row.get("format_b_ebm_additive_terms", "")),
                ("TreeSHAP feature attribution", row.get("format_c_treeshap_feature_attribution", "")),
            ]
            rng.shuffle(formats)
            display = {f"format_{idx}_text": text for idx, (_, text) in enumerate(formats, start=1)}
            blinded_cases.append({
                "dataset_label": label,
                "deidentified_case_id": deid,
                "case_bucket": bucket,
                "native_source_mortality_risk": row.get("native_source_mortality_risk", ""),
                "ppost_mortality_risk": row.get("ppost_mortality_risk", ""),
                "compact_record_fraction": normalize_label(row.get("compact_trace_fraction", "")),
                **display,
                "review_instruction": "Rate each format independently before asking for the private key.",
            })
            for idx, (format_name, _) in enumerate(formats, start=1):
                fmt_id = f"F{idx}"
                key_rows.append({
                    "dataset_label": label,
                    "deidentified_case_id": deid,
                    "display_format_id": fmt_id,
                    "true_format": format_name,
                })
                likert_rows.append({
                    "dataset_label": label,
                    "deidentified_case_id": deid,
                    "display_format_id": fmt_id,
                    "clinical_plausibility_1_to_5": "",
                    "usefulness_for_audit_1_to_5": "",
                    "evidence_sufficiency_1_to_5": "",
                    "clinically_suspicious_1_to_5": "",
                    "free_text_comment": "",
                    "time_seconds": "",
                })
            preference_rows.append({
                "dataset_label": label,
                "deidentified_case_id": deid,
                "preferred_display_format_F1_F2_F3": "",
                "preference_reason": "",
                "confidence_1_to_5": "",
            })
        balance_rows.append({
            "dataset": dataset,
            "dataset_label": label,
            "cases": len(rows),
            "format_count_per_case": 3,
            "case_buckets": "; ".join(f"{k}: {v}" for k, v in sorted(buckets.items())),
            "claim_status": "ready for blinded clinician/user scoring; no completed clinician-study claim",
        })

    write_csv(output_dir / "clinician_review_cases_blinded.csv", blinded_cases)
    write_csv(output_dir / "clinician_review_key_private.csv", key_rows)
    write_csv(output_dir / "clinician_likert_scores_blank.csv", likert_rows)
    write_csv(output_dir / "clinician_preference_scores_blank.csv", preference_rows)
    write_csv(output_dir / "clinician_validation_balance_summary.csv", balance_rows)
    write_csv(output_dir / "clinician_validation_analysis_template.csv", [{
        "metric": "mean_plausibility_by_format",
        "input": "filled clinician_likert_scores_blank.csv",
        "test": "paired comparison within deidentified_case_id",
    }, {
        "metric": "preference_rate_by_format",
        "input": "filled clinician_preference_scores_blank.csv + private key",
        "test": "binomial or bootstrap interval",
    }, {
        "metric": "inter_rater_agreement",
        "input": "multiple filled clinician_likert_scores files",
        "test": "Krippendorff alpha or ICC",
    }])
    (output_dir / "clinician_validation_protocol.md").write_text(
        "\n".join([
            "# Blinded Clinician/User Validation Protocol",
            "",
            "Status: protocol and packet only. Do not claim completed clinician validation until scored forms are returned.",
            "",
            "Primary comparison: PPtheta posterior audit record vs. EBM additive terms vs. TreeSHAP/feature attribution.",
            "",
            "Blinding: explanation formats are randomized into F1/F2/F3 per case; the private key is stored separately.",
            "",
            "Scores: clinical plausibility, usefulness for audit, evidence sufficiency, suspiciousness, free text, time, and forced preference.",
            "",
            "Recommended analysis: paired within-case comparison, preference rates with bootstrap intervals, and inter-rater agreement when multiple reviewers are available.",
            "",
            f"Seed: {seed}",
        ])
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    build_protocol(args.input_root, args.output_dir, args.seed)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
