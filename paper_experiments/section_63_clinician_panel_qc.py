#!/usr/bin/env python3
"""Section 63: materialized clinician panels, QC, and replay validity.

The script converts the blinded protocol package into a QC-ready clinician/user
validation bundle. PPtheta panels are materialized from saved posterior audit
records where available. Comparator panels are checked explicitly; if EBM or
TreeSHAP panels are still placeholders, the QC report marks them as requiring
materialization rather than treating the packet as study-ready.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"eICU": "eicu", "MIMIC-III": "mimic3", "MIMIC-IV": "mimic4"}
DATASET_LABELS = {v: k for k, v in DATASETS.items()}
PROTOCOL_ROOT = ROOT / "output/mortality_paper_jobs/local_clinician_validation_protocol_v1"
SOURCE_ROOT = (
    ROOT
    / "output/mortality_paper_jobs"
    / "rahmatullaev_aaai_acceptance_clinician_symbolic_mortality_accept_clinician_symbolic_v1"
)
OUT_DIR = ROOT / "output/mortality_paper_jobs/local_clinician_panel_qc_v1"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"
REPLAY_CSV = GENERATED / "ppost_final_replay_integrity.csv"
CLINICAL_TRACE_CSV = GENERATED / "ppost_final_clinical_trace.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        fieldnames = fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(x: Any) -> str:
    s = str(x)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def normalize(text: Any) -> str:
    return (
        str(text or "")
        .replace("replayable posterior trace", "replayable posterior audit record")
        .replace("posterior trace", "posterior audit record")
        .replace("trace-sufficiency", "compact-sufficiency")
        .replace("source trace artifact", "source posterior-audit artifact")
    )


def pretty_feature(raw: str) -> str:
    mapping = {
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
    return mapping.get(raw, raw.replace("_", " "))


def pretty_rule(rule: str, max_terms: int = 5) -> str:
    parts: list[str] = []
    for raw in re.split(r"\s+AND\s+|;\s*", normalize(rule)):
        raw = raw.strip()
        if not raw or raw == "...":
            continue
        raw = raw.replace("≤", "<=").replace("≥", ">=")
        value = re.search(r"value\(([^)]+)\)\s*(<=|>=|>|<)\s*([-+0-9.eE]+)", raw)
        mask = re.search(r"mask\(([^)]+)\)", raw)
        if value:
            direction = "low" if value.group(2) in {"<=", "<"} else "high"
            parts.append(f"{direction} {pretty_feature(value.group(1))} ({value.group(2)} {value.group(3)})")
        elif mask:
            parts.append(f"measurement pattern for {pretty_feature(mask.group(1))}")
        else:
            parts.append(raw[:100])
    if not parts:
        return ""
    out = "; ".join(parts[:max_terms])
    if len(parts) > max_terms:
        out += "; ..."
    return out


def load_source_packets() -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    packets: dict[tuple[str, str], dict[str, str]] = {}
    keys: dict[tuple[str, str], dict[str, str]] = {}
    for dataset in ("eicu", "mimic3", "mimic4"):
        d = SOURCE_ROOT / dataset / "rahmatullaev_accept_clinician_audit_packet"
        for row in read_csv(d / "clinician_audit_packet.csv"):
            packets[(row.get("dataset_label", ""), row.get("deidentified_case_id", ""))] = row
        for row in read_csv(d / "clinician_audit_packet_key_private.csv"):
            keys[(row.get("dataset_label", ""), row.get("deidentified_case_id", ""))] = row
    return packets, keys


def load_clinical_rows() -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in read_csv(CLINICAL_TRACE_CSV):
        out.setdefault((row.get("dataset", ""), row.get("case_id", "")), []).append(row)
    return out


def load_replay() -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    replay = {r.get("dataset", ""): r for r in read_csv(REPLAY_CSV)}
    samples: dict[tuple[str, str], dict[str, Any]] = {}
    for label, row in replay.items():
        path = ROOT / row.get("trace_file", "")
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for sample in data.get("samples", []):
            samples[(label, str(sample.get("x_id", "")))] = sample
    return replay, samples


def branch_lines(branches: list[dict[str, Any]], prefix: str, limit: int) -> list[str]:
    lines: list[str] = []
    for i, branch in enumerate(branches[:limit], start=1):
        rule = pretty_rule(str(branch.get("rule", "")))
        score = branch.get("support_score", "")
        activation = branch.get("p_z_aggregated", "")
        score_txt = f"{float(score):.3f}" if isinstance(score, (int, float)) and math.isfinite(float(score)) else str(score)[:8]
        act_txt = f"{float(activation):.3f}" if isinstance(activation, (int, float)) and math.isfinite(float(activation)) else str(activation)[:8]
        lines.append(f"{prefix} {i}: {rule} [support {score_txt}, activation {act_txt}]")
    return lines


def ppost_panel(label: str, case_id: str, source_row: dict[str, str], clinical: dict[tuple[str, str], list[dict[str, str]]], samples: dict[tuple[str, str], dict[str, Any]]) -> tuple[str, str, str]:
    sample = samples.get((label, case_id))
    if sample:
        proba = sample.get("class_proba", ["", ""])
        mortality = proba[1] if isinstance(proba, list) and len(proba) > 1 else ""
        lines = [
            f"PPtheta posterior audit record. Mortality risk {float(mortality):.3f}; predicted class {sample.get('predicted_class')}; threshold {sample.get('decision_threshold')}.",
            *branch_lines(sample.get("top_branches", []), "Supporting branch", 3),
            *branch_lines(sample.get("opposing_branches", []), "Opposing branch", 2),
        ]
        return " ".join(x for x in lines if x), "materialized_from_replay_record", "yes"
    rows = clinical.get((label, case_id), [])
    if rows:
        case = next((r for r in rows if "case" in r.get("example", "")), rows[0])
        evid = [pretty_rule(r.get("evidence", "")) or normalize(r.get("evidence", "")) for r in rows]
        evid = [e for e in evid if e]
        risk = case.get("ppost_mortality_probability") or source_row.get("ppost_mortality_risk", "")
        native = case.get("native_mortality_probability") or source_row.get("native_source_mortality_risk", "")
        text = f"PPtheta posterior audit record. Native risk {native}; PPtheta risk {risk}. " + " ".join(evid[:4])
        return text.strip(), "materialized_from_clinical_record", "no"
    text = normalize(source_row.get("format_a_ppost_posterior_trace", ""))
    if re.search(r"pp_theta|tabpfn|xgb|rulefit|figs", text, flags=re.I):
        text = (
            f"PPtheta posterior audit record. Native risk {source_row.get('native_source_mortality_risk')}; "
            f"PPtheta risk {source_row.get('ppost_mortality_risk')}. Detailed branch-level evidence "
            "was not materialized for this packet case in the current saved artifacts."
        )
        return text, "partial_needs_branch_record", "no"
    return text, "missing", "no"


def comparator_panel(text: str, kind: str) -> tuple[str, str]:
    text = normalize(text)
    if not text or "blinded comparator slot" in text.lower() or "generated from" in text.lower():
        return f"MATERIALIZATION REQUIRED: {kind} panel is not present in the current saved artifacts.", "needs_materialization"
    return text, "materialized"


def has_internal_names(text: str) -> bool:
    return bool(re.search(r"pp_theta|tabpfn_distill|xgb|mimic[34]|eicu|rahmatullaev|\\.json|\\.csv", text, flags=re.I))


def panel_qc(text: str, status: str) -> str:
    issues: list[str] = []
    if status in {"missing", "needs_materialization"} or status.startswith("partial"):
        issues.append(status)
    if len(text.strip()) < 80:
        issues.append("too_short")
    if has_internal_names(text):
        issues.append("internal_names")
    return "pass" if not issues else ";".join(sorted(set(issues)))


def run(output_dir: Path, seed: int) -> None:
    rng = random.Random(seed)
    source_packets, source_keys = load_source_packets()
    clinical = load_clinical_rows()
    replay, samples = load_replay()
    public_rows = read_csv(PROTOCOL_ROOT / "clinician_review_cases_blinded.csv")
    format_key_rows = read_csv(PROTOCOL_ROOT / "clinician_review_key_private.csv")
    key_by_case = {(r["dataset_label"], r["deidentified_case_id"], r["display_format_id"]): r["true_format"] for r in format_key_rows}

    panels: list[dict[str, Any]] = []
    blinded: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []

    for row in public_rows:
        label = row["dataset_label"]
        deid = row["deidentified_case_id"]
        source = source_packets.get((label, deid), {})
        key = source_keys.get((label, deid), {})
        case_id = key.get("source_case_id", "")
        pp_text, pp_status, case_match = ppost_panel(label, case_id, source, clinical, samples)
        ebm_text, ebm_status = comparator_panel(source.get("format_b_ebm_additive_terms", ""), "EBM additive-term")
        shap_text, shap_status = comparator_panel(source.get("format_c_treeshap_feature_attribution", ""), "TreeSHAP/feature-attribution")
        format_panels = [
            ("PPtheta posterior audit record", pp_text, pp_status, panel_qc(pp_text, pp_status)),
            ("EBM additive terms", ebm_text, ebm_status, panel_qc(ebm_text, ebm_status)),
            ("TreeSHAP feature attribution", shap_text, shap_status, panel_qc(shap_text, shap_status)),
        ]
        for name, text, status, qc in format_panels:
            panels.append({
                "dataset_label": label,
                "deidentified_case_id": deid,
                "source_case_id_private": case_id,
                "case_bucket": normalize(row.get("case_bucket", "")),
                "format_name": name,
                "panel_status": status,
                "qc_status": qc,
                "panel_text": text,
            })
        shuffled = format_panels[:]
        rng.shuffle(shuffled)
        blinded.append({
            "dataset_label": label,
            "deidentified_case_id": deid,
            "case_bucket": normalize(row.get("case_bucket", "")),
            "native_source_mortality_risk": row.get("native_source_mortality_risk", ""),
            "ppost_mortality_risk": row.get("ppost_mortality_risk", ""),
            "compact_record_fraction": normalize(row.get("compact_record_fraction", "")),
            "format_1_text": shuffled[0][1],
            "format_2_text": shuffled[1][1],
            "format_3_text": shuffled[2][1],
            "review_instruction": "Rate F1, F2, and F3 independently before accessing the private key.",
        })
        for i, (name, _, status, qc) in enumerate(shuffled, start=1):
            private_rows.append({
                "dataset_label": label,
                "deidentified_case_id": deid,
                "display_format_id": f"F{i}",
                "true_format": name,
                "panel_status": status,
                "qc_status": qc,
            })
        replay_row = replay.get(label, {})
        replay_rows.append({
            "dataset_label": label,
            "deidentified_case_id": deid,
            "source_case_id_private": case_id,
            "case_level_replay_record_found": case_match,
            "dataset_replay_status": replay_row.get("status", "missing"),
            "dataset_replay_samples": replay_row.get("samples", ""),
            "dataset_decision_matches_argmax": replay_row.get("decision_matches_argmax", ""),
            "dataset_complete_top_branch_records": replay_row.get("complete_top_branch_records", ""),
            "dataset_max_probability_sum_error": replay_row.get("max_probability_sum_error", ""),
        })

    summary: list[dict[str, Any]] = []
    for label in DATASETS:
        part = [p for p in panels if p["dataset_label"] == label]
        cases = {p["deidentified_case_id"] for p in part}
        for fmt in ("PPtheta posterior audit record", "EBM additive terms", "TreeSHAP feature attribution"):
            fp = [p for p in part if p["format_name"] == fmt]
            summary.append({
                "dataset_label": label,
                "format_name": fmt,
                "cases": len(cases),
                "materialized_panels": sum(p["panel_status"].startswith("materialized") for p in fp),
                "needs_materialization": sum("needs_materialization" in p["qc_status"] for p in fp),
                "partial_panels": sum("partial" in p["qc_status"] for p in fp),
                "internal_name_flags": sum("internal_names" in p["qc_status"] for p in fp),
                "qc_pass_panels": sum(p["qc_status"] == "pass" for p in fp),
            })
    replay_summary = []
    for label in DATASETS:
        part = [r for r in replay_rows if r["dataset_label"] == label]
        replay_summary.append({
            "dataset_label": label,
            "packet_cases": len(part),
            "case_level_replay_records_found": sum(r["case_level_replay_record_found"] == "yes" for r in part),
            "dataset_replay_status": replay.get(label, {}).get("status", "missing"),
            "dataset_replay_samples": replay.get(label, {}).get("samples", ""),
        })

    write_csv(output_dir / "materialized_clinician_panels.csv", panels)
    write_csv(output_dir / "clinician_review_cases_blinded_materialized.csv", blinded)
    write_csv(output_dir / "clinician_review_key_private_materialized.csv", private_rows)
    write_csv(output_dir / "clinician_panel_qc_summary.csv", summary)
    write_csv(output_dir / "clinician_packet_replay_validity.csv", replay_rows)
    write_csv(output_dir / "clinician_packet_replay_summary.csv", replay_summary)

    rows = [
        [r["dataset_label"], r["format_name"], r["materialized_panels"], r["needs_materialization"], r["partial_panels"], r["qc_pass_panels"]]
        for r in summary
    ]
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Format & Materialized & Needs materialization & Partial & QC pass \\",
        r"\midrule",
    ]
    for r in rows:
        tex.append(" & ".join(tex_escape(x) for x in r) + r" \\")
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Clinician/user validation packet QC. PP$\theta$ panels are materialized from posterior audit records where available; comparator panels are marked when EBM or TreeSHAP explanations still need materialization before a completed user study can be claimed.}",
        r"\label{tab:ppost-clinician-panel-qc}",
        r"\end{table*}",
        "",
    ]
    (GENERATED / "ppost_clinician_panel_qc_table.tex").write_text("\n".join(tex), encoding="utf-8")

    replay_tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Dataset & Packet cases & Case records & Dataset replay \\",
        r"\midrule",
    ]
    for r in replay_summary:
        replay_tex.append(
            f"{tex_escape(r['dataset_label'])} & {r['packet_cases']} & {r['case_level_replay_records_found']} & {tex_escape(r['dataset_replay_status'])} " + r"\\"
        )
    replay_tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Replay-validity status for clinician/user packet cases. Case records count packet cases found in saved posterior audit-record JSON; dataset replay summarizes the full replay-integrity check used by the paper.}",
        r"\label{tab:ppost-clinician-replay-validity}",
        r"\end{table}",
        "",
    ]
    (GENERATED / "ppost_clinician_replay_validity_table.tex").write_text("\n".join(replay_tex), encoding="utf-8")

    md = [
        "# Clinician Panel QC",
        "",
        f"Cases: {len(blinded)}",
        f"Panel rows: {len(panels)}",
        "",
        "This is a materialization and QC artifact, not completed clinician validation.",
        "PPtheta panels are materialized from saved posterior audit records when possible.",
        "EBM and TreeSHAP comparator panels are explicitly flagged when absent.",
        "",
        "## Replay Summary",
    ]
    md.extend(str(r) for r in replay_summary)
    (output_dir / "clinician_panel_qc.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    run(args.output_dir, args.seed)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
