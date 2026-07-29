#!/usr/bin/env python3
"""Aggregate Section 66 symbolic clean PPtheta results into paper-ready tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output/mortality_paper_jobs/rahmatullaev_symbolic_clean_ppost_mortality_symbolic_clean_v1"
OUT = ROOT / "output/mortality_paper_jobs/local_symbolic_clean_tables_v1"
GENERATED = ROOT / "paper/aaai_pppost_mortality/generated"

STAGE_FILES = {
    "rahmatullaev_symbolic_rulefit_calibrated": "rulefit_calibrated_evidence",
    "rahmatullaev_symbolic_figs_bounded": "figs_bounded_residual",
    "rahmatullaev_symbolic_auditselect": "rulefit_figs_auditselect",
    "rahmatullaev_symbolic_family_ppost": "symbolic_family_ppost",
    "rahmatullaev_symbolic_thresholding": "calibration_constrained_thresholding",
}
STAGE_LABELS = {
    "rulefit_calibrated_evidence": "RuleFit calibrated evidence",
    "figs_bounded_residual": "FIGS bounded residual",
    "rulefit_figs_auditselect": "AuditSelect symbolic",
    "symbolic_family_ppost": "Symbolic family PP$\\theta$",
    "calibration_constrained_thresholding": "Cal.-constrained thresholding",
}
VARIANT_LABELS = {
    "pp_theta_post_ebm_bounded_residual_gate": "bounded residual gate",
    "pp_theta_post_family_utility_pruned_topk": "utility-pruned top-k",
    "pp_theta_post_rule_family_calibrated": "rule-family calibrated",
    "pp_theta_post_rule_family_sensitivity": "rule-family sensitivity",
    "pp_theta_post_operating_calibrated": "calibrated operating point",
    "pp_theta_post_operating_mcc": "MCC operating point",
    "pp_theta_post_operating_sens90": "sensitivity@90 operating point",
}
DATASET_ORDER = {"eICU": 0, "MIMIC-III": 1, "MIMIC-IV": 2}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def f(x: Any, default: float=float('nan')) -> float:
    try:
        y=float(x)
    except Exception:
        return default
    return y if math.isfinite(y) else default


def fmt(x: Any, digits: int = 3, signed: bool = False) -> str:
    y=f(x)
    if not math.isfinite(y):
        return "--"
    return f"{y:+.{digits}f}" if signed else f"{y:.{digits}f}"


def tex_escape(x: Any) -> str:
    s=str(x)
    return (s.replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%')
            .replace('$', r'\$').replace('#', r'\#').replace('_', r'\_')
            .replace('{', r'\{').replace('}', r'\}'))


def collect_selected() -> list[dict[str, Any]]:
    rows=[]
    for dataset_dir in sorted(RUN_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for stage, stem in STAGE_FILES.items():
            selected = read_csv(dataset_dir / stage / f"{stem}_selected.csv")
            for r in selected:
                item=dict(r)
                item['stage'] = stage
                item['stage_label'] = STAGE_LABELS.get(stem, stem)
                item['variant_label'] = VARIANT_LABELS.get(item.get('variant',''), item.get('variant',''))
                rows.append(item)
    return rows


def choose_auditselect(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    for label in ("eICU", "MIMIC-III", "MIMIC-IV"):
        ds=[r for r in rows if r.get('dataset_label') == label and r.get('stage_name') == 'rulefit_figs_auditselect']
        if ds:
            out.append(ds[0])
    return out


def collect_controls(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    for sel in selected_rows:
        dataset_key = sel.get('dataset_key') or {'eICU':'eicu','MIMIC-III':'mimic3','MIMIC-IV':'mimic4'}[sel['dataset_label']]
        stage = sel['stage']
        stem = STAGE_FILES[stage]
        rows = read_csv(RUN_ROOT / dataset_key / stage / f"{stem}_controls_summary.csv")
        by={r.get('control'): r for r in rows}
        observed=by.get('observed', {})
        native=by.get('native_source', {})
        perm=by.get('patient_permuted', {})
        flat=by.get('flattened_T4', {})
        out.append({
            'dataset_label': sel['dataset_label'],
            'rule_source': sel.get('rule_source',''),
            'variant': sel.get('variant',''),
            'variant_label': sel.get('variant_label',''),
            'observed_mcc': observed.get('mcc',''),
            'native_mcc': native.get('mcc',''),
            'patient_permuted_mcc': perm.get('mcc',''),
            'permuted_gap_mcc': perm.get('control_gap_mcc',''),
            'flattened_mcc': flat.get('mcc',''),
            'observed_brier': observed.get('brier_score',''),
            'flattened_brier': flat.get('brier_score',''),
            'delta_brier_flattened_minus_observed': f(flat.get('brier_score')) - f(observed.get('brier_score')),
            'observed_ece': observed.get('ece_10',''),
            'flattened_ece': flat.get('ece_10',''),
            'delta_ece_flattened_minus_observed': f(flat.get('ece_10')) - f(observed.get('ece_10')),
        })
    return out


def write_selected_tex(rows: list[dict[str, Any]]) -> None:
    lines=[
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Source $\rightarrow$ PP$\theta$ mode & Native MCC & PP$\theta$ MCC & $\Delta$MCC & $\Delta$Sens. & $\Delta$Brier & $\Delta$ECE \\",
        r"\midrule",
    ]
    line_end = chr(92) * 2
    for r in rows:
        src_name = {'rulefit': 'RuleFit', 'figs': 'FIGS'}.get(r.get('rule_source',''), r.get('rule_source',''))
        src = f"{src_name} $\\rightarrow$ {r.get('variant_label','')}"
        lines.append(
            f"{tex_escape(r.get('dataset_label',''))} & {src} & {fmt(r.get('native_mcc'))} & {fmt(r.get('ppost_mcc'))} & "
            f"{fmt(r.get('delta_mcc'), signed=True)} & {fmt(r.get('delta_sensitivity'), signed=True)} & "
            f"{fmt(r.get('delta_brier_score'), signed=True)} & {fmt(r.get('delta_ece_10'), signed=True)} {line_end}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / 'ppost_symbolic_clean_auditselect_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def write_controls_tex(rows: list[dict[str, Any]]) -> None:
    lines=[
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Obs. MCC & Native MCC & Permuted MCC & Perm. gap & $\Delta$Brier flat & $\Delta$ECE flat \\",
        r"\midrule",
    ]
    line_end = chr(92) * 2
    for r in rows:
        lines.append(
            f"{tex_escape(r.get('dataset_label',''))} & {fmt(r.get('observed_mcc'))} & {fmt(r.get('native_mcc'))} & "
            f"{fmt(r.get('patient_permuted_mcc'))} & {fmt(r.get('permuted_gap_mcc'), signed=True)} & "
            f"{fmt(r.get('delta_brier_flattened_minus_observed'), signed=True)} & {fmt(r.get('delta_ece_flattened_minus_observed'), signed=True)} {line_end}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GENERATED / 'ppost_symbolic_clean_controls_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    selected_all=collect_selected()
    auditselect=choose_auditselect(selected_all)
    controls=collect_controls(auditselect)
    write_csv(OUT / 'symbolic_clean_all_selected.csv', selected_all)
    write_csv(OUT / 'symbolic_clean_auditselect_selected.csv', auditselect)
    write_csv(OUT / 'symbolic_clean_auditselect_controls.csv', controls)
    write_selected_tex(auditselect)
    write_controls_tex(controls)
    md=["# Symbolic clean PPtheta aggregation", "", "Selected AuditSelect rows:"]
    for r in auditselect:
        md.append(f"- {r['dataset_label']}: {r['rule_source']} -> {r['variant_label']}, delta MCC {fmt(r['delta_mcc'], signed=True)}, delta Brier {fmt(r['delta_brier_score'], signed=True)}, delta ECE {fmt(r['delta_ece_10'], signed=True)}")
    md.append("")
    md.append("Controls: patient permutation collapses MCC, while flattening preserves decisions but worsens probability calibration.")
    (OUT / 'symbolic_clean_tables.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(OUT)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
