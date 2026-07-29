#!/usr/bin/env python3
"""Paper Section 34: clinical task feasibility and relabeling.

Audits which additional ICU tasks can be derived from the same first-48h
MIMIC-III, MIMIC-IV, and eICU cohorts used by the mortality benchmark. The
script is conservative by default: it writes only CSV/Markdown feasibility
reports. Pass --write to emit relabeled NPZ caches that reuse the existing
mortality feature tensors after verifying row alignment against hospital
mortality labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT.parent
DEFAULT_INPUT_DIR = ROOT / "data" / "processed" / "mortality"
DEFAULT_REPORT_DIR = ROOT / "output" / "paper" / "34_clinical_task_feasibility"
DEFAULT_CACHE_OUTPUT_DIR = ROOT / "data" / "processed" / "clinical_tasks"
DATASETS = ("mimic3", "mimic4", "eicu")
TASKS = (
    "hospital_mortality",
    "icu_mortality",
    "icu_los_gt_3d_at48",
    "icu_los_gt_7d_at48",
    "hospital_los_gt_7d_at48",
)


def _as_binary_mortality(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").round().astype("Int64")
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    out[s.isin(["1", "true", "t", "yes", "y", "expired", "dead", "deceased"])] = 1
    out[s.isin(["0", "false", "f", "no", "n", "alive", "survived"])] = 0
    return out


def _stratified_limit(df: pd.DataFrame, y_col: str, max_samples: int | None, seed: int) -> pd.DataFrame:
    if not max_samples or max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    classes = sorted(df[y_col].dropna().unique().tolist())
    per_class = max(1, max_samples // max(1, len(classes)))
    for cls in classes:
        cls_df = df[df[y_col] == cls]
        take = min(len(cls_df), per_class)
        if take == len(cls_df):
            parts.append(cls_df)
        else:
            idx = rng.choice(cls_df.index.to_numpy(), size=take, replace=False)
            parts.append(cls_df.loc[idx])
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed)
    if len(out) > max_samples:
        out = out.iloc[:max_samples]
    return out.reset_index(drop=True)


def _mimic3_stays(raw_root: Path, max_samples: int | None, seed: int) -> pd.DataFrame:
    root = raw_root / "MIMIC-III"
    adm = pd.read_csv(
        root / "ADMISSIONS.csv.gz",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "HOSPITAL_EXPIRE_FLAG"],
    )
    icu = pd.read_csv(
        root / "ICUSTAYS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"],
    )
    icu["INTIME"] = pd.to_datetime(icu["INTIME"], errors="coerce")
    stays = icu.merge(adm, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    stays = stays.dropna(subset=["INTIME", "ICUSTAY_ID", "HADM_ID", "SUBJECT_ID", "HOSPITAL_EXPIRE_FLAG"])
    stays = stays.sort_values(["SUBJECT_ID", "INTIME", "ICUSTAY_ID"]).groupby("SUBJECT_ID", as_index=False).first()
    stays["hospital_mortality"] = _as_binary_mortality(stays["HOSPITAL_EXPIRE_FLAG"])
    stays = stays.dropna(subset=["hospital_mortality"]).copy()
    stays["hospital_mortality"] = stays["hospital_mortality"].astype(int)
    stays = _stratified_limit(stays, "hospital_mortality", max_samples, seed)
    stays["icu_los_days"] = pd.to_numeric(stays["LOS"], errors="coerce")
    stays["ADMITTIME"] = pd.to_datetime(stays["ADMITTIME"], errors="coerce")
    stays["DISCHTIME"] = pd.to_datetime(stays["DISCHTIME"], errors="coerce")
    stays["OUTTIME"] = pd.to_datetime(stays["OUTTIME"], errors="coerce")
    stays["DEATHTIME"] = pd.to_datetime(stays["DEATHTIME"], errors="coerce")
    stays["hospital_los_days"] = (stays["DISCHTIME"] - stays["ADMITTIME"]).dt.total_seconds() / 86400.0
    stays["icu_mortality"] = ((stays["DEATHTIME"].notna()) & (stays["DEATHTIME"] <= stays["OUTTIME"])).astype(int)
    stays["stay_id"] = stays["ICUSTAY_ID"].astype(str)
    return stays.reset_index(drop=True)


def _mimic4_stays(raw_root: Path, max_samples: int | None, seed: int) -> pd.DataFrame:
    root = raw_root / "mimic-4" / "physionet.org" / "files" / "mimiciv" / "3.1"
    adm = pd.read_csv(
        root / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"],
    )
    icu = pd.read_csv(
        root / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
    )
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    stays = icu.merge(adm, on=["subject_id", "hadm_id"], how="inner")
    stays = stays.dropna(subset=["intime", "stay_id", "hadm_id", "subject_id", "hospital_expire_flag"])
    stays = stays.sort_values(["subject_id", "intime", "stay_id"]).groupby("subject_id", as_index=False).first()
    stays["hospital_mortality"] = _as_binary_mortality(stays["hospital_expire_flag"])
    stays = stays.dropna(subset=["hospital_mortality"]).copy()
    stays["hospital_mortality"] = stays["hospital_mortality"].astype(int)
    stays = _stratified_limit(stays, "hospital_mortality", max_samples, seed)
    stays["icu_los_days"] = pd.to_numeric(stays["los"], errors="coerce")
    stays["admittime"] = pd.to_datetime(stays["admittime"], errors="coerce")
    stays["dischtime"] = pd.to_datetime(stays["dischtime"], errors="coerce")
    stays["outtime"] = pd.to_datetime(stays["outtime"], errors="coerce")
    stays["deathtime"] = pd.to_datetime(stays["deathtime"], errors="coerce")
    stays["hospital_los_days"] = (stays["dischtime"] - stays["admittime"]).dt.total_seconds() / 86400.0
    stays["icu_mortality"] = ((stays["deathtime"].notna()) & (stays["deathtime"] <= stays["outtime"])).astype(int)
    stays["stay_id"] = stays["stay_id"].astype(str)
    return stays.reset_index(drop=True)


def _eicu_stays(raw_root: Path, max_samples: int | None, seed: int) -> pd.DataFrame:
    root = raw_root / "eICU" / "physionet.org" / "files" / "eicu-crd" / "2.0"
    patient = pd.read_csv(
        root / "patient.csv.gz",
        usecols=[
            "patientunitstayid",
            "uniquepid",
            "unitvisitnumber",
            "unitdischargeoffset",
            "hospitaldischargeoffset",
        ],
        low_memory=False,
    )
    apache = pd.read_csv(
        root / "apachePatientResult.csv.gz",
        usecols=["patientunitstayid", "actualhospitalmortality", "actualicumortality", "actualiculos", "actualhospitallos"],
        low_memory=False,
    )
    stays = patient.merge(apache, on="patientunitstayid", how="inner")
    stays["hospital_mortality"] = _as_binary_mortality(stays["actualhospitalmortality"])
    stays = stays.dropna(subset=["patientunitstayid", "uniquepid", "hospital_mortality"]).copy()
    stays["hospital_mortality"] = stays["hospital_mortality"].astype(int)
    stays["unitvisitnumber"] = pd.to_numeric(stays["unitvisitnumber"], errors="coerce").fillna(0)
    stays = stays.sort_values(["uniquepid", "unitvisitnumber", "patientunitstayid"]).groupby("uniquepid", as_index=False).first()
    stays = _stratified_limit(stays, "hospital_mortality", max_samples, seed)
    stays["icu_los_days"] = pd.to_numeric(stays["actualiculos"], errors="coerce")
    missing = stays["icu_los_days"].isna()
    stays.loc[missing, "icu_los_days"] = pd.to_numeric(stays.loc[missing, "unitdischargeoffset"], errors="coerce") / 1440.0
    stays["hospital_los_days"] = pd.to_numeric(stays["actualhospitallos"], errors="coerce")
    missing = stays["hospital_los_days"].isna()
    stays.loc[missing, "hospital_los_days"] = pd.to_numeric(stays.loc[missing, "hospitaldischargeoffset"], errors="coerce") / 1440.0
    stays["icu_mortality"] = _as_binary_mortality(stays["actualicumortality"]).astype("float")
    stays["stay_id"] = stays["patientunitstayid"].astype(str)
    return stays.reset_index(drop=True)


STAY_LOADERS: Dict[str, Callable[[Path, int | None, int], pd.DataFrame]] = {
    "mimic3": _mimic3_stays,
    "mimic4": _mimic4_stays,
    "eicu": _eicu_stays,
}


def _task_labels(stays: pd.DataFrame, task: str) -> pd.Series:
    if task == "hospital_mortality":
        return stays["hospital_mortality"]
    if task == "icu_mortality":
        return stays["icu_mortality"]
    at48 = stays["icu_los_days"] >= 2.0
    if task == "icu_los_gt_3d_at48":
        return (stays["icu_los_days"] > 3.0).astype("float").where(at48)
    if task == "icu_los_gt_7d_at48":
        return (stays["icu_los_days"] > 7.0).astype("float").where(at48)
    if task == "hospital_los_gt_7d_at48":
        return (stays["hospital_los_days"] > 7.0).astype("float").where(at48 & stays["hospital_los_days"].notna())
    raise ValueError(f"Unknown task: {task}")


def _load_cache(input_dir: Path, dataset: str, kind: str):
    return np.load(input_dir / f"{dataset}_mortality_48h_{kind}.npz", allow_pickle=True)


def _alignment_status(input_dir: Path, dataset: str, stays: pd.DataFrame) -> tuple[bool, str]:
    tab = _load_cache(input_dir, dataset, "tabular")
    y_cache = np.asarray(tab["y"]).astype(int)
    y_raw = np.asarray(stays["hospital_mortality"]).astype(int)
    if len(y_cache) != len(y_raw):
        return False, f"length mismatch cache={len(y_cache)} raw={len(y_raw)}"
    if not np.array_equal(y_cache, y_raw):
        diff = int(np.sum(y_cache != y_raw))
        return False, f"hospital_mortality mismatch rows={diff}"
    return True, "ok"


def _summarize_dataset(raw_root: Path, input_dir: Path, dataset: str, max_samples: int | None, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    stays = STAY_LOADERS[dataset](raw_root, max_samples, seed)
    aligned, alignment_message = _alignment_status(input_dir, dataset, stays)
    rows = []
    for task in TASKS:
        labels = _task_labels(stays, task)
        valid = labels.dropna().astype(int)
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "n_cache_rows": len(stays),
                "n_valid": len(valid),
                "n_positive": int(valid.sum()) if len(valid) else 0,
                "positive_rate": float(valid.mean()) if len(valid) else float("nan"),
                "n_dropped": int(len(stays) - len(valid)),
                "at_risk_48h": int((stays["icu_los_days"] >= 2.0).sum()),
                "alignment_ok": int(aligned),
                "alignment_message": alignment_message,
            }
        )
    return stays, rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _copy_relabeled_cache(input_dir: Path, output_dir: Path, dataset: str, task: str, labels: pd.Series, modalities: set[str]) -> list[Path]:
    valid_mask = labels.notna().to_numpy()
    y = labels.dropna().astype(int).to_numpy(dtype=np.int64)
    out_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    if "tabular" in modalities:
        src = _load_cache(input_dir, dataset, "tabular")
        out = output_dir / f"{dataset}_{task}_48h_tabular.npz"
        np.savez_compressed(
            out,
            X=np.asarray(src["X"])[valid_mask],
            y=y,
            feature_names=np.asarray(src["feature_names"]),
            class_names=np.asarray(["negative", "positive"]),
            dataset_name=np.asarray(f"{dataset}_{task}_48h"),
        )
        out_paths.append(out)
    if "temporal" in modalities:
        src = _load_cache(input_dir, dataset, "temporal")
        out = output_dir / f"{dataset}_{task}_48h_temporal.npz"
        np.savez_compressed(
            out,
            X_ts=np.asarray(src["X_ts"])[valid_mask],
            mask=np.asarray(src["mask"])[valid_mask],
            y=y,
            var_names=np.asarray(src["var_names"]),
            dataset_name=np.asarray(f"{dataset}_{task}_48h"),
        )
        out_paths.append(out)
    meta = {
        "dataset": dataset,
        "task": task,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else None,
        "source_cache": str(input_dir),
        "modalities": sorted(modalities),
        "label_definition": TASK_DESCRIPTIONS[task],
    }
    meta_path = output_dir / f"{dataset}_{task}_48h_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    out_paths.append(meta_path)
    return out_paths


TASK_DESCRIPTIONS = {
    "hospital_mortality": "Hospital mortality label used by the original benchmark.",
    "icu_mortality": "Death before or at ICU discharge; eICU uses actual ICU mortality.",
    "icu_los_gt_3d_at48": "Among stays still in ICU at 48h, predict total ICU LOS > 3 days.",
    "icu_los_gt_7d_at48": "Among stays still in ICU at 48h, predict total ICU LOS > 7 days.",
    "hospital_los_gt_7d_at48": "Among stays still in ICU at 48h, predict total hospital LOS > 7 days.",
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated datasets: mimic3,mimic4,eicu")
    p.add_argument("--tasks", default=",".join(TASKS), help="Comma-separated tasks or 'all'.")
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--cache-output-dir", default=str(DEFAULT_CACHE_OUTPUT_DIR))
    p.add_argument("--modalities", default="tabular", help="Comma-separated modalities to write: tabular,temporal")
    p.add_argument("--max-samples", type=int, default=0, help="Use the same stratified limit as mortality_preprocess; 0 means full cache.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write", action="store_true", help="Write relabeled NPZ caches after alignment checks.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    requested_tasks = list(TASKS) if args.tasks == "all" else [x.strip() for x in args.tasks.split(",") if x.strip()]
    modalities = {x.strip() for x in args.modalities.split(",") if x.strip()}
    raw_root = Path(args.raw_root).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    report_dir = Path(args.report_dir)
    cache_output_dir = Path(args.cache_output_dir)
    max_samples = args.max_samples if args.max_samples > 0 else None

    rows: list[dict] = []
    written: list[str] = []
    stays_by_dataset: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        stays, ds_rows = _summarize_dataset(raw_root, input_dir, dataset, max_samples, args.seed)
        stays_by_dataset[dataset] = stays
        rows.extend(ds_rows)

    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "clinical_task_feasibility.csv"
    _write_csv(csv_path, rows)

    if args.write:
        for row in rows:
            dataset = row["dataset"]
            task = row["task"]
            if task not in requested_tasks:
                continue
            if not row["alignment_ok"]:
                raise RuntimeError(f"Refusing to write {dataset}/{task}: {row['alignment_message']}")
            labels = _task_labels(stays_by_dataset[dataset], task)
            paths = _copy_relabeled_cache(input_dir, cache_output_dir, dataset, task, labels, modalities)
            written.extend(str(p) for p in paths)

    md = report_dir / "CLINICAL_TASK_FEASIBILITY.md"
    lines = [
        "# Clinical Task Feasibility",
        "",
        f"Input feature cache: `{input_dir}`",
        f"Raw root: `{raw_root}`",
        f"Feasibility CSV: `{csv_path}`",
        "",
        "| Dataset | Task | Valid N | Positive N | Positive rate | Dropped | Alignment |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['task']} | {row['n_valid']} | {row['n_positive']} | "
            f"{row['positive_rate']:.4f} | {row['n_dropped']} | {row['alignment_message']} |"
        )
    lines.extend([
        "",
        "## Methodological Notes",
        "",
        "LOS tasks use an at-risk cohort: only stays still in ICU at 48h receive the LOS label. This avoids leaking early discharge through missing post-discharge measurements.",
        "The script verifies that recreated hospital mortality labels match the existing mortality NPZ row order before writing relabeled caches.",
    ])
    if written:
        lines.extend(["", "## Written Caches", ""] + [f"- `{p}`" for p in written])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rows={len(rows)} report={csv_path}")
    if written:
        print(f"written={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
