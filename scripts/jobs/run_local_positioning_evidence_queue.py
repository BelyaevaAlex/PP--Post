#!/usr/bin/env python3
"""Run AAAI positioning evidence jobs sequentially on one local GPU.

The queue runs exactly the stages needed for the paper-ready Section 61 tables:
AuditSelect final deployment, native-error correction, trace perturbations, and
compact sufficiency curves.  It is intentionally sequential so one local GPU is
kept busy without oversubscribing memory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "jobs" / "run_mortality_paper_job.sh"
DATASETS = ("eicu", "mimic3", "mimic4")
EVIDENCE_MODE = "rahmatullaev_aaai_evidence_v2"
ACCEPT_MODE = "rahmatullaev_aaai_acceptance_clinician_symbolic"
DEFAULT_VER = "mortality_positioning_local_v1"

JOBS = (
    (EVIDENCE_MODE, "rahmatullaev_v2_native_wrong_correction", "native_wrong_correction_summary.csv"),
    (EVIDENCE_MODE, "rahmatullaev_v2_rich_randomized_controls", "rich_randomized_controls_summary.csv"),
    (EVIDENCE_MODE, "rahmatullaev_v2_extended_trace_curve", "extended_trace_curve_summary.csv"),
    (ACCEPT_MODE, "rahmatullaev_accept_clean_interpretable_calibrated", "clean_interpretable_calibrated_selected.csv"),
    (ACCEPT_MODE, "rahmatullaev_accept_symbolic_family_calibrated", "symbolic_family_calibrated_selected.csv"),
)


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"started_at": now(), "jobs": {}}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def stage_output_name(stage: str, done_file: str) -> str:
    stem = done_file[:-4] if done_file.endswith(".csv") else done_file
    # run_compare outputs use fixed filenames inside each stage directory.
    return done_file


def done(repo: Path, mode: str, ver: str, dataset: str, stage: str, done_file: str) -> bool:
    path = repo / "output" / "mortality_paper_jobs" / f"{mode}_{ver}" / dataset / stage / stage_output_name(stage, done_file)
    return path.exists() and path.stat().st_size > 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--gpu", default="0")
    p.add_argument("--ver", default=DEFAULT_VER)
    p.add_argument("--state", default=str(REPO / "output" / "local_queues" / "positioning_evidence_local_v1_state.json"))
    p.add_argument("--log", default=str(REPO / "logs" / "local_queues" / "positioning_evidence_local_v1_queue.log"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-extended-trace", action="store_true", help="Skip the heaviest compact-curve stage.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state_path = Path(args.state)
    queue_log = Path(args.log)
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in JOBS if not (args.skip_extended_trace and j[1] == "rahmatullaev_v2_extended_trace_curve")]

    state = load_state(state_path)
    state.pop("finished_at", None)
    state.pop("failed_at", None)
    state.pop("aggregation_returncode", None)
    state.update({"ver": args.ver, "gpu": args.gpu, "updated_at": now(), "datasets": args.datasets})
    save_state(state_path, state)

    base_env = os.environ.copy()
    base_env.update({
        "BASE": str(REPO.parent),
        "VER": args.ver,
        "PAPER_PRESET": "full_tabpfn",
        "TABPFN_STAGES": "1",
        "TABPFN_DEVICE": "cuda",
        "TABPFN_IGNORE_PRETRAINING_LIMITS": "1",
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PPPOST_BOOTSTRAP_N": os.environ.get("PPPOST_BOOTSTRAP_N", "600"),
    })

    with queue_log.open("a", buffering=1, encoding="utf-8") as qlog:
        print(f"[{now()}] positioning evidence local queue start ver={args.ver} gpu={args.gpu}", file=qlog)
        for dataset in args.datasets:
            for mode, stage, done_file in jobs:
                axis = f"{mode}/{dataset}/{stage}"
                if not args.force and done(REPO, mode, args.ver, dataset, stage, done_file):
                    state["jobs"].setdefault(axis, {}).update({"status": "skipped_done", "updated_at": now()})
                    save_state(state_path, state)
                    print(f"[{now()}] skip done {axis}", file=qlog)
                    continue
                env = base_env.copy()
                env.update({"MODE": mode, "DATASET": dataset, "STAGE": stage})
                rec = state["jobs"].setdefault(axis, {})
                rec.update({"status": "running", "started_at": now(), "updated_at": now()})
                save_state(state_path, state)
                print(f"[{now()}] start {axis}", file=qlog)
                rc = subprocess.run(["bash", str(RUNNER)], cwd=str(REPO), env=env).returncode
                rec.update({"status": "completed" if rc == 0 else "failed", "returncode": rc, "finished_at": now(), "updated_at": now()})
                save_state(state_path, state)
                print(f"[{now()}] finish {axis} rc={rc}", file=qlog)
                if rc != 0:
                    state["failed_at"] = now()
                    save_state(state_path, state)
                    print(f"[{now()}] stopping queue after failure {axis}", file=qlog)
                    return rc
        # Build paper-ready tables after the queue, pointing section 61 at this local run.
        env = base_env.copy()
        env.update({
            "PPPOST_POSITIONING_EVIDENCE_ROOT": str(REPO / "output" / "mortality_paper_jobs" / f"{EVIDENCE_MODE}_{args.ver}"),
            "PPPOST_POSITIONING_ACCEPT_ROOT": str(REPO / "output" / "mortality_paper_jobs" / f"{ACCEPT_MODE}_{args.ver}"),
        })
        print(f"[{now()}] aggregate section_61", file=qlog)
        rc = subprocess.run(["python3", "paper_experiments/section_61_positioning_evidence_tables.py"], cwd=str(REPO), env=env).returncode
        state["aggregation_returncode"] = rc
        state["finished_at"] = now()
        state["updated_at"] = now()
        save_state(state_path, state)
        print(f"[{now()}] positioning evidence local queue complete aggregation_rc={rc}", file=qlog)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
