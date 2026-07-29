#!/usr/bin/env python3
"""Run PPtheta proof mortality jobs sequentially on a local GPU."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "jobs" / "run_mortality_paper_job.sh"
DATASETS = ("mimic3", "mimic4", "eicu")
STAGES = (
    "rahmatullaev_proof_evidence_ablation",
    "rahmatullaev_proof_selective_utility",
    "rahmatullaev_proof_strong_base_repair",
    "rahmatullaev_proof_audit_sufficiency",
    "rahmatullaev_proof_operating_points",
    "rahmatullaev_proof_randomized_controls",
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


def proof_done(repo: Path, mode: str, ver: str, dataset: str, stage: str) -> bool:
    out = repo / "output" / "mortality_paper_jobs" / f"{mode}_{ver}" / dataset / stage
    summary = out / "ppost_proof_summary.csv"
    return summary.exists() and summary.stat().st_size > 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    p.add_argument("--gpu", default="0")
    p.add_argument("--mode", default="rahmatullaev_ppost_proof")
    p.add_argument("--ver", default="mortality_ppost_proof_local_v1")
    p.add_argument("--state", default=str(REPO / "output" / "local_queues" / "ppost_proof_local_v1_state.json"))
    p.add_argument("--log", default=str(REPO / "logs" / "local_queues" / "ppost_proof_local_v1_queue.log"))
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state_path = Path(args.state)
    queue_log = Path(args.log)
    queue_log.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path)
    state.update({
        "mode": args.mode,
        "ver": args.ver,
        "gpu": args.gpu,
        "updated_at": now(),
    })
    save_state(state_path, state)

    base_env = os.environ.copy()
    base_env.update({
        "BASE": str(REPO.parent),
        "MODE": args.mode,
        "VER": args.ver,
        "PAPER_PRESET": "full_tabpfn",
        "TABPFN_STAGES": "1",
        "TABPFN_DEVICE": "cuda",
        "TABPFN_IGNORE_PRETRAINING_LIMITS": "1",
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PPPOST_EBM_ANCHOR_OUTER_BAGS": os.environ.get("PPPOST_EBM_ANCHOR_OUTER_BAGS", "2"),
        "PPPOST_EBM_ANCHOR_INTERACTIONS": os.environ.get("PPPOST_EBM_ANCHOR_INTERACTIONS", "4"),
        "PPPOST_EBM_ANCHOR_MAX_BINS": os.environ.get("PPPOST_EBM_ANCHOR_MAX_BINS", "128"),
    })

    with queue_log.open("a", buffering=1, encoding="utf-8") as qlog:
        print(f"[{now()}] local PPtheta proof queue start mode={args.mode} ver={args.ver} gpu={args.gpu}", file=qlog)
        for dataset in args.datasets:
            for stage in args.stages:
                axis = f"{dataset}/{stage}"
                if not args.force and proof_done(REPO, args.mode, args.ver, dataset, stage):
                    state["jobs"].setdefault(axis, {}).update({"status": "skipped_done", "updated_at": now()})
                    save_state(state_path, state)
                    print(f"[{now()}] skip done {axis}", file=qlog)
                    continue
                env = base_env.copy()
                env.update({"DATASET": dataset, "STAGE": stage})
                rec = state["jobs"].setdefault(axis, {})
                rec.update({"status": "running", "started_at": now(), "updated_at": now()})
                save_state(state_path, state)
                print(f"[{now()}] start {axis}", file=qlog)
                rc = subprocess.run(["bash", str(RUNNER)], cwd=str(REPO), env=env).returncode
                rec.update({"status": "completed" if rc == 0 else "failed", "returncode": rc, "finished_at": now(), "updated_at": now()})
                save_state(state_path, state)
                print(f"[{now()}] finish {axis} rc={rc}", file=qlog)
                if rc != 0:
                    print(f"[{now()}] stopping queue after failure {axis}", file=qlog)
                    return rc
        state["finished_at"] = now()
        state["updated_at"] = now()
        save_state(state_path, state)
        print(f"[{now()}] local PPtheta proof queue complete", file=qlog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
