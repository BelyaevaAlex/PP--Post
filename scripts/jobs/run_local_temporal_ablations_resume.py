#!/usr/bin/env python3
"""Resume mortality temporal ablation jobs on free local GPUs.

This intentionally does not reuse the stale full temporal queue state.  It is
for the common recovery case where temporal_main jobs are already complete, one
ablation job may still be running, and the remaining ablation datasets should
be launched as GPUs become free.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import run_local_temporal_queue as base_queue

DEFAULT_VER = "mortality_full_tabpfn_v6_l4_onepass"
DEFAULT_DATASETS = ("mimic3", "mimic4")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _axis(ver: str, dataset: str) -> str:
    return f"full_tabpfn.{ver}.{dataset}.temporal_ablations"


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _state_path(repo: Path, ver: str) -> Path:
    return (
        repo
        / "output"
        / "mortality_paper_jobs"
        / f"full_tabpfn_{ver}"
        / "local_queue_resume_state.json"
    )


def _log_root(repo: Path, ver: str) -> Path:
    return repo / "logs" / "mortality_paper_jobs" / f"full_tabpfn_{ver}"


def _out_root(repo: Path, ver: str) -> Path:
    return repo / "output" / "mortality_paper_jobs" / f"full_tabpfn_{ver}"


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def _completed_from_outputs(repo: Path, ver: str, dataset: str, folds: int) -> bool:
    out_dir = _out_root(repo, ver) / dataset / "temporal_ablations"
    summaries = sorted(out_dir.glob("ablations_*_summary.csv"))
    fold_csvs = sorted(
        p for p in out_dir.glob("ablations_*.csv")
        if not p.name.endswith("_summary.csv")
    )
    if not summaries or not fold_csvs:
        return False
    min_rows = 14 * folds
    for csv_path in fold_csvs:
        try:
            rows = sum(1 for _ in csv_path.open(errors="replace")) - 1
        except OSError:
            continue
        if rows >= min_rows:
            return True
    return False


def _gpu_compute_pids(gpu: str) -> list[int]:
    cmd = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    out: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(int(line.split(",")[0].strip()))
        except ValueError:
            continue
    return out


def _make_base_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        ver=args.ver,
        datasets=args.datasets,
        stages=["temporal_ablations"],
        gpus=args.gpus,
        folds=args.folds,
        epochs=args.epochs,
        expensive_epochs=args.expensive_epochs,
        rule_n_estimators=args.rule_n_estimators,
        rule_max_leaf_nodes=args.rule_max_leaf_nodes,
        l4_batch_size=args.l4_batch_size,
        attention_max_samples=args.attention_max_samples,
    )


def _make_state(args: argparse.Namespace, repo: Path) -> dict:
    jobs: dict[str, dict] = {}
    for dataset in args.datasets:
        axis = _axis(args.ver, dataset)
        status = "completed" if _completed_from_outputs(repo, args.ver, dataset, args.folds) else "queued"
        jobs[axis] = {
            "dataset": dataset,
            "stage": "temporal_ablations",
            "status": status,
            "gpu": None,
            "pid": None,
            "rc": 0 if status == "completed" else None,
            "started_at": None,
            "ended_at": _now() if status == "completed" else None,
            "runner_log": str(_log_root(repo, args.ver) / f"{dataset}_temporal_ablations.out"),
            "child_log": str(
                _log_root(repo, args.ver)
                / "local_queue_resume"
                / f"{dataset}_temporal_ablations.child.out"
            ),
        }
    return {
        "ver": args.ver,
        "created_at": _now(),
        "updated_at": _now(),
        "repo": str(repo),
        "gpus": args.gpus,
        "external_busy": args.busy,
        "jobs": jobs,
    }


def _start_job(
    repo: Path,
    args: argparse.Namespace,
    base_args: argparse.Namespace,
    state: dict,
    axis: str,
    gpu: str,
) -> subprocess.Popen:
    meta = state["jobs"][axis]
    dataset = meta["dataset"]
    child_log = Path(meta["child_log"])
    child_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(repo / "scripts" / "jobs" / "run_mortality_paper_job.sh")]
    env = base_queue._job_env(repo, base_args, dataset, "temporal_ablations", gpu)
    with child_log.open("ab", buffering=0) as fh:
        fh.write(
            f"\n[resume-queue] start {_now()} axis={axis} gpu={gpu} "
            f"cmd={' '.join(cmd)}\n".encode()
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    meta.update({
        "status": "running",
        "gpu": gpu,
        "pid": proc.pid,
        "rc": None,
        "started_at": _now(),
        "ended_at": None,
    })
    return proc


def _parse_busy(values: list[str]) -> dict[str, dict]:
    busy: dict[str, dict] = {}
    for value in values:
        parts = value.split(":", 2)
        if len(parts) < 2:
            raise SystemExit("--busy must look like GPU:PID[:label]")
        gpu, pid_text = parts[0], parts[1]
        label = parts[2] if len(parts) == 3 else "external"
        busy[gpu] = {"pid": int(pid_text), "label": label}
    return busy


def run(args: argparse.Namespace) -> int:
    repo = _repo()
    state_file = _state_path(repo, args.ver)
    log_dir = _log_root(repo, args.ver) / "local_queue_resume"
    log_dir.mkdir(parents=True, exist_ok=True)
    base_args = _make_base_args(args)

    if state_file.exists() and not args.force:
        print(f"state exists: {state_file}", file=sys.stderr)
        print("use --force to recreate resume state", file=sys.stderr)
        return 2

    state = _make_state(args, repo)
    _save_state(state_file, state)
    running: dict[str, subprocess.Popen] = {}
    stop_after_running = False

    def _handle_signal(signum, frame):  # noqa: ANN001
        nonlocal stop_after_running
        stop_after_running = True
        state["stop_requested_at"] = _now()
        state["updated_at"] = _now()
        _save_state(state_file, state)
        print(f"[resume-queue] signal {signum}; no new jobs will be started", flush=True)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print(f"[resume-queue] repo={repo}", flush=True)
    print(f"[resume-queue] state={state_file}", flush=True)
    print(f"[resume-queue] gpus={args.gpus} jobs={len(state['jobs'])}", flush=True)
    print(f"[resume-queue] external_busy={args.busy}", flush=True)

    while True:
        used_gpus: set[str] = set()
        for gpu, info in list(args.busy.items()):
            pid = int(info["pid"])
            if _alive(pid):
                used_gpus.add(gpu)
            else:
                print(
                    f"[resume-queue] external freed gpu={gpu} "
                    f"pid={pid} label={info.get('label')}",
                    flush=True,
                )
                del args.busy[gpu]

        for axis, proc in list(running.items()):
            meta = state["jobs"][axis]
            rc = proc.poll()
            if rc is None:
                used_gpus.add(str(meta["gpu"]))
                continue
            meta["rc"] = rc
            meta["ended_at"] = _now()
            meta["status"] = "completed" if rc == 0 else "failed"
            print(f"[resume-queue] done axis={axis} rc={rc}", flush=True)
            del running[axis]

        if not stop_after_running:
            for gpu in args.gpus:
                if gpu in used_gpus:
                    continue
                gpu_pids = _gpu_compute_pids(gpu)
                if gpu_pids:
                    used_gpus.add(gpu)
                    print(
                        f"[resume-queue] gpu={gpu} busy_by_compute_pids={gpu_pids}",
                        flush=True,
                    )
                    continue
                next_axis = None
                for axis, meta in state["jobs"].items():
                    if meta["status"] == "queued":
                        next_axis = axis
                        break
                if next_axis is None:
                    continue
                proc = _start_job(repo, args, base_args, state, next_axis, gpu)
                running[next_axis] = proc
                used_gpus.add(gpu)
                print(f"[resume-queue] launched axis={next_axis} gpu={gpu} pid={proc.pid}", flush=True)

        state["external_busy"] = args.busy
        state["updated_at"] = _now()
        _save_state(state_file, state)
        statuses = [meta["status"] for meta in state["jobs"].values()]
        if not running and all(s in {"completed", "failed"} for s in statuses):
            break
        if stop_after_running and not running:
            break
        time.sleep(args.poll_seconds)

    state["finished_at"] = _now()
    state["updated_at"] = _now()
    _save_state(state_file, state)
    failed = [axis for axis, meta in state["jobs"].items() if meta["status"] == "failed"]
    print(f"[resume-queue] finished failed={len(failed)}", flush=True)
    for axis in failed:
        print(f"[resume-queue] failed axis={axis} rc={state['jobs'][axis].get('rc')}", flush=True)
    return 1 if failed else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ver", default=DEFAULT_VER)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=["eicu", "mimic3", "mimic4"])
    p.add_argument("--gpus", default="0,1", help="Comma-separated local GPU ids.")
    p.add_argument("--busy", action="append", default=[], help="External busy GPU as GPU:PID[:label].")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--expensive-epochs", type=int, default=80)
    p.add_argument("--rule-n-estimators", type=int, default=8)
    p.add_argument("--rule-max-leaf-nodes", type=int, default=1024)
    p.add_argument("--l4-batch-size", type=int, default=256)
    p.add_argument("--attention-max-samples", type=int, default=2048)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    args.gpus = [x.strip() for x in str(args.gpus).split(",") if x.strip()]
    if not args.gpus:
        raise SystemExit("--gpus must contain at least one GPU id")
    args.busy = _parse_busy(args.busy)
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
