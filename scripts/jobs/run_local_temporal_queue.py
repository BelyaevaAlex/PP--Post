#!/usr/bin/env python3
"""Local GPU queue for PP--Post mortality temporal paper jobs."""
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

DEFAULT_DATASETS = ("eicu", "mimic3", "mimic4")
DEFAULT_STAGES = ("temporal_main", "temporal_ablations")
DEFAULT_VER = "mortality_full_tabpfn_v4_local_temporal"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _axis(ver: str, dataset: str, stage: str) -> str:
    return f"full_tabpfn.{ver}.{dataset}.{stage}"


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _state_path(repo: Path, ver: str) -> Path:
    return repo / "output" / "mortality_paper_jobs" / f"full_tabpfn_{ver}" / "local_queue_state.json"


def _log_root(repo: Path, ver: str) -> Path:
    return repo / "logs" / "mortality_paper_jobs" / f"full_tabpfn_{ver}"


def _out_root(repo: Path, ver: str) -> Path:
    return repo / "output" / "mortality_paper_jobs" / f"full_tabpfn_{ver}"


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def _load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _make_state(args: argparse.Namespace, repo: Path, jobs: list[tuple[str, str]]) -> dict:
    return {
        "ver": args.ver,
        "mode": "full_tabpfn",
        "created_at": _now(),
        "updated_at": _now(),
        "repo": str(repo),
        "gpus": args.gpus,
        "max_parallel": len(args.gpus),
        "env": {
            "FOLDS": str(args.folds),
            "EPOCHS": str(args.epochs),
            "EXPENSIVE_EPOCHS": str(args.expensive_epochs),
            "TEMPORAL_BASELINES": "all",
            "INCLUDE_TABPFN_TS_DISTILL": "1",
            "INCLUDE_TABPFN_TS_BASELINE": "1",
            "TS_TEACHER_BACKEND": "tabpfn_ts",
            "TS_TEACHER_DEVICE": "cuda",
            "TEMPORAL_RULE_N_ESTIMATORS": str(args.rule_n_estimators),
            "TEMPORAL_RULE_MAX_LEAF_NODES": str(args.rule_max_leaf_nodes),
            "TEMPORAL_L4_BATCH_SIZE": str(args.l4_batch_size),
            "TEMPORAL_ATTENTION_MAX_SAMPLES": str(args.attention_max_samples),
        },
        "jobs": {
            _axis(args.ver, dataset, stage): {
                "dataset": dataset,
                "stage": stage,
                "status": "queued",
                "gpu": None,
                "pid": None,
                "rc": None,
                "started_at": None,
                "ended_at": None,
                "runner_log": str(_log_root(repo, args.ver) / f"{dataset}_{stage}.out"),
                "child_log": str(_log_root(repo, args.ver) / "local_queue" / f"{dataset}_{stage}.child.out"),
            }
            for stage in args.stages
            for dataset in args.datasets
            if (dataset, stage) in jobs
        },
    }


def _job_env(repo: Path, args: argparse.Namespace, dataset: str, stage: str, gpu: str) -> dict[str, str]:
    ckpt_dir = repo / "data" / "tabpfn_checkpoints"
    env = os.environ.copy()
    env.update({
        "BASE": str(repo.parent),
        "DATASET": dataset,
        "STAGE": stage,
        "MODE": "full_tabpfn",
        "VER": args.ver,
        "PAPER_PRESET": "full_tabpfn",
        "TABPFN_STAGES": "1",
        "TABPFN_DEVICE": "cuda",
        "TABPFN_IGNORE_PRETRAINING_LIMITS": "1",
        "INCLUDE_TABPFN_TS_DISTILL": "1",
        "INCLUDE_TABPFN_TS_BASELINE": "1",
        "TS_TEACHER_BACKEND": "tabpfn_ts",
        "TS_TEACHER_DEVICE": "cuda",
        "TS_TEACHER_WORKERS": "1",
        "TEMPORAL_BASELINES": "all",
        "TEMPORAL_RULE_N_ESTIMATORS": str(args.rule_n_estimators),
        "TEMPORAL_RULE_MAX_LEAF_NODES": str(args.rule_max_leaf_nodes),
        "TEMPORAL_L4_BATCH_SIZE": str(args.l4_batch_size),
        "TEMPORAL_ATTENTION_MAX_SAMPLES": str(args.attention_max_samples),
        "FOLDS": str(args.folds),
        "EPOCHS": str(args.epochs),
        "EXPENSIVE_EPOCHS": str(args.expensive_epochs),
        "FULL_CACHE_DIR": str(repo / "data" / "processed" / "mortality"),
        "OUT_ROOT": str(_out_root(repo, args.ver)),
        "LOG_ROOT": str(_log_root(repo, args.ver)),
        "TABPFN_CKPT_DIR": str(ckpt_dir),
        "TABPFN_CLASSIFIER_MODEL_PATH": str(ckpt_dir / "tabpfn-v3-classifier-v3_default.ckpt"),
        "TABPFN_TS_MODEL_PATH": str(ckpt_dir / "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"),
        "PY_OVERRIDE": str(repo / ".venv" / "bin" / "python"),
        "CUDA_VISIBLE_DEVICES": gpu,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "TF_GPU_ALLOCATOR": "cuda_malloc_async",
        "TF_CPP_MIN_LOG_LEVEL": "1",
    })
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(key, None)
    return env


def _start_job(repo: Path, args: argparse.Namespace, state: dict, axis: str, gpu: str) -> subprocess.Popen:
    meta = state["jobs"][axis]
    dataset = meta["dataset"]
    stage = meta["stage"]
    child_log = Path(meta["child_log"])
    child_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(repo / "scripts" / "jobs" / "run_mortality_paper_job.sh")]
    with child_log.open("ab", buffering=0) as fh:
        fh.write(f"\n[local-queue] start {_now()} axis={axis} gpu={gpu} cmd={' '.join(cmd)}\n".encode())
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=_job_env(repo, args, dataset, stage, gpu),
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


def _print_status(state: dict | None) -> int:
    if state is None:
        print("No local temporal queue state found.")
        return 1
    counts: dict[str, int] = {}
    for meta in state["jobs"].values():
        counts[meta["status"]] = counts.get(meta["status"], 0) + 1
    print(f"ver={state['ver']} updated={state.get('updated_at')} counts={counts}")
    for axis, meta in sorted(state["jobs"].items()):
        pid = meta.get("pid")
        alive = _alive(pid) if meta.get("status") == "running" else False
        print(
            f"{axis}: {meta['status']} gpu={meta.get('gpu')} "
            f"pid={pid} alive={alive} rc={meta.get('rc')} log={meta.get('runner_log')}"
        )
    return 0


def run(args: argparse.Namespace) -> int:
    repo = _repo()
    jobs = [(dataset, stage) for stage in args.stages for dataset in args.datasets]
    state_file = _state_path(repo, args.ver)
    log_root = _log_root(repo, args.ver)
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "local_queue").mkdir(parents=True, exist_ok=True)

    if args.status:
        return _print_status(_load_state(state_file))

    if state_file.exists() and not args.force:
        print(f"state exists: {state_file}", file=sys.stderr)
        print("use --force to recreate this local temporal version", file=sys.stderr)
        return 2

    state = _make_state(args, repo, jobs)
    if args.dry_run:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    _save_state(state_file, state)
    running: dict[str, subprocess.Popen] = {}
    stop_after_running = False

    def _handle_signal(signum, frame):  # noqa: ANN001
        nonlocal stop_after_running
        stop_after_running = True
        state["stop_requested_at"] = _now()
        state["updated_at"] = _now()
        _save_state(state_file, state)
        print(f"[local-queue] signal {signum}; no new jobs will be started", flush=True)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print(f"[local-queue] repo={repo}", flush=True)
    print(f"[local-queue] state={state_file}", flush=True)
    print(f"[local-queue] gpus={args.gpus} jobs={len(state['jobs'])}", flush=True)

    while True:
        used_gpus = set()
        for axis, proc in list(running.items()):
            meta = state["jobs"][axis]
            rc = proc.poll()
            if rc is None:
                used_gpus.add(str(meta["gpu"]))
                continue
            meta["rc"] = rc
            meta["ended_at"] = _now()
            meta["status"] = "completed" if rc == 0 else "failed"
            print(f"[local-queue] done axis={axis} rc={rc}", flush=True)
            del running[axis]

        if not stop_after_running:
            for gpu in args.gpus:
                if gpu in used_gpus:
                    continue
                next_axis = None
                for axis, meta in state["jobs"].items():
                    if meta["status"] == "queued":
                        next_axis = axis
                        break
                if next_axis is None:
                    continue
                proc = _start_job(repo, args, state, next_axis, gpu)
                running[next_axis] = proc
                used_gpus.add(gpu)
                print(f"[local-queue] launched axis={next_axis} gpu={gpu} pid={proc.pid}", flush=True)

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
    print(f"[local-queue] finished failed={len(failed)}", flush=True)
    for axis in failed:
        print(f"[local-queue] failed axis={axis} rc={state['jobs'][axis].get('rc')}", flush=True)
    return 1 if failed else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ver", default=DEFAULT_VER)
    p.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(DEFAULT_DATASETS))
    p.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES), choices=list(DEFAULT_STAGES))
    p.add_argument("--gpus", default="0,1", help="Comma-separated local GPU ids.")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--expensive-epochs", type=int, default=80)
    p.add_argument("--rule-n-estimators", type=int, default=8)
    p.add_argument("--rule-max-leaf-nodes", type=int, default=1024)
    p.add_argument("--l4-batch-size", type=int, default=256)
    p.add_argument("--attention-max-samples", type=int, default=2048)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)
    args.gpus = [x.strip() for x in str(args.gpus).split(",") if x.strip()]
    if not args.gpus:
        raise SystemExit("--gpus must contain at least one GPU id")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args(sys.argv[1:])))
