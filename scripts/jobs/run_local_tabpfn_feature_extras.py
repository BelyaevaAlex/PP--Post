#!/usr/bin/env python3
"""Local queue for ordinary TabPFN temporal-feature extra rows.

Runs temporal.compare_temporal in extras-only mode:
  --levels none --baselines tabpfn_l1 tabpfn_l2 tabpfn_l3
for eICU, MIMIC-III and MIMIC-IV mortality datasets.  Outputs are kept in
per-dataset temporal_main_tabpfn_features_extra directories so they can be
merged into the existing temporal_main tables after completion.
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

DATASETS = ("eicu", "mimic3", "mimic4")
DEFAULT_VER = "mortality_full_tabpfn_v6_l4_onepass"
STAGE = "temporal_main_tabpfn_features_extra"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _out_root(repo: Path, ver: str) -> Path:
    return repo / "output" / "mortality_paper_jobs" / f"full_tabpfn_{ver}"


def _log_root(repo: Path, ver: str) -> Path:
    return repo / "logs" / "mortality_paper_jobs" / f"full_tabpfn_{ver}" / "tabpfn_feature_extras"


def _state_path(repo: Path, ver: str) -> Path:
    return _out_root(repo, ver) / "local_tabpfn_feature_extras_state.json"


def _axis(ver: str, dataset: str) -> str:
    return f"tabpfn_feature_extras.{ver}.{dataset}"


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _make_state(args: argparse.Namespace, repo: Path) -> dict:
    jobs = {}
    for dataset in args.datasets:
        jobs[_axis(args.ver, dataset)] = {
            "dataset": dataset,
            "status": "queued",
            "gpu": None,
            "pid": None,
            "rc": None,
            "started_at": None,
            "ended_at": None,
            "out_dir": str(_out_root(repo, args.ver) / dataset / STAGE),
            "log": str(_log_root(repo, args.ver) / f"{dataset}.out"),
        }
    return {
        "ver": args.ver,
        "stage": STAGE,
        "created_at": _now(),
        "updated_at": _now(),
        "repo": str(repo),
        "gpus": list(args.gpus),
        "tabpfn_feature_n_estimators": args.tabpfn_feature_n_estimators,
        "datasets": list(args.datasets),
        "jobs": jobs,
    }


def _cmd(repo: Path, args: argparse.Namespace, dataset: str) -> list[str]:
    ckpt = repo / "data" / "tabpfn_checkpoints" / "tabpfn-v3-classifier-v3_default.ckpt"
    out_dir = _out_root(repo, args.ver) / dataset / STAGE
    return [
        str(repo / ".venv" / "bin" / "python"), "-u", "-m", "temporal.compare_temporal",
        "--datasets", f"{dataset}_mortality",
        "--levels", "none",
        "--baselines", "tabpfn_l1", "tabpfn_l2", "tabpfn_l3",
        "--folds", str(args.folds),
        "--epochs", "1",
        "--tabpfn-feature-n-estimators", str(args.tabpfn_feature_n_estimators),
        "--tabpfn-feature-device", "cuda",
        "--tabpfn-classifier-model-path", str(ckpt),
        "--output-dir", str(out_dir),
    ]


def _env(repo: Path, gpu: str) -> dict[str, str]:
    ckpt_dir = repo / "data" / "tabpfn_checkpoints"
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": gpu,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "TABPFN_DEVICE": "cuda",
        "TABPFN_IGNORE_PRETRAINING_LIMITS": "1",
        "TABPFN_CLASSIFIER_MODEL_PATH": str(ckpt_dir / "tabpfn-v3-classifier-v3_default.ckpt"),
        "TABPFN_DISABLE_TELEMETRY": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(key, None)
    return env


def _start(repo: Path, args: argparse.Namespace, state: dict, axis: str, gpu: str) -> subprocess.Popen:
    meta = state["jobs"][axis]
    dataset = meta["dataset"]
    log = Path(meta["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    Path(meta["out_dir"]).mkdir(parents=True, exist_ok=True)
    cmd = _cmd(repo, args, dataset)
    with log.open("ab", buffering=0) as fh:
        fh.write(f"\n[tabpfn-feature-queue] start {_now()} axis={axis} gpu={gpu}\n".encode())
        fh.write(("[cmd] " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=_env(repo, gpu),
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
        print("No tabpfn feature extras state found.")
        return 1
    counts: dict[str, int] = {}
    for meta in state["jobs"].values():
        counts[meta["status"]] = counts.get(meta["status"], 0) + 1
    print(f"ver={state['ver']} updated={state.get('updated_at')} counts={counts}")
    for axis, meta in sorted(state["jobs"].items()):
        pid = meta.get("pid")
        alive = _alive(pid) if meta.get("status") == "running" else False
        print(
            f"{axis}: {meta['status']} gpu={meta.get('gpu')} pid={pid} "
            f"alive={alive} rc={meta.get('rc')} out={meta.get('out_dir')} log={meta.get('log')}"
        )
    return 0


def run(args: argparse.Namespace) -> int:
    repo = _repo()
    state_file = _state_path(repo, args.ver)
    if args.status:
        return _print_status(_load(state_file))
    if state_file.exists() and not args.force:
        print(f"state exists: {state_file}; pass --force to recreate", file=sys.stderr)
        return 2
    state = _make_state(args, repo)
    if args.dry_run:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    _save(state_file, state)
    procs: dict[str, subprocess.Popen] = {}
    free_gpus = list(args.gpus)
    try:
        while True:
            for axis, proc in list(procs.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                meta = state["jobs"][axis]
                meta["status"] = "completed" if rc == 0 else "failed"
                meta["rc"] = rc
                meta["ended_at"] = _now()
                free_gpus.append(str(meta["gpu"]))
                del procs[axis]
                _save(state_file, state)
            queued = [a for a, m in state["jobs"].items() if m["status"] == "queued"]
            while free_gpus and queued:
                gpu = free_gpus.pop(0)
                axis = queued.pop(0)
                procs[axis] = _start(repo, args, state, axis, gpu)
                _save(state_file, state)
            state["updated_at"] = _now()
            _save(state_file, state)
            if not procs and not queued:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        for axis, proc in procs.items():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
            state["jobs"][axis]["status"] = "terminated"
            state["jobs"][axis]["ended_at"] = _now()
        state["updated_at"] = _now()
        _save(state_file, state)
        raise
    return _print_status(state)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ver", default=DEFAULT_VER)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--gpus", nargs="+", default=["0", "1"])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--tabpfn-feature-n-estimators", type=int, default=1)
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--status", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
