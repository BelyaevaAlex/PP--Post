#!/usr/bin/env python3
"""Download the gated TabPFN v3 checkpoints used by PPtheta-Post.

The checkpoints live in the Hugging Face gated repo
``Prior-Labs/tabpfn_3``. Accept the model terms in the browser first,
then authenticate either with ``hf auth login`` or by setting ``HF_TOKEN``.

Example
-------

    python download_tabpfn_ts_weights.py --kind ts
    python download_tabpfn_ts_weights.py --kind classifier

The script stores files in the same cache location read by the temporal
teacher and prints the relevant environment export line for reproducible
runs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "Prior-Labs/tabpfn_3"
CHECKPOINTS = {
    "ts": "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
    "classifier": "tabpfn-v3-classifier-v3_default.ckpt",
    "regressor": "tabpfn-v3-regressor-v3_default.ckpt",
}
EXPORT_VARS = {
    "ts": "TABPFN_TS_MODEL_PATH",
    "classifier": "TABPFN_CLASSIFIER_MODEL_PATH",
    "regressor": "TABPFN_REGRESSOR_MODEL_PATH",
}


def default_tabpfn_checkpoint_dir() -> Path:
    """Match tabpfn's default user cache directory without importing temporal."""
    try:
        from platformdirs import user_cache_dir

        return Path(user_cache_dir("tabpfn"))
    except Exception:
        return Path.home() / ".cache" / "tabpfn"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download gated TabPFN v3 checkpoints for PPtheta-Post.",
    )
    p.add_argument(
        "--kind",
        choices=sorted(CHECKPOINTS),
        default="ts",
        help=(
            "Named checkpoint target: ts for temporal L2T/L3T, classifier "
            "for tabular TabPFN baseline/distillation, regressor for generic "
            "tabular regression uses."
        ),
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repo id containing the gated checkpoint.",
    )
    p.add_argument(
        "--filename",
        default=None,
        help="Checkpoint filename inside the Hugging Face repo. Overrides --kind.",
    )
    p.add_argument(
        "--output-dir",
        default=str(default_tabpfn_checkpoint_dir()),
        help="Directory where the checkpoint should be stored.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN / cached login.",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even when a cached copy exists.",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only verify an already downloaded local copy.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = args.filename or CHECKPOINTS[args.kind]
    direct_path = output_dir / filename

    if args.local_files_only and direct_path.exists():
        path = direct_path.resolve()
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Downloaded: {path}")
        print(f"Size: {size_mb:.1f} MB")
        export_var = EXPORT_VARS.get(args.kind, "TABPFN_MODEL_PATH")
        print(f"export {export_var}='{path}'")
        return 0

    token = args.token or os.environ.get("HF_TOKEN")
    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=filename,
        token=token,
        local_dir=str(output_dir),
        force_download=args.force_download,
        local_files_only=args.local_files_only,
    )
    path = Path(path).expanduser().resolve()
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Downloaded: {path}")
    print(f"Size: {size_mb:.1f} MB")
    export_var = EXPORT_VARS.get(args.kind, "TABPFN_MODEL_PATH")
    print(f"export {export_var}='{path}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
