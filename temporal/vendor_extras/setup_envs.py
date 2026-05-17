"""Provision per-baseline virtualenvs for vendored IMTS SOTA models.

Each vendored baseline lives in ``temporal/vendor/<name>/`` (a git
submodule) and its dependency manifest in
``temporal/vendor_extras/<name>.txt``.  This script creates a
*dedicated* virtualenv for each baseline at
``temporal/vendor/.venvs/<name>/`` and installs the manifest into
it.  The main PPθ-Post environment is never touched.

At runtime, :func:`temporal.baselines_vendored._baseline_venv_site_packages`
discovers the per-baseline ``site-packages`` and prepends it to
``sys.path`` only while the baseline's import is in flight (see
``_isolated_top_level_namespace``).  The next baseline therefore
sees a clean slate, even if their transitive dependencies pin
incompatible package versions.

Usage
-----

::

    # Bootstrap all baselines.
    python -m temporal.vendor_extras.setup_envs

    # Bootstrap a subset.
    python -m temporal.vendor_extras.setup_envs interpgn raindrop

    # Re-create from scratch (deletes existing .venvs/<name>/).
    python -m temporal.vendor_extras.setup_envs --force interpgn

    # Use a specific Python interpreter (defaults to current sys.executable).
    python -m temporal.vendor_extras.setup_envs --python /opt/python3.10/bin/python3 seft

The script is idempotent: re-running it without ``--force`` keeps
the existing virtualenv and only runs ``pip install -U`` against
the manifest.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


THIS_DIR = Path(__file__).resolve().parent
VENDOR_DIR = THIS_DIR.parent / "vendor"
VENVS_DIR = VENDOR_DIR / ".venvs"

ALL_BASELINES = (
    "sand",
    "mtan",
    "grud",
    "raindrop",
    "interpgn",
    "seft",
    "camelot",
)


def _manifest_path(name: str) -> Path:
    return THIS_DIR / f"{name}.txt"


def _venv_path(name: str) -> Path:
    return VENVS_DIR / name


def _venv_python(name: str) -> Path:
    venv = _venv_path(name)
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _venv_site_packages(name: str) -> Path | None:
    """Return the ``site-packages`` path inside the per-baseline venv,
    or ``None`` if the venv has not been created yet."""
    venv = _venv_path(name)
    if not venv.exists():
        return None
    if os.name == "nt":
        candidate = venv / "Lib" / "site-packages"
        return candidate if candidate.exists() else None
    lib = venv / "lib"
    if not lib.exists():
        return None
    for entry in sorted(lib.iterdir()):
        if entry.is_dir() and entry.name.startswith("python"):
            sp = entry / "site-packages"
            if sp.exists():
                return sp
    return None


def _create_venv(name: str, python: str, force: bool) -> None:
    venv = _venv_path(name)
    if venv.exists():
        if force:
            print(f"[setup-envs] {name}: removing existing venv at {venv}")
            shutil.rmtree(venv)
        else:
            print(f"[setup-envs] {name}: re-using existing venv at {venv}")
            return
    VENVS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup-envs] {name}: creating venv with {python} at {venv}")
    subprocess.run(
        [python, "-m", "venv", str(venv)],
        check=True,
    )


def _pip_install(name: str, manifest: Path) -> None:
    py = _venv_python(name)
    if not py.exists():
        raise RuntimeError(
            f"venv for '{name}' has no python executable at {py} — "
            f"venv creation likely failed"
        )
    print(f"[setup-envs] {name}: upgrading pip in venv")
    subprocess.run(
        [str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"],
        check=True,
    )
    print(f"[setup-envs] {name}: installing {manifest.name}")
    subprocess.run(
        [str(py), "-m", "pip", "install", "--upgrade", "-r", str(manifest)],
        check=True,
    )


def setup_one(name: str, python: str, force: bool) -> None:
    manifest = _manifest_path(name)
    if not manifest.exists():
        raise FileNotFoundError(
            f"no manifest for baseline '{name}' at {manifest} — "
            f"available: {', '.join(ALL_BASELINES)}"
        )
    _create_venv(name, python, force)
    _pip_install(name, manifest)
    sp = _venv_site_packages(name)
    print(f"[setup-envs] {name}: site-packages = {sp}")


def setup_many(names: Iterable[str], python: str, force: bool) -> None:
    failures: list[tuple[str, str]] = []
    for name in names:
        try:
            setup_one(name, python, force)
        except subprocess.CalledProcessError as e:
            failures.append((name, f"pip exited with {e.returncode}"))
        except FileNotFoundError as e:
            failures.append((name, str(e)))
        except Exception as e:  # noqa: BLE001
            failures.append((name, f"{type(e).__name__}: {e}"))
    if failures:
        print()
        print("[setup-envs] ❌ failed baselines:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        sys.exit(1)
    print()
    print("[setup-envs] ✅ all requested baselines provisioned")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m temporal.vendor_extras.setup_envs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "names",
        nargs="*",
        choices=list(ALL_BASELINES),
        default=list(ALL_BASELINES),
        help="Baseline names to bootstrap (default: all).",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for venv creation "
             "(default: current sys.executable).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate any existing .venvs/<name>/ directories.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    names = args.names if args.names else list(ALL_BASELINES)
    setup_many(names, python=args.python, force=args.force)


if __name__ == "__main__":
    main()
