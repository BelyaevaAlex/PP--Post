"""Vendoring fallback for tabular standalone baselines (lazy / dormant).

The default tabular baselines in :mod:`tabular.baselines` rely on
``pip install interpret imodels tabpfn`` — the same versions that ship
with the main PPtheta-Post env.  This works in 95 % of cases but does
break on machines where, for example, ``tabpfn`` wants a
``torch>=2.5`` that conflicts with the project pin, or ``interpret``
needs a newer ``numpy`` than other PPtheta-Post deps allow.

When that happens we fall back to the same scheme already used by
:mod:`temporal.baselines_vendored`:

1. Add each upstream as a git submodule under
   ``tabular/vendor/<name>/``.
2. Provision a per-baseline virtualenv at
   ``tabular/vendor/.venvs/<name>/`` from a manifest in
   ``tabular/vendor_extras/<name>.txt``.
3. At import time, prepend that venv's ``site-packages`` to
   ``sys.path`` inside :func:`_isolated_top_level_namespace` so the
   conflicting transitive deps resolve into the vendored venv and the
   main env stays clean.

This module deliberately ships **as a skeleton only**.  Until a real
clash is reported the registry stays empty and ``compare_datasets.py``
keeps using :mod:`tabular.baselines` directly.  When a clash hits, the
fix is mechanical:

* port the per-baseline adapter from :mod:`tabular.baselines` into a
  new ``VendoredXBaseline`` here (subclass of
  :class:`tabular.baselines.TabularBaselineBase`),
* wrap the upstream import inside
  :func:`_isolated_top_level_namespace`,
* register it in :data:`VENDORED_TABULAR_BASELINE_REGISTRY` so
  ``make_tabular_baseline("x", prefer_vendored=True)`` (or a future
  CLI flag) routes to it.

The two helpers below — :func:`_baseline_venv_site_packages` and
:func:`_isolated_top_level_namespace` — are direct ports of
``temporal.baselines_vendored._baseline_venv_site_packages`` and
``temporal.baselines_vendored._isolated_top_level_namespace``.  Keeping
the names identical means whoever maintains the temporal track already
knows the contract.
"""

from __future__ import annotations

import contextlib
import os
import sys
import sysconfig
from typing import Dict, Iterator, Optional

VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
PER_BASELINE_VENVS_DIR = os.path.join(VENDOR_DIR, ".venvs")


def _baseline_venv_site_packages(name: str) -> Optional[str]:
    """Return the ``site-packages`` path for ``tabular/vendor/.venvs/<name>/``.

    Returns ``None`` if the venv has not been provisioned yet — callers
    should treat that as "vendoring not available, fall back to the
    pip-installed import".
    """
    venv_root = os.path.join(PER_BASELINE_VENVS_DIR, name)
    if not os.path.isdir(venv_root):
        return None
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        os.path.join(venv_root, "lib", pyver, "site-packages"),
        os.path.join(venv_root, "Lib", "site-packages"),  # Windows layout
        sysconfig.get_path("purelib", vars={"base": venv_root}),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


@contextlib.contextmanager
def _isolated_top_level_namespace(
    top_level_names: list[str],
    *,
    baseline_name: Optional[str] = None,
) -> Iterator[None]:
    """Mount a per-baseline venv for the duration of a vendored import.

    On enter:
        * snapshot ``sys.path`` and the subset of ``sys.modules`` whose
          top-level names match ``top_level_names`` (so vendored deps
          can't poison the main namespace);
        * if ``baseline_name`` is given and the per-baseline venv exists,
          prepend its ``site-packages`` to ``sys.path``.

    On exit:
        * restore ``sys.path``;
        * remove any newly-imported modules whose top-level name is in
          ``top_level_names``.

    See :func:`temporal.baselines_vendored._isolated_top_level_namespace`
    for the original implementation; this is intentionally a strict
    subset because the tabular track does not yet have to juggle
    incompatible ``models``/``layers`` namespaces.
    """
    saved_path = list(sys.path)
    saved_modules: Dict[str, object] = {
        k: v for k, v in sys.modules.items()
        if any(k == n or k.startswith(n + ".") for n in top_level_names)
    }
    venv_sp = _baseline_venv_site_packages(baseline_name) if baseline_name else None
    if venv_sp is not None:
        sys.path.insert(0, venv_sp)
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for mod_name in list(sys.modules):
            if mod_name in saved_modules:
                continue
            if any(mod_name == n or mod_name.startswith(n + ".") for n in top_level_names):
                sys.modules.pop(mod_name, None)


# Empty registry — populated only when a per-baseline vendoring is needed.
VENDORED_TABULAR_BASELINE_REGISTRY: Dict[str, type] = {}


def has_vendored(name: str) -> bool:
    """Quick check whether a vendored adapter is registered for ``name``."""
    return name in VENDORED_TABULAR_BASELINE_REGISTRY
