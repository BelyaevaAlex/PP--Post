"""§6 — Vendored adapters that wrap **authors' original** baseline code
from ``temporal/vendor/<name>``.

Each vendored repository is included as a git submodule (see
``temporal/vendor/README.md`` for licences and pinned commit hashes).
This module imports each upstream model lazily and wraps it in a thin
:class:`BaselineBase`-compatible adapter so it slots directly into the
fold/seed loop of :mod:`temporal.compare_temporal`.

Under the *only-vendored* SOTA strategy this module is the **only**
source of truth for the five PyTorch-based IMTS SOTA baselines — GRU-D,
SAnD, mTAN, Raindrop and InterpGN — and there is no re-implementation
fallback.  If a particular machine cannot import an upstream repo
(e.g. Raindrop without CUDA + ``torch_geometric``), the driver logs a
``[skipped]`` entry instead of silently substituting our own code.

Import isolation
----------------
Several upstream repos use *unprefixed* top-level package names
(``models``, ``utils``, ``layers``).  In particular, ``vendor/mtan/src``
ships a ``models.py`` *file* whose presence on ``sys.path`` makes
``from models.InterpGN import ...`` raise ``'models' is not a package``.
:func:`_isolated_top_level_namespace` snapshots and restores
``sys.path`` + ``sys.modules`` around each problematic import so
co-loaded vendored repos do not stomp on each other.

Caveats
-------
* **Raindrop** depends on ``torch_geometric`` and hard-codes ``.cuda()``
  in the forward pass; on a CPU-only machine the adapter raises a clear
  ``RuntimeError`` pointing at ``temporal/vendor/raindrop/README.md``.
* **InterpGN (vendored 2025 version)** uses a Shapelet-Bottleneck head
  fronted by an FCN (or another DNN) backbone.  Default configuration
  is built by :func:`default_interpgn_configs` and mirrors the official
  ``reproduce/run_uea.sh`` script.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import (  # noqa: E402  reuse helpers, do not duplicate
    BaselineBase,
    _device,
    _predict_torch_proba,
    _to_tensor,
    _train_torch_model,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(THIS_DIR, "vendor")
PER_BASELINE_VENVS_DIR = os.path.join(VENDOR_DIR, ".venvs")


# ═════════════════════════════════════════════════════════════════════════
# Per-baseline virtualenv discovery
# ═════════════════════════════════════════════════════════════════════════
#
# Each vendored baseline can ship its transitive Python dependencies
# in a *dedicated* virtualenv at ``temporal/vendor/.venvs/<name>/``,
# created by ``python -m temporal.vendor_extras.setup_envs <name>``
# from the manifest in ``temporal/vendor_extras/<name>.txt``.
#
# At runtime, ``_isolated_top_level_namespace`` prepends that venv's
# ``site-packages`` directory to ``sys.path`` for the duration of a
# baseline's import.  This means:
#
#   * exotic deps (``reformer_pytorch`` for InterpGN, TF for SeFT /
#     CAMELOT, ``torch_geometric`` for Raindrop) are installed only
#     for the baseline that needs them — the main PPθ-Post env stays
#     clean;
#   * different baselines can pin incompatible versions of the same
#     package without stepping on each other (the per-baseline venv
#     is isolated on entry, restored on exit);
#   * the bootstrap is fully optional — if a venv is not present, the
#     adapter falls back to importing from the main environment, and
#     compare_temporal.py logs ``[skipped]`` cleanly when a transitive
#     dependency is missing there too.

def _baseline_venv_site_packages(name: str) -> Optional[str]:
    """Return the ``site-packages`` path for a per-baseline venv at
    ``temporal/vendor/.venvs/<name>/``, or ``None`` if no such venv
    has been provisioned yet.

    Called by :func:`_isolated_top_level_namespace`; usually you do
    not need to call this directly.
    """
    venv = os.path.join(PER_BASELINE_VENVS_DIR, name)
    if not os.path.isdir(venv):
        return None
    if os.name == "nt":
        candidate = os.path.join(venv, "Lib", "site-packages")
        return candidate if os.path.isdir(candidate) else None
    lib = os.path.join(venv, "lib")
    if not os.path.isdir(lib):
        return None
    for entry in sorted(os.listdir(lib)):
        if entry.startswith("python"):
            sp = os.path.join(lib, entry, "site-packages")
            if os.path.isdir(sp):
                return sp
    return None


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

def _ensure_submodule(name: str) -> str:
    """Return the absolute path to a vendored submodule directory or raise
    a helpful error if it has not been cloned (``git submodule update --init``)."""
    sub = os.path.join(VENDOR_DIR, name)
    if not os.path.isdir(sub) or not os.listdir(sub):
        raise RuntimeError(
            f"vendored submodule '{name}' is not initialised; "
            f"run `git submodule update --init --recursive` from the "
            f"PPθ-Post repository root."
        )
    return sub


def _load_module_from(path: str, mod_name: str):
    """Load a single .py file as an importable module under ``mod_name``."""
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _isolated_top_level_namespace(
    extra_path: str,
    conflict_top_levels: Iterable[str] = (),
    baseline_name: Optional[str] = None,
):
    """Temporarily expose a vendored submodule (and, optionally, its
    per-baseline virtualenv) on ``sys.path`` while isolating a fixed
    set of top-level package names (e.g. ``models``, ``utils``,
    ``layers``) so they cannot clash with names already registered
    by another co-loaded vendored repo.

    Why this exists
    ---------------
    Several upstream repos in ``temporal/vendor`` use *unprefixed*
    package names — most painfully ``mTAN`` (``vendor/mtan/src/models.py``
    is a *file*, not a package) and ``InterpGN``
    (``vendor/interpgn/models/`` is a PEP-420 namespace package).
    Once mTAN is imported, ``sys.modules['models']`` points at the
    mTAN file and ``vendor/mtan/src`` stays on ``sys.path``.  Even if
    we evict the cached module, Python's finder still discovers the
    plain ``models.py`` *before* the namespace package and a subsequent
    ``from models.InterpGN import ...`` raises::

        ModuleNotFoundError: No module named 'models.InterpGN';
        'models' is not a package

    To get a clean slate we therefore:

    1. Snapshot + clear conflicting entries in ``sys.modules``.
    2. Snapshot + drop any sibling ``temporal/vendor/<other>`` path
       from ``sys.path`` (so e.g. mTAN's ``models.py`` cannot be found
       while we're resolving InterpGN's ``models/`` namespace package).
    3. Optionally prepend the per-baseline venv's ``site-packages``
       directory to ``sys.path`` so transitive dependencies installed
       via :mod:`temporal.vendor_extras.setup_envs` become visible
       *only* inside the context.
    4. Pin ``extra_path`` (the vendored submodule root) to the front
       of ``sys.path``.

    On exit both ``sys.path`` and the conflicting-namespace slice of
    ``sys.modules`` are restored to their pre-context state, so the
    *next* adapter is not affected by leftover bindings from this one.
    The model object built inside the context survives unaffected
    because all of its transitive imports have already been resolved
    by the time we exit (no lazy ``import`` inside ``forward()``).

    Parameters
    ----------
    extra_path
        Directory pushed to the front of ``sys.path`` for the duration
        of the context (typically the vendored submodule root).
    conflict_top_levels
        Top-level module names that are known to clash between vendored
        repos (e.g. ``("models", "utils", "layers")``).  Both the bare
        names and any ``<name>.<sub>...`` children are evicted from
        ``sys.modules`` on enter and restored on exit.  Default:
        ``()`` — no namespace eviction; useful for adapters whose
        upstream uses prefixed package names (e.g. ``import sand.X``).
    baseline_name
        If given, the function looks up
        ``temporal/vendor/.venvs/<baseline_name>/.../site-packages``
        via :func:`_baseline_venv_site_packages` and prepends it to
        ``sys.path``.  When the venv has not been provisioned yet,
        the parameter is silently ignored and the adapter falls back
        to the main Python environment.
    """

    conflict = tuple(conflict_top_levels)
    vendor_root = os.path.realpath(VENDOR_DIR)
    extra_path = os.path.abspath(extra_path)
    venv_site_packages = (
        _baseline_venv_site_packages(baseline_name)
        if baseline_name is not None else None
    )

    def _is_conflict(mod_name: str) -> bool:
        return mod_name in conflict or any(
            mod_name.startswith(c + ".") for c in conflict
        )

    def _is_sibling_vendor_path(p: str) -> bool:
        # Any path that lives under temporal/vendor/* but is NOT the
        # extra_path (or its descendants) we want to expose.  Note:
        # the per-baseline venv at ``vendor/.venvs/<name>/`` is also
        # under ``vendor/``, but is added back explicitly below.
        try:
            real = os.path.realpath(p)
        except OSError:
            return False
        kept_root = os.path.realpath(extra_path)
        return (
            real.startswith(vendor_root + os.sep)
            and real != kept_root
            and not real.startswith(kept_root + os.sep)
        )

    saved_path = sys.path[:]
    saved_modules = {
        k: v for k, v in sys.modules.items() if _is_conflict(k)
    }
    for k in list(sys.modules):
        if _is_conflict(k):
            del sys.modules[k]
    sys.path[:] = [p for p in sys.path if not _is_sibling_vendor_path(p)]
    # The order here matters: extra_path goes *first* so the vendored
    # submodule wins over anything else (including its own venv);
    # then the per-baseline site-packages goes second so transitive
    # deps of that submodule resolve into the per-baseline venv;
    # main-env packages (already on sys.path) are still reachable as
    # a third tier — that's how main PyTorch / NumPy stay shared.
    if extra_path in sys.path:
        sys.path.remove(extra_path)
    if venv_site_packages is not None:
        if venv_site_packages in sys.path:
            sys.path.remove(venv_site_packages)
        sys.path.insert(0, venv_site_packages)
    sys.path.insert(0, extra_path)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for k in list(sys.modules):
            if _is_conflict(k):
                del sys.modules[k]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        importlib.invalidate_caches()


# ═════════════════════════════════════════════════════════════════════════
# 1. SAnD (khirotaka/SAnD) — clean PyTorch model, MIT
# ═════════════════════════════════════════════════════════════════════════

def _build_vendored_sand(input_features, seq_len, n_heads, factor,
                        n_classes, n_layers, d_model, dropout):
    _ensure_submodule("sand")
    # Upstream uses ``from ..core import modules`` so we need ``sand``
    # to be importable as a package (PEP-420 namespace package over
    # ``temporal/vendor/``).  We do NOT need to isolate top-level
    # names because every internal reference is prefixed with
    # ``sand.`` — but we still go through the context manager so the
    # per-baseline venv (if provisioned) gets attached.
    with _isolated_top_level_namespace(
        extra_path=VENDOR_DIR,
        baseline_name="sand",
    ):
        sand_model_module = importlib.import_module("sand.core.model")
        SAnDClass = sand_model_module.SAnD
        return SAnDClass(
            input_features=input_features, seq_len=seq_len,
            n_heads=n_heads, factor=factor, n_class=n_classes,
            n_layers=n_layers, d_model=d_model, dropout_rate=dropout,
        )


@dataclass
class VendoredSAnDBaseline(BaselineBase):
    """Adapter around ``temporal/vendor/sand/core/model.SAnD`` (Song 2018)."""

    n_classes: int = 2
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    factor: int = 4
    dropout: float = 0.2
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 42
    name: str = "SAnD (vendored)"
    needs_torch: bool = True

    def __post_init__(self):
        self._model: Optional[torch.nn.Module] = None

    @staticmethod
    def _forward(model, X_ts, mask):
        # SAnD upstream expects [B, T, V]; we concatenate the mask channel
        # to keep informative-missingness signal (mask doubled into V dim).
        x = torch.cat([torch.nan_to_num(X_ts, nan=0.0), mask], dim=-1)
        return model(x)

    def fit(self, X_ts, mask, y, x_val=None):
        torch.manual_seed(self.seed)
        seq_len = X_ts.shape[1]
        input_features = 2 * X_ts.shape[2]
        self._model = _build_vendored_sand(
            input_features=input_features, seq_len=seq_len,
            n_heads=self.n_heads, factor=self.factor,
            n_classes=self.n_classes, n_layers=self.n_layers,
            d_model=self.d_model, dropout=self.dropout,
        )
        _train_torch_model(
            self._model, self._forward, X_ts, mask, y, x_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
            n_classes=self.n_classes,
        )
        return self

    def predict_proba(self, X_ts, mask):
        return _predict_torch_proba(
            self._model, self._forward, X_ts, mask, batch_size=32,
        )


# ═════════════════════════════════════════════════════════════════════════
# 2. mTAN (reml-lab/mTAN) — clean PyTorch model, MIT
# ═════════════════════════════════════════════════════════════════════════

def _build_vendored_mtan(input_dim, n_ref, nhidden, embed_time, num_heads,
                         device, n_classes):
    sub = _ensure_submodule("mtan")
    src = os.path.join(sub, "src")
    # mTAN's ``src/models.py`` is a top-level *file* (not a package),
    # so it conflicts with InterpGN's ``models/`` namespace package
    # if both end up on sys.path simultaneously.  We isolate the
    # ``models`` / ``utils`` top-level names while we resolve the
    # mTAN encoder; on exit, the per-baseline venv (if provisioned)
    # detaches and main-env packages remain unchanged.
    with _isolated_top_level_namespace(
        extra_path=src,
        conflict_top_levels=("models", "utils"),
        baseline_name="mtan",
    ):
        mtan_models = importlib.import_module("models")
        query = torch.linspace(0.0, 1.0, n_ref, device=device)
        enc = mtan_models.enc_mtan_classif(
            input_dim=input_dim, query=query, nhidden=nhidden,
            embed_time=embed_time, num_heads=num_heads,
            learn_emb=True, freq=10.0, device=str(device),
        )
    # The upstream classifier hard-codes 2 output classes; replace the
    # final ``Linear`` layer to fit our ``n_classes`` if needed.
    if n_classes != 2:
        last = enc.classifier[-1]
        enc.classifier[-1] = torch.nn.Linear(last.in_features, n_classes)
    return enc


@dataclass
class VendoredMTANBaseline(BaselineBase):
    """Adapter around ``temporal/vendor/mtan/src/models.enc_mtan_classif``."""

    n_classes: int = 2
    n_ref: int = 8
    nhidden: int = 32
    embed_time: int = 16
    n_heads: int = 2
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 42
    name: str = "mTAN (vendored)"
    needs_torch: bool = True

    def __post_init__(self):
        self._model: Optional[torch.nn.Module] = None

    @staticmethod
    def _forward(model, X_ts, mask):
        # enc_mtan_classif expects ``x`` of shape [B, T, 2*V] (values
        # concatenated with the mask) and a parallel ``time_steps``
        # tensor [B, T] giving timestamps.
        B, T, V = X_ts.shape
        device = X_ts.device
        time_steps = torch.linspace(0.0, 1.0, T, device=device).expand(B, T)
        x = torch.cat([torch.nan_to_num(X_ts, nan=0.0), mask], dim=-1)
        return model(x, time_steps)

    def fit(self, X_ts, mask, y, x_val=None):
        torch.manual_seed(self.seed)
        device = _device()
        self._model = _build_vendored_mtan(
            input_dim=X_ts.shape[2], n_ref=self.n_ref, nhidden=self.nhidden,
            embed_time=self.embed_time, num_heads=self.n_heads,
            device=device, n_classes=self.n_classes,
        )
        _train_torch_model(
            self._model, self._forward, X_ts, mask, y, x_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
            n_classes=self.n_classes,
        )
        return self

    def predict_proba(self, X_ts, mask):
        return _predict_torch_proba(
            self._model, self._forward, X_ts, mask, batch_size=32,
        )


# ═════════════════════════════════════════════════════════════════════════
# 3. GRU-D (zhiyongc/GRU-D) — single-file model, no LICENSE (submodule only)
# ═════════════════════════════════════════════════════════════════════════

class _VendoredGRUDClassifier(torch.nn.Module):
    """Wraps the upstream ``GRUD`` cell with a classification head.

    The upstream ``GRUD.forward`` returns the *last* hidden state if
    ``output_last=True``; we add a linear classifier on top.
    """

    def __init__(self, n_vars, n_classes, hidden_size, X_mean):
        super().__init__()
        sub = _ensure_submodule("grud")
        # GRU-D upstream is a single-file model (``GRUD.py``).  We
        # isolate the bare ``GRUD`` top-level name in case any other
        # vendored repo grew a same-named module in the future, and
        # also pull in the per-baseline venv (if provisioned) so any
        # transitive deps don't leak.
        with _isolated_top_level_namespace(
            extra_path=sub,
            conflict_top_levels=("GRUD",),
            baseline_name="grud",
        ):
            grud_module = importlib.import_module("GRUD")
            GRUDClass = grud_module.GRUD
            self.cell = GRUDClass(
                input_size=n_vars, cell_size=hidden_size,
                hidden_size=hidden_size, X_mean=X_mean, output_last=True,
            )
        self.head = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, x4d):
        # x4d: [B, 4, T, V] — channels: [X, X_last_obsv, Mask, Delta]
        h = self.cell(x4d)
        return self.head(h)


def _grud_pack_fourchannel(X_ts: np.ndarray, mask: np.ndarray,
                           X_mean_per_feature: np.ndarray) -> np.ndarray:
    """Build the ``[B, 4, T, V]`` tensor expected by the upstream GRU-D cell.

    Channels in order: ``X``, ``X_last_obsv``, ``Mask``, ``Delta``
    (matching the indices in ``GRUD.forward``).

    Parameters
    ----------
    X_mean_per_feature : np.ndarray of shape [V]
        Per-feature mean used to fill the ``X_last_obsv`` channel before
        the first observation of each variable.
    """
    B, T, V = X_ts.shape
    X_filled = np.where(mask > 0.5, X_ts, 0.0).astype(np.float32)
    X_last = np.zeros((B, T, V), dtype=np.float32)
    delta = np.zeros((B, T, V), dtype=np.float32)

    # Per-sample, per-feature: last observed value (init: feature mean)
    # and last observation timestamp (init: -1 = never seen).
    last_value = np.broadcast_to(
        X_mean_per_feature.astype(np.float32), (B, V),
    ).copy()
    last_seen = np.full((B, V), -1.0, dtype=np.float32)
    for t in range(T):
        m_t = mask[:, t, :] > 0.5
        # Delta: time since last observation in the SAME feature; 0 for
        # the very first observed step, and accumulates for missing.
        prev_delta = delta[:, t - 1, :] if t > 0 else np.zeros((B, V), dtype=np.float32)
        delta_t = np.where(
            last_seen >= 0,
            (t - last_seen).astype(np.float32),
            np.zeros((B, V), dtype=np.float32),
        )
        delta_t = np.where(m_t, np.zeros_like(delta_t),
                           delta_t + np.where(m_t, np.zeros_like(prev_delta), prev_delta))
        delta[:, t, :] = delta_t
        X_last[:, t, :] = last_value
        last_value = np.where(m_t, X_filled[:, t, :], last_value)
        last_seen = np.where(
            m_t, np.full_like(last_seen, float(t)), last_seen,
        )
    out = np.stack(
        [X_filled, X_last, mask.astype(np.float32), delta], axis=1,
    )
    return out.astype(np.float32)


@dataclass
class VendoredGRUDBaseline(BaselineBase):
    """Adapter around ``temporal/vendor/grud/GRUD.GRUD`` (Che 2018).

    .. note::

       The upstream zhiyongc/GRU-D implementation ties the per-feature
       decay vector ``γ_h ∈ R^V`` to the hidden-state dimension; thus
       ``hidden_size`` must equal ``n_vars``.  When the user supplies
       a different ``hidden_size`` we silently override it to the
       feature count and emit a warning.
    """

    n_classes: int = 2
    hidden_size: Optional[int] = None     # auto-set to n_vars
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 42
    name: str = "GRU-D (vendored)"
    needs_torch: bool = True

    def __post_init__(self):
        self._model: Optional[torch.nn.Module] = None
        self._X_mean: Optional[np.ndarray] = None

    @staticmethod
    def _forward(model, x4d, _unused):
        return model(x4d)

    def fit(self, X_ts, mask, y, x_val=None):
        torch.manual_seed(self.seed)
        T, n_vars = X_ts.shape[1], X_ts.shape[2]
        # Compute per-(timestep, feature) mean for the upstream model's
        # ``X_mean`` parameter (shape [1, T, V]) — this is what the
        # upstream cell uses when imputing missing values along the
        # sequence.
        with np.errstate(invalid="ignore"):
            X_with_nan = np.where(mask.astype(bool), X_ts, np.nan)
            X_mean_tv = np.nan_to_num(np.nanmean(X_with_nan, axis=0), nan=0.0)
            X_mean_v = np.nan_to_num(np.nanmean(X_with_nan, axis=(0, 1)), nan=0.0)
        self._X_mean = X_mean_tv[None, :, :]                    # [1, T, V]
        self._X_mean_v = X_mean_v.astype(np.float32)            # [V]
        # zhiyongc/GRU-D requires hidden_size == n_vars (per-feature decay).
        hidden = n_vars if self.hidden_size is None else self.hidden_size
        if hidden != n_vars:
            import warnings
            warnings.warn(
                f"VendoredGRUDBaseline: hidden_size={hidden} does not "
                f"match n_vars={n_vars}; the upstream zhiyongc/GRU-D "
                f"implementation only supports hidden_size==n_vars. "
                f"Overriding to {n_vars}.",
                stacklevel=2,
            )
            hidden = n_vars
        self._model = _VendoredGRUDClassifier(
            n_vars=n_vars, n_classes=self.n_classes,
            hidden_size=hidden, X_mean=self._X_mean,
        )
        x4d = _grud_pack_fourchannel(X_ts, mask, self._X_mean_v)
        dummy_mask = np.zeros((X_ts.shape[0], 1), dtype=np.float32)
        _train_torch_model(
            self._model, self._forward, x4d, dummy_mask, y, x_val=None,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
            n_classes=self.n_classes,
        )
        return self

    def predict_proba(self, X_ts, mask):
        x4d = _grud_pack_fourchannel(X_ts, mask, self._X_mean_v)
        dummy_mask = np.zeros((X_ts.shape[0], 1), dtype=np.float32)
        return _predict_torch_proba(
            self._model, self._forward, x4d, dummy_mask, batch_size=32,
        )


# ═════════════════════════════════════════════════════════════════════════
# 4. Raindrop (mims-harvard/Raindrop) — torch_geometric, CUDA-only upstream
# ═════════════════════════════════════════════════════════════════════════

def _patch_os_for_raindrop_import() -> None:
    """Make ``models_rd`` importable on non-Windows machines.

    Upstream ``temporal/vendor/raindrop/code/models_rd.py`` calls
    ``os.add_dll_directory(...)`` at module-import time, but that
    function only exists on Windows ≥ Python 3.8.  We monkey-patch a
    no-op shim on POSIX so the import succeeds; PyTorch + PyG do the
    real CUDA discovery via ``LD_LIBRARY_PATH`` / ``CUDA_HOME``.
    """
    if not hasattr(os, "add_dll_directory"):
        os.add_dll_directory = lambda *_a, **_k: None        # type: ignore[attr-defined]


def _can_import_raindrop() -> Tuple[bool, str]:
    """Soft check: requires ``torch_geometric`` install only.

    The upstream code hard-codes ``.cuda()`` calls in the forward pass,
    so the model only runs on CUDA-capable hardware in practice — but
    we don't pre-check CUDA at adapter init time so the user can build
    the adapter on a CPU box (e.g. for code-tour / dry-run / unit
    tests).  ``fit`` is what actually allocates tensors on GPU.
    """
    if importlib.util.find_spec("torch_geometric") is None:
        return False, ("Raindrop requires ``torch_geometric``; install via "
                       "``pip install torch-geometric``.")
    return True, ""


def _build_vendored_raindrop(d_inp, d_static, max_len, d_model, n_heads,
                             n_layers, n_classes, dropout, global_structure):
    sub = _ensure_submodule("raindrop")
    code = os.path.join(sub, "code")
    _patch_os_for_raindrop_import()
    # Raindrop's ``code/`` exposes ``models_rd``, ``utils_rd`` and
    # ``baselines/`` as top-level imports; isolate the ones that
    # collide with InterpGN / mTAN, and attach the per-baseline venv
    # (which carries ``torch_geometric`` and friends).
    with _isolated_top_level_namespace(
        extra_path=code,
        conflict_top_levels=("models_rd", "utils_rd", "baselines"),
        baseline_name="raindrop",
    ):
        models_rd = importlib.import_module("models_rd")
        Raindrop_v2 = models_rd.Raindrop_v2
        return Raindrop_v2(
            d_inp=d_inp, d_model=d_model, nhead=n_heads, nhid=2 * d_model,
            nlayers=n_layers, dropout=dropout, max_len=max_len,
            d_static=d_static, MAX=100, n_classes=n_classes,
            global_structure=global_structure, sensor_wise_mask=False,
            static=False,
        )


def _make_raindrop_global_structure(n_vars: int) -> torch.Tensor:
    """Default sensor-graph adjacency: identity (self-edges only).

    Upstream computes a cosine-similarity-from-train-set adjacency
    (`generate_global_structure(...)` in `Raindrop.py`); here we let
    the driver re-train per fold and pass an identity matrix so the
    learnable propagation is responsible for picking up the structure.
    """
    return torch.eye(n_vars)


@dataclass
class VendoredRaindropBaseline(BaselineBase):
    """Adapter around ``temporal/vendor/raindrop/code/models_rd.Raindrop_v2``.

    Tested on A100 (CUDA + ``torch_geometric``).  Upstream hard-codes
    ``.cuda()`` so a CUDA device is required at fit time; ``__post_init__``
    only checks for the ``torch_geometric`` install so the adapter can
    be constructed for unit tests on CPU-only machines.
    """

    n_classes: int = 2
    d_model: int = 36          # divisible by n_heads + by d_inp by upstream design
    n_heads: int = 2
    n_layers: int = 2
    dropout: float = 0.3
    epochs: int = 30
    lr: float = 1e-4
    batch_size: int = 32
    seed: int = 42
    name: str = "Raindrop (vendored)"
    needs_torch: bool = True

    def __post_init__(self):
        self._model: Optional[torch.nn.Module] = None
        ok, why = _can_import_raindrop()
        if not ok:
            raise RuntimeError(
                f"vendored Raindrop unavailable: {why} See "
                "temporal/vendor/README.md."
            )

    @staticmethod
    def _forward(model, X_ts, mask):
        # Upstream API: src=[T, B, 2*V] (values + missing_mask concat),
        # static=[B, d_static] (we pass dummy zeros), times=[B, T],
        # lengths=[B] (number of non-zero rows per sample).
        B, T, V = X_ts.shape
        device = X_ts.device
        src = torch.cat(
            [torch.nan_to_num(X_ts, nan=0.0), mask], dim=-1,
        ).permute(1, 0, 2).contiguous()                        # [T, B, 2V]
        times = torch.linspace(0.0, 1.0, T, device=device).expand(B, T)
        # Lengths: count timesteps that have ANY observation.
        lengths = (mask.sum(dim=-1) > 0).sum(dim=-1).clamp(min=1)
        static = torch.zeros((B, 0), device=device)             # static=False
        out, _, _ = model(src, static, times, lengths)
        return out

    def fit(self, X_ts, mask, y, x_val=None):
        torch.manual_seed(self.seed)
        n_vars = X_ts.shape[2]
        # d_model must be divisible by n_heads AND by n_vars (upstream
        # constraint: ``self.d_ob = d_model // d_inp``, integer).
        if self.d_model % self.n_heads != 0 or self.d_model % n_vars != 0:
            raise ValueError(
                f"VendoredRaindropBaseline: d_model={self.d_model} must "
                f"be divisible by both n_heads={self.n_heads} and "
                f"n_vars={n_vars}; pick a multiple."
            )
        global_structure = _make_raindrop_global_structure(n_vars)
        if torch.cuda.is_available():
            global_structure = global_structure.cuda()
        self._model = _build_vendored_raindrop(
            d_inp=n_vars, d_static=0, max_len=X_ts.shape[1],
            d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, n_classes=self.n_classes,
            dropout=self.dropout, global_structure=global_structure,
        )
        _train_torch_model(
            self._model, self._forward, X_ts, mask, y, x_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
            n_classes=self.n_classes,
        )
        return self

    def predict_proba(self, X_ts, mask):
        return _predict_torch_proba(
            self._model, self._forward, X_ts, mask, batch_size=32,
        )


# ═════════════════════════════════════════════════════════════════════════
# 5. InterpGN (YunshiWen/InterpretGatedNetwork) — Shapelet head, no LICENSE
# ═════════════════════════════════════════════════════════════════════════

def default_interpgn_configs(
    seq_len: int, enc_in: int, num_class: int,
    dnn_type: str = "FCN", num_shapelet: int = 10,
    lambda_div: float = 0.1, lambda_reg: float = 0.1,
    epsilon: float = 1.0, dropout: float = 0.0,
    sbm_cls: str = "linear", distance_func: str = "euclidean",
    beta_schedule: str = "constant",
    memory_efficient: bool = False,
):
    """Build the default ``configs`` Namespace expected by the vendored
    InterpGN forward.

    Defaults match the values in the official reproduce script
    ``temporal/vendor/interpgn/reproduce/run_uea.sh``:

    * model = ``InterpGN``, dnn_type = ``FCN``
    * num_shapelet = 10, lambda_div = 0.1, lambda_reg = 0.1
    * epsilon = 1.0, beta_schedule = ``constant``, gating_value = 1

    Parameters
    ----------
    seq_len : int     — number of timesteps in the input grid.
    enc_in  : int     — number of input variables (channels).
    num_class : int   — number of output classes.
    """
    from argparse import Namespace
    return Namespace(
        # Bottleneck / shapelet config
        seq_len=seq_len, enc_in=enc_in, num_class=num_class,
        num_shapelet=num_shapelet, lambda_div=lambda_div,
        lambda_reg=lambda_reg, epsilon=epsilon, dropout=dropout,
        sbm_cls=sbm_cls, distance_func=distance_func,
        beta_schedule=beta_schedule, memory_efficient=memory_efficient,
        # DNN backbone
        dnn_type=dnn_type,
        # Bookkeeping fields the upstream code reads but does not
        # always require for the FCN backbone.
        e_layers=2, d_model=64, n_heads=4, d_ff=256,
        factor=1, activation="gelu", embed="fixed",
        freq="h", patch_len=8, stride=4,
        top_k=5, num_kernels=6, c_out=num_class,
        moving_avg=25, output_attention=False,
    )


def _build_vendored_interpgn(configs):
    """Instantiate ``vendor/interpgn/models/InterpGN.InterpGN`` without
    polluting the global import namespace.

    Upstream uses bare top-level packages — ``models``, ``utils`` and
    ``layers`` — which clash with mTAN (``vendor/mtan/src/models.py``)
    and any other co-loaded submodule that re-uses those names.  We
    therefore wrap the whole import sequence in
    :func:`_isolated_top_level_namespace`, so the vendored ``models``
    package is visible *only* while we resolve ``models.InterpGN`` and
    its transitive dependencies; once the model object is constructed,
    the previous bindings (including any mTAN-specific ``models``
    module) are restored and the next adapter sees a clean slate.

    Resolving the class through :func:`importlib.import_module` instead
    of ``from models.InterpGN import InterpGN`` keeps the import strictly
    local: no symbol leaks into this file's globals, so a second call —
    e.g. across CV folds — re-runs the isolation cleanly.
    """
    sub = _ensure_submodule("interpgn")
    with _isolated_top_level_namespace(
        extra_path=sub,
        conflict_top_levels=("models", "utils", "layers"),
        baseline_name="interpgn",
    ):
        interpgn_module = importlib.import_module("models.InterpGN")
        InterpGNClass = interpgn_module.InterpGN
        return InterpGNClass(configs)


@dataclass
class VendoredInterpGNBaseline(BaselineBase):
    """Adapter around ``temporal/vendor/interpgn/models/InterpGN.InterpGN``
    (Wen et al., AAAI 2025).

    Default configuration mirrors the official UEA reproduce script
    (``--model InterpGN --dnn_type FCN --num_shapelet 10
    --lambda_div 0.1 --lambda_reg 0.1 --epsilon 1 --beta_schedule
    constant --gating_value 1``).  Pass ``configs=...`` to override.

    The upstream model returns ``(logits, ModelInfo)`` where
    ``ModelInfo.eta`` is the per-sample gating fraction; we expose
    ``mean_gate_routing`` for paper §6.5 reporting.
    """

    n_classes: int = 2
    dnn_type: str = "FCN"
    num_shapelet: int = 10
    lambda_div: float = 0.1
    lambda_reg: float = 0.1
    epsilon: float = 1.0
    epochs: int = 60
    lr: float = 5e-3
    batch_size: int = 32
    seed: int = 42
    configs: Optional[object] = None
    name: str = "InterpGN (vendored)"
    needs_torch: bool = True

    def __post_init__(self):
        self._model: Optional[torch.nn.Module] = None
        self._mean_gate: float = 0.5
        sub = os.path.join(VENDOR_DIR, "interpgn")
        if not os.path.isdir(sub) or not os.listdir(sub):
            raise RuntimeError(
                "vendored InterpGN unavailable: submodule not initialised. "
                "Run `git submodule update --init` then retry."
            )

    @staticmethod
    def _forward(model, X_ts, mask):
        # Upstream expects [B, T, V]; we concatenate the missingness
        # mask along the channel axis so the convolutional backbone has
        # access to the missingness signal too.
        x = torch.cat([torch.nan_to_num(X_ts, nan=0.0), mask], dim=-1)
        out = model(x)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0]
        return out

    def fit(self, X_ts, mask, y, x_val=None):
        torch.manual_seed(self.seed)
        if self.configs is None:
            self.configs = default_interpgn_configs(
                seq_len=X_ts.shape[1],
                enc_in=2 * X_ts.shape[2],          # values + mask channels
                num_class=self.n_classes,
                dnn_type=self.dnn_type,
                num_shapelet=self.num_shapelet,
                lambda_div=self.lambda_div,
                lambda_reg=self.lambda_reg,
                epsilon=self.epsilon,
            )
        self._model = _build_vendored_interpgn(self.configs)
        _train_torch_model(
            self._model, self._forward, X_ts, mask, y, x_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
            n_classes=self.n_classes,
        )
        # Compute mean gating fraction for §6.5 caveat reporting.
        self._model.eval()
        with torch.no_grad():
            x_full = torch.cat([
                _to_tensor(np.nan_to_num(X_ts, nan=0.0)),
                _to_tensor(mask),
            ], dim=-1)
            out = self._model(x_full)
            if isinstance(out, tuple) and len(out) == 2:
                _, info = out
                self._mean_gate = float(info.eta.mean().item())
        return self

    def predict_proba(self, X_ts, mask):
        return _predict_torch_proba(
            self._model, self._forward, X_ts, mask, batch_size=32,
        )

    @property
    def mean_gate_routing(self) -> float:
        """Average ``eta`` (fraction routed to the deep DNN path)."""
        return self._mean_gate


# ═════════════════════════════════════════════════════════════════════════
# Registry + fallback driver
# ═════════════════════════════════════════════════════════════════════════

VENDORED_REGISTRY = {
    "sand":      VendoredSAnDBaseline,
    "mtan":      VendoredMTANBaseline,
    "gru_d":     VendoredGRUDBaseline,
    "raindrop":  VendoredRaindropBaseline,
    "interp_gn": VendoredInterpGNBaseline,
}


def make_vendored(name: str, n_classes: int, **kwargs) -> BaselineBase:
    """Instantiate a vendored PyTorch adapter by registry key.

    Raises :class:`RuntimeError` if the underlying submodule or
    optional extras are missing.  Since the SOTA baselines no longer
    have a re-implementation fallback, callers should catch this and
    skip the offending row in the comparison report.
    """
    key = name.lower()
    if key not in VENDORED_REGISTRY:
        raise KeyError(
            f"unknown vendored baseline {name!r}; "
            f"choose from {sorted(VENDORED_REGISTRY)}"
        )
    return VENDORED_REGISTRY[key](n_classes=n_classes, **kwargs)
