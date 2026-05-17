"""TensorFlow / Keras vendored baseline adapters.

Wraps the **authors' original TensorFlow / Keras** code shipped as
``git submodules`` under ``temporal/vendor/``:

    * ``seft``    → :class:`VendoredSeFTBaseline`     (Horn et al., ICML 2020)
    * ``camelot`` → :class:`VendoredCAMELOTBaseline`  (Aguiar et al., ICML 2022)

Design contract
---------------
* All adapters subclass :class:`temporal.baselines.BaselineBase` so they
  speak the uniform ``fit(X_ts, mask, y) → predict_proba(X_ts, mask)``
  contract used by :mod:`temporal.compare_temporal`.
* TensorFlow itself is heavy and Apple-Silicon-fragile; we therefore
  import it **lazily** inside the per-baseline isolated context manager
  (:func:`temporal.baselines_vendored._isolated_top_level_namespace`).
  When :mod:`temporal.vendor_extras.setup_envs` has been run for the
  baseline, TF resolves out of
  ``temporal/vendor/.venvs/<name>/.../site-packages`` and never enters
  the main PPθ-Post environment.  If TF is unavailable in either
  location, :func:`_lazy_import_tf` raises :class:`RuntimeError` with
  bootstrap instructions and the comparison driver emits a single
  ``[skipped]`` row.
* The authors' source lives in ``temporal/vendor/<name>``; we never copy
  or fork it.  Each adapter mounts the upstream layout via the shared
  context manager so the original imports
  (``import seft.models.deep_set_attention`` /
  ``import src.models.camelot.model``) work without modification.

This file is the *TensorFlow track*; the *PyTorch track* lives in
:mod:`temporal.baselines_vendored`.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .baselines import BaselineBase
# Re-use the per-baseline venv discovery + isolated namespace machinery
# from the PyTorch track so all seven vendored adapters share one
# implementation of "find the venv, prepend its site-packages,
# isolate top-level conflicts, restore on exit".
from .baselines_vendored import (
    _baseline_venv_site_packages,
    _isolated_top_level_namespace,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(THIS_DIR, "vendor")


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _ensure_submodule(name: str) -> str:
    """Return the absolute path to ``temporal/vendor/<name>`` and verify it
    has been initialised; raise a clear :class:`RuntimeError` otherwise."""
    path = os.path.join(VENDOR_DIR, name)
    if not (os.path.isdir(path) and os.listdir(path)):
        raise RuntimeError(
            f"git submodule ``temporal/vendor/{name}`` is not initialised; "
            f"run ``git submodule update --init --recursive``."
        )
    return path


def _lazy_import_tf(baseline_name: Optional[str] = None):
    """Import TensorFlow lazily.

    If ``baseline_name`` is provided and the per-baseline virtualenv at
    ``temporal/vendor/.venvs/<baseline_name>/`` has been provisioned by
    :mod:`temporal.vendor_extras.setup_envs`, that venv's
    ``site-packages`` is prepended to ``sys.path`` *before* the import
    so TF resolves out of the dedicated env (TF 2.10–2.15 in our
    curated manifest), keeping the main PPθ-Post environment free of
    a heavy TF install.

    Raises
    ------
    RuntimeError
        If TensorFlow is not importable from any of the searched
        locations (per-baseline venv, then main env).  The driver
        translates this into a single ``[skipped]`` row.
    """
    venv_sp = _baseline_venv_site_packages(baseline_name) if baseline_name else None
    if venv_sp is not None and venv_sp not in sys.path:
        sys.path.insert(0, venv_sp)
    if importlib.util.find_spec("tensorflow") is None:
        raise RuntimeError(
            "tensorflow is not installed; vendored TF baselines (SeFT, "
            "CAMELOT) require TensorFlow ≥ 2.10.  Bootstrap a per-baseline "
            "venv with ``python -m temporal.vendor_extras.setup_envs "
            f"{baseline_name or '<seft|camelot>'}`` or install TF in the "
            "main environment."
        )
    import tensorflow as tf  # type: ignore
    return tf


def _to_one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return out


# ─────────────────────────────────────────────────────────────────────────
# SeFT — Set Functions for Time Series (Horn et al., ICML 2020)
# Upstream: https://github.com/BorgwardtLab/Set_Functions_for_Time_Series
# ─────────────────────────────────────────────────────────────────────────

def _seft_pack_input(
    X_ts: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Convert ``(X_ts, mask)`` into the
    ``(demo, times, values, measurements, lengths)`` triplet
    representation expected by ``DeepSetAttentionModel``.

    * ``demo``         : shape ``[B, 1]`` (single dummy demographic)
    * ``times``        : shape ``[B, L]`` of normalised timestamps
    * ``values``       : shape ``[B, L, 1]`` of observed scalar values
    * ``measurements`` : shape ``[B, L]`` of integer variable indices
    * ``lengths``      : shape ``[B]`` of per-sample triplet counts

    Padded to ``L = max number of observations`` across the batch with
    zeros; the upstream code masks invalid positions via ``lengths``.
    """
    B, T, V = X_ts.shape
    triplets = []
    max_len = 0
    for b in range(B):
        m = mask[b].astype(bool)
        t_idx, v_idx = np.nonzero(m)
        if t_idx.size == 0:
            t_idx = np.array([0], dtype=np.int64)
            v_idx = np.array([0], dtype=np.int64)
        vals = np.nan_to_num(X_ts[b, t_idx, v_idx], nan=0.0).astype(np.float32)
        t_norm = t_idx.astype(np.float32) / max(T - 1, 1)
        triplets.append((t_norm, v_idx.astype(np.int32), vals))
        max_len = max(max_len, t_idx.size)

    demo = np.zeros((B, 1), dtype=np.float32)
    times = np.zeros((B, max_len), dtype=np.float32)
    values = np.zeros((B, max_len, 1), dtype=np.float32)
    measurements = np.zeros((B, max_len), dtype=np.int32)
    lengths = np.zeros((B,), dtype=np.int32)
    for b, (t, v, x) in enumerate(triplets):
        n = t.shape[0]
        times[b, :n] = t
        measurements[b, :n] = v
        values[b, :n, 0] = x
        lengths[b] = n
    return demo, times, values, measurements, lengths


def _build_vendored_seft(n_classes: int, n_vars: int):
    """Construct a default :class:`DeepSetAttentionModel` with the
    upstream-default hyper-parameters and the modality count fixed to
    ``n_vars``.

    Defaults follow ``DeepSetAttentionModel.get_default(task)`` from the
    upstream repository (n_phi_layers=3, phi_width=32, n_heads=4, etc.).
    """
    sub = _ensure_submodule("seft")
    # SeFT exposes ``seft.<sub>`` packages under the prefix ``seft.``, so
    # there's no top-level name conflict with the other vendored repos.
    # We still go through the context manager to mount the per-baseline
    # venv's site-packages (TF 2.x lives in there for SeFT).
    with _isolated_top_level_namespace(
        extra_path=sub,
        baseline_name="seft",
    ):
        tf = _lazy_import_tf("seft")
        deep_set = importlib.import_module("seft.models.deep_set_attention")
        DeepSetAttentionModel = deep_set.DeepSetAttentionModel

        output_activation = "softmax" if n_classes > 1 else "sigmoid"
        model = DeepSetAttentionModel(
            output_activation=output_activation,
            output_dims=n_classes,
            n_phi_layers=3,
            phi_width=32,
            n_psi_layers=2,
            psi_width=64,
            psi_latent_width=128,
            dot_prod_dim=128,
            n_heads=4,
            attn_dropout=0.1,
            latent_width=128,
            phi_dropout=0.0,
            n_rho_layers=3,
            rho_width=32,
            rho_dropout=0.0,
            max_timescale=100.0,
            n_positional_dims=4,
        )
        model._n_modalities = n_vars
    return model, tf


@dataclass
class VendoredSeFTBaseline(BaselineBase):
    """Vendored adapter for ``BorgwardtLab/Set_Functions_for_Time_Series``.

    Uses the upstream :class:`DeepSetAttentionModel` with its **default**
    hyper-parameters (3 φ-layers / 32-wide, 4 attention heads,
    128-d latent), modality count set to ``X_ts.shape[-1]``.

    The adapter only wires Keras compile / fit / predict; the more
    elaborate training routine in ``seft.training_routine`` (Polynomial
    LR schedule, warm-up, hyper-parameter search) is intentionally not
    plugged in — we follow PPθ-Post's uniform training budget instead.
    """

    n_classes: int = 2
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 42
    name: str = "SeFT (vendored)"
    needs_torch: bool = False
    _model: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Touch lazy imports so the driver can fall back fast if missing.
        _ensure_submodule("seft")
        _lazy_import_tf("seft")

    def fit(self, X_ts, mask, y, x_val=None):
        np.random.seed(self.seed)
        n_vars = X_ts.shape[-1]
        model, tf = _build_vendored_seft(self.n_classes, n_vars)
        tf.random.set_seed(self.seed)

        demo, times, values, measurements, lengths = _seft_pack_input(X_ts, mask)
        y_oh = _to_one_hot(np.asarray(y), self.n_classes)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            (demo, times, values, measurements, lengths),
            y_oh,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            shuffle=True,
        )
        self._model = model
        return self

    def predict_proba(self, X_ts, mask):
        if self._model is None:
            raise RuntimeError("VendoredSeFTBaseline.fit must be called first")
        demo, times, values, measurements, lengths = _seft_pack_input(X_ts, mask)
        proba = self._model.predict(
            (demo, times, values, measurements, lengths),
            batch_size=max(32, self.batch_size),
            verbose=0,
        )
        return np.asarray(proba, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────
# CAMELOT — Aguiar et al., ICML 2022
# Upstream: https://github.com/hrna-ox/camelot-icml
# ─────────────────────────────────────────────────────────────────────────

def _camelot_pack_input(X_ts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Concatenate ``X_ts`` (NaN→0) and ``mask`` along feature axis to
    produce a dense ``[B, T, 2*V]`` tensor — the input format expected
    by the upstream :class:`CAMELOT` ``call()``."""
    X_filled = np.nan_to_num(X_ts, nan=0.0).astype(np.float32)
    m = mask.astype(np.float32)
    return np.concatenate([X_filled, m], axis=-1)


def _build_vendored_camelot(n_classes: int, input_shape: Tuple[int, int, int]):
    """Construct a default upstream :class:`CAMELOT` model using the
    parameters from ``camelot_default_config.json`` (10 clusters,
    32-d latent, dropout 0.6, regulariser 0.01)."""
    sub = _ensure_submodule("camelot")
    # CAMELOT's upstream uses absolute ``src.models.camelot.model``
    # imports; the bare top-level name ``src`` could clash with anything
    # else exposing a ``src/`` directory.  We isolate ``src`` for the
    # duration of the import and mount the per-baseline venv (TF 2.x +
    # tslearn live there for CAMELOT).
    with _isolated_top_level_namespace(
        extra_path=sub,
        conflict_top_levels=("src",),
        baseline_name="camelot",
    ):
        tf = _lazy_import_tf("camelot")
        camelot_module = importlib.import_module("src.models.camelot.model")
        CAMELOT = camelot_module.CAMELOT

        model = CAMELOT(
            num_clusters=10,
            latent_dim=32,
            output_dim=n_classes,
            seed=4347,
            alpha=0.01,
            beta=0.01,
            regulariser_params=(0.01, 0.01),
            dropout=0.6,
            encoder_params={"hidden_layers": 1, "hidden_nodes": 20},
            identifier_params={"hidden_layers": 1, "hidden_nodes": 20},
            predictor_params={"hidden_layers": 1, "hidden_nodes": 20},
            cluster_rep_lr=0.001,
            optimizer_init="adam",
            weighted_loss=True,
        )
        model.build(input_shape)
    return model, tf


@dataclass
class VendoredCAMELOTBaseline(BaselineBase):
    """Vendored adapter for ``hrna-ox/camelot-icml``.

    Wraps the upstream :class:`CAMELOT` Keras model with the parameters
    from its ``camelot_default_config.json``.  We use the *plain*
    ``compile`` / ``fit`` route — the ``initialise_model`` (KMeans-based
    cluster representation pre-training) used by the upstream
    ``Model.fit`` driver is **omitted** because it depends on a richer
    train/val/test data dictionary; if you want literature-faithful
    CAMELOT numbers, switch to the upstream ``Model.fit`` driver
    directly.

    Caveat: the upstream model is sensitive to cluster initialisation
    and to a custom train_step that interleaves Predictor / Encoder /
    cluster-rep gradient updates; we keep ``run_eagerly=True`` to avoid
    XLA-related crashes when ``train_step`` is exercised.
    """

    n_classes: int = 2
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 42
    name: str = "CAMELOT (vendored)"
    needs_torch: bool = False
    _model: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        _ensure_submodule("camelot")
        _lazy_import_tf("camelot")

    def fit(self, X_ts, mask, y, x_val=None):
        np.random.seed(self.seed)
        X = _camelot_pack_input(X_ts, mask)
        y_oh = _to_one_hot(np.asarray(y), self.n_classes)

        model, tf = _build_vendored_camelot(self.n_classes, X.shape)
        tf.random.set_seed(self.seed)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            run_eagerly=True,
        )
        model.fit(
            X, y_oh,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            shuffle=True,
        )
        self._model = model
        return self

    def predict_proba(self, X_ts, mask):
        if self._model is None:
            raise RuntimeError("VendoredCAMELOTBaseline.fit must be called first")
        X = _camelot_pack_input(X_ts, mask)
        proba = self._model.predict(
            X,
            batch_size=max(32, self.batch_size),
            verbose=0,
        )
        return np.asarray(proba, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────
# Registry / factory
# ─────────────────────────────────────────────────────────────────────────

VENDORED_TF_REGISTRY = {
    "seft":    VendoredSeFTBaseline,
    "camelot": VendoredCAMELOTBaseline,
}


def make_vendored_tf(name: str, n_classes: int, **kwargs) -> BaselineBase:
    """Instantiate a vendored TensorFlow baseline by registry key.

    Raises :class:`RuntimeError` if TensorFlow is unavailable or the
    submodule has not been initialised.
    """
    key = name.lower()
    if key not in VENDORED_TF_REGISTRY:
        raise KeyError(
            f"unknown vendored TF baseline {name!r}; choose from "
            f"{sorted(VENDORED_TF_REGISTRY)}"
        )
    cls = VENDORED_TF_REGISTRY[key]
    return cls(n_classes=n_classes, **kwargs)
