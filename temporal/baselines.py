"""§6 — External baselines for the PPθ-Post-Temporal comparison.

This module ships the local baseline track for shallow / canonical
models that have *no dedicated upstream repository* worth vendoring,
plus the standalone TabPFN-TS black-box row:

    * :class:`LRStatsBaseline`            — logistic regression on L1
      summary statistics (interpretable shallow baseline).
    * :class:`XGBStatsBaseline`           — XGBoost on L2 multi-window
      statistics (non-interpretable shallow baseline).
    * :class:`TabPFNTSBaseline`           — black-box TabPFN-TS temporal
      representation with a classifier head.
    * :class:`TransformerIMTSBaseline`    — vanilla Transformer encoder
      over per-timestep ``(value, mask)`` snapshots — a generic deep
      baseline distinct from any IMTS-specialised SOTA.

For the seven IMTS-specific SOTA baselines (GRU-D, SAnD, mTAN, SeFT,
Raindrop, CAMELOT, InterpGN) we exclusively use the **authors' original
code** as git submodules — see :mod:`temporal.baselines_vendored` (PyTorch
track) and :mod:`temporal.baselines_vendored_tf` (TensorFlow / Keras
track).  This guarantees a paper-faithful comparison and removes any
ambiguity about whose implementation is being evaluated.

All baselines speak the uniform contract::

    fit(X_ts: [N, T, V], mask: [N, T, V], y: [N], x_val=None) -> self
    predict_proba(X_ts, mask) -> np.ndarray [N, K]

so they slot directly into :mod:`temporal.compare_temporal` without
per-baseline glue code.
"""

from __future__ import annotations

import math
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from .tabularize import multi_window_flatten, summary_flatten  # noqa: E402
from .tabpfn_ts_distill import TabPFNTSClassifierTeacher  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Common interface
# ─────────────────────────────────────────────────────────────────────────

class BaselineBase(ABC):
    """Common (X_ts, mask, y) interface for every external baseline."""

    name: str = "base"
    needs_torch: bool = False

    @abstractmethod
    def fit(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        x_val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> "BaselineBase":
        ...

    @abstractmethod
    def predict_proba(
        self, X_ts: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        ...


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(arr), dtype=dtype, device=_device())


def _train_torch_model(
    model: torch.nn.Module,
    forward_fn,
    X_ts: np.ndarray, mask: np.ndarray, y: np.ndarray,
    x_val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    epochs: int, lr: float, batch_size: int, n_classes: int,
    weight_decay: float = 1e-5, patience: int = 10,
) -> torch.nn.Module:
    """Generic training loop with cross-entropy, Adam optimiser and an
    optional held-out early-stopping signal.

    ``forward_fn(model, X_ts_batch, mask_batch)`` must return logits of
    shape ``[B, n_classes]``.
    """
    device = _device()
    model.to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    X_ts_t = _to_tensor(X_ts)
    mask_t = _to_tensor(mask)
    y_t = _to_tensor(y, dtype=torch.long)
    n = X_ts_t.shape[0]

    best_val = math.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            logits = forward_fn(model, X_ts_t[idx], mask_t[idx])
            loss = F.cross_entropy(logits, y_t[idx])
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()

        if x_val is not None:
            model.eval()
            with torch.no_grad():
                logits_v = forward_fn(
                    model, _to_tensor(x_val[0]), _to_tensor(x_val[1]),
                )
                v_loss = F.cross_entropy(
                    logits_v, _to_tensor(x_val[2], dtype=torch.long),
                ).item()
            if v_loss < best_val - 1e-4:
                best_val = v_loss
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > patience:
                    break

    model.load_state_dict(best_state)
    return model


def _predict_torch_proba(
    model: torch.nn.Module,
    forward_fn,
    X_ts: np.ndarray, mask: np.ndarray, batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    X_ts_t = _to_tensor(X_ts)
    mask_t = _to_tensor(mask)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, X_ts_t.shape[0], batch_size):
            logits = forward_fn(
                model, X_ts_t[s:s + batch_size], mask_t[s:s + batch_size],
            )
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


# ─────────────────────────────────────────────────────────────────────────
# Shallow baselines
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class LRStatsBaseline(BaselineBase):
    """Logistic regression on L1 summary statistics."""
    n_classes: int = 2
    seed: int = 42
    name: str = "LR-stats"

    def __post_init__(self):
        self._scaler: Optional[StandardScaler] = None
        self._clf: Optional[LogisticRegression] = None

    def fit(self, X_ts, mask, y, x_val=None):
        X = summary_flatten(X_ts, mask)
        self._scaler = StandardScaler().fit(X)
        self._clf = LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=self.seed,
        ).fit(self._scaler.transform(X), y)
        return self

    def predict_proba(self, X_ts, mask):
        X = summary_flatten(X_ts, mask)
        proba = self._clf.predict_proba(self._scaler.transform(X))
        if proba.shape[1] < self.n_classes:
            full = np.zeros((proba.shape[0], self.n_classes))
            for i, c in enumerate(self._clf.classes_):
                full[:, int(c)] = proba[:, i]
            return full
        return proba


@dataclass
class XGBStatsBaseline(BaselineBase):
    """XGBoost on L2 multi-window statistics."""
    n_classes: int = 2
    n_windows: int = 4
    seed: int = 42
    name: str = "XGB-stats"

    def __post_init__(self):
        self._clf = None

    def fit(self, X_ts, mask, y, x_val=None):
        import xgboost as xgb
        X = multi_window_flatten(X_ts, mask, n_windows=self.n_windows)
        objective = "binary:logistic" if self.n_classes == 2 else "multi:softprob"
        params = dict(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=self.seed, n_jobs=-1, objective=objective,
            tree_method="hist",
        )
        if self.n_classes > 2:
            params["num_class"] = self.n_classes
        self._clf = xgb.XGBClassifier(**params).fit(X, y)
        return self

    def predict_proba(self, X_ts, mask):
        X = multi_window_flatten(X_ts, mask, n_windows=self.n_windows)
        proba = self._clf.predict_proba(X)
        if proba.shape[1] < self.n_classes:
            full = np.zeros((proba.shape[0], self.n_classes))
            for i, c in enumerate(self._clf.classes_):
                full[:, int(c)] = proba[:, i]
            return full
        return proba


@dataclass
class TabPFNTSBaseline(BaselineBase):
    """Standalone black-box TabPFN-TS baseline.

    TabPFN-TS is a forecasting model, not a direct classifier.  This
    baseline therefore evaluates the black-box temporal teacher itself:
    TabPFN-TS representation features are fed to a classifier head
    (TabPFN by default), with no PPtheta-Post rule extraction.
    """

    n_classes: int = 2
    seed: int = 42
    ts_backend: str = "tabpfn_ts"
    ts_max_rows: int = 4096
    ts_model_path: Optional[str] = None
    ts_device: str = "cpu"
    ts_n_estimators: int = 8
    ts_num_workers: int = 1
    head: str = "tabpfn"
    classifier_model_path: Optional[str] = None
    classifier_device: str = "cpu"
    classifier_n_estimators: int = 8
    name: str = "TabPFN-TS"

    def __post_init__(self):
        self._teacher: Optional[TabPFNTSClassifierTeacher] = None

    def fit(self, X_ts, mask, y, x_val=None):
        self._teacher = TabPFNTSClassifierTeacher(
            n_classes=self.n_classes,
            seed=self.seed,
            ts_backend=self.ts_backend,
            ts_max_rows=self.ts_max_rows,
            ts_model_path=self.ts_model_path,
            ts_device=self.ts_device,
            ts_n_estimators=self.ts_n_estimators,
            ts_num_workers=self.ts_num_workers,
            head=self.head,
            classifier_model_path=self.classifier_model_path,
            classifier_device=self.classifier_device,
            classifier_n_estimators=self.classifier_n_estimators,
        ).fit(X_ts, mask, y)
        backend = self._teacher.ts_backend_used
        head = self._teacher.head_used_ or self.head
        self.name = f"TabPFN-TS-{backend}-{head}"
        return self

    def predict_proba(self, X_ts, mask):
        if self._teacher is None:
            raise RuntimeError("TabPFNTSBaseline must be .fit() first")
        return self._teacher.predict_proba(X_ts, mask)


# ─────────────────────────────────────────────────────────────────────────
# Transformer-IMTS — vanilla Transformer encoder over (value, mask)
# snapshots.  This is *not* an IMTS-specialised SOTA — it is a generic
# deep-learning baseline used to gauge the gap between a naive Transformer
# and the irregular-time-series-aware methods in the vendored track.
# ─────────────────────────────────────────────────────────────────────────

class _PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _TransformerIMTSNet(torch.nn.Module):
    def __init__(
        self, n_vars: int, n_classes: int, d_model: int = 64,
        n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = torch.nn.Linear(2 * n_vars, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = torch.nn.Linear(d_model, n_classes)

    def forward(self, X_ts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # X_ts: [B, T, V]; mask: [B, T, V]
        snap = torch.cat([torch.nan_to_num(X_ts, nan=0.0), mask], dim=-1)
        x = self.pos(self.in_proj(snap))
        # ignore padding-style timesteps where the entire mask row is zero
        all_missing = (mask.sum(dim=-1) == 0)
        # avoid having "all timesteps masked out" → forward fails; keep at least one True
        all_missing[:, 0] = False
        x = self.encoder(x, src_key_padding_mask=all_missing)
        valid = (~all_missing).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


@dataclass
class TransformerIMTSBaseline(BaselineBase):
    """Vanilla Transformer encoder over per-timestep (value, mask)
    snapshots — no irregular-time handling beyond the mask channel.
    """
    n_classes: int = 2
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    epochs: int = 60
    lr: float = 1e-3
    batch_size: int = 64
    dropout: float = 0.1
    seed: int = 42
    name: str = "Transformer-IMTS"
    needs_torch: bool = True

    def __post_init__(self):
        self._model: Optional[torch.nn.Module] = None

    @staticmethod
    def _forward(model, X_ts, mask):
        return model(X_ts, mask)

    def fit(self, X_ts, mask, y, x_val=None):
        torch.manual_seed(self.seed)
        n_vars = X_ts.shape[-1]
        self._model = _TransformerIMTSNet(
            n_vars=n_vars, n_classes=self.n_classes, d_model=self.d_model,
            n_heads=self.n_heads, n_layers=self.n_layers, dropout=self.dropout,
        )
        _train_torch_model(
            self._model, self._forward, X_ts, mask, y, x_val,
            epochs=self.epochs, lr=self.lr, batch_size=self.batch_size,
            n_classes=self.n_classes,
        )
        return self

    def predict_proba(self, X_ts, mask):
        return _predict_torch_proba(self._model, self._forward, X_ts, mask)


# ═════════════════════════════════════════════════════════════════════════
# Registry
# ═════════════════════════════════════════════════════════════════════════
#
# Only local baselines for which we have no dedicated upstream
# repository worth vendoring live here, plus TabPFN-TS.  All
# IMTS-specialised SOTA baselines (GRU-D, SAnD, mTAN, SeFT, Raindrop,
# CAMELOT, InterpGN) are
# served from :mod:`temporal.baselines_vendored` (PyTorch) and
# :mod:`temporal.baselines_vendored_tf` (TensorFlow).
# ═════════════════════════════════════════════════════════════════════════

BASELINE_REGISTRY = {
    "lr":          LRStatsBaseline,
    "xgb":         XGBStatsBaseline,
    "tabpfn_ts":   TabPFNTSBaseline,
    "transformer": TransformerIMTSBaseline,
}

DEFAULT_BASELINES = tuple(BASELINE_REGISTRY.keys())


def make_baseline(name: str, n_classes: int, **kwargs) -> BaselineBase:
    """Instantiate a baseline by registry key with a uniform signature."""
    key = name.lower()
    if key not in BASELINE_REGISTRY:
        raise KeyError(
            f"unknown baseline {name!r}; "
            f"choose from {sorted(BASELINE_REGISTRY)}"
        )
    return BASELINE_REGISTRY[key](n_classes=n_classes, **kwargs)
