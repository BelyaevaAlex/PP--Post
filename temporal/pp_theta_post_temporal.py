"""§4.4 — L4 PPθ-Post-Temporal: per-timestep branch latent ``z(b, X, t)``.

Architecture
------------
1. The input time-series tensor ``X_ts`` of shape ``[N, T, V]`` together
   with its observation ``mask`` is reshaped into ``N · T`` per-timestep
   snapshots of length ``2V`` (values concatenated with the mask).
2. An ``ExtraTreesClassifier`` is fitted on these snapshots, with the
   patient label replicated across all of its timesteps.  The resulting
   branches encode *instantaneous* clauses such as
   ``HR ≤ 90 ∧ mask(Lactate) = 1``.
3. The branch ensemble is converted into PPtheta-Post condition-aware
   rule activations.  At inference time we run ``branch_probs`` on every
   timestep snapshot and obtain a ``[N, T, B]`` tensor of per-timestep
   latent probabilities ``P(z(b, X, t))``.
4. A configurable temporal aggregation collapses the time dimension into
   a ``[N, B]`` matrix that is **directly compatible** with the nine
   PPθ-Post inference variants — only the way ``z`` is produced differs.

Aggregation modes
-----------------
``mean``      ``z_b = mean_t  P(z(b, x_t))``
``max``       ``z_b = max_t   P(z(b, x_t))``
``exists``    noisy-or over time:  ``1 − ∏_t (1 − P(z(b, x_t)))``
``forall``    all-active:           ``∏_t P(z(b, x_t))``
``k_of_t``    P(at least ``k`` timesteps active) via Lyapunov normal
              approximation of the Poisson-binomial CDF.
``last``      ``z_b = P(z(b, x_T))``  (last observation only).
``attention`` learned softmax weights ``α_t`` over time, applied as
              ``z_b = Σ_t α_t · P(z(b, x_t))``.  See
              :class:`TemporalAttentionAggregator`.

Notes
-----
* The same per-timestep condition-aware rule activation model is reused
  for explanations: the
  per-time matrix of latents is exposed via
  :py:meth:`TemporalRuleNetwork.predict_branch_probs_per_time` for
  attribution and visualisation.
* All operations are pure NumPy at inference time; PyTorch is only used
  internally for the neural prior path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier

from rule_network import RuleNetwork
from rule_network_model import RuleNetworkModel
from branch_schema import Branch


AggregationMode = str  # one of: mean, max, exists, forall, k_of_t, last, attention


VALID_AGGREGATIONS: Tuple[str, ...] = (
    "mean", "max", "exists", "forall", "k_of_t", "last", "attention",
)


# ─────────────────────────────────────────────────────────────────────────
# Aggregation utilities
# ─────────────────────────────────────────────────────────────────────────

def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def temporal_aggregate(
    z_per_time: np.ndarray,
    mode: AggregationMode = "mean",
    k: Optional[int] = None,
    attention_weights: Optional[np.ndarray] = None,
    top_k_time: Optional[int] = None,
) -> np.ndarray:
    """Collapse a per-timestep latent tensor into a per-sample latent.

    Parameters
    ----------
    z_per_time : np.ndarray
        Either ``[T, B]`` for a single sample or ``[N, T, B]`` for a batch.
    mode : str
        See module docstring for the list of supported modes.
    k : int, optional
        Required for ``mode == "k_of_t"``.  Interpreted as ``ceil(k_frac · T)``
        when ``0 < k <= 1``, otherwise as an absolute timestep count.
    attention_weights : np.ndarray, optional
        Required for ``mode == "attention"``.  Shape ``[T]`` (shared across
        branches) or ``[T, B]`` (per-branch / multi-head pooling).
    top_k_time : int, optional
        If provided, only the ``top_k_time`` most active timesteps per
        (sample, branch) are retained before aggregation; the remaining
        timesteps are zeroed.  This is the canonical fix for the
        existential / noisy-or-over-time saturation problem when ``T``
        is large (PL-tNoisyOr-topk variant).  Accepts a fractional value
        in ``(0, 1]``, interpreted as ``ceil(top_k_time · T)``.

    Returns
    -------
    np.ndarray of shape ``[B]`` or ``[N, B]``.
    """
    if mode not in VALID_AGGREGATIONS:
        raise ValueError(f"unknown temporal aggregation mode {mode!r}")
    arr = np.asarray(z_per_time, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]   # [1, T, B]
        squeeze = True
    elif arr.ndim == 3:
        squeeze = False
    else:
        raise ValueError("z_per_time must have shape [T, B] or [N, T, B]")
    N, T, B = arr.shape

    # ── Top-k-time filter ────────────────────────────────────────────
    # Zeroing low-activation timesteps mitigates noisy-or saturation
    # when many irrelevant timesteps would otherwise contribute small
    # but non-zero probabilities.
    if top_k_time is not None:
        k_t = (
            int(np.ceil(top_k_time * T))
            if (0.0 < top_k_time <= 1.0)
            else int(top_k_time)
        )
        k_t = max(1, min(k_t, T))
        if k_t < T:
            keep_idx = np.argpartition(-arr, k_t - 1, axis=1)[:, :k_t, :]
            mask = np.zeros_like(arr)
            np.put_along_axis(mask, keep_idx, 1.0, axis=1)
            arr = arr * mask

    if mode == "mean":
        out = arr.mean(axis=1)
    elif mode == "max":
        out = arr.max(axis=1)
    elif mode == "exists":
        log1m = np.log1p(-np.clip(arr, 0.0, 1.0 - 1e-12))
        out = 1.0 - np.exp(log1m.sum(axis=1))
    elif mode == "forall":
        log_p = np.log(np.clip(arr, 1e-12, 1.0))
        out = np.exp(log_p.sum(axis=1))
    elif mode == "k_of_t":
        if k is None:
            raise ValueError("k must be provided for mode='k_of_t'")
        k_abs = int(np.ceil(k * T)) if (0.0 < k <= 1.0) else int(k)
        k_abs = max(1, min(k_abs, T))
        mu = arr.sum(axis=1)                      # [N, B]
        var = (arr * (1.0 - arr)).sum(axis=1)
        sigma = np.sqrt(np.maximum(var, 1e-12))
        z = (mu - k_abs + 0.5) / sigma
        out = _normal_cdf(z)
    elif mode == "last":
        out = arr[:, -1, :]
    elif mode == "attention":
        if attention_weights is None:
            raise ValueError("attention_weights must be provided for mode='attention'")
        w = np.asarray(attention_weights, dtype=np.float64)
        if w.ndim == 1:
            # shared α ∈ R^T  →  broadcast to all samples and branches.
            if w.shape[0] != T:
                raise ValueError("attention_weights time dimension does not match")
            w_norm = w / max(w.sum(), 1e-12)
            out = (w_norm[None, :, None] * arr).sum(axis=1)
        elif w.ndim == 2:
            # per-branch α ∈ R^{T, B}  →  broadcast over samples only.
            if w.shape[0] != T or w.shape[1] != B:
                raise ValueError(
                    f"attention_weights shape {w.shape} != ({T}, {B})"
                )
            w_norm = w / np.clip(w.sum(axis=0, keepdims=True), 1e-12, None)
            out = (w_norm[None, :, :] * arr).sum(axis=1)
        else:
            raise ValueError("attention_weights must be 1-D or 2-D")
    else:
        raise AssertionError(mode)

    return out[0] if squeeze else out


# ─────────────────────────────────────────────────────────────────────────
# Per-timestep feature flattening (raw value + mask)
# ─────────────────────────────────────────────────────────────────────────

def _flatten_per_timestep(
    X_ts: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Convert ``[N, T, V]`` into per-timestep snapshots of shape
    ``[N · T, 2V]``.  NaNs in ``X_ts`` are replaced with ``0`` and the
    mask is concatenated so the trees can split on missingness.
    """
    if X_ts.ndim != 3 or mask.ndim != 3 or X_ts.shape != mask.shape:
        raise ValueError("X_ts and mask must both have shape [N, T, V]")
    N, T, V = X_ts.shape
    values = np.where(mask.astype(bool), X_ts, 0.0).astype(np.float32)
    snapshots = np.concatenate(
        [values.reshape(N * T, V), mask.reshape(N * T, V).astype(np.float32)],
        axis=1,
    )
    return snapshots


def _unflatten_per_timestep(
    flat: np.ndarray,
    n_samples: int,
    n_timesteps: int,
) -> np.ndarray:
    """Inverse of :func:`_flatten_per_timestep` — reshape predictions
    back into ``[N, T, ...]``."""
    return flat.reshape(n_samples, n_timesteps, *flat.shape[1:])


# ─────────────────────────────────────────────────────────────────────────
# Optional learnable temporal attention aggregator
# ─────────────────────────────────────────────────────────────────────────

class TemporalAttentionAggregator(torch.nn.Module):
    """Learnable softmax attention over time, optionally per-branch.

    Three configurations are supported:

    * ``mode="shared"`` (default) — a single weight vector
      ``α ∈ R^T`` is shared across all branches.  Equivalent to the
      original implementation.
    * ``mode="per_branch"`` — every branch has its own weight vector
      ``α_b ∈ R^T``.  Stored as ``[T, B]`` parameter; useful when
      different branches encode different temporal scales (e.g. one
      branch fires early on a deteriorating trajectory, another late).
    * ``mode="multi_head"`` — ``H`` shared heads ``α_h ∈ R^T`` are
      learned together with a head-mixing matrix ``M ∈ R^{H, B}`` that
      assigns each branch to a soft combination of heads.  Total
      parameter count is ``H · T + H · B`` instead of ``T · B`` for the
      per-branch case, so this is a parameter-efficient interpolation.

    The :meth:`weights` API returns a numpy array compatible with
    :func:`temporal_aggregate(mode="attention")`.  When the aggregator is
    operating in ``per_branch`` or ``multi_head`` mode the returned
    weights have shape ``[T, B]``; ``temporal_aggregate`` accepts both
    ``[T]`` (shared) and ``[T, B]`` (per-branch) layouts.
    """

    VALID_MODES = ("shared", "per_branch", "multi_head")

    def __init__(
        self,
        T: int,
        n_branches: Optional[int] = None,
        mode: str = "shared",
        n_heads: int = 4,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got {mode!r}"
            )
        self.T = T
        self.n_branches = n_branches
        self.mode = mode
        self.n_heads = n_heads

        if mode == "shared":
            self.logits = torch.nn.Parameter(torch.zeros(T))
        elif mode == "per_branch":
            if n_branches is None:
                raise ValueError("n_branches is required for mode='per_branch'")
            self.logits = torch.nn.Parameter(torch.zeros(T, n_branches))
        else:
            if n_branches is None:
                raise ValueError("n_branches is required for mode='multi_head'")
            self.head_logits = torch.nn.Parameter(torch.zeros(n_heads, T))
            self.mix_logits = torch.nn.Parameter(
                torch.zeros(n_heads, n_branches)
            )

    def _alpha(self) -> torch.Tensor:
        """Return the effective attention tensor of shape ``[T]`` (shared)
        or ``[T, B]`` (per-branch / multi-head)."""
        if self.mode == "shared":
            return torch.softmax(self.logits, dim=0)            # [T]
        if self.mode == "per_branch":
            return torch.softmax(self.logits, dim=0)            # [T, B]
        # multi_head: α_{t,b} = Σ_h softmax_b(M)_{h,b} · softmax_t(H)_{h,t}
        head_w = torch.softmax(self.head_logits, dim=1)         # [H, T]
        mix_w = torch.softmax(self.mix_logits, dim=0)           # [H, B]
        return torch.einsum("ht,hb->tb", head_w, mix_w)         # [T, B]

    def weights(self) -> np.ndarray:
        with torch.no_grad():
            return self._alpha().cpu().numpy()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, T, B]
        alpha = self._alpha()
        if alpha.dim() == 1:
            return (alpha[None, :, None] * z).sum(dim=1)
        # alpha: [T, B]
        return (alpha[None, :, :] * z).sum(dim=1)

    def fit(
        self,
        z_per_time: np.ndarray,
        theta: np.ndarray,
        y: np.ndarray,
        epochs: int = 300,
        lr: float = 0.05,
    ) -> np.ndarray:
        """Fit attention parameters so that
        ``softmax(weighted-mean over branches × θ)`` matches ``y``.
        Mirrors :func:`problog_inference.learn_branch_weights` but with
        a temporal aggregation that may be branch-specific.
        """
        z_t = torch.tensor(z_per_time, dtype=torch.float32)
        th_t = torch.tensor(theta, dtype=torch.float32)
        y_t = torch.tensor(y.ravel(), dtype=torch.long)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            agg = self(z_t)                       # [N, B]
            num = agg @ th_t                      # [N, K]
            num = torch.clamp(num, 1e-12, None)
            den = num.sum(dim=1, keepdim=True)
            p = num / den
            loss = torch.nn.functional.nll_loss(torch.log(p + 1e-15), y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self.weights()


# ─────────────────────────────────────────────────────────────────────────
# Main TemporalRuleNetwork
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PPThetaPostTemporal:
    """L4 PPθ-Post-Temporal classifier.

    Per-timestep condition-aware rule activation feeds a temporal
    aggregation that emits the
    per-sample latent ``z(b, X) ∈ R^B`` consumed by the standard PPθ-Post
    inference variants.  The class delegates per-snapshot computation to
    a vanilla :class:`RuleNetworkModel`, so the L4 extension is fully
    additive and does not modify the static ``RuleNetwork`` /
    ``rule_network_model`` modules.

    The legacy alias ``TemporalRuleNetwork`` is provided for backwards
    compatibility but is deprecated and will be removed in a future
    release.
    """

    var_names: Sequence[str]
    n_classes: int
    aggregation: AggregationMode = "mean"
    k: Optional[float] = None
    n_estimators: Optional[int] = None
    max_leaf_nodes: Optional[int] = None
    seed: int = 42
    device: Optional[str] = None
    epochs: int = 200
    learning_rate: float = 0.01

    forest: Optional[ExtraTreesClassifier] = None
    rule_network: Optional[RuleNetworkModel] = None
    branches: List[Branch] = None
    n_features_per_timestep: Optional[int] = None
    T: Optional[int] = None
    attention: Optional[TemporalAttentionAggregator] = None

    def __post_init__(self) -> None:
        if self.aggregation not in VALID_AGGREGATIONS:
            raise ValueError(
                f"unsupported aggregation {self.aggregation!r}; "
                f"choose from {VALID_AGGREGATIONS}"
            )

    # ── training ────────────────────────────────────────────
    def fit(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        x_val: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> "PPThetaPostTemporal":
        """Train the per-timestep ensemble + RuleNetwork on ``X_ts``.

        Parameters
        ----------
        X_ts, mask : np.ndarray of shape [N, T, V]
        y : np.ndarray of shape [N]
        x_val : optional tuple ``(X_val, mask_val, y_val)`` used as a
            validation set for the RuleNetwork's early stopping.  When
            ``None``, the training data itself is reused (no early stop).
        """
        if X_ts.ndim != 3 or mask.shape != X_ts.shape:
            raise ValueError("X_ts and mask must both have shape [N, T, V]")
        N, T, V = X_ts.shape
        self.T = T
        self.n_features_per_timestep = 2 * V

        snapshots = _flatten_per_timestep(X_ts, mask)
        y_repeat = np.repeat(y, T)

        log2_d = int(np.floor(np.log2(max(self.n_features_per_timestep, 2))))
        n_est = self.n_estimators or max(2, self.n_classes + log2_d)
        max_leaves = self.max_leaf_nodes or (2 ** (log2_d + 4))

        self.forest = ExtraTreesClassifier(
            n_estimators=n_est,
            max_leaf_nodes=max_leaves,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.forest.fit(snapshots, y_repeat)

        self.rule_network = RuleNetworkModel(device=self.device)
        self.rule_network.build_model_from_ensemble(self.forest)
        self.branches = list(self.rule_network.branches)

        if x_val is not None:
            X_val, mask_val, y_val = x_val
            val_snap = _flatten_per_timestep(X_val, mask_val)
            y_val_rep = np.repeat(y_val, X_val.shape[1])
        else:
            val_snap = snapshots
            y_val_rep = y_repeat

        self.rule_network.fit(
            snapshots.astype(np.float32),
            y_repeat.astype(np.int64),
            val_snap.astype(np.float32),
            y_val_rep.astype(np.int64),
            learning_rate=self.learning_rate,
            epochs=self.epochs,
        )
        return self

    # ── inference ───────────────────────────────────────────
    def predict_branch_probs_per_time(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Return the full per-timestep latent tensor of shape ``[N, T, B]``.

        Useful for explanations and L4 ProbLog export.
        """
        self._check_fitted()
        N, T, V = X_ts.shape
        if T != self.T:
            raise ValueError(
                f"input time dimension {T} != fitted T={self.T}"
            )
        snapshots = _flatten_per_timestep(X_ts, mask)
        bp = self.rule_network.predict_branch_proba(snapshots)
        bp = bp.detach().cpu().numpy() if isinstance(bp, torch.Tensor) else bp
        return _unflatten_per_timestep(bp, N, T)

    def predict_branch_probs(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
        aggregation: Optional[AggregationMode] = None,
        k: Optional[float] = None,
    ) -> np.ndarray:
        """Return aggregated per-sample latent of shape ``[N, B]``.

        Aggregation mode defaults to ``self.aggregation`` but may be
        overridden per call (useful for ablations and the comparison
        driver).
        """
        z_per_time = self.predict_branch_probs_per_time(X_ts, mask)
        mode = aggregation or self.aggregation
        attention_weights = None
        if mode == "attention":
            if self.attention is None:
                raise RuntimeError(
                    "attention mode requires fit_attention() to be called first"
                )
            attention_weights = self.attention.weights()
        return temporal_aggregate(
            z_per_time, mode=mode, k=k or self.k,
            attention_weights=attention_weights,
        )

    # ── auxiliary ───────────────────────────────────────────
    def fit_attention(
        self,
        X_ts: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        mode: str = "shared",
        n_heads: int = 4,
        epochs: int = 300,
        lr: float = 0.05,
    ) -> np.ndarray:
        """Fit the learnable attention pooling on the training set.

        Parameters
        ----------
        mode : str
            One of ``"shared"`` / ``"per_branch"`` / ``"multi_head"`` —
            see :class:`TemporalAttentionAggregator`.
        n_heads : int
            Number of attention heads when ``mode="multi_head"``.
        """
        self._check_fitted()
        z_per_time = self.predict_branch_probs_per_time(X_ts, mask)
        self.attention = TemporalAttentionAggregator(
            T=self.T,
            n_branches=len(self.branches),
            mode=mode,
            n_heads=n_heads,
        )
        return self.attention.fit(
            z_per_time=z_per_time,
            theta=theta,
            y=y,
            epochs=epochs,
            lr=lr,
        )

    def _check_fitted(self) -> None:
        if self.rule_network is None or self.branches is None:
            raise RuntimeError("PPThetaPostTemporal has not been fitted")


# Backwards-compatibility alias — deprecated, will be removed in a
# future release.  New code should use ``PPThetaPostTemporal`` directly.
TemporalRuleNetwork = PPThetaPostTemporal
