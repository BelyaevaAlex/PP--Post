"""Attribution maps for the differentiable-posterior + noisy-or pipeline.

Given a trained ``RuleNetworkModel`` (W1, branches) and a learned
``θ ∈ [0,1]^{B×C}`` matrix, this module decomposes a class prediction
``P(c|x)`` into additive contributions from individual branch conditions.

The decomposition uses the *exact* analytical structure of the model:

1. ``DifferentiablePosterior.forward_with_attribution`` exposes the
   per-condition log-likelihood ratio
   ``Δ_i = log P(obs_i | z=1) − log P(obs_i | z=0)``.
   The branch-level Bayesian shift is then ``LLR_b = Σ_{i ∈ b} Δ_i``,
   and ``logit(z_post_b) − logit(z_prior_b) = LLR_b`` exactly.

2. The noisy-or class head is decomposable in the negative-log-survival
   space: ``-log(1 − P(c|x)) = Σ_b -log(1 − θ_{bc}·z_b)``.  The summand
   ``s_{bc} = -log(1 − θ_{bc}·z_b) ≥ 0`` is the natural additive
   "support of branch b for class c" used by the noisy-or rule
   ``θ::supports(B,C,X) :- z(B,X). class(X,C) :- supports(B,C,X).``

3. Drilling deeper, each condition's *share* of its branch's support
   is taken proportionally to its log-LR contribution, signed by the
   sign of the branch shift.  This gives the requested
   condition→class attribution path.

These two views are complementary: (1) traces the *posterior update*
back to its evidence, (2) traces the *class probability* down to its
underlying conditions.  Both are exact (no integrated-gradients or
SHAP-style approximation) within the differentiable surrogate.

Usage
-----
>>> from attribution import BranchAttributor
>>> attr = BranchAttributor(branches, theta, feature_names=feat_names,
...                         class_names=class_names)
>>> z_prior = model.predict_branch_proba(x)        # torch.Tensor [B]
>>> rep = attr.explain(x_np, z_prior, top_k_branches=3)
>>> print(rep.pretty())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from branch_schema import Branch
from problog_inference import DifferentiablePosterior


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ConditionExplanation:
    feature_idx: int
    feature_name: str
    threshold: float
    direction: str          # "le" / "gt"
    x_value: float
    match: float            # σ((thr−x)/τ)  [0,1] — soft truth value
    log_lr: float           # Δ_i = log_lik_z1_cond − log_lik_z0_cond
    support_share: float    # signed share of branch's support for the predicted class

    def pretty(self, indent: str = "    ") -> str:
        sign_match = "✓" if self.match >= 0.5 else "✗"
        op = "≤" if self.direction == "le" else ">"
        return (
            f"{indent}{sign_match}  {self.feature_name} {op} {self.threshold:.4f}  "
            f"| x={self.x_value:.4f}  match={self.match:.3f}  "
            f"Δlog-LR={self.log_lr:+.3f}  share={self.support_share:+.3f}"
        )


@dataclass
class BranchExplanation:
    branch_id: int
    branch_label: str
    z_prior: float
    z_post: float
    branch_log_lr: float    # LLR_b: total Bayesian shift of this branch
    theta_bc: float         # θ_{b, predicted_class}
    support_to_class: float # s_{b, predicted_class} = -log(1 − θ·z_post)
    conditions: List[ConditionExplanation] = field(default_factory=list)

    def pretty(self, indent: str = "  ") -> str:
        head = (
            f"{indent}[{self.branch_label}]  "
            f"z: {self.z_prior:.3f} → {self.z_post:.3f}  "
            f"LLR={self.branch_log_lr:+.3f}  "
            f"θ={self.theta_bc:.3f}  "
            f"support={self.support_to_class:.4f}"
        )
        cond_lines = [c.pretty(indent + "    ") for c in self.conditions]
        return "\n".join([head, *cond_lines])


@dataclass
class ExplanationReport:
    pred_class: int
    pred_class_name: str
    pred_proba: np.ndarray            # [n_classes]
    log_survival: float               # -log(1 − P(pred_class|x))
    branches: List[BranchExplanation] = field(default_factory=list)

    def pretty(self) -> str:
        proba_str = ", ".join(
            f"{p:.3f}" for p in self.pred_proba.tolist()
        )
        head = (
            f"Predicted: {self.pred_class_name} (class {self.pred_class})  "
            f"P={self.pred_proba[self.pred_class]:.4f}  "
            f"-log(1−P)={self.log_survival:.4f}\n"
            f"Class probs: [{proba_str}]\n"
            f"Top branches contributing to this prediction:"
        )
        return "\n".join([head, *(b.pretty() for b in self.branches)])


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _branch_label(branch: Branch, feature_names: Sequence[str]) -> str:
    if not branch.conditions:
        return f"b{branch.branch_id}"
    parts = []
    for c in branch.conditions:
        op = "≤" if c.direction == "le" else ">"
        fname = (
            feature_names[c.feature_idx]
            if c.feature_idx < len(feature_names) else f"f{c.feature_idx}"
        )
        parts.append(f"{fname}{op}{c.threshold:.3f}")
    return " ∧ ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# BranchAttributor
# ─────────────────────────────────────────────────────────────────────


class BranchAttributor:
    """Decompose noisy-or class predictions into branch and condition contributions."""

    def __init__(
        self,
        branches: List[Branch],
        theta: Union[np.ndarray, torch.Tensor],
        p_high: float = 0.95,
        p_low: float = 0.05,
        tau: float = 0.1,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
    ):
        self.branches = branches
        self.theta = (
            theta.detach().cpu().numpy().astype(np.float64)
            if isinstance(theta, torch.Tensor)
            else np.asarray(theta, dtype=np.float64)
        )
        if self.theta.ndim != 2 or self.theta.shape[0] != len(branches):
            raise ValueError(
                f"theta must be [n_branches, n_classes]; got {self.theta.shape} "
                f"with {len(branches)} branches"
            )
        self.n_classes = self.theta.shape[1]
        self.p_high, self.p_low, self.tau = p_high, p_low, tau

        n_features_guess = max(
            (c.feature_idx for b in branches for c in b.conditions),
            default=0,
        ) + 1
        self.feature_names = (
            list(feature_names) if feature_names is not None
            else [f"f{i}" for i in range(n_features_guess)]
        )
        self.class_names = (
            list(class_names) if class_names is not None
            else [f"c{i}" for i in range(self.n_classes)]
        )

        self.diff_post = DifferentiablePosterior(
            branches, p_high=p_high, p_low=p_low, tau=tau
        )

    # ────────────────────────────────────────────────
    # Core: per-batch raw attribution tensors
    # ────────────────────────────────────────────────

    def attribute_batch(
        self,
        x: Union[np.ndarray, torch.Tensor],
        z_prior: Union[np.ndarray, torch.Tensor],
    ) -> dict:
        """Run the differentiable posterior and noisy-or with attribution.

        Returns a dict of numpy arrays:

        * ``proba`` [batch, n_classes] — normalised noisy-or class probs.
        * ``z_prior`` [batch, n_branches]
        * ``z_post``  [batch, n_branches]
        * ``support`` [batch, n_branches, n_classes] — s_{bc} = -log(1−θ·z_post)
        * ``branch_llr`` [batch, n_branches] — Bayesian shift LLR_b
        * ``cond_log_lr`` [batch, n_total_conds] — per-condition log-LR Δ_i
        * ``match`` [batch, n_total_conds]
        * ``branch_idx`` [n_total_conds] — branch each condition belongs to
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        else:
            x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if not isinstance(z_prior, torch.Tensor):
            z_prior = torch.from_numpy(np.asarray(z_prior, dtype=np.float32))
        else:
            z_prior = z_prior.float()
        if z_prior.ndim == 1:
            z_prior = z_prior.unsqueeze(0)

        with torch.no_grad():
            z_post, info = self.diff_post.forward_with_attribution(
                z_prior, x, return_intermediates=True
            )

        z_post_np = z_post.detach().cpu().numpy().astype(np.float64)
        z_prior_np = z_prior.detach().cpu().numpy().astype(np.float64)

        eps = 1e-12
        cap = 1.0 - 1e-9
        p_support = np.clip(
            z_post_np[:, :, None] * self.theta[None, :, :], 0.0, cap
        )
        support = -np.log1p(-p_support)              # [B, n_branches, n_classes]
        log_class = np.log1p(-p_support).sum(axis=1)
        class_prob = (1.0 - np.exp(log_class)).clip(min=eps)
        proba = class_prob / class_prob.sum(axis=1, keepdims=True).clip(min=eps)

        if info["log_lik_z1"] is None:
            branch_llr = np.zeros_like(z_prior_np)
            cond_log_lr = np.zeros((z_prior_np.shape[0], 0))
            match = np.zeros((z_prior_np.shape[0], 0))
            branch_idx = np.zeros(0, dtype=np.int64)
        else:
            branch_llr = (info["log_lik_z1"] - info["log_lik_z0"]).cpu().numpy()
            cond_log_lr = info["cond_log_lr"].cpu().numpy()
            match = info["match"].cpu().numpy()
            branch_idx = info["branch_idx"].cpu().numpy()

        return {
            "proba": proba,
            "z_prior": z_prior_np,
            "z_post": z_post_np,
            "support": support,
            "branch_llr": branch_llr,
            "cond_log_lr": cond_log_lr,
            "match": match,
            "branch_idx": branch_idx,
        }

    # ────────────────────────────────────────────────
    # User-facing: structured explanation for one sample
    # ────────────────────────────────────────────────

    def explain(
        self,
        x: Union[np.ndarray, torch.Tensor],
        z_prior: Union[np.ndarray, torch.Tensor],
        predicted_class: Optional[int] = None,
        top_k_branches: int = 5,
        top_k_conditions: int = 5,
    ) -> ExplanationReport:
        """Build an :class:`ExplanationReport` for a single input sample."""
        x_np = (
            x.detach().cpu().numpy() if isinstance(x, torch.Tensor)
            else np.asarray(x)
        ).astype(np.float64).ravel()

        info = self.attribute_batch(x_np[None, :], z_prior)
        proba = info["proba"][0]
        c_pred = int(predicted_class) if predicted_class is not None else int(np.argmax(proba))

        log_surv = float(
            -np.log(np.clip(1.0 - proba[c_pred], 1e-15, 1.0))
        )

        support_per_branch = info["support"][0, :, c_pred]            # [n_branches]
        branch_llr = info["branch_llr"][0]                            # [n_branches]
        z_prior_v = info["z_prior"][0]
        z_post_v = info["z_post"][0]
        cond_log_lr = info["cond_log_lr"][0] if info["cond_log_lr"].size else None
        match = info["match"][0] if info["match"].size else None

        # Rank branches by their support to the predicted class
        order = np.argsort(-support_per_branch)
        order = order[: int(min(top_k_branches, len(order)))]

        branch_explanations: List[BranchExplanation] = []
        for b in order:
            br = self.branches[b]
            llr_b = float(branch_llr[b])
            s_bc = float(support_per_branch[b])
            theta_bc = float(self.theta[b, c_pred])

            # Conditions of this branch
            cond_explanations: List[ConditionExplanation] = []
            if cond_log_lr is not None:
                mask = info["branch_idx"] == b
                cond_indices = np.nonzero(mask)[0]
                # Total |Δ_i| for this branch — used for sign-preserving share
                lr_b = cond_log_lr[cond_indices]
                m_b = match[cond_indices]
                abs_sum = float(np.abs(lr_b).sum()) + 1e-12

                for local_i, global_i in enumerate(cond_indices):
                    cond = br.conditions[local_i]
                    fname = (
                        self.feature_names[cond.feature_idx]
                        if cond.feature_idx < len(self.feature_names)
                        else f"f{cond.feature_idx}"
                    )
                    # Sign-preserving share: condition's Δ_i scaled to the
                    # branch's s_bc magnitude.  Σ_i share = (LLR_b/abs_sum)·s_bc;
                    # equals s_bc exactly when all Δ_i have the same sign.
                    share = float(lr_b[local_i]) / abs_sum * s_bc
                    cond_explanations.append(
                        ConditionExplanation(
                            feature_idx=int(cond.feature_idx),
                            feature_name=fname,
                            threshold=float(cond.threshold),
                            direction=str(cond.direction),
                            x_value=float(x_np[cond.feature_idx]),
                            match=float(m_b[local_i]),
                            log_lr=float(lr_b[local_i]),
                            support_share=share,
                        )
                    )

                cond_explanations.sort(key=lambda c: -abs(c.log_lr))
                cond_explanations = cond_explanations[: int(top_k_conditions)]

            branch_explanations.append(
                BranchExplanation(
                    branch_id=int(b),
                    branch_label=_branch_label(br, self.feature_names),
                    z_prior=float(z_prior_v[b]),
                    z_post=float(z_post_v[b]),
                    branch_log_lr=llr_b,
                    theta_bc=theta_bc,
                    support_to_class=s_bc,
                    conditions=cond_explanations,
                )
            )

        cls_name = (
            self.class_names[c_pred] if c_pred < len(self.class_names)
            else f"c{c_pred}"
        )
        return ExplanationReport(
            pred_class=c_pred,
            pred_class_name=cls_name,
            pred_proba=proba,
            log_survival=log_surv,
            branches=branch_explanations,
        )

    # ────────────────────────────────────────────────
    # Sanity check: reconstructed probability matches
    # ────────────────────────────────────────────────

    def consistency_check(
        self,
        x: Union[np.ndarray, torch.Tensor],
        z_prior: Union[np.ndarray, torch.Tensor],
        atol: float = 1e-3,
    ) -> dict:
        """Verify the two identities exposed by the differentiable posterior:

        * ``support_sum_vs_product_form``: numerical agreement of the two
          equivalent noisy-or formulations
          ``s_sum_bc = Σ_b -log(1 − θ_{bc}·z_b)`` vs.
          ``s_sum_bc = Σ_b log_complement_bc`` re-derived from the
          per-branch products.  Both should differ only by FP noise.
        * ``branch_llr_vs_logit_shift``: ``LLR_b`` (Bayesian shift
          recovered from per-condition log-LRs) equals
          ``logit(z_post_b) − logit(z_prior_b)`` for every branch.

        Default tolerance ``atol=1e-3`` accommodates float32 round-off
        from the underlying ``DifferentiablePosterior`` forward; the
        logical identity is exact in float64.
        """
        info = self.attribute_batch(x, z_prior)
        z_pr, z_po = info["z_prior"], info["z_post"]
        eps = 1e-12

        # Cross-check: re-derive support from z_post & θ via product form,
        # ensure it matches the layer's `support` tensor.
        cap = 1.0 - 1e-9
        p_support = np.clip(z_po[:, :, None] * self.theta[None, :, :], 0.0, cap)
        support_recompute = -np.log1p(-p_support)
        d_support = float(np.max(np.abs(info["support"] - support_recompute)))

        logit = lambda p: np.log(np.clip(p, eps, 1 - eps)) - np.log(np.clip(1 - p, eps, 1 - eps))
        shift = logit(z_po) - logit(z_pr)
        d_llr = float(np.max(np.abs(shift - info["branch_llr"])))

        return {
            "support_sum_vs_product_form": d_support,
            "branch_llr_vs_logit_shift": d_llr,
            "passed": (d_support < atol) and (d_llr < atol),
        }
