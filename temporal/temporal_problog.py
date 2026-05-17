"""L3 / L4 — ProbLog export for temporal Branches.

Two new atom families are introduced.

L3 — interval atoms
    ``gt_mean(b0, hr, 0, 12, 95.0, X)`` reads as: in sample ``X``,
    the mean of variable ``hr`` over hours ``0..12`` is greater than
    ``95.0``.  Direction (``gt`` / ``le``) and statistic (``mean``,
    ``std``, ``slope``) are part of the functor name.

L4 — temporal latent atoms
    ``z(b0, X, T)`` describes branch activation **at timestep T**;
    ``z_overall(b0, X)`` aggregates across time according to one of
    the supported temporal modes (``exists``, ``forall``, ``mean``,
    ``noisy_or``, ``k_of_t``).

The L3 helpers reuse the existing ``Branch`` / ``Condition`` schema by
*translating* the integer ``feature_idx`` of each condition to a
:class:`~temporal.interval_forest.IntervalFeatureMeta` entry — there is
no need to subclass ``Condition``.  For raw (non-temporal) branches the
helpers degrade gracefully and emit standard atoms identical to
:mod:`problog_export`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from branch_schema import Branch, Condition
from .interval_forest import IntervalFeatureMeta


# ─────────────────────────────────────────────────────────────────────────
# L3: temporal atoms over interval features
# ─────────────────────────────────────────────────────────────────────────

def _quote_for_prolog(name: str) -> str:
    """Wrap an identifier in single quotes if it would otherwise be
    parsed by Prolog as a variable (uppercase first character) or
    contains characters that break atom syntax."""
    if not name:
        return "''"
    safe = name[0].islower() and name.replace("_", "").isalnum()
    return name if safe else f"'{name}'"


def feature_meta_to_atom(
    branch: Branch,
    condition: Condition,
    feature_meta: Optional[Sequence[IntervalFeatureMeta]],
    x_symbol: str = "X",
) -> str:
    """Render a single condition as a temporal ProbLog atom.

    Falls back to the standard ``le(fJ,tT_N,X)`` form whenever no metadata
    is supplied (covers L1 / L2 branches and unit tests).

    Variable names with an uppercase first letter (e.g. ``HR``) would be
    parsed by Prolog as logical variables; this function quotes them so
    the engine treats them as atoms.
    """
    if feature_meta is None:
        thr_sym = f"t{branch.tree_id}_{condition.node_id}"
        return (
            f"{condition.direction}({branch.branch_id},"
            f"f{condition.feature_idx},{thr_sym},{x_symbol})"
        )
    meta = feature_meta[condition.feature_idx]
    functor = f"{condition.direction}_{meta.stat}"
    var = _quote_for_prolog(meta.variable_name)
    return (
        f"{functor}({branch.branch_id},{var},"
        f"{meta.interval_start},{meta.interval_end},"
        f"{condition.threshold:.6g},{x_symbol})"
    )


def temporal_branch_to_rule(
    branch: Branch,
    feature_meta: Optional[Sequence[IntervalFeatureMeta]],
    x_symbol: str = "X",
) -> str:
    """Translate a Branch into a ProbLog ``branch_struct/2`` rule whose
    body is composed of L3 temporal atoms.
    """
    if not branch.conditions:
        return f"branch_struct({branch.branch_id}, {x_symbol})."
    atoms = ", ".join(
        feature_meta_to_atom(branch, cond, feature_meta, x_symbol)
        for cond in branch.conditions
    )
    return f"branch_struct({branch.branch_id}, {x_symbol}) :- {atoms}."


def export_temporal_branches_to_problog(
    branches: Iterable[Branch],
    feature_meta: Optional[Sequence[IntervalFeatureMeta]],
    output_path: str = "knowledge_base_temporal.pl",
) -> str:
    """Write ``branch_struct`` rules with temporal atoms to disk."""
    path = Path(output_path)
    lines: List[str] = ["% Auto-generated temporal ProbLog rules", ""]
    for branch in branches:
        lines.append(temporal_branch_to_rule(branch, feature_meta))
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def export_temporal_problog_program(
    branches: Sequence[Branch],
    branch_probs_single: np.ndarray,
    interval_feature_row: np.ndarray,
    feature_meta: Optional[Sequence[IntervalFeatureMeta]],
    x_id: int | str,
    n_classes: int,
    p_high: float = 0.95,
    p_low: float = 0.05,
    min_theta: float = 1e-6,
) -> str:
    """Emit a complete ProbLog program for one sample using temporal atoms.

    Mirrors :func:`problog_export.export_full_problog_program` but with
    temporal functors.  Used both for human-readable explanations and for
    spot-checking the analytical posterior against the ProbLog engine.
    """
    from problog_export import (
        threshold_facts,
        classification_head_rules,
        class_aggregation_rules,
        query_rules_for_sample,
    )

    probs = np.asarray(branch_probs_single)
    row = np.asarray(interval_feature_row)
    lines: List[str] = []

    lines.extend(threshold_facts(branches))
    lines.append("")

    for branch in branches:
        lines.append(temporal_branch_to_rule(branch, feature_meta, x_symbol="X"))
    lines.append("")

    lines.append(f"% Latent branch activations for sample {x_id}")
    for br_idx, branch in enumerate(branches):
        pz = float(probs[br_idx])
        pz = max(min(pz, 1.0 - 1e-8), 1e-8)
        lines.append(f"{pz:.8f}::z({branch.branch_id},{x_id}).")
    lines.append("")

    for branch in branches:
        lines.append(
            f"not_z({branch.branch_id},X) :- \\+ z({branch.branch_id},X)."
        )
    lines.append("")

    lines.append("% Manifestation: temporal atoms as symptoms of z (depth-normalised)")
    for branch in branches:
        m = len(branch.conditions)
        if m > 0:
            p_h = p_high ** (1.0 / m)
            p_l = p_low ** (1.0 / m)
        else:
            p_h, p_l = p_high, p_low
        for cond in branch.conditions:
            atom = feature_meta_to_atom(branch, cond, feature_meta, x_symbol="X")
            lines.append(f"{p_h:.8f}::{atom} :- z({branch.branch_id},X).")
            lines.append(f"{p_l:.8f}::{atom} :- not_z({branch.branch_id},X).")
    lines.append("")

    lines.append(f"% Evidence for sample {x_id}")
    for branch in branches:
        for cond in branch.conditions:
            atom = feature_meta_to_atom(
                branch, cond, feature_meta, x_symbol=str(x_id)
            )
            value = float(row[cond.feature_idx])
            holds = (
                value <= cond.threshold
                if cond.direction == "le"
                else value > cond.threshold
            )
            if holds:
                lines.append(f"evidence({atom}).")
            else:
                lines.append(f"evidence({atom}, false).")
    lines.append("")

    lines.extend(classification_head_rules(list(branches), n_classes, min_theta))
    lines.append("")
    lines.extend(class_aggregation_rules(n_classes))
    lines.append("")
    lines.extend(query_rules_for_sample(x_id, n_classes))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# L4: temporal latent z(b, X, T) and aggregation rules
# ─────────────────────────────────────────────────────────────────────────

TEMPORAL_AGGREGATION_RULES: Dict[str, str] = {
    # P(z_overall(b, X)) = 1 - prod_t (1 - P(z(b, X, t)))
    "exists":   "z_overall({bid},X) :- z({bid},X,_).",
    # All timesteps must be active — defined recursively over the timestep list.
    "forall":   (
        "z_overall({bid},X) :- "
        "\\+ ( member(T, Timesteps), \\+ z({bid},X,T) )."
    ),
    # Mean, noisy-or-time, k-of-T are not pure ProbLog single-atom rules; they
    # are computed analytically (see :func:`temporal.aggregate_z_over_time`).
    "mean":     "% mean aggregation handled analytically",
    "noisy_or": "% noisy-or-over-time handled analytically",
    "k_of_t":   "% k-of-T aggregation handled analytically",
}


def temporal_latent_facts(
    branches: Sequence[Branch],
    z_per_time: np.ndarray,
    x_id: int | str,
) -> List[str]:
    """Emit ``pZ::z(b, x_id, t).`` facts for L4 latent atoms.

    Parameters
    ----------
    branches : sequence of Branch (length B)
    z_per_time : np.ndarray of shape [T, B]
        Per-timestep branch latent probabilities.
    x_id : sample identifier.
    """
    arr = np.asarray(z_per_time, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(branches):
        raise ValueError("z_per_time must have shape [T, n_branches]")
    lines: List[str] = []
    for t in range(arr.shape[0]):
        for b_idx, branch in enumerate(branches):
            pz = float(arr[t, b_idx])
            pz = max(min(pz, 1.0 - 1e-8), 1e-8)
            lines.append(
                f"{pz:.8f}::z({branch.branch_id},{x_id},{t})."
            )
    return lines


def temporal_aggregation_rule(
    branch: Branch,
    mode: str = "exists",
) -> str:
    """Return the ProbLog rule that aggregates per-timestep latents into
    a branch-level latent for the given branch.

    Only ``exists`` and ``forall`` are pure-ProbLog rules; the other modes
    delegate to numerical aggregation in
    :func:`temporal_inference.aggregate_z_over_time`.
    """
    if mode not in TEMPORAL_AGGREGATION_RULES:
        raise ValueError(f"unsupported temporal aggregation mode {mode!r}")
    template = TEMPORAL_AGGREGATION_RULES[mode]
    return template.format(bid=branch.branch_id)
