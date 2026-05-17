import json
from pathlib import Path
from typing import Iterable, List
import numpy as np
from branch_schema import Branch, Condition


def _threshold_symbol(branch: Branch, condition: Condition) -> str:
    return f"t{branch.tree_id}_{condition.node_id}"


def _condition_atom(
    branch: Branch,
    condition: Condition,
    x_symbol: str = 'X',
    scoped_branch: bool = False,
) -> str:
    if scoped_branch:
        return (
            f"{condition.direction}"
            f"({branch.branch_id},f{condition.feature_idx},{_threshold_symbol(branch, condition)},{x_symbol})"
        )
    return f"{condition.direction}(f{condition.feature_idx},{_threshold_symbol(branch, condition)},{x_symbol})"


def branch_to_rule(branch: Branch, x_symbol: str = 'X', scoped_conditions: bool = False) -> str:
    if not branch.conditions:
        return f"branch_struct({branch.branch_id}, {x_symbol})."
    atoms = ', '.join(
        _condition_atom(branch, cond, x_symbol=x_symbol, scoped_branch=scoped_conditions)
        for cond in branch.conditions
    )
    return f"branch_struct({branch.branch_id}, {x_symbol}) :- {atoms}."


def threshold_facts(branches: Iterable[Branch]) -> List[str]:
    thresholds = {}
    for branch in branches:
        for cond in branch.conditions:
            thresholds[(branch.tree_id, cond.node_id)] = cond.threshold
    return [
        f"threshold(t{tree_id}_{node_id},{threshold:.10g})."
        for (tree_id, node_id), threshold in sorted(thresholds.items())
    ]


def _condition_holds(condition: Condition, row) -> bool:
    value = float(row[int(condition.feature_idx)])
    threshold = float(condition.threshold)
    if condition.direction == 'le':
        return value <= threshold
    if condition.direction == 'gt':
        return value > threshold
    raise ValueError(f"Unsupported condition direction: {condition.direction}")


def observed_condition_evidence(
    branches: List[Branch],
    observed_data,
    x_ids=None,
) -> List[str]:
    if observed_data is None:
        return []

    array = np.asarray(observed_data)
    if array.ndim != 2:
        raise ValueError("observed_data must be a 2D array-like object")

    if x_ids is None:
        x_ids = list(range(array.shape[0]))
    else:
        x_ids = list(x_ids)

    if len(x_ids) != array.shape[0]:
        raise ValueError("x_ids length must match observed_data rows")

    lines = []
    for row_idx, x_id in enumerate(x_ids):
        row = array[row_idx]
        for branch in branches:
            for cond in branch.conditions:
                atom = _condition_atom(branch, cond, x_symbol=str(x_id), scoped_branch=True)
                if _condition_holds(cond, row):
                    lines.append(f"evidence({atom}).")
                else:
                    lines.append(f"evidence({atom}, false).")
    return lines


def export_branches_to_problog(branches: List[Branch], output_path: str = 'knowledge_base.pl') -> str:
    path = Path(output_path)
    lines = ['% Auto-generated ProbLog rules from RuleNetwork branches', '']
    lines.extend(threshold_facts(branches))
    if branches:
        lines.append('')
    for branch in branches:
        lines.append(branch_to_rule(branch))
    path.write_text('\n'.join(lines), encoding='utf-8')
    return str(path)


def export_branches_to_json(branches: List[Branch], output_path: str = 'branches.json') -> str:
    path = Path(output_path)
    path.write_text(json.dumps([b.to_dict() for b in branches], indent=2), encoding='utf-8')
    return str(path)


def export_branches_to_problog_latent(
    branches: List[Branch],
    branch_probs: dict,
    observed_data=None,
    output_path: str = 'knowledge_base_latent.pl',
    p_high: float = 0.95,
    p_low: float = 0.05,
) -> str:
    """Export ProbLog knowledge base with latent branch activations and manifestations.

    branches: list of Branch objects
    branch_probs: mapping (x_id -> list/array of branch probabilities)
        e.g. {0: [0.8, 0.1, ...], 1: [0.2, ...], ...}
    observed_data: 2D array-like, rows aligned with sorted branch_probs keys.
        Used to emit evidence(...) for observed branch conditions.
    p_high: probability of condition being true if z=1
    p_low: probability of condition being true if z=0
    """
    path = Path(output_path)
    lines = ['% Auto-generated ProbLog rules from RuleNetwork latent branches', '']

    lines.extend(threshold_facts(branches))
    if branches:
        lines.append('')

    for branch in branches:
        lines.append(branch_to_rule(branch, scoped_conditions=True))

    if branch_probs:
        lines.append('')
        for x_id, probs in sorted(branch_probs.items()):
            for br_idx, branch in enumerate(branches):
                pz = float(probs[br_idx])
                lines.append(f"{pz:.8f}::z({branch.branch_id},{x_id}).")

        lines.append('')
        for branch in branches:
            lines.append(f"not_z({branch.branch_id},X) :- \\+ z({branch.branch_id},X).")

        lines.append('')

        for branch in branches:
            m = len(branch.conditions)
            if m > 0:
                # Depth-normalised manifestation: p_high^(1/m) per condition
                p_h = p_high ** (1.0 / m)
                p_l = p_low ** (1.0 / m)
            else:
                p_h, p_l = p_high, p_low
            for cond in branch.conditions:
                atom = _condition_atom(branch, cond, x_symbol='X', scoped_branch=True)
                lines.append(f"{p_h:.8f}::{atom} :- z({branch.branch_id},X).")
                lines.append(f"{p_l:.8f}::{atom} :- not_z({branch.branch_id},X).")

    if observed_data is not None:
        x_ids = sorted(branch_probs.keys()) if branch_probs else None
        evidence_lines = observed_condition_evidence(branches, observed_data, x_ids=x_ids)
        if evidence_lines:
            lines.append('')
            lines.append('% Observed evidence for branch conditions')
            lines.extend(evidence_lines)

    path.write_text('\n'.join(lines), encoding='utf-8')
    return str(path)


# ---------------------------------------------------------------------------
# Classification head: replaces frozen W2 with ProbLog rules
# ---------------------------------------------------------------------------

def _class_proportions_to_theta(branch: Branch) -> List[float]:
    """Normalize class_proportions of a branch to a probability distribution theta.

    Returns list of theta_k such that sum(theta_k) = 1.
    If all proportions are zero, returns uniform distribution.
    """
    props = branch.class_proportions
    if props is None:
        return []
    total = sum(props)
    if total <= 0:
        n = len(props)
        return [1.0 / n] * n if n > 0 else []
    return [p / total for p in props]


def classification_head_rules(
    branches: List[Branch],
    n_classes: int,
    min_theta: float = 1e-6,
) -> List[str]:
    """Generate ProbLog rules that replace the frozen W2 linear head.

    For each branch b and class k:
        theta_bk :: supports(k, b, X) :- z(b, X).

    Multiple rules for the same ``supports(k, _, X)`` are combined by ProbLog
    via the noisy-or (independent cause) semantics:
        P(supports(k, X)) = 1 - prod_b (1 - theta_bk * P(z(b,X)))

    Parameters
    ----------
    branches : list of Branch
    n_classes : int
    min_theta : float
        Minimum theta value to avoid degenerate 0-probability rules.
    """
    lines: List[str] = []
    lines.append('% Classification head: supports(Class, Branch, X) :- z(Branch, X)')
    lines.append('% theta_bk initialized from W2 class proportions (normalized)')
    for branch in branches:
        theta = _class_proportions_to_theta(branch)
        if not theta:
            continue
        for k in range(min(n_classes, len(theta))):
            t = max(theta[k], min_theta)
            t = min(t, 1.0 - min_theta)
            lines.append(
                f"{t:.8f}::supports({k},{branch.branch_id},X) :- z({branch.branch_id},X)."
            )
    return lines


def class_aggregation_rules(n_classes: int) -> List[str]:
    """Generate class aggregation rules.

    class(X, K) :- supports(K, _, X).   (any branch supporting K suffices)
    """
    lines: List[str] = []
    lines.append('')
    lines.append('% Class aggregation: class is supported if any branch supports it')
    for k in range(n_classes):
        lines.append(f"class(X,{k}) :- supports({k},_,X).")
    return lines


def query_rules_for_sample(x_id, n_classes: int) -> List[str]:
    """Generate query(...) directives for a single sample."""
    return [f"query(class({x_id},{k}))." for k in range(n_classes)]


def export_full_problog_program(
    branches: List[Branch],
    branch_probs_single,
    observed_row,
    x_id,
    n_classes: int,
    p_high: float = 0.95,
    p_low: float = 0.05,
    min_theta: float = 1e-6,
) -> str:
    """Build a complete ProbLog program for a single sample.

    This is the self-contained program that ProbLog can run to produce
    P(class(x_id, k)) for each class k.

    Parameters
    ----------
    branches : list of Branch
    branch_probs_single : array-like of shape [n_branches]
        P(z(b, x_id)) for every branch, from neural network.
    observed_row : array-like of shape [n_features]
        Feature vector for the sample.
    x_id : int or str
        Sample identifier.
    n_classes : int
    p_high, p_low : float
        Manifestation probabilities.
    min_theta : float
        Minimum theta for classification head.

    Returns
    -------
    str : complete ProbLog program text.
    """
    probs = np.asarray(branch_probs_single)
    row = np.asarray(observed_row)
    lines: List[str] = []

    # 1. Thresholds
    lines.extend(threshold_facts(branches))
    lines.append('')

    # 2. Structural rules (for interpretability / debugging)
    for branch in branches:
        lines.append(branch_to_rule(branch, scoped_conditions=True))
    lines.append('')

    # 3. Latent z(b, x_id) from neural network
    lines.append(f'% Latent branch activations for sample {x_id}')
    for br_idx, branch in enumerate(branches):
        pz = float(probs[br_idx])
        pz = max(min(pz, 1.0 - 1e-8), 1e-8)  # clamp to avoid 0/1
        lines.append(f"{pz:.8f}::z({branch.branch_id},{x_id}).")
    lines.append('')

    # 4. Negation helper
    for branch in branches:
        lines.append(f"not_z({branch.branch_id},X) :- \\+ z({branch.branch_id},X).")
    lines.append('')

    # 5. Manifestation rules (depth-normalised: p_high^(1/m) per condition)
    lines.append('% Manifestation: conditions as symptoms of z  (depth-normalised)')
    for branch in branches:
        m = len(branch.conditions)
        if m > 0:
            p_h = p_high ** (1.0 / m)
            p_l = p_low ** (1.0 / m)
        else:
            p_h, p_l = p_high, p_low
        for cond in branch.conditions:
            atom = _condition_atom(branch, cond, x_symbol='X', scoped_branch=True)
            lines.append(f"{p_h:.8f}::{atom} :- z({branch.branch_id},X).")
            lines.append(f"{p_l:.8f}::{atom} :- not_z({branch.branch_id},X).")
    lines.append('')

    # 6. Evidence: observed conditions
    lines.append(f'% Evidence for sample {x_id}')
    for branch in branches:
        for cond in branch.conditions:
            atom = _condition_atom(branch, cond, x_symbol=str(x_id), scoped_branch=True)
            if _condition_holds(cond, row):
                lines.append(f"evidence({atom}).")
            else:
                lines.append(f"evidence({atom}, false).")
    lines.append('')

    # 7. Classification head (replaces W2)
    lines.extend(classification_head_rules(branches, n_classes, min_theta))
    lines.append('')

    # 8. Class aggregation
    lines.extend(class_aggregation_rules(n_classes))
    lines.append('')

    # 9. Queries
    lines.extend(query_rules_for_sample(x_id, n_classes))

    return '\n'.join(lines)
