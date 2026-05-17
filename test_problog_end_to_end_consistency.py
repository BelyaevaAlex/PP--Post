import math
from pathlib import Path

import numpy as np
from problog import get_evaluatable
from problog.program import PrologString
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier

from rule_network_model import RuleNetworkModel
from problog_export import export_branches_to_problog_latent


P_HIGH = 0.95
P_LOW = 0.05


def condition_holds(cond, row) -> bool:
    value = float(row[int(cond.feature_idx)])
    threshold = float(cond.threshold)
    if cond.direction == 'le':
        return value <= threshold
    if cond.direction == 'gt':
        return value > threshold
    raise ValueError(f"Unsupported direction: {cond.direction}")


def branch_holds(branch, row) -> bool:
    return all(condition_holds(cond, row) for cond in branch.conditions)


def expected_posterior(prior, branch, row, p_high=P_HIGH, p_low=P_LOW) -> float:
    like_z = float(prior)
    like_not_z = float(1.0 - prior)
    depth = max(len(branch.conditions), 1)
    p_high = p_high ** (1.0 / depth)
    p_low = p_low ** (1.0 / depth)
    for cond in branch.conditions:
        holds = condition_holds(cond, row)
        if holds:
            like_z *= p_high
            like_not_z *= p_low
        else:
            like_z *= (1.0 - p_high)
            like_not_z *= (1.0 - p_low)
    denom = like_z + like_not_z
    return like_z / denom if denom else 0.0


def test_problog_end_to_end_consistency():
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    X_train, y_train = X[:120], y[:120]
    X_test, y_test = X[120:140], y[120:140]

    model = RuleNetworkModel(device='cpu')
    tree_ensemble = ExtraTreesClassifier(n_estimators=8, max_leaf_nodes=32, random_state=42)
    tree_ensemble.fit(X_train, y_train)

    model.build_model_from_ensemble(tree_ensemble)
    model.fit(X_train, y_train, X_test, y_test, epochs=20, learning_rate=0.01)

    observed = X_test[:5]
    priors = model.predict_branch_proba(observed).numpy()
    branch_probs = {i: priors[i] for i in range(priors.shape[0])}

    latent_path = 'tmp_problog_latent.pl'
    export_branches_to_problog_latent(
        model.branches,
        branch_probs,
        observed_data=observed,
        output_path=latent_path,
        p_high=P_HIGH,
        p_low=P_LOW,
    )

    candidate_indices = [idx for idx, branch in enumerate(model.branches) if branch.conditions][:5]
    assert candidate_indices, "Need at least one non-empty branch for consistency test"

    query_lines = []
    checked_pairs = []
    for x_id in range(observed.shape[0]):
        for branch_idx in candidate_indices:
            branch = model.branches[branch_idx]
            query_lines.append(f"query(branch_struct({branch.branch_id},{x_id})).")
            query_lines.append(f"query(z({branch.branch_id},{x_id})).")
            checked_pairs.append((x_id, branch_idx))

    text = Path(latent_path).read_text(encoding='utf-8') + '\n' + '\n'.join(query_lines) + '\n'
    result = get_evaluatable().create_from(PrologString(text)).evaluate()
    result_by_name = {str(key): float(value) for key, value in result.items()}

    true_count = 0
    false_count = 0
    for x_id, branch_idx in checked_pairs:
        branch = model.branches[branch_idx]
        row = observed[x_id]
        holds = branch_holds(branch, row)
        branch_key = f"branch_struct({branch.branch_id},{x_id})"
        z_key = f"z({branch.branch_id},{x_id})"

        expected_struct = 1.0 if holds else 0.0
        prob_struct = result_by_name[branch_key]
        assert math.isclose(prob_struct, expected_struct, abs_tol=1e-9), (
            branch_key,
            prob_struct,
            expected_struct,
        )

        prior = float(priors[x_id][branch_idx])
        expected_z = expected_posterior(prior, branch, row)
        prob_z = result_by_name[z_key]
        assert math.isclose(prob_z, expected_z, rel_tol=1e-3, abs_tol=1e-4), (
            z_key,
            prob_z,
            expected_z,
        )
        if holds:
            true_count += 1
        else:
            false_count += 1

    assert true_count > 0
    assert false_count > 0

    print('checked_pairs', len(checked_pairs))
    print('true_count', true_count)
    print('false_count', false_count)
    print('test_problog_end_to_end_consistency OK')


if __name__ == '__main__':
    test_problog_end_to_end_consistency()
