import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from rule_network_model import RuleNetworkModel


def test_condition_aware_rule_activations():
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    X_train, y_train = X[:120], y[:120]
    X_test, y_test = X[120:140], y[120:140]

    model = RuleNetworkModel(device='cpu')
    tree_ensemble = ExtraTreesClassifier(n_estimators=8, max_leaf_nodes=32, random_state=0)
    tree_ensemble.fit(X_train, y_train)

    model.build_model_from_ensemble(tree_ensemble)
    model.fit(X_train, y_train, X_test, y_test, epochs=20, learning_rate=0.01)

    pz = model.predict_branch_proba(X_test)
    pz_numpy = pz.numpy() if hasattr(pz, 'numpy') else np.asarray(pz)
    condition_z = model.predict_condition_branch_proba(X_test)
    condition_z_numpy = (
        condition_z.numpy() if hasattr(condition_z, 'numpy') else np.asarray(condition_z)
    )
    rule_head, diagnostics = model.predict_rule_head_proba(
        X_test,
        activation="hybrid",
        return_diagnostics=True,
        include_activation_diagnostics=True,
    )

    print('pz shape', pz_numpy.shape)
    print('pz min/max', float(pz_numpy.min()), float(pz_numpy.max()))
    print('first row', pz_numpy[0, :5])

    assert pz_numpy.shape[0] == X_test.shape[0]
    assert pz_numpy.shape[1] == model.hidden_neurons
    assert np.all(pz_numpy >= 0.0) and np.all(pz_numpy <= 1.0)
    assert condition_z_numpy.shape == pz_numpy.shape
    assert np.all(condition_z_numpy >= 0.0) and np.all(condition_z_numpy <= 1.0)
    assert rule_head.shape == (X_test.shape[0], model.out_features)
    assert np.allclose(rule_head.numpy().sum(axis=1), 1.0)
    assert diagnostics["condition_z"].shape == pz_numpy.shape

    print('test_condition_aware_rule_activations OK')


if __name__ == '__main__':
    test_condition_aware_rule_activations()
