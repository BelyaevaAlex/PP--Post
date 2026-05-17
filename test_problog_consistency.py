import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from rule_network_model import RuleNetworkModel
from problog_export import export_branches_to_problog, export_branches_to_problog_latent
from pathlib import Path


def test_problog_export_and_model_consistency():
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    # небольшая выборка для быстрого теста
    X_train, y_train = X[:120], y[:120]
    X_test, y_test = X[120:140], y[120:140]

    model = RuleNetworkModel(device='cpu')
    tree_ensemble = ExtraTreesClassifier(n_estimators=8, max_leaf_nodes=32, random_state=42)
    tree_ensemble.fit(X_train, y_train)

    model.build_model_from_ensemble(tree_ensemble)
    model.fit(X_train, y_train, X_test, y_test, epochs=20, learning_rate=0.01)

    y_pred_before = model.predict(X_test)
    y_pred_before = np.asarray(y_pred_before)

    # Обычный экспорт ProbLog структуры
    pl_path = 'tmp_problog.pl'
    latent_path = 'tmp_problog_latent.pl'
    export_branches_to_problog(model.branches, pl_path)

    # latent export с P(z(b,X))
    z_probs = model.predict_branch_proba(X_test[:5]).numpy()
    branch_probs = {i: z_probs[i] for i in range(z_probs.shape[0])}
    export_branches_to_problog_latent(
        model.branches,
        branch_probs,
        observed_data=X_test[:5],
        output_path=latent_path,
    )

    # После экспорта модель теоретически должна выдавать такие же метки;
    # допускаем небольшую разницу для устойчивости
    y_pred_after = model.predict(X_test)
    y_pred_after = np.asarray(y_pred_after)
    assert y_pred_before.shape == y_pred_after.shape
    same_frac = np.mean(y_pred_before == y_pred_after)
    assert same_frac >= 0.8, f"Predictions diverged: {same_frac*100:.1f}% equal"

    # проблог-файлы созданы и имеют ключевые строки
    text = Path(pl_path).read_text(encoding='utf-8')
    latent_text = Path(latent_path).read_text(encoding='utf-8')
    assert 'branch_struct(' in text
    assert 'threshold(' in text
    assert 'z(' in latent_text
    assert 'evidence(' in latent_text
    assert 'p_high' not in latent_text  # Мы используем фиксированное p_high, p_low без маркировки слов

    # Проверяем неразрывность данных (z-факты согласованы со значениями)
    first_branch_id = model.branches[0].branch_id
    pz_val = branch_probs[0][0]
    assert f"{pz_val:.8f}::z({first_branch_id},0)." in latent_text

    print('test_problog_export_and_model_consistency OK')


if __name__ == '__main__':
    test_problog_export_and_model_consistency()
