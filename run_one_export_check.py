from pathlib import Path
import json

from sklearn.datasets import load_wine
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

from rule_network_model import RuleNetworkModel
from problog_export import export_branches_to_json, export_branches_to_problog


SEED = 0
OUT_DIR = Path("debug_export")
OUT_DIR.mkdir(exist_ok=True)


def extract_expected_leaf_paths(tree):
    """Independent reference extraction of all root-to-leaf decision paths."""
    results = []

    def walk(node_id, path_conditions):
        left_id = tree.children_left[node_id]
        right_id = tree.children_right[node_id]
        is_leaf = left_id == -1 and right_id == -1

        if is_leaf:
            results.append(
                {
                    "leaf_node_id": int(node_id),
                    "conditions": list(path_conditions),
                }
            )
            return

        feature_idx = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])

        walk(
            left_id,
            path_conditions
            + [
                (
                    feature_idx,
                    threshold,
                    "le",
                    int(node_id),
                )
            ],
        )
        walk(
            right_id,
            path_conditions
            + [
                (
                    feature_idx,
                    threshold,
                    "gt",
                    int(node_id),
                )
            ],
        )

    walk(0, [])
    return results


def extract_expected_parent_of_leaf_paths(tree):
    """Independent reference extraction of parent-of-leaf paths.

    In the rule-network backbone one hidden neuron is created for each
    internal node that has at least one leaf child.  The conditions describe
    the path from the root to that parent node, not all the way to a
    specific leaf.
    """
    is_leaf = (tree.children_left == -1) & (tree.children_right == -1)
    results = []

    def walk(node_id, path_conditions):
        left_id = tree.children_left[node_id]
        right_id = tree.children_right[node_id]

        if is_leaf[node_id]:
            return

        has_left_leaf = left_id != -1 and is_leaf[left_id]
        has_right_leaf = right_id != -1 and is_leaf[right_id]

        feature_idx = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])

        if has_left_leaf or has_right_leaf:
            results.append(
                {
                    "parent_node_id": int(node_id),
                    "conditions": list(path_conditions),
                    "split_feature_idx": feature_idx,
                    "split_threshold": threshold,
                }
            )

        # Recurse into non-leaf children
        if not has_left_leaf and left_id != -1:
            walk(
                left_id,
                path_conditions
                + [
                    (feature_idx, threshold, "le", int(node_id)),
                ],
            )

        if not has_right_leaf and right_id != -1:
            walk(
                right_id,
                path_conditions
                + [
                    (feature_idx, threshold, "gt", int(node_id)),
                ],
            )

    walk(0, [])
    return results


def branch_signature(branch):
    return tuple(
        (
            int(cond.feature_idx),
            float(cond.threshold),
            str(cond.direction),
            int(cond.node_id),
        )
        for cond in branch.conditions
    )


def expected_signature(expected_path):
    return tuple(expected_path["conditions"])


def print_paths(title, paths, limit=5):
    print(f"\n=== {title} ===")
    for idx, path in enumerate(paths[:limit]):
        if not path:
            print(f"{idx}: <empty path>")
            continue
        formatted = " AND ".join(
            f"f{feature_idx} {('<=' if direction == 'le' else '>')} {threshold:.6g} [node {node_id}]"
            for feature_idx, threshold, direction, node_id in path
        )
        print(f"{idx}: {formatted}")
    if len(paths) > limit:
        print(f"... {len(paths) - limit} more")


def main():
    data = load_wine(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    n_features = X_train.shape[1]
    n_labels = len(set(y_train))
    max_depth_proxy = round(__import__("numpy").log2(n_features)) + 4
    n_estimators = n_labels + round(__import__("numpy").log2(n_features))
    max_leaf_nodes = 2 ** max_depth_proxy

    forest = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_leaf_nodes=max_leaf_nodes,
        random_state=SEED,
    )
    forest.fit(X_train, y_train)

    print("Dataset: sklearn wine")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Forest config:", {"n_estimators": n_estimators, "max_leaf_nodes": max_leaf_nodes})

    first_tree = forest.estimators_[0].tree_
    first_tree_text = export_text(
        forest.estimators_[0],
        feature_names=list(X_train.columns),
        max_depth=5,
    )
    print("\n=== First tree (text view, truncated depth) ===")
    print(first_tree_text)

    # --- Parent-of-leaf extraction (matches the rule-network backbone) ---
    expected_per_tree = {}
    for tree_id, estimator in enumerate(forest.estimators_):
        expected_per_tree[tree_id] = extract_expected_parent_of_leaf_paths(estimator.tree_)

    first_tree_expected = expected_per_tree[0]
    first_tree_leaf_count = sum(
        1
        for node_id in range(first_tree.node_count)
        if first_tree.children_left[node_id] == -1 and first_tree.children_right[node_id] == -1
    )
    print("\n=== First tree path counts ===")
    print("Leaf nodes in tree:", first_tree_leaf_count)
    print("Expected parent-of-leaf branches:", len(first_tree_expected))

    model = RuleNetworkModel(device="cpu")
    model.build_model_from_ensemble(forest)

    export_branches_to_json(model.branches, str(OUT_DIR / "branches.json"))
    export_branches_to_problog(model.branches, str(OUT_DIR / "knowledge_base.pl"))

    actual_per_tree = {}
    for tree_id in range(len(forest.estimators_)):
        actual_per_tree[tree_id] = [branch for branch in model.branches if branch.tree_id == tree_id]

    first_tree_actual = actual_per_tree[0]
    print("Built RuleNetwork branches:", len(first_tree_actual))

    # Condition-level comparison (path from root to parent, excluding parent split)
    expected_signatures = {
        tuple(path["conditions"]) for path in first_tree_expected
    }
    actual_signatures = {branch_signature(branch) for branch in first_tree_actual}

    print_paths(
        "Expected first-tree parent-of-leaf paths",
        [list(sig) for sig in sorted(expected_signatures)],
    )
    print_paths(
        "Actual first-tree RuleNetwork paths",
        [list(sig) for sig in sorted(actual_signatures)],
    )

    missing = expected_signatures - actual_signatures
    extra = actual_signatures - expected_signatures

    print("\n=== Verification by tree ===")
    total_expected = 0
    total_actual = 0
    for tree_id in range(len(forest.estimators_)):
        expected_count = len(expected_per_tree[tree_id])
        actual_count = len(actual_per_tree[tree_id])
        total_expected += expected_count
        total_actual += actual_count
        status = "OK" if expected_count == actual_count else "MISMATCH"
        print(f"tree {tree_id}: expected {expected_count}, actual {actual_count} -> {status}")

    print("\n=== Global verification ===")
    print("Total expected parent-of-leaf branches:", total_expected)
    print("Total built RuleNetwork branches:", total_actual)
    print("Missing first-tree paths:", len(missing))
    print("Extra first-tree paths:", len(extra))

    if missing:
        print_paths("Missing paths", [list(sig) for sig in sorted(missing)], limit=10)
    if extra:
        print_paths("Extra paths", [list(sig) for sig in sorted(extra)], limit=10)

    exported_json = json.loads((OUT_DIR / "branches.json").read_text(encoding="utf-8"))
    exported_rules = [
        line
        for line in (OUT_DIR / "knowledge_base.pl").read_text(encoding="utf-8").splitlines()
        if line.startswith("branch_struct(")
    ]
    print("\n=== Export verification ===")
    print("branches.json entries:", len(exported_json))
    print("knowledge_base.pl branch_struct rules:", len(exported_rules))

    # With parent-of-leaf extraction, #branches <= #leaves
    assert len(first_tree_actual) == len(first_tree_expected), (
        f"Branch count mismatch: {len(first_tree_actual)} actual vs "
        f"{len(first_tree_expected)} expected parent-of-leaf branches"
    )
    assert not missing, f"Missing {len(missing)} paths"
    assert not extra, f"Extra {len(extra)} paths"
    assert total_actual == total_expected
    assert len(exported_json) == total_expected
    assert len(exported_rules) == total_expected

    print("\nParent-of-leaf branch extraction verified successfully.")
    print("Saved to:")
    print(" ", OUT_DIR / "branches.json")
    print(" ", OUT_DIR / "knowledge_base.pl")


if __name__ == "__main__":
    main()
