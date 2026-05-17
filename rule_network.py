import torch
from torch import Tensor
from sklearn import ensemble
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from branch_schema import Branch, Condition


def extract_branches_from_sklearn_tree(tree, tree_id, branch_offset):
    """Parent-of-leaf branches from a single sklearn ``Tree`` object.

    Returns the list in the same order as the original DFS walk used by
    ``build_model_from_ensemble``.  ``branch_offset`` is added to the
    auto-generated ``branch_id`` so that ids stay unique across trees.
    """
    is_leaf = (tree.children_left == -1) & (tree.children_right == -1)
    out: list = []

    def walk(index, path_conditions):
        left_i = tree.children_left[index]
        right_i = tree.children_right[index]
        has_left_leaf = left_i != -1 and is_leaf[left_i]
        has_right_leaf = right_i != -1 and is_leaf[right_i]

        split_feature = int(tree.feature[index])
        split_threshold = (
            float(tree.threshold[index]) if split_feature >= 0 else None
        )

        if has_left_leaf or has_right_leaf:
            n_samples_total = tree.n_node_samples[0]
            node_samples = tree.n_node_samples[index]
            factor = node_samples / n_samples_total
            dist = factor * tree.value[index][0]  # parent distribution

            branch = Branch(
                branch_id=f"b{branch_offset + len(out)}",
                tree_id=tree_id,
                parent_node_id=int(index),
                conditions=list(path_conditions),
                class_proportions=dist.tolist(),
                split_feature_idx=(
                    split_feature if split_feature >= 0 else None
                ),
                split_threshold=split_threshold,
                split_node_id=(
                    int(index) if split_feature >= 0 else None
                ),
            )
            out.append(branch)

        if not has_left_leaf and left_i != -1:
            left_path = list(path_conditions)
            if split_feature >= 0:
                left_path.append(Condition(
                    feature_idx=split_feature,
                    threshold=split_threshold,
                    direction='le',
                    node_id=int(index),
                ))
            walk(left_i, left_path)

        if not has_right_leaf and right_i != -1:
            right_path = list(path_conditions)
            if split_feature >= 0:
                right_path.append(Condition(
                    feature_idx=split_feature,
                    threshold=split_threshold,
                    direction='gt',
                    node_id=int(index),
                ))
            walk(right_i, right_path)

    walk(0, [])
    return out


def extract_branches_from_sklearn_ensemble(tree_ensemble):
    """Walk ``tree_ensemble.estimators_`` and return branches-per-tree.

    Output matches ``RuleNetwork.all_branch_conditions`` 1-in-1: outer
    list indexed by tree id, inner list = parent-of-leaf rules in DFS
    order with monotonically increasing ``branch_id``.
    """
    branches_per_tree: list = []
    offset = 0
    for tree_id, estimator in enumerate(tree_ensemble.estimators_):
        br = extract_branches_from_sklearn_tree(
            estimator.tree_, tree_id=tree_id, branch_offset=offset,
        )
        branches_per_tree.append(br)
        offset += len(br)
    return branches_per_tree


class RuleNetwork(nn.Module):
    """Condition-aware rule-activation backbone.

    This is the lightweight local backbone used by PPtheta-Post.  It
    follows the NSToolkit condition-aware convention: a symbolic rule
    pool is converted into differentiable branch activations, while the
    explicit Branch / Condition objects remain available for posterior
    evidence updates and attribution.

    Key design decisions:
    * One latent activation per parent-of-leaf rule.
    * Class distributions (w2) are taken from the rule source node.
    * w2 is frozen (requires_grad=False); theta-based heads can replace it.
    * Forward: BN0 -> Linear(w1*m1) -> BN1 -> Sigmoid -> BN2 -> Linear(w2)
    """

    def __init__(
            self,
            task: str = "classification",
            device=None,
            dtype=torch.float,
    ) -> None:
        assert (
            task == "classification"
        ), f"""RuleNetwork is only implemented for classification,
            found {task}"""
        super().__init__()
        self.task = task
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.hidden_neurons = 0
        self.in_features = None
        self.out_features = None
        self.bn0 = None
        self.w1 = None
        self.m1 = None
        self.bn1 = None
        self.bn2 = None
        self.w2 = None
        self.all_branch_conditions = []
        self.branches = []
        self.branch_id_to_hidden_index = {}
        self.rule_reliability_ = None
        
    def forward(self, input: Tensor) -> Tensor:
        x = self.bn0(input)
        if self.training:
            x = F.linear(x, self.w1 * self.m1)
        else:
            x = F.linear(x, self.w1)
        x = self.bn1(x)
        x = torch.sigmoid(x)
        x = self.bn2(x)
        x = F.linear(x, self.w2)
        return x

    def build_model_from_ensemble(self, tree_ensemble: ensemble) -> nn.Module:
        """Build condition-aware rule activations from a fitted sklearn ensemble.

        Thin wrapper that extracts parent-of-leaf branches from
        ``tree_ensemble.estimators_`` (sklearn-tree API) and delegates
        to :meth:`build_model_from_branches`.  Kept as the public entry
        point for backward compatibility; alternative rule sources
        (XGBoost, CatBoost, FIGS, RuleFit, …) should produce ``Branch``
        objects and call :meth:`build_model_from_branches` directly.
        """
        branches_per_tree = extract_branches_from_sklearn_ensemble(tree_ensemble)
        self.build_model_from_branches(
            branches_per_tree,
            in_features=int(tree_ensemble.n_features_in_),
            out_features=int(tree_ensemble.n_classes_),
        )

    def build_model_from_branches(
        self,
        branches_per_tree,
        in_features: int,
        out_features: int,
    ) -> nn.Module:
        """Build condition-aware rule activations from explicit branches.

        Generic entry point: any rule source (sklearn ensembles, XGBoost,
        CatBoost, FIGS, RuleFit, …) can produce a list-of-lists of
        ``Branch`` (outer = tree id, inner = parent-of-leaf rules in that
        tree) and call this method.  Per-tree grouping is required only
        to compute feature importance the same way the sklearn path does
        — within a tree, w1 weights are normalised by max usage of each
        feature across that tree's branches.

        For every branch, ``class_proportions`` must be set and
        ``split_feature_idx`` should be set whenever the parent node has
        at least one leaf child whose split feature is meaningful — both
        feed w2 and w1 respectively, exactly as the sklearn path does.
        """
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_neurons = 0
        self.branches = []
        self.all_branch_conditions = []
        self.branch_id_to_hidden_index = {}
        self.rule_reliability_ = None

        feature_importance = []
        all_proportions = []

        for branches_in_tree in branches_per_tree:
            self.all_branch_conditions.append(list(branches_in_tree))
            self.hidden_neurons += len(branches_in_tree)
            class_proportion = []
            importance = torch.zeros(self.in_features).float()
            for branch in branches_in_tree:
                if branch.class_proportions is None:
                    raise ValueError(
                        f"branch {branch.branch_id!r} has no class_proportions; "
                        "rule source must populate them"
                    )
                cp = np.asarray(branch.class_proportions, dtype=np.float64)
                if cp.shape[0] != self.out_features:
                    raise ValueError(
                        f"branch {branch.branch_id!r} class_proportions has "
                        f"length {cp.shape[0]} but out_features={self.out_features}"
                    )
                class_proportion.append(cp)
                for feat_idx in branch.feature_indices_for_w1():
                    if 0 <= feat_idx < self.in_features:
                        importance[feat_idx] += 1
            if importance.max() > 0:
                importance = importance / importance.max()
            feature_importance.append(importance)
            all_proportions.append(class_proportion)
            self.branches.extend(branches_in_tree)

        for hidden_idx, branch in enumerate(self.branches):
            self.branch_id_to_hidden_index[branch.branch_id] = hidden_idx

        def get_w1(size, device, dtype):
            w1 = torch.zeros(size, dtype=dtype)
            i = 0
            for t, branches_in_tree in enumerate(self.all_branch_conditions):
                for branch in branches_in_tree:
                    feature_indices = branch.feature_indices_for_w1()
                    feature_indices = [
                        f for f in feature_indices if 0 <= f < self.in_features
                    ]
                    if feature_indices:
                        w1[i][feature_indices] = feature_importance[t][feature_indices]
                    i += 1
            w1 *= 1 / np.sqrt(self.in_features)
            return w1.to(device)

        def get_w2(size, device, dtype):
            w2 = torch.zeros(size, dtype=dtype)
            i = 0
            for t, proportions_in_tree in enumerate(all_proportions):
                for classes_involved_in_branch in proportions_in_tree:
                    w2[:, i] = torch.from_numpy(
                        np.asarray(classes_involved_in_branch, dtype=np.float64)
                    )
                    i += 1
            w2 *= 1 / np.sqrt(self.in_features)
            return w2.to(device)

        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        w1 = get_w1(
            (self.hidden_neurons, self.in_features),
            device=self.device, dtype=self.dtype,
        )
        self.m1 = (w1 != 0)
        self.w1 = nn.Parameter(w1)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        w2 = get_w2(
            (self.out_features, self.hidden_neurons),
            device=self.device, dtype=self.dtype,
        )
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self.rule_reliability_ = torch.ones(
            self.hidden_neurons, dtype=self.dtype, device=self.device
        )
        print(self.hidden_neurons, "hidden")

    def build_from_dict(self, fn_dict) -> nn.Module:
        self.w1 = nn.Parameter(fn_dict['w1'])
        self.m1 = (self.w1 != 0)
        self.hidden_neurons = self.w1.shape[0]
        self.w2 = nn.Parameter(fn_dict['w2'], requires_grad=False)
        self.out_features = self.w2.shape[0]
        self.in_features = self.w1.shape[1]
        self.bn0 = nn.BatchNorm1d(self.in_features, device=self.device)
        self.bn1 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons, device=self.device)
        self.rule_reliability_ = torch.ones(
            self.hidden_neurons, dtype=self.dtype, device=self.device
        )

    def branch_probs(self, input: Tensor) -> Tensor:
        """Return branch latent probabilities P(z(b,X)=true | X)."""
        x = self.bn0(input)
        if self.training:
            x = F.linear(x, self.w1 * self.m1)
        else:
            x = F.linear(x, self.w1)
        x = self.bn1(x)
        x = torch.sigmoid(x)
        # z(b, X) probability for each branch
        return x
