from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Condition:
    feature_idx: int
    threshold: float
    direction: str   # "le" or "gt"
    node_id: int

    def to_dict(self):
        return {
            "feature_idx": int(self.feature_idx),
            "threshold": float(self.threshold),
            "direction": str(self.direction),
            "node_id": int(self.node_id),
        }


@dataclass
class Branch:
    branch_id: str
    tree_id: int
    parent_node_id: int
    conditions: List[Condition] = field(default_factory=list)
    class_proportions: Optional[List[float]] = None

    # split metadata для совместимости с оригинальным W1
    split_feature_idx: Optional[int] = None
    split_threshold: Optional[float] = None
    split_node_id: Optional[int] = None

    def feature_indices_for_w1(self):
        feats = [int(c.feature_idx) for c in self.conditions]
        if self.split_feature_idx is not None:
            feats.append(int(self.split_feature_idx))
        return feats

    def to_dict(self):
        return {
            "branch_id": str(self.branch_id),
            "tree_id": int(self.tree_id),
            "parent_node_id": int(self.parent_node_id),
            "conditions": [c.to_dict() for c in self.conditions],
            "class_proportions": None
            if self.class_proportions is None
            else [float(x) for x in self.class_proportions],
            "split_feature_idx": None
            if self.split_feature_idx is None
            else int(self.split_feature_idx),
            "split_threshold": None
            if self.split_threshold is None
            else float(self.split_threshold),
            "split_node_id": None
            if self.split_node_id is None
            else int(self.split_node_id),
        }