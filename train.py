# Adapted from https://github.com/SquareResearchCenter-AI/BEExAI/src/beexai/training/
"""Training models and evaluating their performance."""

from typing import Callable, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,roc_curve,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score, roc_auc_score)
try:
    from beexai.utils.convert import convert_to_numpy
    from beexai.utils.path import get_path
except ImportError:
    def convert_to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def get_path(path: str, check_dir: bool = False):
        from pathlib import Path
        p = Path(path)
        if check_dir:
            p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
from rule_network_model import RuleNetworkModel
from problog_export import (
    export_branches_to_json,
    export_branches_to_problog,
    export_branches_to_problog_latent,
)


class Trainer:
    """Trainer class
    """

    def __init__(
        self,
        model_name: str,
        task: str,
        model_params: Optional[dict] = None,
        device: str = None,
    ):
        assert task in ["classification"], f"Task must be classification, got {task}"
        self.models = {
            "XGBClassifier": XGBClassifier,
            "RuleNetwork": RuleNetworkModel
        }
        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}
        self.task = task
        self.device = device
        assert (
            self.model_name in self.models
        ), f"Model name must be one of {self.models.keys()}"
        if model_name == "RuleNetwork":
            self.model = RuleNetworkModel(**self.model_params, device=device, task=task)
        else:
            self.model = self.models[model_name](**self.model_params)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        seed: int,
        loss_file: Optional[str] = None,
        branch_json_path: Optional[str] = None,
        branch_problog_path: Optional[str] = None,
    ) -> Callable:
        """Perform training on the whole training set.
        Returns:
            callable: trained model
        """
        if self.model_name == "RuleNetwork":
            n_samples, n_features = x_train.shape
            n_labels = len(list(set(list(y_train.ravel()))))
            the_number = round(np.log2( n_features ))+4
            n_estimators = n_labels+round(np.log2( n_features ))
            print(n_estimators,"trees, with ",2**the_number,"leaves (maximum)",the_number)
            tree_ensemble = ExtraTreesClassifier(n_estimators=n_estimators,max_leaf_nodes=2**the_number,random_state=seed)
            tree_ensemble.fit(x_train, y_train.ravel())
            self.model.build_model_from_ensemble(tree_ensemble)
            if branch_json_path is not None:
                export_branches_to_json(self.model.branches, branch_json_path)
            if branch_problog_path is not None:
                export_branches_to_problog(self.model.branches, branch_problog_path)
            self.model = self.model.fit(
                x_train,
                y_train,
                x_val,
                y_val,
                loss_file=loss_file,
            )
            self.branch_probs_data = None
            if x_train is not None and self.model is not None:
                try:
                    # Cкоро по умолчанию все training + val: включаем x_train
                    probs = self.model.predict_branch_proba(x_train)
                    self.branch_probs_data = {i: probs[i].numpy() for i in range(probs.shape[0])}
                except Exception:
                    self.branch_probs_data = None

            if branch_problog_path is not None and hasattr(self, 'branch_probs_data') and self.branch_probs_data is not None:
                latent_path = branch_problog_path.replace('.pl', '_latent.pl')
                export_branches_to_problog_latent(
                    self.model.branches,
                    self.branch_probs_data,
                    observed_data=x_train,
                    output_path=latent_path,
                )
        else:
            self.model.fit(x_train, y_train)
        return self.model

    def get_metrics(self, x:np.ndarray, y:np.ndarray) -> dict:
        """Get metrics for the model. Accuracy and f1 score for
        classification, mse and r2 score for regression.
        Returns:
            dict: dictionary of metrics
        """
        pred = self.model.predict(x)
        pred = convert_to_numpy(pred)
        metrics = {
            "accuracy": round(accuracy_score(pred,y),6),
            "f1 score": round(f1_score(pred, y, average="weighted"),6),
        }
        return metrics

    def get_metrics_problog(self, x: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
        """Get metrics using ProbLog inference (replaces W2 with symbolic rules).

        Only available for RuleNetwork models with branches stored.
        Returns neural metrics alongside ProbLog metrics for comparison.
        """
        if self.model_name != "RuleNetwork":
            raise ValueError("ProbLog inference is only available for RuleNetwork")

        # Neural prediction (baseline)
        pred_neural = self.model.predict(x)
        pred_neural = convert_to_numpy(pred_neural)

        # ProbLog prediction
        pred_problog = self.model.predict_problog(x, verbose=verbose)
        pred_problog = convert_to_numpy(pred_problog)

        metrics = {
            "neural_accuracy": round(accuracy_score(pred_neural, y), 6),
            "neural_f1": round(f1_score(pred_neural, y, average="weighted"), 6),
            "problog_accuracy": round(accuracy_score(pred_problog, y), 6),
            "problog_f1": round(f1_score(pred_problog, y, average="weighted"), 6),
        }
        return metrics

    def save_model(self, path: str):
        """Save the model"""
        path = get_path(path, check_dir=True)
        is_nn = self.model_name == "RuleNetwork"
        if is_nn:
            torch.save(self.model.state_dict(), path)
        else:
            joblib.dump(self.model, path)

    def load_model(self, path: str):
        """Load the model"""
        path = get_path(path)
        is_nn = self.model_name == "RuleNetwork"
        if is_nn:
            fn_dict = torch.load(path,weights_only=True)
            self.model.build_from_dict(fn_dict)
            self.model.load_state_dict(fn_dict)
            self.model.eval()
            self.model.to(self.device)
        else:
            self.model = joblib.load(path)


