import torch
import torch.nn as nn
import copy
from torch.nn import functional as F
import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple, Union
from tqdm import tqdm
from rule_network import RuleNetwork
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def convert_to_tensor(data: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert various data types to PyTorch tensor.
    
    Args:
        data: Input data as DataFrame, ndarray or Tensor
        
    Returns:
        torch.Tensor: Converted data
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, pd.DataFrame):
        return torch.from_numpy(data.values)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        if self.y is not None:
            y = self.y[idx]
            return x, y
        else:
            return x

    def __len__(self):
        return self.X.shape[0]


class RuleNetworkModel(RuleNetwork):
    """Condition-aware rule-activation classifier with fit / predict helpers.

    This is the PPtheta-Post local counterpart of the nstoolkit
    condition-aware rule-head API: the backbone provides differentiable
    rule activations, while this wrapper handles training, prediction and
    ProbLog-aware posterior heads.
    """

    def __init__(
        self,
        task: str="classification",
        device: str = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.task = task
        self.dtype = dtype
        
    def train_step(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        criterion: nn.modules.loss,
        optimizer: torch.optim.Optimizer,
        p: float = 0.4
    ) -> float:
        """Train the model for one epoch.
        """
        dataset = TabularDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=min(256,dataset.__len__()), shuffle=True,drop_last=True)#630
        loss_sum = 0
        for x, y in dataloader:
            y = y.to(self.device)
            y_pred = self.forward(x.to(self.device)) 
            y = y.squeeze(1).long()
            ce = criterion(y_pred,y) 
            pt = torch.exp(-ce)  # pt = prob of correct class
            f = 0.5 * (1 - pt) ** 2.5 * ce
            loss = p*f.mean() + (1-p)*ce.mean()
            
            # Add sparsity loss if model has it
            if hasattr(self, 'get_sparsity_loss'):
                sparsity_loss = self.get_sparsity_loss()
                loss = loss + sparsity_loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        return loss_sum / len(dataloader) #(i+1)

    def val_step(
        self,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        criterion: nn.modules.loss,
        p: float = 0.4
    ) -> float:
        """Validate the model for one epoch.
        """
        dataset = TabularDataset(x_val, y_val)
        dataloader = DataLoader(dataset, batch_size=min(256,dataset.__len__()), shuffle=True,drop_last=True)
        loss_sum = 0
        self.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                y = y.to(self.device)
                y_pred = self.forward(x.to(self.device))
                y = y.squeeze(1).long()
                ce = criterion(y_pred,y) 
                pt = torch.exp(-ce)
                f = 0.5 * (1 - pt) ** 2.5 * ce
                loss = p*f.mean() + (1-p)*ce.mean()
                loss_sum += loss.item()
        self.train()
        return loss_sum / len(dataloader)#(i+1)
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        learning_rate = 0.01,
        epochs = 1500,
        loss_file: Optional[str] = None,
    ) -> Any:
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).reshape(-1,1)
        x_val = torch.from_numpy(x_val).float()
        y_val = torch.from_numpy(y_val).reshape(-1,1)
        y_train = y_train.float()
        y_val = y_val.float()
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        min_val_loss=10000000
        best_state = None
        patience = 0
        max_patience=100
        progress_bar = tqdm(range(epochs))
        loss_history = []
        val_loss_history = []
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=180)
        for i,_ in enumerate(progress_bar):
            loss = self.train_step(x_train, y_train, criterion, optimizer)
            progress_bar.set_description(f"Loss: {loss:.6f}")
            loss_history.append(loss)
            val_loss = self.val_step(x_val, y_val, criterion)
            val_loss_history.append(val_loss)
            scheduler.step(val_loss)
            if val_loss<min_val_loss:
                min_val_loss = val_loss
                best_state = copy.deepcopy(self.state_dict())
                patience = 0
            else:
                patience+=1
            if patience==max_patience:
                break
        del scheduler
        
        _ = plt.figure(figsize=(12, 8))
        plt.plot(loss_history[5:], label="train")
        if i<epochs-1:
            self.load_state_dict(best_state)
        plt.plot(val_loss_history[5:], label="val")
        plt.legend()
        if loss_file is not None:
            plt.savefig(loss_file)
        plt.close()

        return self

    def predict(
        self, x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        x_test_copy = convert_to_tensor(x_test).float()
        dataset = TabularDataset(x_test_copy)
        dataloader = DataLoader(dataset, batch_size=min(200,dataset.__len__()), shuffle=False)
        res = []
        with torch.no_grad():
            for x in dataloader:
                res1 = torch.softmax(self.forward(x.to(self.device)),dim=1)
                res.append(res1.to('cpu'))
            res = torch.vstack(res)
            res= torch.argmax(res,dim=1)
        if was_training:
            self.train()
        return res

    def predict_branch_proba(self, x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Predict per-branch latent probabilities p(z(b,X)=true) across examples."""
        was_training = self.training
        self.eval()
        x_test_copy = convert_to_tensor(x_test).float().to(self.device)
        dataset = TabularDataset(x_test_copy)
        dataloader = DataLoader(dataset, batch_size=min(200,dataset.__len__()), shuffle=False)
        res = []
        with torch.no_grad():
            for x in dataloader:
                res1 = self.branch_probs(x.to(self.device))
                res.append(res1.to('cpu'))
            res = torch.vstack(res)
        if was_training:
            self.train()
        return res

    def predict_condition_branch_proba(
        self,
        x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        tau: float = 1.0,
        soft_and: str = "geomean",
        batch_size: int | None = 4096,
    ) -> torch.Tensor:
        """Predict branch activations from explicit stored conditions.

        This is the NSToolkit condition-aware path: each branch score is
        a differentiable soft-AND over its symbolic ``Condition`` atoms
        rather than only the learned neural hidden unit.
        """
        from problog_inference import compute_condition_activation

        if not self.branches:
            raise RuntimeError("No branches stored — call build_model_from_ensemble first")
        x_np = convert_to_tensor(x_test).float().cpu().numpy()
        if batch_size is not None and len(x_np) > int(batch_size):
            chunks = []
            for start in range(0, len(x_np), int(batch_size)):
                chunk = x_np[start:start + int(batch_size)]
                z_chunk = compute_condition_activation(
                    self.branches, chunk, tau=tau, soft_and=soft_and,
                )
                chunks.append(torch.from_numpy(z_chunk).float())
            return torch.vstack(chunks)
        z = compute_condition_activation(
            self.branches, x_np, tau=tau, soft_and=soft_and,
        )
        return torch.from_numpy(z).float()

    def get_rule_reliability(self) -> np.ndarray:
        """Return the current per-rule reliability vector."""
        if self.rule_reliability_ is None:
            return np.ones(self.hidden_neurons, dtype=np.float32)
        if isinstance(self.rule_reliability_, torch.Tensor):
            return self.rule_reliability_.detach().cpu().numpy().astype(np.float32)
        return np.asarray(self.rule_reliability_, dtype=np.float32)

    def set_rule_reliability(self, reliability: np.ndarray) -> "RuleNetworkModel":
        """Attach per-rule reliability weights to the model."""
        r = np.asarray(reliability, dtype=np.float32).reshape(-1)
        if r.shape[0] != self.hidden_neurons:
            raise ValueError(
                f"Expected {self.hidden_neurons} reliability values, got {r.shape[0]}"
            )
        self.rule_reliability_ = torch.tensor(
            np.clip(r, 0.0, 1.0), dtype=self.dtype, device=self.device,
        )
        return self

    def fit_rule_reliability(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        theta_np: Optional[np.ndarray] = None,
        min_coverage: float = 1e-3,
    ) -> np.ndarray:
        """Estimate and store validation reliability for every rule."""
        from problog_inference import build_theta_matrix, estimate_rule_reliability

        if theta_np is None:
            theta_np = build_theta_matrix(self.branches, self.out_features)
        reliability = estimate_rule_reliability(
            self.branches,
            X_val,
            y_val,
            self.out_features,
            theta=theta_np,
            min_coverage=min_coverage,
        )
        self.set_rule_reliability(reliability)
        return reliability

    def predict_rule_head_proba(
        self,
        x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        theta_np: Optional[np.ndarray] = None,
        activation: str = "hybrid",
        tau: float = 1.0,
        soft_and: str = "geomean",
        hybrid_lam: float = 0.5,
        aggregation: str = "weighted_mean",
        use_reliability: bool = True,
        conflict_penalty: float = 0.0,
        return_diagnostics: bool = False,
        include_activation_diagnostics: bool = False,
        batch_size: int | None = 4096,
    ):
        """Advanced condition-aware rule-head prediction.

        Matches the NSToolkit rule head:
        ``activation='neural'`` keeps learned hidden activations,
        ``'condition'`` uses explicit soft rule satisfaction, and
        ``'hybrid'`` geometrically combines both.
        """
        from problog_inference import (
            aggregate_noisy_or,
            aggregate_weighted_mean,
            apply_rule_reliability,
            build_theta_matrix,
            combine_rule_activations,
            compute_condition_activation,
            compute_rule_uncertainty,
        )

        x_np = convert_to_tensor(x_test).float().cpu().numpy()
        if not self.branches:
            raise RuntimeError("No branches stored — call build_model_from_ensemble first")
        if batch_size is not None and len(x_np) > int(batch_size):
            proba_chunks = []
            diag_chunks: dict[str, list] = {}
            for start in range(0, len(x_np), int(batch_size)):
                chunk = x_np[start:start + int(batch_size)]
                out = self.predict_rule_head_proba(
                    chunk,
                    theta_np=theta_np,
                    activation=activation,
                    tau=tau,
                    soft_and=soft_and,
                    hybrid_lam=hybrid_lam,
                    aggregation=aggregation,
                    use_reliability=use_reliability,
                    conflict_penalty=conflict_penalty,
                    return_diagnostics=return_diagnostics,
                    include_activation_diagnostics=include_activation_diagnostics,
                    batch_size=None,
                )
                if return_diagnostics:
                    p_chunk, d_chunk = out
                    proba_chunks.append(p_chunk)
                    for key, value in d_chunk.items():
                        if isinstance(value, np.ndarray) and key != "reliability":
                            diag_chunks.setdefault(key, []).append(value)
                        elif key not in diag_chunks:
                            diag_chunks[key] = value
                else:
                    proba_chunks.append(out)
            proba = torch.vstack(proba_chunks)
            if return_diagnostics:
                diagnostics = {}
                for key, value in diag_chunks.items():
                    diagnostics[key] = (
                        np.concatenate(value, axis=0)
                        if isinstance(value, list)
                        else value
                    )
                return proba, diagnostics
            return proba

        neural_z = self.predict_branch_proba(x_test).numpy()
        condition_z = compute_condition_activation(
            self.branches, x_np, tau=tau, soft_and=soft_and,
        )
        z = combine_rule_activations(
            neural_z, condition_z, mode=activation, lam=hybrid_lam,
        )
        theta = (
            np.asarray(theta_np, dtype=np.float64)
            if theta_np is not None
            else build_theta_matrix(self.branches, self.out_features)
        )
        reliability = self.get_rule_reliability() if use_reliability else None
        z_eff = apply_rule_reliability(z, reliability)

        if aggregation == "weighted_mean":
            proba = aggregate_weighted_mean(
                z_eff, theta, conflict_penalty=conflict_penalty,
            )
        elif aggregation == "noisy_or":
            proba = aggregate_noisy_or(z_eff, theta)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        if return_diagnostics:
            diagnostics = compute_rule_uncertainty(
                proba, z, theta, reliability=reliability,
            )
            diagnostics["reliability"] = reliability
            if include_activation_diagnostics:
                diagnostics.update({
                    "neural_z": neural_z,
                    "condition_z": condition_z,
                    "effective_z": z_eff,
                })
            return torch.from_numpy(proba).float(), diagnostics
        return torch.from_numpy(proba).float()

    def predict_rule_uncertainty(
        self,
        x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        theta_np: Optional[np.ndarray] = None,
        **kwargs,
    ) -> dict:
        """Return uncertainty/conflict diagnostics for rule-head predictions."""
        _, diagnostics = self.predict_rule_head_proba(
            x_test, theta_np=theta_np, return_diagnostics=True, **kwargs,
        )
        return diagnostics

    def predict_proba(self, x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        x_test_copy = convert_to_tensor(x_test).float().to(self.device)
        dataset = TabularDataset(x_test_copy)
        dataloader = DataLoader(dataset, batch_size=min(200,dataset.__len__()), shuffle=False)
        res = []
        with torch.no_grad():
            for x in dataloader:
                res1 = torch.softmax(self.forward(x.to(self.device)),dim=1)
                res.append(res1.to('cpu'))
            res = torch.vstack(res)
        if was_training:
            self.train()
        return res

    def predict_problog(
        self,
        x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        mode: str = "fast",
        verbose: bool = True,
    ) -> torch.Tensor:
        """Predict using ProbLog inference instead of softmax(W2 @ h).

        Pipeline:
            1. Neural net computes P(z(b,X)) for every branch
            2. For each sample, a ProbLog program is built with:
               - z(b,x_id) latent facts
               - classification head: theta_bk::supports(k,b,X) :- z(b,X)
               - class aggregation:  class(X,K) :- supports(K,_,X)
               - (in "full" mode: + manifestation rules + evidence)
            3. ProbLog inference yields P(class(x_id, k))
            4. argmax gives predicted class

        Parameters
        ----------
        mode : str
            "fast" — only z + classification head (recommended)
            "full" — adds manifestation + evidence (slower, for research)

        Returns
        -------
        torch.Tensor of shape [n_samples] — predicted class labels
        """
        from problog_inference import ProbLogClassifier

        if not self.branches:
            raise RuntimeError("No branches stored — call build_model_from_ensemble first")

        bp = self.predict_branch_proba(x_test)
        bp_np = bp.numpy()
        x_np = convert_to_tensor(x_test).float().numpy()

        clf = ProbLogClassifier(
            branches=self.branches,
            n_classes=self.out_features,
            mode=mode,
        )
        preds = clf.predict(bp_np, x_np, verbose=verbose)
        return torch.from_numpy(preds).long()

    def predict_problog_proba(
        self,
        x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        mode: str = "fast",
        verbose: bool = True,
    ) -> torch.Tensor:
        """Predict class probabilities using ProbLog inference.

        Returns
        -------
        torch.Tensor of shape [n_samples, n_classes]
        """
        from problog_inference import ProbLogClassifier

        if not self.branches:
            raise RuntimeError("No branches stored — call build_model_from_ensemble first")

        bp = self.predict_branch_proba(x_test)
        bp_np = bp.numpy()
        x_np = convert_to_tensor(x_test).float().numpy()

        clf = ProbLogClassifier(
            branches=self.branches,
            n_classes=self.out_features,
            mode=mode,
        )
        proba = clf.predict_proba(bp_np, x_np, verbose=verbose)
        return torch.from_numpy(proba).float()

    def predict_problog_with_explanations(
        self,
        x_test: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        mode: str = "fast",
        top_k_branches: int = 5,
        verbose: bool = True,
    ):
        """Predict with per-sample symbolic explanations.

        Returns
        -------
        predictions : torch.Tensor of shape [n_samples]
        explanations : list of dicts (one per sample) with:
            - predicted_class
            - class_probabilities
            - top_branches: list of contributing branches with theta, P(z), conditions
        """
        from problog_inference import ProbLogClassifier

        if not self.branches:
            raise RuntimeError("No branches stored — call build_model_from_ensemble first")

        bp = self.predict_branch_proba(x_test)
        bp_np = bp.numpy()
        x_np = convert_to_tensor(x_test).float().numpy()

        clf = ProbLogClassifier(
            branches=self.branches,
            n_classes=self.out_features,
            mode=mode,
        )
        preds, explanations = clf.predict_with_explanations(
            bp_np, x_np, top_k_branches=top_k_branches, verbose=verbose,
        )
        return torch.from_numpy(preds).long(), explanations

    # ── ProbLog-aware training methods ─────────────────────

    def fit_problog_finetune(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        theta_np: np.ndarray,
        epochs: int = 50,
        lr: float = 0.001,
    ) -> "RuleNetworkModel":
        """Fine-tune W1 + BN using ProbLog weighted-mean as output head.

        After standard RuleNetwork training, this adjusts the hidden-layer
        weights so that activations P(z) work better with θ-based
        weighted-mean aggregation rather than the frozen W2 head.
        """
        x_tr = torch.from_numpy(x_train).float()
        y_tr = torch.from_numpy(y_train.ravel()).long()
        x_v = torch.from_numpy(x_val).float()
        y_v = torch.from_numpy(y_val.ravel()).long()
        th = torch.tensor(theta_np, dtype=torch.float32).to(self.device)

        params = [self.w1]
        for m in [self.bn0, self.bn1, self.bn2]:
            params.extend(m.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        best_vl, pat, best_sd = float("inf"), 0, None
        ds = TabularDataset(x_tr, y_tr.unsqueeze(1).float())
        loader = DataLoader(ds, batch_size=min(256, len(x_tr)),
                            shuffle=True, drop_last=True)

        for _ in range(epochs):
            self.train()
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.squeeze(1).long().to(self.device)
                h = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xb), self.w1 * self.m1)))
                p = (h @ th) / (h.sum(1, keepdim=True) + 1e-15)
                loss = F.nll_loss(torch.log(p + 1e-15), yb)
                opt.zero_grad(); loss.backward(); opt.step()

            self.eval()
            with torch.no_grad():
                hv = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(x_v.to(self.device)),
                             self.w1 * self.m1)))
                pv = (hv @ th) / (hv.sum(1, keepdim=True) + 1e-15)
                vl = F.nll_loss(torch.log(pv + 1e-15),
                                y_v.to(self.device)).item()
            if vl < best_vl:
                best_vl = vl
                best_sd = {k: v.clone() for k, v in self.state_dict().items()}
                pat = 0
            else:
                pat += 1
            if pat >= 15:
                break

        if best_sd:
            self.load_state_dict(best_sd)
        self.eval()
        return self

    def fit_dual_head(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        theta_np: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        lambda_w2: float = 0.5,
    ) -> "RuleNetworkModel":
        """Train from scratch with dual W2 + ProbLog-wmean loss.

        Combined loss = λ · CE(W2_head) + (1−λ) · NLL(wmean_head).
        Both heads share the same hidden layer; W2 remains frozen.
        This trains W1 to produce activations useful for *both* the
        frozen rule-network head and the probabilistic wmean head.
        """
        x_tr = torch.from_numpy(x_train).float()
        y_tr = torch.from_numpy(y_train.ravel()).long()
        x_v = torch.from_numpy(x_val).float()
        y_v = torch.from_numpy(y_val.ravel()).long()
        th = torch.tensor(theta_np, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = CosineAnnealingWarmRestarts(opt, T_0=180)

        best_vl, pat = float("inf"), 0
        ds = TabularDataset(x_tr, y_tr.unsqueeze(1).float())
        loader = DataLoader(ds, batch_size=min(256, len(x_tr)),
                            shuffle=True, drop_last=True)

        pbar = tqdm(range(epochs), desc="DualHead")
        for ep in pbar:
            self.train()
            el = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.squeeze(1).long().to(self.device)
                # Shared hidden layer
                h = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xb), self.w1 * self.m1)))
                # W2 head
                out_w2 = F.linear(self.bn2(h), self.w2)
                loss_w2 = criterion(out_w2, yb)
                # ProbLog wmean head
                p = (h @ th) / (h.sum(1, keepdim=True) + 1e-15)
                loss_pl = F.nll_loss(torch.log(p + 1e-15), yb)
                loss = lambda_w2 * loss_w2 + (1 - lambda_w2) * loss_pl
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item()
            pbar.set_description(f"DH: {el/max(len(loader),1):.4f}")

            self.eval()
            with torch.no_grad():
                xv = x_v.to(self.device)
                hv = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xv), self.w1 * self.m1)))
                vl_w2 = criterion(F.linear(self.bn2(hv), self.w2),
                                  y_v.to(self.device)).item()
                pv = (hv @ th) / (hv.sum(1, keepdim=True) + 1e-15)
                vl_pl = F.nll_loss(torch.log(pv + 1e-15),
                                   y_v.to(self.device)).item()
                vl = lambda_w2 * vl_w2 + (1 - lambda_w2) * vl_pl
            sch.step(vl)
            if vl < best_vl:
                best_vl = vl
                best_state_dh = copy.deepcopy(self.state_dict())
                pat = 0
            else:
                pat += 1
            if pat >= 100:
                break

        if ep < epochs - 1 and best_state_dh is not None:
            self.load_state_dict(best_state_dh)
        del sch
        self.eval()
        return self

    def fit_dual_head_e2e(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        theta_np: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        lambda_w2: float = 0.5,
    ) -> tuple:
        """Dual-head training with **learnable θ and α** (end-to-end).

        Combined loss = λ · CE(W2_head) + (1−λ) · NLL(wmean(θ_trainable, α)).
        Unlike fit_dual_head, θ and α are co-optimised with W1, so the ProbLog
        head becomes as aligned with W1 as W2 is.

        Returns (self, theta_learned, alpha_learned).
        """
        x_tr = torch.from_numpy(x_train).float()
        y_tr = torch.from_numpy(y_train.ravel()).long()
        x_v = torch.from_numpy(x_val).float()
        y_v = torch.from_numpy(y_val.ravel()).long()

        n_br = theta_np.shape[0]
        # Trainable θ (softmax-parameterised) and α (softplus-parameterised)
        theta_logits = nn.Parameter(
            torch.tensor(np.log(np.clip(theta_np, 1e-6, 1.0)),
                         dtype=torch.float32).to(self.device))
        alpha_raw = nn.Parameter(torch.zeros(n_br, device=self.device))

        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(
            [{'params': self.parameters(), 'lr': lr},
             {'params': [theta_logits, alpha_raw], 'lr': lr}])
        sch = CosineAnnealingWarmRestarts(opt, T_0=180)

        best_vl, pat = float("inf"), 0
        ds = TabularDataset(x_tr, y_tr.unsqueeze(1).float())
        loader = DataLoader(ds, batch_size=min(256, len(x_tr)),
                            shuffle=True, drop_last=True)

        pbar = tqdm(range(epochs), desc="DH-E2E")
        for ep in pbar:
            self.train()
            el = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.squeeze(1).long().to(self.device)
                h = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xb), self.w1 * self.m1)))
                # W2 head
                out_w2 = F.linear(self.bn2(h), self.w2)
                loss_w2 = criterion(out_w2, yb)
                # ProbLog wmean head with trainable θ, α
                th_soft = torch.softmax(theta_logits, dim=1)
                a = F.softplus(alpha_raw)
                wz = h * a.unsqueeze(0)
                p = (wz @ th_soft) / (wz.sum(1, keepdim=True) + 1e-15)
                loss_pl = F.nll_loss(torch.log(p + 1e-15), yb)
                loss = lambda_w2 * loss_w2 + (1 - lambda_w2) * loss_pl
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item()
            pbar.set_description(f"E2E: {el/max(len(loader),1):.4f}")

            self.eval()
            with torch.no_grad():
                xv = x_v.to(self.device)
                hv = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xv), self.w1 * self.m1)))
                vl_w2 = criterion(F.linear(self.bn2(hv), self.w2),
                                  y_v.to(self.device)).item()
                th_s = torch.softmax(theta_logits, dim=1)
                a_s = F.softplus(alpha_raw)
                wzv = hv * a_s.unsqueeze(0)
                pv = (wzv @ th_s) / (wzv.sum(1, keepdim=True) + 1e-15)
                vl_pl = F.nll_loss(torch.log(pv + 1e-15),
                                   y_v.to(self.device)).item()
                vl = lambda_w2 * vl_w2 + (1 - lambda_w2) * vl_pl
            sch.step(vl)
            if vl < best_vl:
                best_vl = vl
                best_ckpt_e2e = {
                    'model': copy.deepcopy(self.state_dict()),
                    'theta_logits': theta_logits.data.clone(),
                    'alpha_raw': alpha_raw.data.clone(),
                }
                pat = 0
            else:
                pat += 1
            if pat >= 100:
                break

        if ep < epochs - 1 and best_ckpt_e2e is not None:
            self.load_state_dict(best_ckpt_e2e['model'])
            theta_logits.data = best_ckpt_e2e['theta_logits']
            alpha_raw.data = best_ckpt_e2e['alpha_raw']
        del sch
        self.eval()

        theta_out = torch.softmax(theta_logits, dim=1).detach().cpu().numpy()
        alpha_out = F.softplus(alpha_raw).detach().cpu().numpy()
        return self, theta_out, alpha_out

    def fit_problog_pure(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        theta_np: np.ndarray,
        epochs: int = 500,
        lr: float = 0.01,
        loss_file: Optional[str] = None,
    ) -> "RuleNetworkModel":
        """Train W1 from scratch using ONLY ProbLog weighted-mean loss.

        Unlike ``fit_dual_head`` which combines W2 + ProbLog losses,
        this method uses **exclusively** the probabilistic head:
            P(k|X) = Σ_b (z_b · θ_{b,k}) / Σ_b z_b   (weighted mean)
            Loss   = NLL(log P, y)

        The frozen W2 head is never used during training.  This means
        that W1 (and BN parameters) are optimised solely to produce
        branch activations z = σ(BN₁(W1·M1 · BN₀(x))) that work well
        with the ProbLog θ-based weighted-mean aggregation.

        At inference time, the resulting model can be combined with any
        ProbLog aggregation variant (wmean, noisy-or, posterior, top-K,
        learned α, temperature, etc.).

        Parameters
        ----------
        x_train, y_train : training data
        x_val, y_val : validation data (for early stopping)
        theta_np : [n_branches, n_classes] — theta matrix from build_theta_matrix
        epochs : int — maximum training epochs (default: 500)
        lr : float — learning rate
        loss_file : optional path to save loss plot
        """
        x_tr = torch.from_numpy(x_train).float()
        y_tr = torch.from_numpy(y_train.ravel()).long()
        x_v = torch.from_numpy(x_val).float()
        y_v = torch.from_numpy(y_val.ravel()).long()
        th = torch.tensor(theta_np, dtype=torch.float32).to(self.device)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = CosineAnnealingWarmRestarts(opt, T_0=180)

        best_vl, pat = float("inf"), 0
        ds = TabularDataset(x_tr, y_tr.unsqueeze(1).float())
        loader = DataLoader(ds, batch_size=min(256, len(x_tr)),
                            shuffle=True, drop_last=True)

        loss_history, val_loss_history = [], []
        pbar = tqdm(range(epochs), desc="PurePL")
        for ep in pbar:
            self.train()
            el = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.squeeze(1).long().to(self.device)
                # Branch activations: z_b = σ(BN₁(W1·M1 · BN₀(x)))
                h = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xb), self.w1 * self.m1)))
                # Weighted mean: P(k|X) = Σ(z_b · θ_bk) / Σ z_b
                p = (h @ th) / (h.sum(1, keepdim=True) + 1e-15)
                loss = F.nll_loss(torch.log(p + 1e-15), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item()
            avg_loss = el / max(len(loader), 1)
            loss_history.append(avg_loss)
            pbar.set_description(f"PP: {avg_loss:.4f}")

            # Validation
            self.eval()
            with torch.no_grad():
                xv = x_v.to(self.device)
                hv = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xv), self.w1 * self.m1)))
                pv = (hv @ th) / (hv.sum(1, keepdim=True) + 1e-15)
                vl = F.nll_loss(torch.log(pv + 1e-15),
                                y_v.to(self.device)).item()
            val_loss_history.append(vl)
            sch.step(vl)

            if vl < best_vl:
                best_vl = vl
                best_state_pp = copy.deepcopy(self.state_dict())
                pat = 0
            else:
                pat += 1
            if pat >= 100:
                break

        if ep < epochs - 1 and best_state_pp is not None:
            self.load_state_dict(best_state_pp)

        # Plot loss
        _ = plt.figure(figsize=(12, 8))
        plt.plot(loss_history[5:], label="train")
        plt.plot(val_loss_history[5:], label="val")
        plt.title("Pure ProbLog Training Loss")
        plt.legend()
        if loss_file is not None:
            plt.savefig(loss_file)
        plt.close()

        del sch
        self.eval()
        return self

    def fit_problog_pure_theta(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        theta_np: np.ndarray,
        epochs: int = 500,
        lr: float = 0.01,
    ) -> Tuple["RuleNetworkModel", np.ndarray]:
        """Train W1 + θ end-to-end using ONLY ProbLog weighted-mean loss.

        Unlike ``fit_problog_pure`` where θ is frozen, here θ is a
        **trainable parameter** that starts from the forest-derived values
        but is refined jointly with W1.  This addresses the θ-mismatch
        problem: as W1 changes during training, θ adapts accordingly.

        Returns
        -------
        self : RuleNetworkModel (updated in place)
        theta_learned : np.ndarray [n_branches, n_classes] — learned θ
        """
        x_tr = torch.from_numpy(x_train).float()
        y_tr = torch.from_numpy(y_train.ravel()).long()
        x_v = torch.from_numpy(x_val).float()
        y_v = torch.from_numpy(y_val.ravel()).long()

        # θ as trainable parameter (logit-space for unconstrained optim)
        th_logit = torch.nn.Parameter(
            torch.log(torch.tensor(theta_np, dtype=torch.float32) + 1e-8)
        )

        # Optimise W1 (rule-network params) + theta together.
        opt = torch.optim.Adam(
            list(self.parameters()) + [th_logit], lr=lr
        )
        sch = CosineAnnealingWarmRestarts(opt, T_0=180)

        best_vl, pat = float("inf"), 0
        best_th = theta_np.copy()
        ds = TabularDataset(x_tr, y_tr.unsqueeze(1).float())
        loader = DataLoader(ds, batch_size=min(256, len(x_tr)),
                            shuffle=True, drop_last=True)

        pbar = tqdm(range(epochs), desc="PP-θ")
        for ep in pbar:
            self.train()
            el = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.squeeze(1).long().to(self.device)
                th = F.softmax(th_logit, dim=1)
                h = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xb), self.w1 * self.m1)))
                p = (h @ th) / (h.sum(1, keepdim=True) + 1e-15)
                loss = F.nll_loss(torch.log(p + 1e-15), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item()
            avg_loss = el / max(len(loader), 1)
            pbar.set_description(f"PPθ: {avg_loss:.4f}")

            # Validation
            self.eval()
            with torch.no_grad():
                th = F.softmax(th_logit, dim=1)
                xv = x_v.to(self.device)
                hv = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xv), self.w1 * self.m1)))
                pv = (hv @ th) / (hv.sum(1, keepdim=True) + 1e-15)
                vl = F.nll_loss(torch.log(pv + 1e-15),
                                y_v.to(self.device)).item()
            sch.step(vl)

            if vl < best_vl:
                best_vl = vl
                best_state_ppth = copy.deepcopy(self.state_dict())
                best_th = F.softmax(th_logit, dim=1).detach().cpu().numpy()
                pat = 0
            else:
                pat += 1
            if pat >= 100:
                break

        if ep < epochs - 1 and best_state_ppth is not None:
            self.load_state_dict(best_state_ppth)

        self.eval()
        return self, best_th

    # ──────────────────────────────────────────────────────────
    # Posterior-through end-to-end ProbLog training
    # ──────────────────────────────────────────────────────────

    def fit_problog_posterior_e2e(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        theta_np: np.ndarray,
        epochs: int = 400,
        lr: float = 0.01,
        tau: float = 0.1,
        batch_size: int = 256,
    ) -> Tuple["RuleNetworkModel", np.ndarray]:
        """Train W1 + θ end-to-end with **differentiable Bayesian posterior**.

        This is the TRUE neuro-symbolic ProbLog training:
            1. z_prior = σ(BN(W1 · M1 · x))
            2. z_post  = P(z | evidence)  — differentiable!
            3. P(k|x)  = Σ_b z_post_b · θ_bk / Σ_b z_post_b
            4. loss = NLL(P(k|x), y)
            5. Backprop through (4)→(3)→(2)→(1) → update W1 and θ

        The posterior is computed using soft sigmoid condition matching
        so that gradients flow through the evidence evaluation.

        Parameters
        ----------
        x_train, y_train : training data
        x_val, y_val     : validation data (for early stopping)
        theta_np : initial theta from forest [n_branches, n_classes]
        epochs   : max training epochs
        lr       : learning rate
        tau      : temperature for soft condition matching
        batch_size : training and validation mini-batch size. Validation is
                     streamed in chunks so large held-out splits do not need
                     one full posterior tensor in memory.

        Returns
        -------
        self         : RuleNetworkModel (updated in place)
        theta_learned : np.ndarray [n_branches, n_classes]
        """
        from problog_inference import DifferentiablePosterior

        x_tr = torch.from_numpy(x_train).float()
        y_tr = torch.from_numpy(y_train.ravel()).long()
        x_v = torch.from_numpy(x_val).float()
        y_v = torch.from_numpy(y_val.ravel()).long()

        # Differentiable posterior module
        diff_post = DifferentiablePosterior(
            self.branches, p_high=0.95, p_low=0.05, tau=tau
        )

        # θ as trainable parameter (logit-space)
        th_logit = torch.nn.Parameter(
            torch.log(torch.tensor(theta_np, dtype=torch.float32) + 1e-8)
        )

        opt = torch.optim.Adam(
            list(self.parameters()) + [th_logit], lr=lr
        )
        sch = CosineAnnealingWarmRestarts(opt, T_0=180)

        best_vl, pat = float("inf"), 0
        best_state_ppth_post = None
        best_th = theta_np.copy()
        ds = TabularDataset(x_tr, y_tr.unsqueeze(1).float())
        train_bs = max(1, min(int(batch_size), len(x_tr)))
        loader = DataLoader(ds, batch_size=train_bs,
                            shuffle=True, drop_last=True)
        val_ds = TabularDataset(x_v, y_v.unsqueeze(1).float())
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, min(train_bs, len(x_v))),
            shuffle=False,
            drop_last=False,
        )

        pbar = tqdm(range(epochs), desc="PPθ-Post")
        for ep in pbar:
            self.train()
            el = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.squeeze(1).long().to(self.device)
                th = F.softmax(th_logit, dim=1)

                # z_prior from neural net
                z_prior = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xb), self.w1 * self.m1)))

                # Differentiable Bayesian posterior P(z|evidence)
                z_post = diff_post(z_prior, xb)

                # Weighted mean aggregation with posterior z
                p = (z_post @ th) / (z_post.sum(1, keepdim=True) + 1e-15)
                loss = F.nll_loss(torch.log(p + 1e-15), yb)

                opt.zero_grad()
                loss.backward()
                opt.step()
                el += loss.item()

            avg_loss = el / max(len(loader), 1)
            pbar.set_description(f"PPθ-Post: {avg_loss:.4f}")

            # Validation
            self.eval()
            val_loss, val_n = 0.0, 0
            with torch.no_grad():
                th = F.softmax(th_logit, dim=1)
                for xb_v, yb_v in val_loader:
                    xb_v = xb_v.to(self.device)
                    yb_v = yb_v.squeeze(1).long().to(self.device)
                    z_prior_v = torch.sigmoid(self.bn1(
                        F.linear(self.bn0(xb_v), self.w1 * self.m1)))
                    z_post_v = diff_post(z_prior_v, xb_v)
                    pv = (
                        z_post_v @ th
                    ) / (z_post_v.sum(1, keepdim=True) + 1e-15)
                    loss_v = F.nll_loss(
                        torch.log(pv + 1e-15),
                        yb_v,
                        reduction="sum",
                    )
                    val_loss += float(loss_v.item())
                    val_n += int(len(yb_v))
                vl = val_loss / max(val_n, 1)
            sch.step(vl)

            if vl < best_vl:
                best_vl = vl
                best_state_ppth_post = copy.deepcopy(self.state_dict())
                best_th = F.softmax(th_logit, dim=1).detach().cpu().numpy()
                pat = 0
            else:
                pat += 1
            if pat >= 100:
                break

        if ep < epochs - 1 and best_state_ppth_post is not None:
            self.load_state_dict(best_state_ppth_post)

        self.eval()
        return self, best_th

    # ──────────────────────────────────────────────────────────
    # e2e-NoisyOr: differentiable noisy-or surrogate of native ProbLog
    # ──────────────────────────────────────────────────────────

    def fit_e2e_noisy_or(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        theta_np: np.ndarray,
        epochs: int = 400,
        lr: float = 0.01,
        tau: float = 0.1,
        use_posterior: bool = True,
        consistency_weight: float = 0.0,
        consistency_every: int = 5,
        consistency_n: int = 64,
        consistency_seed: int = 0,
        batch_size: int = 256,
    ) -> Tuple["RuleNetworkModel", np.ndarray]:
        """Train W1 + θ end-to-end with **differentiable noisy-or** aggregation.

        This is a direct PyTorch surrogate of the native ProbLog program
        ``θ::supports(B,C,X) :- branch(B,X). class(X,C) :- supports(B,C,X).``
        whose semantics under independent evidence is exactly noisy-or:
        ``P(class=c|x) = 1 − Π_b (1 − θ_{bc} · z_b)``.

        The same posterior P(z|evidence) that the native ProbLog engine
        computes by enumeration is computed here in closed form by
        :class:`DifferentiablePosterior` (soft sigmoid condition matching
        + log-sum-exp Bayes update), so a single forward pass replaces
        seconds of native ProbLog inference per batch and — crucially —
        admits gradients w.r.t. both W1 and θ.

        Differences from :meth:`fit_problog_posterior_e2e`:
            * θ is parametrised element-wise via ``σ(θ_logit) ∈ (0,1)``
              (independent branch→class probabilities), matching noisy-or
              semantics rather than a softmax distribution.
            * Aggregation is noisy-or in log-space (numerically stable
              via ``log1p(−θ·z)``) instead of weighted mean.
            * ``use_posterior=False`` falls back to the prior branch
              activations, mirroring the cheap ``pl_fast`` ProbLog mode.

        Parameters
        ----------
        x_train, y_train : training data
        x_val, y_val     : validation data (early stopping)
        theta_np         : initial θ from forest [n_branches, n_classes]
        epochs, lr       : optimiser settings
        tau              : temperature for soft condition matching
        use_posterior    : if True, run DifferentiablePosterior on the
                           branch activations (PL-full surrogate); if
                           False, use the raw prior z (PL-fast surrogate).
        consistency_weight : λ for the optional ProbLog-anchor loss
                           ``λ · MSE(p_torch(x), p_problog(x))`` evaluated
                           on a small subsample of training points using
                           the **analytical** ProbLog inference (exact
                           Bayesian posterior + numpy noisy-or with the
                           current θ). 0 disables the regulariser.
        consistency_every : run the anchor step every N epochs.
        consistency_n     : number of training points to sample for the
                           anchor (kept small — analytical ProbLog is
                           the slow ground truth here).
        consistency_seed  : RNG seed for the anchor subsample.
        batch_size        : training and validation mini-batch size. The
                            validation pass is chunked for large datasets.

        Returns
        -------
        self          : RuleNetworkModel (updated in place)
        theta_learned : np.ndarray [n_branches, n_classes]
        """
        from problog_inference import (
            DifferentiablePosterior,
            ProbLogClassifier,
            aggregate_noisy_or,
        )

        x_tr = torch.from_numpy(x_train).float()
        y_tr = torch.from_numpy(y_train.ravel()).long()
        x_v = torch.from_numpy(x_val).float()
        y_v = torch.from_numpy(y_val.ravel()).long()

        diff_post = (
            DifferentiablePosterior(
                self.branches, p_high=0.95, p_low=0.05, tau=tau
            )
            if use_posterior else None
        )

        th_init = np.clip(theta_np.astype(np.float32), 1e-4, 1 - 1e-4)
        th_logit = torch.nn.Parameter(
            torch.log(torch.from_numpy(th_init) / (1.0 - torch.from_numpy(th_init)))
        )

        opt = torch.optim.Adam(
            list(self.parameters()) + [th_logit], lr=lr
        )
        sch = CosineAnnealingWarmRestarts(opt, T_0=180)

        best_vl, pat = float("inf"), 0
        best_th = theta_np.copy()
        best_state_e2e_nor = None
        ds = TabularDataset(x_tr, y_tr.unsqueeze(1).float())
        train_bs = max(1, min(int(batch_size), len(x_tr)))
        loader = DataLoader(ds, batch_size=train_bs,
                            shuffle=True, drop_last=True)
        val_ds = TabularDataset(x_v, y_v.unsqueeze(1).float())
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, min(train_bs, len(x_v))),
            shuffle=False,
            drop_last=False,
        )

        EPS = 1e-12
        CAP = 1.0 - 1e-6

        def _noisy_or_probs(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            """Row-normalised noisy-or class probabilities. Returns [B, C]."""
            p_support = z.unsqueeze(2) * theta.unsqueeze(0)
            p_support = p_support.clamp(0.0, CAP)
            log_class = torch.log1p(-p_support).sum(dim=1)
            class_prob = (1.0 - torch.exp(log_class)).clamp_min(EPS)
            return class_prob / class_prob.sum(dim=1, keepdim=True).clamp_min(EPS)

        def _noisy_or_logp(z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            """log of row-normalised noisy-or probabilities. Returns [B, C]."""
            return torch.log(_noisy_or_probs(z, theta).clamp_min(EPS))

        n_classes_inferred = theta_np.shape[1]
        use_consistency = consistency_weight > 0 and consistency_every > 0
        clf_anchor = (
            ProbLogClassifier(self.branches, n_classes_inferred, mode="full")
            if use_consistency else None
        )
        rng = np.random.RandomState(consistency_seed) if use_consistency else None
        last_consistency = float("nan")

        pbar = tqdm(range(epochs), desc="e2e-NoisyOr")
        for ep in pbar:
            self.train()
            el = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.squeeze(1).long().to(self.device)
                th = torch.sigmoid(th_logit)

                z_prior = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(xb), self.w1 * self.m1)))
                z = diff_post(z_prior, xb) if diff_post is not None else z_prior

                logp = _noisy_or_logp(z, th)
                loss = F.nll_loss(logp, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()
                el += loss.item()

            avg_loss = el / max(len(loader), 1)
            if use_consistency:
                pbar.set_description(
                    f"e2e-NoisyOr: {avg_loss:.4f}  consist={last_consistency:.4f}")
            else:
                pbar.set_description(f"e2e-NoisyOr: {avg_loss:.4f}")

            # ── Optional ProbLog-anchor consistency step ───────────
            if use_consistency and ((ep + 1) % consistency_every == 0):
                self.eval()
                n_pool = len(x_train)
                k = int(min(consistency_n, n_pool))
                idx = rng.choice(n_pool, size=k, replace=False)
                x_sub = x_train[idx]

                with torch.no_grad():
                    bp_sub = self.predict_branch_proba(x_sub)
                    bp_sub_np = (bp_sub.detach().cpu().numpy()
                                 if isinstance(bp_sub, torch.Tensor) else np.asarray(bp_sub))
                    theta_now = torch.sigmoid(th_logit).detach().cpu().numpy()
                    post_sub = clf_anchor.get_posterior_z(bp_sub_np, x_sub)
                    p_ref = aggregate_noisy_or(post_sub, theta_now)
                p_ref_t = torch.from_numpy(p_ref).float().to(self.device)

                self.train()
                x_sub_t = torch.from_numpy(x_sub).float().to(self.device)
                th = torch.sigmoid(th_logit)
                z_prior_s = torch.sigmoid(self.bn1(
                    F.linear(self.bn0(x_sub_t), self.w1 * self.m1)))
                z_s = diff_post(z_prior_s, x_sub_t) if diff_post is not None else z_prior_s
                p_torch = _noisy_or_probs(z_s, th)
                cons_loss = consistency_weight * F.mse_loss(p_torch, p_ref_t)
                opt.zero_grad()
                cons_loss.backward()
                opt.step()
                last_consistency = cons_loss.item()

            self.eval()
            val_loss, val_n = 0.0, 0
            with torch.no_grad():
                th = torch.sigmoid(th_logit)
                for xb_v, yb_v in val_loader:
                    xb_v = xb_v.to(self.device)
                    yb_v = yb_v.squeeze(1).long().to(self.device)
                    z_prior_v = torch.sigmoid(self.bn1(
                        F.linear(self.bn0(xb_v), self.w1 * self.m1)))
                    z_v = (
                        diff_post(z_prior_v, xb_v)
                        if diff_post is not None else z_prior_v
                    )
                    logp_v = _noisy_or_logp(z_v, th)
                    loss_v = F.nll_loss(logp_v, yb_v, reduction="sum")
                    val_loss += float(loss_v.item())
                    val_n += int(len(yb_v))
                vl = val_loss / max(val_n, 1)
            sch.step(vl)

            if vl < best_vl:
                best_vl = vl
                best_state_e2e_nor = copy.deepcopy(self.state_dict())
                best_th = torch.sigmoid(th_logit).detach().cpu().numpy()
                pat = 0
            else:
                pat += 1
            if pat >= 100:
                break

        if best_state_e2e_nor is not None:
            self.load_state_dict(best_state_e2e_nor)

        self.eval()
        return self, best_th
