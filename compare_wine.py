#!/usr/bin/env python3
"""Comparison of RuleNetwork inference modes on Wine dataset.

Models compared (9 total):
    1. ExtraTrees           — pure ensemble baseline (no neural network)
    2. RuleNetwork-Neural     — neural forward: softmax(W2 · Sigmoid(BN(W1·m1·x)))
    3. PL-fast              — noisy-or over prior P(z), no evidence
    4. PL-full              — analytical Bayesian posterior P(z|ev) + noisy-or
    5. PL-wmean             — posterior + weighted-mean aggregation (no noisy-or)
    6. PL-fast-match        — P(z) * match_ratio + noisy-or (lightweight evidence)
    7. PL-wm-match          — P(z) * match_ratio + weighted-mean (best interpretable)
    8. PL-βAdMatch-wm       — match-informed adaptive tempered posterior + wmean (best Bayesian)
    9. PL-ens-3way          — α₁·Neural + α₂·wmean + α₃·fast-match (best overall)
    + ProbLog-engine        — ProbLog engine per sample (slow, verification only)

All PL variants share the same trained RuleNetwork; only inference head differs.

Results are saved to: compare_wine_results.txt
"""

import sys
import time
import datetime
import warnings
import numpy as np
import torch
from io import StringIO
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

from rule_network_model import RuleNetworkModel
from problog_inference import (
    ProbLogClassifier,
    build_theta_matrix,
    aggregate_noisy_or,
    aggregate_weighted_mean,
    compute_match_ratio,
    find_optimal_3way,
    compute_match_informed_adaptive_posterior,
)

# ─────────────────────────────────────────────────────────────
# Tee: print to console AND capture to string
# ─────────────────────────────────────────────────────────────
class Tee:
    def __init__(self):
        self.buf = StringIO()
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s)
        self.buf.write(s)
    def flush(self):
        self.stdout.flush()
    def getvalue(self):
        return self.buf.getvalue()

tee = Tee()
sys.stdout = tee

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SEED = 42
EPOCHS = 200
N_FOLDS = 5
# ProbLog-engine verification: run on first N samples of each fold
PROBLOG_VERIFY_N = 5
PROBLOG_VERIFY_TOP_K = 10
OUTPUT_FILE = "compare_wine_results.txt"

# βAdMatch hyperparameters
MATCH_BOOST = 0.5  # how much match_ratio influences adaptive β

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
data = load_wine()
X, y = data.data.astype(np.float32), data.target
feature_names = list(data.feature_names)
class_names = list(data.target_names)
n_features = X.shape[1]
n_classes = len(set(y))

print("=" * 120)
print("  RuleNetwork Inference Comparison on Wine Dataset  (9 models)")
print("  Baselines: ExtraTrees · RuleNetwork-Neural")
print("  ProbLog:   fast · full · wmean · fast-match · wm-match · βAdMatch-wm · ens-3way")
print("=" * 120)
print(f"Date             : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset          : Wine (sklearn)")
print(f"Samples          : {X.shape[0]}")
print(f"Features         : {n_features}")
print(f"Classes          : {n_classes}  {class_names}")
print(f"Class distr      : {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Epochs           : {EPOCHS}")
print(f"CV Folds         : {N_FOLDS}")
print(f"Random seed      : {SEED}")
print(f"ProbLog-engine   : verification on first {PROBLOG_VERIFY_N} samples/fold "
      f"(top-{PROBLOG_VERIFY_TOP_K} branches)")
print(f"Match boost      : {MATCH_BOOST} (for PL-βAdMatch-wm)")
print()

# Formulas from paper
the_number = round(np.log2(n_features)) + 4
n_estimators = n_classes + round(np.log2(n_features))
max_leaf = 2 ** the_number
print(f"ExtraTrees config (NSToolkit-style symbolic backbone):")
print(f"  n_estimators   = n_classes + floor(log2(n_features)) = {n_classes} + {round(np.log2(n_features))} = {n_estimators}")
print(f"  max_leaf_nodes = 2^(floor(log2(n_features))+4) = 2^{the_number} = {max_leaf}")
print()

# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def compute_all_metrics(y_true, y_pred, y_proba=None):
    m = {}
    m["accuracy"]          = accuracy_score(y_true, y_pred)
    m["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    m["f1_weighted"]       = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    m["f1_macro"]          = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1_micro"]          = f1_score(y_true, y_pred, average="micro", zero_division=0)
    m["precision_weighted"]= precision_score(y_true, y_pred, average="weighted", zero_division=0)
    m["precision_macro"]   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["recall_weighted"]   = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    m["recall_macro"]      = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m["mcc"]               = matthews_corrcoef(y_true, y_pred)
    m["cohen_kappa"]       = cohen_kappa_score(y_true, y_pred)
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == n_classes:
        try:
            m["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            m["roc_auc_ovr"] = float("nan")
        try:
            m["roc_auc_ovo"] = roc_auc_score(y_true, y_proba, multi_class="ovo", average="weighted")
        except Exception:
            m["roc_auc_ovo"] = float("nan")
        try:
            pc = np.clip(y_proba, 1e-15, 1 - 1e-15)
            pc = pc / pc.sum(axis=1, keepdims=True)
            m["log_loss"] = log_loss(y_true, pc, labels=list(range(n_classes)))
        except Exception:
            m["log_loss"] = float("nan")
    else:
        m["roc_auc_ovr"] = m["roc_auc_ovo"] = m["log_loss"] = float("nan")
    return m


def fmt(arr):
    a = np.array(arr); a = a[~np.isnan(a)]
    return f"{a.mean():7.4f} ± {a.std():.4f}" if len(a) else "       N/A      "


# ─────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

MODEL_NAMES = [
    "extratrees", "neural",
    "problog_fast", "problog_full", "full_wmean",
    "fast_match", "wmean_match",
    "adapt_match_wmean",
    "ens_3way",
]
MODEL_LABELS = {
    "extratrees":        "ExtraTrees",
    "neural":            "RuleNetwork-Neural",
    "problog_fast":      "PL-fast",
    "problog_full":      "PL-full",
    "full_wmean":        "PL-wmean",
    "fast_match":        "PL-fast-match",
    "wmean_match":       "PL-wm-match",
    "adapt_match_wmean": "PL-βAdMatch-wm",
    "ens_3way":          "PL-ens-3way",
}

all_metrics   = {n: [] for n in MODEL_NAMES}
all_confusion = {n: [] for n in MODEL_NAMES}
all_times = {
    "et_train": [], "et_infer": [],
    "bn_train": [],
    "neural_infer": [], "fast_infer": [], "full_infer": [],
    "wmean_infer": [], "fast_match_infer": [],
    "wmean_match_infer": [], "adapt_match_wmean_infer": [],
    "ens_3way_infer": [],
    "problog_engine_infer": [],
}
fold_branches = []

agreement = {
    "et_vs_neural": [], "neural_vs_fast": [],
    "neural_vs_full": [], "fast_vs_full": [],
    "neural_vs_wmean": [], "neural_vs_fast_match": [],
    "neural_vs_wmean_match": [], "neural_vs_adapt_match_wmean": [],
    "neural_vs_ens_3way": [],
}

# Store posterior diagnostics
posterior_shift = []  # mean |P(z|ev) - P(z)| per fold

total_start = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("-" * 90)
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}  |  train={len(train_idx)}  test={len(test_idx)}")
    print("-" * 90)

    # Reset seeds per fold for full reproducibility
    fold_seed = SEED + fold_idx
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    # ── 0. ExtraTrees ──────────────────────────────────────
    t0 = time.time()
    et = ExtraTreesClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf,
        random_state=SEED + fold_idx,
    )
    et.fit(X_train, y_train.ravel())
    dt_et_train = time.time() - t0
    all_times["et_train"].append(dt_et_train)

    t0 = time.time()
    pred_et = et.predict(X_test); proba_et = et.predict_proba(X_test)
    dt_et_infer = time.time() - t0
    all_times["et_infer"].append(dt_et_infer)

    m_et = compute_all_metrics(y_test, pred_et, proba_et)
    m_et["infer_time_s"] = dt_et_infer
    all_metrics["extratrees"].append(m_et)
    all_confusion["extratrees"].append(confusion_matrix(y_test, pred_et, labels=range(n_classes)))
    print(f"  [ExtraTrees]    Acc={m_et['accuracy']:.4f}  F1w={m_et['f1_weighted']:.4f}  "
          f"MCC={m_et['mcc']:.4f}  AUC={m_et['roc_auc_ovr']:.4f}  "
          f"Train={dt_et_train:.3f}s  Infer={dt_et_infer:.4f}s")

    # ── 1. RuleNetwork training ──────────────────────────────
    model = RuleNetworkModel(task="classification")
    t0 = time.time()
    model.build_model_from_ensemble(et)
    n_branches = len(model.branches)
    fold_branches.append(n_branches)
    model = model.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS)
    dt_bn = time.time() - t0
    all_times["bn_train"].append(dt_bn)
    print(f"  Branches: {n_branches}  |  RuleNetwork training: {dt_bn:.2f}s")

    bp = model.predict_branch_proba(X_test).numpy()

    # ── 2. Neural ──────────────────────────────────────────
    t0 = time.time()
    pred_neural = model.predict(X_test).numpy()
    proba_neural = model.predict_proba(X_test).numpy()
    dt_n = time.time() - t0
    all_times["neural_infer"].append(dt_n)

    m_n = compute_all_metrics(y_test, pred_neural, proba_neural)
    m_n["infer_time_s"] = dt_n
    all_metrics["neural"].append(m_n)
    all_confusion["neural"].append(confusion_matrix(y_test, pred_neural, labels=range(n_classes)))
    print(f"  [Neural]        Acc={m_n['accuracy']:.4f}  F1w={m_n['f1_weighted']:.4f}  "
          f"MCC={m_n['mcc']:.4f}  AUC={m_n['roc_auc_ovr']:.4f}  Infer={dt_n:.4f}s")

    # ── 3. ProbLog-fast ────────────────────────────────────
    t0 = time.time()
    clf_fast = ProbLogClassifier(model.branches, n_classes, mode="fast")
    proba_fast = clf_fast.predict_proba(bp)
    pred_fast = np.argmax(proba_fast, axis=1)
    dt_f = time.time() - t0
    all_times["fast_infer"].append(dt_f)

    m_f = compute_all_metrics(y_test, pred_fast, proba_fast)
    m_f["infer_time_s"] = dt_f
    all_metrics["problog_fast"].append(m_f)
    all_confusion["problog_fast"].append(confusion_matrix(y_test, pred_fast, labels=range(n_classes)))
    print(f"  [ProbLog-fast]  Acc={m_f['accuracy']:.4f}  F1w={m_f['f1_weighted']:.4f}  "
          f"MCC={m_f['mcc']:.4f}  AUC={m_f['roc_auc_ovr']:.4f}  Infer={dt_f:.4f}s")

    # ── 4. ProbLog-full (analytical posterior) ─────────────
    t0 = time.time()
    clf_full = ProbLogClassifier(model.branches, n_classes, mode="full")
    proba_full = clf_full.predict_proba(bp, X_test)
    pred_full = np.argmax(proba_full, axis=1)
    dt_full = time.time() - t0
    all_times["full_infer"].append(dt_full)

    m_full = compute_all_metrics(y_test, pred_full, proba_full)
    m_full["infer_time_s"] = dt_full
    all_metrics["problog_full"].append(m_full)
    all_confusion["problog_full"].append(confusion_matrix(y_test, pred_full, labels=range(n_classes)))
    print(f"  [ProbLog-full]  Acc={m_full['accuracy']:.4f}  F1w={m_full['f1_weighted']:.4f}  "
          f"MCC={m_full['mcc']:.4f}  AUC={m_full['roc_auc_ovr']:.4f}  Infer={dt_full:.4f}s")

    # ── Posterior diagnostics ──────────────────────────────
    posterior_z = clf_full.get_posterior_z(bp, X_test)
    shift = np.abs(posterior_z - bp).mean()
    posterior_shift.append(shift)
    print(f"  Posterior shift: mean |P(z|ev) - P(z)| = {shift:.6f}")

    # Theta matrices
    theta_default = build_theta_matrix(model.branches, n_classes)

    # ── 5. ProbLog-full-wmean (weighted mean aggregation) ───
    t0 = time.time()
    proba_wmean = aggregate_weighted_mean(posterior_z, theta_default)
    pred_wmean = np.argmax(proba_wmean, axis=1)
    dt_wmean = time.time() - t0
    all_times["wmean_infer"].append(dt_wmean)

    m_wmean = compute_all_metrics(y_test, pred_wmean, proba_wmean)
    m_wmean["infer_time_s"] = dt_wmean
    all_metrics["full_wmean"].append(m_wmean)
    all_confusion["full_wmean"].append(confusion_matrix(y_test, pred_wmean, labels=range(n_classes)))
    print(f"  [PL-full-wmean] Acc={m_wmean['accuracy']:.4f}  F1w={m_wmean['f1_weighted']:.4f}  "
          f"MCC={m_wmean['mcc']:.4f}  AUC={m_wmean['roc_auc_ovr']:.4f}  Infer={dt_wmean:.4f}s")

    # ── 8. ProbLog-fast-match (P(z) * match_ratio) ──────────
    t0 = time.time()
    match_ratio = compute_match_ratio(model.branches, X_test)
    z_corrected = bp * match_ratio
    proba_fast_mr = aggregate_noisy_or(z_corrected, theta_default)
    pred_fast_mr = np.argmax(proba_fast_mr, axis=1)
    dt_fast_mr = time.time() - t0
    all_times["fast_match_infer"].append(dt_fast_mr)

    m_fast_mr = compute_all_metrics(y_test, pred_fast_mr, proba_fast_mr)
    m_fast_mr["infer_time_s"] = dt_fast_mr
    all_metrics["fast_match"].append(m_fast_mr)
    all_confusion["fast_match"].append(confusion_matrix(y_test, pred_fast_mr, labels=range(n_classes)))
    print(f"  [PL-fast-match] Acc={m_fast_mr['accuracy']:.4f}  F1w={m_fast_mr['f1_weighted']:.4f}  "
          f"MCC={m_fast_mr['mcc']:.4f}  AUC={m_fast_mr['roc_auc_ovr']:.4f}  Infer={dt_fast_mr:.4f}s")

    # ── 7. PL-wmean-match (P(z)*match → wmean) ────────────
    t0 = time.time()
    z_match = bp * match_ratio
    proba_wm_match = aggregate_weighted_mean(z_match, theta_default)
    pred_wm_match = np.argmax(proba_wm_match, axis=1)
    dt_wm = time.time() - t0
    all_times["wmean_match_infer"].append(dt_wm)

    m_wm = compute_all_metrics(y_test, pred_wm_match, proba_wm_match)
    m_wm["infer_time_s"] = dt_wm
    all_metrics["wmean_match"].append(m_wm)
    all_confusion["wmean_match"].append(confusion_matrix(y_test, pred_wm_match, labels=range(n_classes)))
    print(f"  [PL-wm-match]   Acc={m_wm['accuracy']:.4f}  F1w={m_wm['f1_weighted']:.4f}  "
          f"MCC={m_wm['mcc']:.4f}  AUC={m_wm['roc_auc_ovr']:.4f}  Infer={dt_wm:.4f}s")

    # ── 8. PL-βAdMatch-wm (match-informed adaptive posterior → wmean)
    t0 = time.time()
    z_adm = compute_match_informed_adaptive_posterior(
        model.branches, bp, X_test, beta_base=0.5, match_boost=MATCH_BOOST,
    )
    proba_adm = aggregate_weighted_mean(z_adm, theta_default)
    pred_adm = np.argmax(proba_adm, axis=1)
    dt_adm = time.time() - t0
    all_times["adapt_match_wmean_infer"].append(dt_adm)

    m_adm = compute_all_metrics(y_test, pred_adm, proba_adm)
    m_adm["infer_time_s"] = dt_adm
    all_metrics["adapt_match_wmean"].append(m_adm)
    all_confusion["adapt_match_wmean"].append(confusion_matrix(y_test, pred_adm, labels=range(n_classes)))
    shift_adm = np.abs(z_adm - bp).mean()
    print(f"  [PL-βAdMatch]   Acc={m_adm['accuracy']:.4f}  F1w={m_adm['f1_weighted']:.4f}  "
          f"MCC={m_adm['mcc']:.4f}  AUC={m_adm['roc_auc_ovr']:.4f}  shift={shift_adm:.4f}")

    # ── 9. PL-ens-3way (w1·Neural + w2·wmean + w3·fast-match, opt) ─
    # Need train-set predictions for weight optimisation
    bp_train = model.predict_branch_proba(X_train).numpy()
    proba_neural_train = model.predict_proba(X_train).numpy()
    clf_full_train = ProbLogClassifier(model.branches, n_classes, mode="full")
    posterior_z_train = clf_full_train.get_posterior_z(bp_train, X_train)
    proba_wmean_train = aggregate_weighted_mean(posterior_z_train, theta_default)
    match_ratio_train = compute_match_ratio(model.branches, X_train)
    t0 = time.time()
    proba_fast_mr_train = aggregate_noisy_or(
        bp_train * match_ratio_train, theta_default
    )
    best_w3, _ = find_optimal_3way(
        proba_neural_train, proba_wmean_train, proba_fast_mr_train, y_train, step=0.1,
    )
    proba_3w = best_w3[0] * proba_neural + best_w3[1] * proba_wmean + best_w3[2] * proba_fast_mr
    proba_3w = proba_3w / proba_3w.sum(axis=1, keepdims=True)
    pred_3w = np.argmax(proba_3w, axis=1)
    dt_3w = time.time() - t0
    all_times["ens_3way_infer"].append(dt_3w)

    m_3w = compute_all_metrics(y_test, pred_3w, proba_3w)
    m_3w["infer_time_s"] = dt_3w
    all_metrics["ens_3way"].append(m_3w)
    all_confusion["ens_3way"].append(confusion_matrix(y_test, pred_3w, labels=range(n_classes)))
    print(f"  [PL-ens-3way]   Acc={m_3w['accuracy']:.4f}  F1w={m_3w['f1_weighted']:.4f}  "
          f"MCC={m_3w['mcc']:.4f}  AUC={m_3w['roc_auc_ovr']:.4f}  w={best_w3}")

    # ── ProbLog-engine verification (small subset) ─────
    n_verify = min(PROBLOG_VERIFY_N, len(X_test))
    t0 = time.time()
    clf_engine = ProbLogClassifier(model.branches, n_classes, mode="full_problog")
    proba_engine = clf_engine.predict_proba(
        bp[:n_verify], X_test[:n_verify],
        verbose=True, top_k_branches=PROBLOG_VERIFY_TOP_K,
    )
    pred_engine = np.argmax(proba_engine, axis=1)
    dt_engine = time.time() - t0
    all_times["problog_engine_infer"].append(dt_engine)

    # Compare analytical vs ProbLog engine on the verification subset
    proba_full_sub = proba_full[:n_verify]
    pred_full_sub = np.argmax(proba_full_sub, axis=1)
    match_pct = np.mean(pred_full_sub == pred_engine) * 100
    max_prob_diff = np.max(np.abs(proba_full_sub - proba_engine))
    mean_prob_diff = np.mean(np.abs(proba_full_sub - proba_engine))
    print(f"  [ProbLog-engine] Verification ({n_verify} samples, top-{PROBLOG_VERIFY_TOP_K}): "
          f"prediction match={match_pct:.0f}%  max_prob_diff={max_prob_diff:.4f}  "
          f"mean_prob_diff={mean_prob_diff:.4f}  time={dt_engine:.1f}s")

    # ── Agreement ──────────────────────────────────────────
    eq = lambda a, b: np.mean(a == b)
    agreement["et_vs_neural"].append(eq(pred_et, pred_neural))
    agreement["neural_vs_fast"].append(eq(pred_neural, pred_fast))
    agreement["neural_vs_full"].append(eq(pred_neural, pred_full))
    agreement["fast_vs_full"].append(eq(pred_fast, pred_full))
    agreement["neural_vs_wmean"].append(eq(pred_neural, pred_wmean))
    agreement["neural_vs_fast_match"].append(eq(pred_neural, pred_fast_mr))
    agreement["neural_vs_wmean_match"].append(eq(pred_neural, pred_wm_match))
    agreement["neural_vs_adapt_match_wmean"].append(eq(pred_neural, pred_adm))
    agreement["neural_vs_ens_3way"].append(eq(pred_neural, pred_3w))

    print(f"\n  Agreement with Neural:")
    print(f"    ET={agreement['et_vs_neural'][-1]:.3f}  "
          f"fast={agreement['neural_vs_fast'][-1]:.3f}  "
          f"wmean={agreement['neural_vs_wmean'][-1]:.3f}  "
          f"fmatch={agreement['neural_vs_fast_match'][-1]:.3f}  "
          f"wm-m={agreement['neural_vs_wmean_match'][-1]:.3f}  "
          f"βAdM={agreement['neural_vs_adapt_match_wmean'][-1]:.3f}  "
          f"3way={agreement['neural_vs_ens_3way'][-1]:.3f}")

    # ── Classification reports ─────────────────────────────
    for lbl, yp in [
        ("ExtraTrees",       pred_et),         ("Neural",           pred_neural),
        ("PL-fast",          pred_fast),       ("PL-full",          pred_full),
        ("PL-wmean",         pred_wmean),      ("PL-fast-match",    pred_fast_mr),
        ("PL-wm-match",     pred_wm_match),   ("PL-βAdMatch-wm",  pred_adm),
        ("PL-ens-3way",      pred_3w),
    ]:
        print(f"\n  --- {lbl} (fold {fold_idx+1}) ---")
        print(classification_report(y_test, yp, target_names=class_names,
                                    labels=list(range(n_classes)), zero_division=0))
    print()

total_time = time.time() - total_start

# ═════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════
print()
W = 20 + len(MODEL_NAMES) * 26
print("=" * W)
print(f"  SUMMARY: Mean ± Std across {N_FOLDS} folds")
print("=" * W)

metric_keys = [
    ("accuracy",           "Accuracy"),
    ("balanced_accuracy",  "Balanced Acc"),
    ("f1_weighted",        "F1 (weighted)"),
    ("f1_macro",           "F1 (macro)"),
    ("f1_micro",           "F1 (micro)"),
    ("precision_weighted", "Precision (w)"),
    ("precision_macro",    "Precision (m)"),
    ("recall_weighted",    "Recall (w)"),
    ("recall_macro",       "Recall (m)"),
    ("mcc",                "MCC"),
    ("cohen_kappa",        "Cohen Kappa"),
    ("roc_auc_ovr",        "ROC AUC (ovr)"),
    ("roc_auc_ovo",        "ROC AUC (ovo)"),
    ("log_loss",           "Log Loss"),
    ("infer_time_s",       "Infer time (s)"),
]

header = f"{'Metric':<20}"
for n in MODEL_NAMES:
    header += f" {MODEL_LABELS[n]:>24}"
print(header)
print("-" * len(header))

for key, label in metric_keys:
    row = f"  {label:<18}"
    for n in MODEL_NAMES:
        row += f" {fmt([m.get(key, float('nan')) for m in all_metrics[n]]):>24}"
    print(row)

print()

# ── Timing ─────────────────────────────────────────────────
print("=" * W)
print("  TIMING SUMMARY (seconds per fold)")
print("=" * W)
for key, label in [
    ("et_train",    "ExtraTrees training"),
    ("et_infer",    "ExtraTrees inference"),
    ("bn_train",    "RuleNetwork training"),
    ("neural_infer","Inference: Neural"),
    ("fast_infer",  "Inference: ProbLog-fast"),
    ("full_infer",  "Inference: ProbLog-full (analytical)"),
    ("wmean_infer", "Inference: PL-wmean"),
    ("fast_match_infer", "Inference: PL-fast-match"),
    ("wmean_match_infer",       "Inference: PL-wm-match"),
    ("adapt_match_wmean_infer", "Inference: PL-βAdMatch-wm"),
    ("ens_3way_infer",          "Inference: PL-ens-3way"),
    ("problog_engine_infer", f"Inference: ProbLog-engine ({PROBLOG_VERIFY_N} samples)"),
]:
    a = np.array(all_times[key])
    print(f"  {label:<55} mean={a.mean():.4f}  std={a.std():.4f}  total={a.sum():.2f}")

print(f"\n  Total experiment time: {total_time:.1f} s")

# Speedup
mn = np.mean(all_times["neural_infer"])
mf = np.mean(all_times["fast_infer"])
mfull = np.mean(all_times["full_infer"])
me = np.mean(all_times["problog_engine_infer"])
if mf > 0:
    print(f"\n  Speedup: ProbLog-fast vs Neural : {mn/mf:.1f}x" if mn > mf else
          f"\n  Speedup: Neural vs ProbLog-fast : {mf/mn:.1f}x" if mn > 0 else "")
if mfull > 0 and mn > 0:
    print(f"  Speedup: ProbLog-full(analytical) vs Neural : {'%.1f' % (mn/mfull)}x")
if me > 0 and mfull > 0:
    n_test_avg = X.shape[0] / N_FOLDS * (N_FOLDS - 1) / N_FOLDS  # rough avg test size
    estimated_engine_full = me / PROBLOG_VERIFY_N * n_test_avg
    print(f"  Estimated ProbLog-engine for full test: {estimated_engine_full:.0f}s per fold")
    print(f"  Speedup: analytical vs ProbLog-engine : {estimated_engine_full / mfull:.0f}x")
print()

# ── Agreement ──────────────────────────────────────────────
print("=" * W)
print("  PREDICTION AGREEMENT (with Neural)")
print("=" * W)
for key, label in [
    ("et_vs_neural",              "ExtraTrees ↔ Neural"),
    ("neural_vs_fast",            "Neural ↔ PL-fast"),
    ("neural_vs_full",            "Neural ↔ PL-full"),
    ("fast_vs_full",              "PL-fast ↔ PL-full"),
    ("neural_vs_wmean",           "Neural ↔ PL-wmean"),
    ("neural_vs_fast_match",      "Neural ↔ PL-fast-match"),
    ("neural_vs_wmean_match",     "Neural ↔ PL-wm-match"),
    ("neural_vs_adapt_match_wmean","Neural ↔ PL-βAdMatch-wm"),
    ("neural_vs_ens_3way",        "Neural ↔ PL-ens-3way"),
]:
    a = np.array(agreement[key])
    vals = "  ".join(f"{v:.3f}" for v in a)
    print(f"  {label:<35} {a.mean():.4f} ± {a.std():.4f}  [{vals}]")
print()

# ── Posterior diagnostics ──────────────────────────────────
print("=" * W)
print("  POSTERIOR DIAGNOSTICS: mean |P(z|evidence) - P(z_prior)|")
print("=" * W)
ps = np.array(posterior_shift)
print(f"  Per fold: {['%.6f' % v for v in ps]}")
print(f"  Mean: {ps.mean():.6f} ± {ps.std():.6f}")
print(f"  Interpretation: larger shift = evidence strongly updates branch beliefs")
print()

# ── Confusion matrices ─────────────────────────────────────
print("=" * W)
print("  CONFUSION MATRICES (summed across folds)")
print("=" * W)
for name in MODEL_NAMES:
    cm = sum(all_confusion[name])
    print(f"\n  {MODEL_LABELS[name]}")
    pad = "    "
    print(pad + "".join(f"{'Pred_'+cn:>12}" for cn in class_names))
    for i in range(n_classes):
        row_str = pad + f"True_{class_names[i]:<8}"
        for j in range(n_classes):
            row_str += f"{cm[i,j]:>8}"
        print(row_str)
print()

# ── Architecture ───────────────────────────────────────────
print("=" * W)
print("  ARCHITECTURE DETAILS")
print("=" * W)
print(f"  Branches per fold       : {fold_branches}")
print(f"  Mean branches           : {np.mean(fold_branches):.1f}")
print(f"")
print(f"  Neural forward pass:")
print(f"    x -> BN0(x) -> Linear(w1*m1, x) -> BN1(h) -> Sigmoid(h) -> BN2(h) -> Linear(w2, h) -> softmax")
print(f"")
print(f"  ProbLog-fast (noisy-or, no evidence):")
print(f"    P(class(x,k)) = 1 - prod_b(1 - theta_bk * P(z(b,x)))")
print(f"    normalized to sum = 1")
print(f"")
print(f"  ProbLog-full (analytical posterior + noisy-or):")
print(f"    For each branch b with m conditions: p_h = 0.95^(1/m), p_l = 0.05^(1/m)")
print(f"    P(ev|z=1) = p_h^n_match * (1-p_h)^n_miss")
print(f"    P(ev|z=0) = p_l^n_match * (1-p_l)^n_miss")
print(f"    P(z|ev) = Bayes update of P(z) with likelihood ratio")
print(f"    P(class=k) = 1 - prod_b(1 - theta_bk * P(z|ev))  [noisy-or]")
print(f"")
print(f"  PL-wmean (weighted mean):")
print(f"    P(class=k) = sum_b(theta_bk * P(z|ev)) / sum  [no noisy-or saturation]")
print(f"")
print(f"  PL-fast-match:")
print(f"    z_corr = P(z) * match_ratio  [match_ratio = n_match/n_total]")
print(f"    P(class=k) = 1 - prod_b(1 - theta_bk * z_corr)  [lightweight evidence]")
print(f"")
print(f"  PL-wm-match (best interpretable, match × prior → wmean):")
print(f"    z = P(z) * match_ratio → weighted mean aggregation")
print(f"    Combines learned P(z) with deterministic condition check")
print(f"")
print(f"  PL-βAdMatch-wm (match-informed adaptive posterior, best Bayesian):")
print(f"    β_bx = β_depth_b × (ε + match_boost × match_ratio_bx)")
print(f"    β_depth_b = beta_base × sqrt(m_ref / m_b)  [depth adaptive]")
print(f"    log P(z|ev) ∝ log P(z) + β_bx × log L(ev|z)")
print(f"    Then aggregate via weighted mean: P(class=k) ∝ Σ θ_bk * z_tempered_b")
print(f"    Key: per-sample, per-branch tempering — max fine-grained evidence weighting")
print(f"")
print(f"  PL-ens-3way (best overall, uses neural):")
print(f"    P = α₁·Neural + α₂·PL-wmean + α₃·PL-fast-match")
print(f"    Weights optimized on training set via grid search")
print(f"")
print(f"  theta_bk: normalized class proportions from parent-of-leaf nodes (= W2)")
print()

# ── Per-class F1 ───────────────────────────────────────────
print("=" * W)
print("  PER-CLASS F1 (from summed confusion matrix)")
print("=" * W)
for name in MODEL_NAMES:
    cm = sum(all_confusion[name]).astype(float)
    f1s = []
    for i in range(n_classes):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2*p*r/(p+r) if (p+r) > 0 else 0.0)
    vals = "  ".join(f"{class_names[i]}={f1s[i]:.4f}" for i in range(n_classes))
    print(f"  {MODEL_LABELS[name]:<30} {vals}")
print()

# ── Sample explanations (last fold) ────────────────────────
print("=" * W)
print("  SAMPLE EXPLANATIONS (last fold, 3 samples, using posterior z)")
print("=" * W)

from problog_export import _class_proportions_to_theta

posterior_z = clf_full.get_posterior_z(bp, X_test)

for i in range(min(3, len(X_test))):
    p_n = proba_neural[i]
    p_f = proba_fast[i]
    p_full = proba_full[i]
    true_label = int(y_test[i])
    pred_n = int(np.argmax(p_n))
    pred_f = int(np.argmax(p_f))
    pred_full_i = int(np.argmax(p_full))

    print(f"\n  Sample {i}: true={true_label} ({class_names[true_label]})")
    print(f"    Neural:       [{', '.join(f'{v:.4f}' for v in p_n)}] -> {pred_n}")
    print(f"    ProbLog-fast: [{', '.join(f'{v:.4f}' for v in p_f)}] -> {pred_f}")
    print(f"    ProbLog-full: [{', '.join(f'{v:.4f}' for v in p_full)}] -> {pred_full_i}")

    # Top-5 branches by posterior support
    branch_support = []
    for br_idx, branch in enumerate(model.branches):
        theta = _class_proportions_to_theta(branch)
        if theta and pred_full_i < len(theta):
            pz_prior = float(bp[i, br_idx])
            pz_post = float(posterior_z[i, br_idx])
            score = theta[pred_full_i] * pz_post
            branch_support.append((branch, theta[pred_full_i], pz_prior, pz_post, score))
    branch_support.sort(key=lambda x: x[4], reverse=True)

    print(f"    Top-5 branches supporting class {pred_full_i} ({class_names[pred_full_i]}):")
    for rank, (br, th, pz_prior, pz_post, sc) in enumerate(branch_support[:5]):
        conds = " AND ".join(
            f"{feature_names[c.feature_idx] if c.feature_idx < len(feature_names) else f'f{c.feature_idx}'}"
            f" {c.direction} {c.threshold:.2f}"
            for c in br.conditions
        )
        if not conds:
            conds = "(root)"
        delta = pz_post - pz_prior
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
        print(f"      {rank+1}. {br.branch_id} | theta={th:.3f}  "
              f"P(z)={pz_prior:.3f}->{pz_post:.3f} {arrow}  "
              f"score={sc:.4f}")
        print(f"         IF {conds}")

print()
print("=" * W)
print("  END OF REPORT")
print("=" * W)

# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
sys.stdout = tee.stdout
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(tee.getvalue())
print(f"\nResults saved to: {OUTPUT_FILE}")
