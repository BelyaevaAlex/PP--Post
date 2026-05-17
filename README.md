# PPtheta-Post: Condition-Aware Rule Activation with ProbLog Posterior Inference

## Overview

This project uses the **NSToolkit Condition-Aware Rule Activation**
pattern: symbolic backbones mine explicit `Branch` / `Condition`
objects, each condition is evaluated as soft evidence, and branch-level
rule activations become latent probabilistic variables for PPtheta-Post.

Each tree-derived branch becomes an explicit logical rule with a
condition-aware activation score.  The theta-based ProbLog posterior
head then updates these latent activations with observed evidence,
enabling Bayesian inference, interpretable explanations, and multiple
aggregation strategies.

**Temporal paper (PPθ-Post-Temporal)**: target **IEEE ICDM 2026**, track **A\*** (main conference research paper). Deadlines: abstract **30 May 2026**; full paper **6 June 2026** — see `../PAPER_LAYOUT.md` and `../cursor_documentation.md` (confirm on the official ICDM 2026 CFP).

---

## Paper Experiment Scripts

The repository now keeps paper sections as explicit runnable entry points in
`paper_experiments/`:

| # | Section goal | Script |
|---|--------------|--------|
| 01 | Theoretical / model-limit probes | `paper_experiments/section_01_theoretical_limits.py` |
| 02 | Tabular — main methods (PPθ-Post + e2e-NoisyOr on ExtraTrees) | `paper_experiments/section_02_tabular_main_methods.py` |
| 03 | Tabular — inference-variant ablations (which PPθ-Post component matters?) | `paper_experiments/section_03_tabular_ablations.py` |
| 04 | Tabular — rule-source sweep (Track A) + standalone competitors (Track B) | `paper_experiments/section_04_tabular_rule_sources_sweep.py` |
| 05 | Tabular — TabPFN distillation as a rule source (`tabpfn_distill_{xgb,et,cb}`) | `paper_experiments/section_05_tabular_tabpfn_distill.py` |
| 06 | Tabular — ensemble variants (`pl_ens_distill` interpretable, `pl_ens_tabpfn` black-box) | `paper_experiments/section_06_tabular_ensembles.py` |
| 07 | Tabular — interpretability story (post-processor: interpretable / distill-guided / black-box gap) | `paper_experiments/section_07_interpretability_story.py` |
| 08 | Temporal — L1-L4 benchmark + optional vendored baselines | `paper_experiments/section_08_temporal_benchmark.py` |
| 09 | Temporal — aggregation / head ablations | `paper_experiments/section_09_temporal_ablations.py` |
| 10 | Rule-level case studies (top-K rule traces) | `paper_experiments/section_10_case_studies.py` |

The large tabular driver is `compare_datasets.py`.  It is designed for
larger datasets via chunked prediction, configurable tree budgets,
streaming CSV/JSONL output, real loaders (`sklearn:`, `openml:`, `csv:`,
`npz:`), optional subsampling for expensive variants, and no native
`full_problog` path.  The two main methods, **PPtheta-Post** and
**e2e-NoisyOr**, train in mini-batches and validate/predict in chunks.

The driver supports two extension tracks (see §10.8): `--rule-sources`
swaps the symbolic source feeding PPθ-Post (ExtraTrees, XGBoost,
CatBoost, FIGS, RuleFit), and `--baselines` appends standalone
competitors (EBM, FIGS, RuleFit, TabPFN) that do not go through
PPθ-Post inference.

Vendored temporal baselines are still present as Git submodules under
`temporal/vendor/*`.  After a fresh clone run:

```bash
git submodule update --init --recursive
```

Untracked cache files inside vendor submodules are ignored by the parent
repository, so local `__pycache__` directories do not make the vendor code
look missing.

---

## 1. Condition-Aware Rule Activation

### Rule Acquisition

The default symbolic backbone follows the NSToolkit `SymbolicBackboneExtraTrees`
pattern.  An `ExtraTreesClassifier` ensemble is trained, and from each tree
we extract parent-of-leaf nodes: internal nodes whose at least one child is
a leaf.  Each extracted node becomes one rule neuron and one symbolic
`Branch` object.

### Condition-Aware Activation

For each branch `b`, every stored `Condition` is evaluated against the
input.  Hard threshold predicates (`<=`, `>`) are used for exact ProbLog
evidence, while the differentiable posterior uses sigmoid soft matches:

```
match_i(x) = sigmoid((threshold_i - x_f) / tau)   # le condition
match_i(x) = sigmoid((x_f - threshold_i) / tau)   # gt condition
```

The condition scores contribute additively in log-likelihood-ratio space
and update the latent rule activation:

```
logit P(z_b | evidence) =
    logit P(z_b | x) + sum_i condition_log_lr_i(x)
```

This is the key NSToolkit-style condition-aware path used by PPtheta-Post:
the neural rule prior `P(z_b | x)` and explicit symbolic evidence are
combined before the ProbLog/weighted-mean/noisy-or head.

Implemented entry points:

```python
condition_z = model.predict_condition_branch_proba(X, tau=1.0)
proba, info = model.predict_rule_head_proba(
    X,
    activation="hybrid",      # neural | condition | hybrid
    aggregation="weighted_mean",
    return_diagnostics=True,
)
```

### Default Backbone Budget

```
n_estimators = n_classes + floor(log2(n_features))
max_leaf_nodes = 2^(floor(log2(n_features)) + 4)
```

### Weight initialization

| Matrix | Shape | Source | Trainable? |
|--------|-------|--------|------------|
| **m1** | `[hidden, features]` | Binary mask: `(w1 ≠ 0)` | No (buffer) |
| **w1** | `[hidden, features]` | Feature importance × mask | Yes |
| **w2** | `[classes, hidden]` | Class distribution from the rule source node | No (frozen) |

### Neural Prior Path

```
x -> BN0(x) -> Linear(w1 * m1) -> BN1 -> Sigmoid -> BN2 -> Linear(w2) -> softmax -> y_hat
```

---

## 2. Symbolic Representation: Branch / Condition

Each branch is stored as a `Branch` object (`branch_schema.py`):

```python
@dataclass
class Branch:
    branch_id: str              # "b0", "b1", ...
    tree_id: int                # tree index in the ensemble
    parent_node_id: int         # ID of the parent-of-leaf node
    conditions: List[Condition] # path from root TO the parent (le/gt)
    class_proportions: List[float]  # class distribution (→ theta / W2)
    split_feature_idx: int      # split feature of the parent node (→ mask)
    split_threshold: float      # split threshold of the parent node
```

Each `Condition` describes one node on the path:

```python
@dataclass
class Condition:
    feature_idx: int   # feature index
    threshold: float   # threshold value
    direction: str     # "le" (≤) or "gt" (>)
    node_id: int       # tree node ID
```

---

## 3. ProbLog Integration

### 3.1 Latent Variables z(b, X)

The neural part computes, for each sample X and branch b:

```
P(z(b,X) = true | x) = Sigmoid(BN₁(Linear(BN₀(x), w1 · m1)))
```

In ProbLog notation:
```prolog
0.95756::z(b33, 0).    % P(z(b33, sample_0) = true) = 0.958
```

### 3.2 Manifestation Rules (conditions as "symptoms" of z)

Conditions (le/gt) are **manifestations** of the latent event z(b,X).
A single hidden cause governs all conditions in a branch.

For a branch with **m** conditions, normalized probabilities are used:

```prolog
% If z(b0) is active — condition holds with high probability
p_high^(1/m) :: le(b0,f4,t0_0,X) :- z(b0,X).

% If z(b0) is inactive — condition holds with low probability
p_low^(1/m) :: le(b0,f4,t0_0,X) :- not_z(b0,X).
```

The `p_high^(1/m)` normalization ensures that evidence strength is **comparable across branches** with different numbers of conditions, preventing aggressive posterior collapse for deep branches.

### 3.3 Evidence: Observations from Data

For each sample, conditions are computed deterministically and passed as evidence:

```prolog
evidence(le(b0,f4,t0_0,0)).           % x[f4] ≤ threshold → true
evidence(gt(b206,f8,t6_12,4), false). % x[f8] > threshold → false
```

### 3.4 Classification Head: θ (replaces W2)

Class proportions from parent-of-leaf nodes form the classification rules:

```prolog
theta(b0, class_0, 0.9524).
theta(b0, class_1, 0.0238).
theta(b0, class_2, 0.0238).
```

---

## 4. Inference Variants (9 Selected Models)

After extensive experimentation with 30+ variants, **9 essential models** were selected:

### 4.1 Baselines

| Model | Description |
|-------|-------------|
| **ExtraTrees** | Default NSToolkit-compatible symbolic rule source (one of five — see §10.8 for XGBoost / CatBoost / FIGS / RuleFit) |
| **RuleNetwork-Neural** | Neural prior path: `x -> w1*m1 -> Sigmoid -> w2 -> softmax` |
| **Standalone competitors** | EBM, FIGS, RuleFit, TabPFN — run end-to-end, no PPθ-Post inference; see §10.8 track B |

### 4.2 ProbLog Inference Modes

| Model | Formula | Key Idea |
|-------|---------|----------|
| **PL-fast** | `P(k) = 1 - ∏_b(1 - θ_bk · P(z_b))` | Noisy-or with neural prior, no evidence |
| **PL-full** | `P(z\|ev) = Bayes(P(z), L(ev\|z)); P(k) = noisy-or(θ, P(z\|ev))` | Full Bayesian posterior + noisy-or |
| **PL-wmean** | `P(k) ∝ Σ_b θ_bk · P(z_b\|ev)` | Weighted mean (avoids noisy-or saturation) |
| **PL-fast-match** | `z_corr = P(z) · match_ratio; P(k) = noisy-or(θ, z_corr)` | Lightweight evidence via condition matching |
| **PL-wm-match** | `z = P(z) · match_ratio → weighted mean` | Best interpretable: prior × match → wmean |
| **PL-βAdMatch-wm** | `β_bx = β_depth · (ε + boost · match); log P(z\|ev) ∝ log P(z) + β · log L` | Match-informed adaptive Bayesian posterior |
| **PL-ens-3way** | `P = α₁·Neural + α₂·wmean + α₃·fast-match` | Ensemble with optimized weights |

### Key Technical Details

**Analytical posterior** (for PL-full, PL-wmean, PL-βAdMatch-wm):
```
For branch b with m conditions:
  p_h = p_high^(1/m),  p_l = p_low^(1/m)
  P(ev|z=1) = p_h^n_match · (1-p_h)^n_miss
  P(ev|z=0) = p_l^n_match · (1-p_l)^n_miss
  P(z|ev) ∝ P(z) · P(ev|z=1)^β  /  [P(z)·P(ev|z=1)^β + (1-P(z))·P(ev|z=0)^β]
```

**Adaptive β** (for PL-βAdMatch-wm):
```
  β_depth_b = β_base · sqrt(m_ref / m_b)     # depth-adaptive tempering
  β_bx = β_depth_b · (ε + boost · match_bx)  # match-informed per-sample
```

**Match ratio**:
```
  match_ratio = n_satisfied_conditions / n_total_conditions
    ```

---

## 5. Experimental Results (Wine Dataset, 5-fold CV)

### Accuracy & Interpretability

| Model | Accuracy | F1 (weighted) | MCC | ROC AUC (ovr) |
|-------|----------|---------------|-----|---------------|
| ExtraTrees | 0.9605 ± 0.0427 | 0.9602 ± 0.0431 | 0.9407 ± 0.0644 | 0.9936 ± 0.0059 |
| **RuleNetwork-Neural** | 0.9832 ± 0.0137 | 0.9832 ± 0.0138 | 0.9752 ± 0.0203 | 0.9992 ± 0.0015 |
| PL-fast | 0.9163 ± 0.0495 | 0.9158 ± 0.0499 | 0.8804 ± 0.0715 | 0.6588 ± 0.0329 |
| PL-full | 0.9440 ± 0.0309 | 0.9436 ± 0.0311 | 0.9177 ± 0.0460 | 0.7003 ± 0.0197 |
| PL-wmean | 0.9495 ± 0.0329 | 0.9492 ± 0.0331 | 0.9254 ± 0.0490 | 0.9974 ± 0.0041 |
| PL-fast-match | 0.9721 ± 0.0248 | 0.9716 ± 0.0253 | 0.9594 ± 0.0360 | 0.6476 ± 0.0291 |
| **PL-wm-match** | **0.9832 ± 0.0223** | **0.9829 ± 0.0227** | **0.9755 ± 0.0324** | 0.9987 ± 0.0020 |
| PL-βAdMatch-wm | 0.9776 ± 0.0208 | 0.9774 ± 0.0211 | 0.9673 ± 0.0302 | 0.9990 ± 0.0021 |
| **PL-ens-3way** | **0.9832 ± 0.0137** | **0.9832 ± 0.0138** | **0.9752 ± 0.0203** | 0.9331 ± 0.1338 |

### Key Findings

1. **PL-wm-match matches Neural accuracy** (0.9832) while being **fully interpretable** — every prediction traces back to explicit IF-THEN rules weighted by condition satisfaction.
2. **PL-ens-3way** also matches Neural, but relies on the neural component (α₃ ≈ 0.9).
3. **PL-βAdMatch-wm** is the best purely Bayesian variant (0.9776), demonstrating that match-informed adaptive tempering significantly outperforms standard posteriors.
4. **Noisy-or saturates** with ~160 branches, making weighted mean (`wmean`) a more robust aggregation.
5. **Match ratio** is the most effective lightweight evidence mechanism, closely approximating full Bayesian inference.

### Inference Speed

| Model | Inference time |
|-------|---------------|
| PL-wmean | 0.0001s (fastest) |
| PL-wm-match | 0.0001s |
| PL-fast | 0.0004s |
| Neural | 0.0007s |
| PL-fast-match | 0.0012s |
| PL-βAdMatch-wm | 0.0014s |
| PL-full | 0.0015s |
| PL-ens-3way | 0.0017s |
| ProbLog-engine | ~15s per fold (9926× slower than analytical) |

### Sample Explanation

```
Sample 0: true = class_0
  Top-5 supporting branches:
    1. b142 | θ=0.970  P(z)=0.997→1.000  IF alcalinity_of_ash ≤ 19.31 AND proline > 886.54
    2. b123 | θ=0.964  P(z)=0.975→0.999  IF alcalinity_of_ash ≤ 19.20 AND alcohol > 12.45 AND alcohol > 13.53
    3. b62  | θ=0.963  P(z)=0.980→0.999  IF od280/od315 > 2.79 AND color_intensity > 4.97 AND flavanoids ≤ 4.84
    4. b12  | θ=0.952  P(z)=0.996→1.000  IF alcohol > 11.74 AND alcohol > 12.52 AND proline > 635.80 AND nonflavanoid_phenols ≤ 0.36
    5. b61  | θ=0.929  P(z)=0.981→0.999  IF od280/od315 > 2.79 AND color_intensity > 4.97
```

---

## 6. Full Pipeline

```
                    TRAINING
                    ========
ExtraTreesClassifier(data)
         │
         ▼
    ┌─────────────────┐
    │ bf_search:       │
    │ parent-of-leaf   │──→ Branch[] (conditions, class_proportions)
    │ extraction       │
    └─────────────────┘
         │
         ▼
    ┌─────────────────┐
    │ Condition-aware │
    │ rule activation │
    │ w1 + m1 + conds │
    └─────────────────┘
         │
         ▼
    fit(x_train, y_train)
         │
         ▼
    branch_probs(x) → P(z(b,X) | x)

                    EXPORT
                    ======
    Branch[]  ──→ branch_struct(b, X) :- le(...), gt(...)    [structural rules]
    P(z)      ──→ pZ::z(b, x_id).                            [latent variables]
    p_high^(1/m) ──→ 0.95^(1/m)::le(...) :- z(b,X).         [manifestation]
    p_low^(1/m)  ──→ 0.05^(1/m)::le(...) :- not_z(b,X).     [manifestation]
    data      ──→ evidence(le(...,x_id)). / evidence(...,false). [observed]
    theta     ──→ theta(b, class_k, proportion).              [classification head]

                    INFERENCE (9 variants)
                    ========================
    Neural:      w2 * Sigmoid(BN(x*w1*m1)) -> softmax
    PL-fast:     noisy-or(θ, P(z))
    PL-full:     noisy-or(θ, P(z|evidence))
    PL-wmean:    weighted-mean(θ, P(z|evidence))
    PL-fast-m:   noisy-or(θ, P(z) × match_ratio)
    PL-wm-match: weighted-mean(θ, P(z) × match_ratio)
    PL-βAdMatch: weighted-mean(θ, posterior_β(z, evidence, match))
    PL-ens-3way: α₁·Neural + α₂·wmean + α₃·fast-match
```

---

## 7. Project Files

```
PPtheta-Post/
├── rule_network.py            # Condition-aware rule activation backbone
├── rule_network_model.py      # fit/predict wrapper + predict_branch_proba() + predict_problog()
├── branch_schema.py          # Branch, Condition dataclasses
├── problog_export.py         # Export: structural, latent, evidence, classification head
├── problog_inference.py      # All 7 ProbLog inference modes + explanations
├── train.py                  # Trainer: ExtraTrees -> condition-aware rules -> ProbLog export
├── compare_wine.py           # Full comparison of 9 models on Wine (5-fold CV)
├── compare_wine_results.txt  # Detailed results output
├── run_one_export_check.py   # Verification of parent-of-leaf extraction
├── verify_branch_export.py   # Spot-check JSON ↔ ProbLog consistency
├── test_rule_network_branch_probs.py       # Test: P(z) range, shape
├── test_problog_export_latent.py        # Test: latent KB content
├── test_problog_consistency.py          # Test: predict before/after = identical
├── test_problog_end_to_end_consistency.py
├── test_problog_evidence_inference.py
├── test_problog_latent_branch_scoping.py
├── benchmetrics.py           # Benchmark metrics utilities
├── openml_download.py        # OpenML dataset downloader
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## 8. Why This Is Neuro-Symbolic

| Component | Neural | Symbolic |
|-----------|--------|----------|
| **w1 (input weights)** | ✅ trainable | Initialized from RF feature importance |
| **m1 (mask)** | — | ✅ frozen boolean mask from RF |
| **w2 / θ (output)** | — | ✅ frozen class distributions from RF parent nodes |
| **Branch conditions** | — | ✅ IF-THEN rules from trees (le/gt) |
| **z(b,X)** | ✅ P(z) from Sigmoid | ✅ latent variable in ProbLog |
| **Manifestation** | — | ✅ conditions as symptoms of z (p_high/p_low) |
| **Evidence** | — | ✅ deterministic observations from data |
| **Posterior inference** | — | ✅ Bayesian update P(z│evidence) |
| **Match ratio** | — | ✅ deterministic condition satisfaction score |
| **Aggregation** | — | ✅ noisy-or / weighted mean over symbolic θ |

The neural part computes latent state probabilities.
The symbolic part contains rule structure, a manifestation model, and performs formal inference.
They are connected through a shared branch space — each branch is simultaneously a neuron in the network and a rule in ProbLog.

---

## 9. Reproducibility

- Random seed: 42 (fixed for all components)
- Per-fold seed: `SEED + fold_idx` for PyTorch and ExtraTrees
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- 5-fold stratified cross-validation
- All results saved to `compare_wine_results.txt`

---

## 10. Temporal Extension (`temporal/`)

PPθ-Post is lifted onto irregular multivariate time-series benchmarks
(P12, PAM, MIMIC-III demo) through four progressively-deeper levels
that share the same Branch / ProbLog stack and therefore inherit all 9
inference variants from the static pipeline.

### 10.1 Levels

| Level | Idea | Architecture impact |
|------|------|----------------------|
| **L1** | Per-variable summary stats `(mean, std, min, max, first, last, slope, count, frac_obs)` | None — pipeline runs as-is on the flat feature vector |
| **L2** | Multi-window summary stats + cross-window deltas | None — only feature engineering |
| **L3** | Time-Series-Forest backbone over random intervals × `{mean, std, slope}` | Trees split on interval features; metadata table maps `feature_idx → (var, interval, stat)` |
| **L4** | Per-timestep latent `z(b, X, t)` + temporal aggregation | `PPThetaPostTemporal` runs condition-aware rule activation on `[N*T, 2V]` snapshots, aggregates over time |

### 10.2 L3 — temporal ProbLog atoms

Each branch `Condition` is rendered as a temporal atom whose functor
encodes both direction and statistic:

```prolog
0.95^(1/m) :: gt_mean(b0,hr,0,12,95.0,X)   :- z(b0,X).
0.95^(1/m) :: le_slope(b0,lactate,24,48,0.3,X) :- z(b0,X).
0.05^(1/m) :: gt_mean(b0,hr,0,12,95.0,X)   :- not_z(b0,X).
```

Reads as *"in sample X, mean of HR over hours 0–12 is greater than 95"*.
The `feature_meta` table is consumed by
`export_temporal_branches_to_problog` and
`export_temporal_problog_program` without any change to `Branch` /
`Condition`.

### 10.3 L4 — per-timestep latent + temporal aggregation

```
ExtraTreesClassifier(snapshots [N·T, 2V])
                          │
                          ▼
          Condition-aware rule activation
                          │
                          ▼
                   P(z(b, X, t))     [N, T, B]
                          │
                          ▼
            ┌─────────────────────────┐
            │ temporal aggregation:    │
            │   mean | max | exists |  │
            │   forall | k_of_t |      │
            │   last | attention       │
            └─────────────────────────┘
                          │
                          ▼
                   z(b, X)            [N, B]
                          │
                          ▼
       all 9 PPθ-Post inference variants (PL-fast / PL-full /
       PL-wmean / PL-wm-match / PL-βAdMatch-wm / PL-ens-3way / …)
```

The temporal latent atoms can also be exported to ProbLog:

```prolog
0.92::z(b0,X,0).   0.81::z(b0,X,1).   ...   0.95::z(b0,X,T).
% existential aggregation:
z_overall(b0,X) :- z(b0,X,_).
```

`exists` and `forall` are pure-ProbLog rules; `mean`, `noisy_or` and
`k_of_T` are computed analytically by
`temporal_inference.aggregate_z_over_time`.

### 10.4 Default L4 inference variants

Registered in `temporal.temporal_inference.DEFAULT_TEMPORAL_VARIANTS`:

| Name                  | Temporal mode                                                                          | Branch head        |
| --------------------- | -------------------------------------------------------------------------------------- | ------------------ |
| **PL-tMean**          | `mean` over time                                                                       | weighted mean      |
| **PL-tMax**           | `max` over time                                                                        | weighted mean      |
| **PL-tNoisyOr**       | `1 − Π_t (1 − P(z_t))`                                                                 | noisy-or           |
| **PL-tNoisyOr-top10** | top-10 % most-active timesteps zeroed-elsewhere, then noisy-or                         | noisy-or           |
| **PL-tNoisyOr-top25** | top-25 % most-active timesteps, then noisy-or                                          | noisy-or           |
| **PL-tForall**        | `Π_t P(z_t)`                                                                           | weighted mean      |
| **PL-kOfT-25 / 50**   | normal-CDF approx. of Poisson-binomial CDF for ⌈k·T⌉ active timesteps                  | weighted mean      |
| **PL-tLast**          | `P(z_T)` only                                                                          | weighted mean      |
| **PL-tAttn**          | learned softmax weights `α_t` (shared across branches)                                 | weighted mean      |
| **PL-tAttnPB**        | per-branch attention `α_{t,b}` — every branch gets its own temporal pooling            | weighted mean      |
| **PL-tAttnMH**        | multi-head attention with `H` heads + soft head-mixing per branch (parameter-efficient) | weighted mean      |

> ⚠️  Bare `PL-tNoisyOr` / `PL-tForall` saturate for long sequences
> (T ≳ 100): existential probability tends to 1, universal to 0.
> The **`PL-tNoisyOr-topN`** variants address this directly — only the
> top-N % most-active timesteps contribute to the noisy-or, restoring
> discriminative power.  For paper-scale benchmarks prefer
> `PL-tNoisyOr-top10`, `PL-kOfT-50`, or `PL-tAttn{PB,MH}`.

### 10.5 Datasets and comparison drivers

- `temporal.datasets.load_temporal_dataset("p12" | "pam" | "mimic3")` —
  synthetic loaders that mirror the structure of PhysioNet 2012, PAMAP2
  and mimic3-benchmarks (mortality task) at laptop scale.  Real loaders
  drop into the same registry once credentialing is in place.
- `python -m temporal.compare_temporal --datasets p12 pam mimic3
  --levels L1 L2 L3 L4 --folds 3 --epochs 80` — **intra-method
  ablation** across PPθ-Post temporal levels.  Markdown summary saved to
  `output/temporal/compare_temporal_<timestamp>.md`.
- Same driver with `--baselines lr xgb transformer sand mtan gru_d
  seft raindrop camelot interp_gn` (or `--baselines all`) appends 10
  external baselines to the report.  All SOTA rows run from the
  *authors' original code* (`temporal/vendor/*`) — see §10.7.
- `python -m temporal.ablations --datasets pam` — fixes the L4 backbone
  and sweeps only over aggregation modes / hyper-parameters
  (`top_k_time`, attention modes, k-of-T thresholds).
- `python -m temporal.case_studies --dataset p12 --level L3 --top-k 5`
  — emits human-readable top-K rules per sample as JSON for the paper /
  supplementary.
- `python -m temporal.compare_static_on_temporal --datasets pam --levels L3`
  — runs the **17 static PPθ-Post inference variants** from
  `compare_datasets.py` on temporal benchmarks via L1 / L2 / L3
  flattening.  Verifies that the static stack survives temporal data
  without any modification.
- `python -m temporal.problog_spotcheck` — compiles the full L3
  temporal ProbLog program for a tiny instance through the native
  ProbLog engine and checks parity with the analytical posterior to
  within ≈10⁻⁹.  No branch truncation: parity is proved on the full
  reduced-size program (see `--n-estimators` / `--max-leaf-nodes`).

> **Why no `--max-branches` flag?**  A truncated spot-check would only
> validate parity on a subset of the program, leaving the unverified
> branches as a hidden assumption.  We instead shrink the upstream
> forest so the full program compiles in seconds — no information is
> discarded.

### 10.6 Tests

```
python -m temporal.tests.test_tabularize                # L1 / L2 shapes
python -m temporal.tests.test_interval_forest           # L3 backbone + meta
python -m temporal.tests.test_temporal_problog          # L3 / L4 ProbLog export
python -m temporal.tests.test_pp_theta_post_temporal    # L4 fit / predict / aggregations / multi-head attention
python -m temporal.tests.test_problog_spotcheck         # full-program ProbLog ↔ analytical parity
python -m temporal.tests.test_baselines                 # 3 reimpl baselines (LR/XGB/Transformer-IMTS)
python -m temporal.tests.test_baselines_vendored        # 5 PyTorch vendored adapters
python -m temporal.tests.test_baselines_vendored_tf     # 2 TF vendored adapters (SeFT, CAMELOT)
```

> **Backwards compatibility**: `TemporalRuleNetwork` is still importable
> as an alias of `PPThetaPostTemporal` and will be removed in a future
> release; new code should use `PPThetaPostTemporal` directly.

### 10.7 External baselines — vendored-first track

All baselines share the contract
`fit(X_ts, mask, y, x_val=None) → predict_proba(X_ts, mask)` and live
in a single dispatcher `UNIFIED_BASELINE_REGISTRY` exposed by
`temporal.compare_temporal`.  The driver picks the right backend based
on the registry key — no `--vendored` flag, **vendored is the only
track for SOTA**.

**Re-implementation track** (`temporal/baselines.py`) — three
baselines that have no upstream worth vendoring:

| Key | Module class | Notes |
|---|---|---|
| `lr` | `LRStatsBaseline` | Logistic regression on L1 statistics; interpretable shallow baseline |
| `xgb` | `XGBStatsBaseline` | XGBoost on L2 multi-window statistics; non-interpretable shallow |
| `transformer` | `TransformerIMTSBaseline` | Vanilla Transformer encoder over `(value, mask)` snapshots |

**PyTorch vendored track** (`temporal/baselines_vendored.py`) —
authors' original code as git submodules:

| Key | Submodule | Upstream | Commit | License | Runtime |
|---|---|---|---|---|---|
| `sand` | `temporal/vendor/sand` | [khirotaka/SAnD](https://github.com/khirotaka/SAnD) | `b5da888` | MIT | CPU |
| `mtan` | `temporal/vendor/mtan` | [reml-lab/mTAN](https://github.com/reml-lab/mTAN) | `7a3d536` | MIT | CPU |
| `gru_d` | `temporal/vendor/grud` | [zhiyongc/GRU-D](https://github.com/zhiyongc/GRU-D) | `d070b52` | _none_ | CPU; `hidden_size = n_vars` |
| `raindrop` | `temporal/vendor/raindrop` | [mims-harvard/Raindrop](https://github.com/mims-harvard/Raindrop) | `892eb57` | MIT | CUDA + `torch_geometric` |
| `interp_gn` | `temporal/vendor/interpgn` | [YunshiWen/InterpretGatedNetwork](https://github.com/YunshiWen/InterpretGatedNetwork) | `5ea6045` | _none_ | CPU; default FCN backbone via `default_interpgn_configs()` |

**TensorFlow / Keras vendored track**
(`temporal/baselines_vendored_tf.py`) — lazy-TF adapters:

| Key | Submodule | Upstream | License | Runtime |
|---|---|---|---|---|
| `seft` | `temporal/vendor/seft` | [BorgwardtLab/Set_Functions_for_Time_Series](https://github.com/BorgwardtLab/Set_Functions_for_Time_Series) | BSD-3 | TensorFlow ≥ 2.4 |
| `camelot` | `temporal/vendor/camelot` | [hrna-ox/camelot-icml](https://github.com/hrna-ox/camelot-icml) | _none_ | TensorFlow ≥ 2.4 |

Clone all vendored repos before running:
```bash
git submodule update --init --recursive
```

Run the comparison (vendored is the default — no flags):
```bash
python -m temporal.compare_temporal \
    --datasets p12 \
    --baselines all \
    --folds 5 --epochs 80
```

If a vendored adapter cannot initialise (missing TensorFlow, missing
CUDA, missing `torch_geometric`, upstream import error), the driver
emits a single diagnostic line and continues with the remaining
baselines:

```
[skipped] baseline 'raindrop': Raindrop requires `torch_geometric`; …
[skipped] baseline 'seft':     tensorflow is not installed; …
```

> ⚠️  **InterpGN gate-opacity caveat.**  InterpGN's "interpretability"
> comes from **soft routing** between a prototype-based interpretable
> head and a black-box neural head — but the gate `g(X) ∈ [0, 1]` is
> itself a neural network, so the routing decision is opaque even when
> each path is locally interpretable.  This is fundamentally different
> from PPθ-Post, where every prediction carries a complete rule trace.
> The driver logs the average routing fraction `g(X)` so this
> distinction surfaces in the report.

See `temporal/vendor/README.md` for
licence summary and the per-baseline status table.

### 10.8 Tabular comparison — rule sources, standalone baselines, distill, ensembles

The tabular driver (`compare_datasets.py`) supports four parallel tracks
that mirror the temporal-track split.  Every row in the result CSV is
tagged by `(rule_source, variant)` so tracks never overlap.

#### 10.8.1 Track A — alternative rule sources (`tabular/rule_sources.py`)

Each source fits a base estimator on `(X, y)`, then emits parent-of-leaf
`Branch` objects fed to `RuleNetwork.build_model_from_branches(...)`.
The rest of the PPθ-Post pipeline (PL-fast / PL-full / PL-wmean /
e2e-NoisyOr / theta-learn / …) runs unchanged on top of whichever source
was selected.

| Key | Adapter class | Upstream | Notes |
|---|---|---|---|
| `extratrees` | `ExtraTreesRuleSource` | `sklearn.ensemble.ExtraTreesClassifier` | Default; bit-for-bit compatible with the legacy `build_model_from_ensemble` path (proved by `test_rule_network_branches_equivalence.py`) |
| `xgb` | `XGBoostRuleSource` | `xgboost.XGBClassifier` | Recursive JSON-tree walk; empirical cp refined via `booster.predict(pred_leaf=True)` |
| `catboost` | `CatBoostRuleSource` | `catboost.CatBoostClassifier` | Oblivious-tree walk; empirical cp refined via `calc_leaf_indexes` (LSB-first leaf-id mapping `_catboost_leaf_ids_for_parent`) |
| `figs` | `FIGSRuleSource` | [`imodels.FIGSClassifier`](https://github.com/csinva/imodels) | Recursive walk over `figs.Node` trees; empirical cp via vectorized condition-eval |
| `rulefit` | `RuleFitRuleSource` | `imodels.RuleFitClassifier` | Parses text rules → `Condition`; **binary only** (driver emits clean `[skip]` on multiclass) |

Run with one or more sources:

```bash
python compare_datasets.py --datasets sklearn:breast_cancer \
    --rule-sources extratrees,xgb,catboost,figs,rulefit \
    --variants core --folds 3
```

#### 10.8.2 Track B — standalone competitors (`tabular/baselines.py`)

These run end-to-end (`fit(X, y) / predict_proba(X)`) and **do not**
feed PPθ-Post inference — they are the upper-bound / interpretable-
competitor reference.  Tagged in the CSV as `rule_source=_standalone`.

| Key | Adapter class | Upstream | Notes |
|---|---|---|---|
| `ebm` | `EBMBaseline` | `interpret.glassbox.ExplainableBoostingClassifier` | Current glass-box SOTA on tabular data |
| `figs` | `FIGSBaseline` | `imodels.FIGSClassifier` | Same upstream as the `figs` rule source, but used standalone (no `build_model_from_branches`) |
| `rulefit` | `RuleFitBaseline` | `imodels.RuleFitClassifier` | Binary only |
| `tabpfn` | `TabPFNBaseline` | [`tabpfn.TabPFNClassifier`](https://github.com/PriorLabs/TabPFN) | v2.x ships an offline checkpoint (~200 MB); v8 requires a PriorLabs license token, so we pin `tabpfn>=2,<3` |

#### 10.8.3 TabPFN-distill rule sources

TabPFN itself has no symbolic structure, but it can act as a *teacher*.
`TabPFNDistillRuleSource` fits TabPFN on `(X, y)`, then trains a
tree-ensemble student on `argmax(p_soft)` with `sample_weight =
max(p_soft, axis=1)` (hard distillation with confidence weighting), and
extracts branches from the student.  Empirical `class_proportions` are
refined against the **original** `(X, y)`, not the TabPFN-induced
labels — so cp stays sample-grounded.

| Key | Student | Label in CSV |
|---|---|---|
| `tabpfn_distill_xgb` | `xgboost.XGBClassifier` | `TabPFN→XGB` |
| `tabpfn_distill_et` | `sklearn.ensemble.ExtraTreesClassifier` | `TabPFN→ExtraTrees` |
| `tabpfn_distill_cb` | `catboost.CatBoostClassifier` | `TabPFN→CatBoost` |

The student is fully interpretable via the standard branch-extraction
pipeline — only the *choice of which branches to grow* was TabPFN-guided.

#### 10.8.4 Ensemble variants (`pl_ens_tabpfn` vs. `pl_ens_distill`)

Two ensemble variants combine three members per source:

* `pl_ens_tabpfn`: `α₁·TabPFN_proba + α₂·PL-wmean + α₃·source_native`
* `pl_ens_distill`: `α₁·DistilledStudent_proba + α₂·PL-wmean + α₃·source_native`

α is learned per fold on a stratified inner-val split (≤200 samples)
via SLSQP on the probability simplex with multi-start (uniform + each
one-hot biased); the previous Nelder-Mead-on-softmax-logits formulation
silently stalled at uniform.  Optional `--ensemble-shrinkage λ ∈ [0, 1]`
pulls the learned α toward uniform — a Stein-style estimator that
trades a sliver of inner-val log-loss for noticeably better test-set
generalisation when the inner-val is tiny.

Both ensembles cache their pretrained teacher (`_fit_tabpfn_cache`,
`_fit_distill_cache`) **once per fold** and reuse it across every
`rule_source`, so adding the ensemble variant does not multiply
TabPFN-fit cost by the source count.

**Interpretability tradeoff:**

| Variant | Members | End-to-end interpretable? |
|---|---|---|
| `pl_ens_distill` | DistilledStudent + PL-wmean + source-native | **Yes** — every member is a tree-ensemble with extractable branches |
| `pl_ens_tabpfn` | TabPFN + PL-wmean + source-native | **No** — TabPFN contribution is black-box |

Empirically `pl_ens_tabpfn` is usually 1–2 p.p. ahead of `pl_ens_distill`
on accuracy, but on some datasets (e.g. wine with `--distill-student=et`)
the distilled ensemble can win because the ET student fits the soft
labels better than XGB does.

#### 10.8.5 Empirical class-proportion refinement and scale knobs

For non-sklearn rule sources we replace model-internal class-proportion
heuristics (leaf margins, linear coefficients) with the **empirical
class fraction** of training samples that satisfy every branch
condition — the same quantity the sklearn-tree path encodes via
`tree.value[parent_node]`.  Two fast paths plus a generic fallback:

| Source | Refinement mode | Complexity |
|---|---|---|
| `xgb` (+ distill XGB student) | `pred_leaf` (`booster.predict(..., pred_leaf=True)`) + `np.isin` | `O(n × n_trees)` |
| `catboost` (+ distill CB student) | `calc_leaf_indexes` + LSB-first leaf-id map | `O(n × n_trees)` |
| `figs`, `rulefit`, FIGS/RuleFit/ExtraTrees branches in distill | vectorized condition-eval with shared condition cache | `O(n_unique_conditions × n + n_branches × depth)` |

`--refinement-max-samples K` caps the training-sample count used during
refinement (stratified subsample) — `0` (default) uses the full split.
Set to e.g. `50000` on datasets with hundreds of thousands of rows to
bound refinement cost; the empirical estimate stays representative.

Before this refinement was added, `XGBoost+PL-full` was at 0.52 on
breast_cancer and `CatBoost+PL-full` / `RuleFit+PL-*` collapsed to the
modal class.  Post-refinement those rise to 0.88 / 0.91 / 0.94
respectively — see commit history for the before/after table.

#### 10.8.6 CLI cheat sheet (tabular)

| Flag | Default | Purpose |
|---|---|---|
| `--rule-sources` | `extratrees` | Comma list or `all` — Track A sources fed to PPθ-Post |
| `--baselines` | `none` | Comma list, `all`, or `none` — Track B standalone competitors |
| `--variants` | `core` | `core` / `all` / `expensive` / comma list — inference variants |
| `--refinement-max-samples` | `0` | Cap empirical-cp refinement to this many stratified train samples; `0` = full |
| `--ensemble-shrinkage` | `0.0` | Stein-style λ ∈ [0,1] pulling learned ensemble α toward uniform |
| `--distill-student` | `xgb` | Tree student inside `pl_ens_distill` cache: `xgb` / `et` / `cb` |

Common recipes:

```bash
# Track A only (5 rule sources × core variants)
python compare_datasets.py --datasets sklearn:breast_cancer \
    --rule-sources all --variants core --folds 3

# Track A + B (rule sources + standalone competitors)
python compare_datasets.py --datasets sklearn:breast_cancer \
    --rule-sources all --baselines ebm,figs,rulefit,tabpfn \
    --variants core --folds 3

# Distill rule source + ensemble variant + shrinkage
python compare_datasets.py --datasets sklearn:wine \
    --rule-sources extratrees,catboost,tabpfn_distill_et \
    --variants source_native,pl_wmean,pl_ens_distill \
    --distill-student et --ensemble-shrinkage 0.3 \
    --baselines ebm,tabpfn --folds 3
```

#### 10.8.7 Vendoring fallback

Default is **pip-first**: every adapter does lazy `import …` and raises
a clear `ImportError` with the install command if the package is
missing.  When pip itself fails on a specific host (typically because
of numpy / torch version clashes), the same per-baseline venv +
git-submodule machinery used by `temporal.baselines_vendored` is
mirrored at `tabular/baselines_vendored.py` — see its module docstring
for the escalation path.  The vendored registry ships empty; activate
per-baseline only when a clash is reported.
