# Formal expressivity of the differentiable posterior layer

For the paper-level theory block, see also
`THEORY_RULE_EVIDENCE.md`.  That note states the unified
rule-evidence view shared with NSToolkit and then isolates the
PPtheta-Post-specific claims: exact ProbLog equivalence, posterior
log-odds decomposition, soft-to-hard consistency, depth-normalised
evidence, and noisy-or expressivity limits.

This note states the theoretical capabilities and limitations of the
``DifferentiablePosterior`` + noisy-or layer used in ``PPθ-Post`` and
``e2e-NoisyOr``, and points to the empirical probes in
``study_expressivity.py`` that verify each statement.

Throughout: a *branch* `b` is a conjunction `c_{b,1} ∧ … ∧ c_{b,m_b}` of
axis-aligned conditions `c_{b,i} = (x_{f_i} ⋆_i θ_i)` where `⋆_i ∈ {≤, >}`.
The model has hyper-parameters `p_high, p_low ∈ (0,1)` and a temperature
`τ > 0`, and learnable parameters `W_1` (feature→branch attention) and
`θ ∈ [0,1]^{B × C}` (branch→class probabilities).  We write
`m_i(x) = σ((sign·(τ_i − x_{f_i}))/τ)` for the soft truth value of
condition `i` and `Δ_i(x) = m_i · log(p_h^{1/m_b}/p_l^{1/m_b}) +
(1−m_i) · log((1−p_h^{1/m_b})/(1−p_l^{1/m_b}))` for its log-likelihood
ratio contribution.


## What the layer can represent

**Proposition 1 (pointwise consistency, τ → 0).**
Fix `x` such that `x_{f_i} ≠ τ_i` for every condition.  Then
`lim_{τ → 0} m_i(x) ∈ {0,1}` is the indicator of `c_{b,i}(x)`, and
`lim_{τ → 0} z_{b,τ}^{post}(x) = P_{ProbLog}(z_b = 1 | evidence)`.
The convergence is uniform on every compact set bounded away from the
threshold hyperplanes.

*Sketch.* For `τ → 0` the sigmoid degenerates to the Heaviside step
function pointwise off-threshold; substituting into the Bayesian
update yields the analytical posterior with depth-adjusted
probabilities `p_h^{1/m_b}, p_l^{1/m_b}` — exactly the formula used
in ``ProbLogClassifier(mode="full")``.  Empirical convergence rate is
measured in **E1** in `study_expressivity.py`.

**Proposition 2 (additive Bayesian decomposition).**
For every input `x` and every branch `b`,
`logit(z_b^{post}(x)) − logit(z_b^{prior}(x)) = Σ_{i ∈ b} Δ_i(x)`.

This is exact within the layer (no approximation), and it is the
identity that ``BranchAttributor.consistency_check`` verifies to
machine precision.  It justifies the per-condition attribution map:
each condition contributes a *signed log-odds shift* to its branch.

**Proposition 3 (smoothness).**
For every `τ > 0`, `z_b^{post}` is `C^∞` in `x` and in the layer
parameters `(W_1, θ, p_h, p_l, τ)`.  The class output
`P(c|x) = 1 − Π_b (1 − θ_{bc} z_b^{post}(x))` is therefore `C^∞` and
admits gradients used by ``fit_e2e_noisy_or``.

**Proposition 4 (depth-cancellation).**
Under depth-adjusted likelihoods `p_h^{1/m_b}, p_l^{1/m_b}`, when *all*
conditions of a branch fire (`m_i = 1` for every `i ∈ b`) the total
Bayesian shift is `m_b · (1/m_b) · log(p_h/p_l) = log(p_h/p_l)`,
**independent of branch depth**.  Conversely the per-condition
contribution is `(1/m_b)·log(p_h/p_l)` — deeper branches are
less sensitive to losing any single condition match.  Probed in **E3**.


## What the layer cannot represent

**Limitation 1 (XOR within branches).**
A single branch evaluates to the AND of its conditions.  It therefore
cannot represent XOR over its own conditions.  XOR is recoverable only
at the *ensemble* level, by combining ≥ 2 branches via noisy-or.

**Limitation 2 (XOR across branches under noisy-or).**
The noisy-or class head `P(c=1|x) = 1 − Π_b (1 − θ_{bc} z_b)` is
**monotone** in every `z_b` — increasing any single posterior cannot
decrease the class probability.  Therefore there exists no `θ ∈
[0,1]^{B × C}` and no `(z_1, z_2) ∈ {0,1}^2` such that the noisy-or
output realises the XOR truth table.  Probed empirically in **E2**:
exhaustive grid over `θ ∈ [0,1]^{2 × 2}` on a 2-mode XOR dataset
recovers at most 75 % accuracy versus 100 % for the Bayes-optimal
classifier.

**Limitation 3 (branch independence assumption).**
Noisy-or aggregation is exact only when the branches are
*conditionally independent* given the class label.  In a tree-derived
ensemble this assumption is violated whenever two branches share
ancestor splits or partition the same feature subspace.  The diagnostic
in **E4** reports the empirical excess covariance
`E[z_{b_1}·z_{b_2}] − E[z_{b_1}]·E[z_{b_2}]` for every branch pair on
the actual training data; large values warn that the noisy-or output
is biased even at τ = 0.

**Limitation 4 (calibration on threshold boundaries).**
At `x_{f_i} = τ_i` the soft sigmoid match is `m_i = 1/2` regardless of
τ, so the posterior is the prior tilted by the symmetric likelihood
ratio `0.5·log(p_h/p_l) + 0.5·log((1−p_h)/(1−p_l))`.  Decisions at
exact threshold values are intrinsically uncalibrated — the model
cannot decide between `z = 0` and `z = 1` by construction.  This is a
*feature*, not a bug, of the soft layer: it produces well-defined
gradients across decision boundaries that a hard-evidence layer
cannot.

**Limitation 5 (single-class evidence channel).**
Each branch contributes to every class via a single scalar `θ_{bc}`.
This is strictly less expressive than a softmax head over `B` features
(which uses `B·(C−1)` independent affine combinations).  In particular
a branch with `θ_{b,c_1} = θ_{b,c_2}` casts an identical vote for two
classes, and there is no mechanism for a branch to *suppress* class
`c` — `θ_{bc} = 0` only removes its support.


## Empirical artefacts

Run `python study_expressivity.py` from `PPθ-Post/`.  The script writes
`expressivity_report.json` with raw numbers for every probe, and prints
the four summary tables to stdout.

| probe | proves                                              | location in script                |
| ----- | --------------------------------------------------- | --------------------------------- |
| E1    | Prop. 1 (τ → 0 consistency)                         | `probe_tau_consistency`           |
| E2    | Limitations 1, 2 (XOR ceiling of noisy-or)          | `probe_xor_independence`          |
| E3    | Prop. 4 (depth cancellation, per-condition shrink)  | `probe_depth_likelihood`          |
| E4    | Limitation 3 (independence assumption diagnostic)   | `probe_branch_independence`       |

The decomposition exposed by ``DifferentiablePosterior.forward_with_attribution``
and consumed by ``BranchAttributor.consistency_check`` is what makes Prop. 2
verifiable to floating-point precision on every input — the same hook
powers the per-(branch, condition) attribution maps documented in
`attribution.py`.
