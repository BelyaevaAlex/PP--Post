# Main Theory: A Unified Probabilistic Rule-Evidence View

This document gives a shared theoretical layer for both projects:

- `neuro-symbolic-toolkit`
- `PPtheta-Post`

The first part is intentionally the same across both repositories.  It
defines the common mathematical object.  The second part explains how
this common object is used differently in this project.

## Part I: Shared Theory

The common schema is:

```text
rules -> z_b(x) -> r_b z_b(x) theta_b -> p(y | x)
```

Each rule `b` has:

- an explicit logical body, usually a conjunction of feature conditions;
- a rule-evidence value `z_b(x) in [0, 1]`;
- an optional reliability weight `r_b in [0, 1]`;
- a class-support vector `theta_b`.

Define:

```text
a_b(x) = r_b z_b(x)
A(x) = sum_b a_b(x)
```

When at least one rule is active, the weighted rule-evidence predictor is:

```text
p_k(x) = sum_b a_b(x) theta_bk / A(x).
```

If no rule is active, the predictor uses a fixed fallback distribution.

This view covers three families of models:

1. Symbolic rule-network heads, where `z_b(x)` is a rule or branch
   activation and `r_b = 1`.
2. Reliability-gated logical heads, where `z_b(x)` is neural,
   condition-based, or hybrid evidence and `r_b` is validation- or
   teacher-guided reliability.
3. Probabilistic-logic posterior heads, where
   `z_b(x) = P(z_b = 1 | x, evidence_b)`.

### Proposition 1: Common Representation

The weighted symbolic head, the reliability-gated logical head, and the
posterior weighted-mean head are all instances of:

```text
p_k(x) = sum_b r_b z_b(x) theta_bk / sum_b r_b z_b(x).
```

Proof.  For each method, choose the appropriate meaning of `z_b(x)`:
branch activation, logical-condition activation, hybrid activation, or
posterior rule activation.  The rule contribution is always
`r_b z_b(x) theta_b`, followed by aggregation over rules.

### Proposition 2: Exact Posterior Semantics

For a rule `b` with `m_b > 0` conditions, let:

```text
pi_b(x) = P(z_b = 1 | x)
n_plus_b(x) = number of satisfied conditions
n_minus_b(x) = number of unsatisfied conditions
p_h,b = p_high^(1 / m_b)
p_l,b = p_low^(1 / m_b)
```

Under scoped condition atoms and conditional independence given `z_b`,
the posterior is:

```text
P(z_b = 1 | x, evidence_b)
= pi_b L_b^+ / (pi_b L_b^+ + (1 - pi_b) L_b^-)
```

where:

```text
L_b^+ = p_h,b^n_plus_b (1 - p_h,b)^n_minus_b
L_b^- = p_l,b^n_plus_b (1 - p_l,b)^n_minus_b.
```

Proof.  Conditional on `z_b`, the condition observations factorise.
Thus `L_b^+` and `L_b^-` are the likelihoods of the observed condition
evidence under `z_b = 1` and `z_b = 0`.  Bayes' rule gives the displayed
posterior.  If condition atoms are scoped by branch, the branch
posteriors factorise.

### Proposition 3: Posterior Log-Odds Decomposition

For every branch:

```text
logit P(z_b = 1 | x, evidence_b) - logit pi_b(x)
= n_plus_b log(p_h,b / p_l,b)
  + n_minus_b log((1 - p_h,b) / (1 - p_l,b)).
```

For soft conditions, replace `n_plus_b` with the sum of soft matches and
`n_minus_b` with the sum of soft mismatches.

Proof.  Taking the logit of Proposition 2 cancels the normalising
denominator and leaves the prior log-odds plus
`log(L_b^+ / L_b^-)`.  Expanding this likelihood ratio gives the result.

Interpretation.  Each condition has an exact signed contribution to the
branch posterior.  This makes the explanation part of the model's
algebra, not a post-hoc decoration.

### Proposition 4: Soft-to-Hard Consistency

Assume an input is bounded away from every rule threshold.  If each
condition is evaluated with sigmoid temperature `tau`, then as
`tau -> 0`, the soft posterior converges to the hard-evidence posterior.

Proof.  Away from threshold hyperplanes, each sigmoid condition score
converges to its Boolean truth value.  The posterior formula is
continuous when priors are clipped away from `0` and `1`.  Class
probabilities obtained by weighted-mean or noisy-or aggregation converge
by continuity of the aggregation map.

### Proposition 5: Depth-Normalised Evidence

If all conditions in branch `b` match, the total positive evidence shift
is independent of branch depth:

```text
m_b log(p_high^(1/m_b) / p_low^(1/m_b))
= log(p_high / p_low).
```

Proof.  This follows from the logarithm power rule.  A long conjunction
therefore does not dominate a short conjunction merely because it
contains more condition atoms.

### Proposition 6: Rule Deletion Faithfulness

Let `S` be a deleted set of rules:

```text
A_S(x) = sum_{b in S} r_b z_b(x)
A(x) = sum_b r_b z_b(x)
```

If `A(x) > 0` and `A(x) - A_S(x) > 0`, then:

```text
||p(x) - p_{-S}(x)||_1
= A_S(x) / A(x) * ||mu_S(x) - p_{-S}(x)||_1
<= 2 A_S(x) / A(x).
```

Proof.  The full prediction decomposes as:

```text
p(x) = (A_S / A) mu_S(x) + (1 - A_S / A) p_{-S}(x).
```

Subtract `p_{-S}(x)` and take the L1 norm.  The L1 distance between two
probability distributions is at most `2`.

### Proposition 7: Teacher-to-Symbolic Transfer

Let `h_T` be a teacher and `h_S` a symbolic student.  For zero-one loss:

```text
R(h_S) <= R(h_T) + P[h_S(X) != h_T(X)].
```

Proof.  If the student is wrong, then either the teacher is also wrong,
or the student disagrees with the teacher:

```text
{h_S(X) != Y} subset {h_T(X) != Y} union {h_S(X) != h_T(X)}.
```

Taking probabilities gives the bound.

### Proposition 8: Finite Candidate Selection

Let `H` be a finite set of candidate symbolic students or rule heads.
For `n` independent validation examples, with probability at least
`1 - delta`, every candidate satisfies:

```text
|R(h) - Rhat(h)| <= sqrt(log(2 |H| / delta) / (2 n)).
```

Consequently, the validation-selected candidate has risk at most
`2 epsilon` above the best candidate in `H`, where `epsilon` is the
displayed bound.

Proof.  Apply Hoeffding's inequality to each fixed candidate and
union-bound over `H`.

### Proposition 9: Reliability-Gated Monotonicity

For the unnormalised class score:

```text
s_k(x) = sum_b r_b z_b(x) theta_bk,
```

increasing any reliability weight `r_b` cannot decrease any class score.
For the normalised weighted mean:

```text
d p_k(x) / d r_b = z_b(x) (theta_bk - p_k(x)) / A(x).
```

Proof.  The unnormalised score is a sum of nonnegative terms.  The
normalised derivative follows by differentiating the ratio defining
`p_k(x)`.

### Proposition 10: Conformal Selective Prediction

For nonconformity score:

```text
s(x, y) = 1 - p_y(x),
```

split conformal prediction sets:

```text
C(x) = {k : 1 - p_k(x) <= q_alpha}
```

have marginal coverage at least `1 - alpha` under exchangeability of
calibration and test examples:

```text
P[Y in C(X)] >= 1 - alpha.
```

Proof.  This is the standard split-conformal rank argument.  The proof
does not depend on how `p_k(x)` is produced, only on exchangeability and
the calibration quantile.

## Part II: How This Theory Is Used In PPtheta-Post

In `PPtheta-Post`, the common theory is used to support the posterior
probabilistic-logic view:

```text
explicit Branch / Condition rules
-> neural prior P(z_b = 1 | x)
-> ProbLog-style posterior P(z_b = 1 | x, evidence_b)
-> theta-based class aggregation
```

The main project-specific claims are:

1. **Exact ProbLog equivalence.**  
   The vectorised analytic posterior equals the posterior of the scoped
   ProbLog program under the same manifestation assumptions.

2. **Posterior log-odds decomposition.**  
   Each condition contributes an exact signed log-likelihood-ratio term
   to the branch posterior.

3. **Soft-to-hard convergence.**  
   The differentiable soft evidence layer converges to hard symbolic
   evidence as the sigmoid temperature goes to zero, away from threshold
   boundaries.

4. **Depth-normalised evidence.**  
   Long branches do not receive larger total positive evidence merely
   because they contain more conjuncts.

5. **Noisy-or expressivity limits.**  
   The noisy-or head is monotone in rule activations, so it cannot
   represent non-monotone functions such as XOR without additional
   mechanisms.

Rule deletion faithfulness, teacher-to-symbolic transfer, finite
candidate selection, reliability gating, conformal selective prediction,
and stability-aware selection are shared or adjacent tools, but they
should not be the central PPtheta-Post theory claim.  The central
PPtheta-Post claim is exact posterior symbolic inference with explicit
semantics and known expressivity boundaries.
