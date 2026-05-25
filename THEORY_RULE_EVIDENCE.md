# A Unified Probabilistic Rule-Evidence View of PPtheta-Post

This note extracts the theory that should be used for the PPtheta-Post
paper.  It shares the same rule-evidence language as the
neuro-symbolic-toolkit paper, but the PPtheta-Post contribution is
different: exact probabilistic symbolic posterior inference.

The common schema is

```text
rules -> z_b(x) -> r_b z_b(x) theta_b -> p(y | x)
```

where each branch rule `b` has explicit conditions, `z_b(x)` is its
latent rule evidence, `theta_b` is the class-support distribution stored
by the branch, and `r_b` is an optional reliability weight.  In
PPtheta-Post the central object is

```text
z_b(x) = P(z_b = 1 | x, evidence_b)
```

computed from a scoped ProbLog-style latent-variable model.

## Setup

For branch `b`, let the rule body contain `m_b > 0` axis-aligned
conditions.  A root-style branch with no conditions simply uses its
prior as its posterior.  The neural rule network supplies a prior

```text
pi_b(x) = P(z_b = 1 | x).
```

Observed feature values determine which conditions match.  Let
`n_plus_b(x)` be the number of satisfied conditions and `n_minus_b(x)`
the number of unsatisfied conditions.  PPtheta-Post uses
depth-normalised manifestation probabilities

```text
p_h,b = p_high^(1 / m_b),    p_l,b = p_low^(1 / m_b).
```

The depth normalisation is important: it prevents deeper rules from
receiving stronger total positive evidence simply because they contain
more conjuncts.

## Theorem 1: Exact ProbLog Equivalence

For the scoped PPtheta-Post program, the analytic posterior

```text
P(z_b = 1 | x, evidence_b)
= pi_b L_b^+ / (pi_b L_b^+ + (1 - pi_b) L_b^-)
```

with

```text
L_b^+ = p_h,b^n_plus_b (1 - p_h,b)^n_minus_b
L_b^- = p_l,b^n_plus_b (1 - p_l,b)^n_minus_b
```

is exactly the posterior returned by the corresponding ProbLog program.

Proof.  The exported program gives each branch its own latent event
`z(b, X)` and its own scoped condition atoms.  Conditional on `z(b, X)`,
condition observations factorise.  Therefore the likelihood of the
observed condition evidence is the product `L_b^+` when `z_b = 1` and
`L_b^-` when `z_b = 0`.  Bayes' rule gives the displayed expression.
Because condition atoms are scoped by branch, branch posteriors
factorise.  This is exactly the vectorised analytic computation used in
`problog_inference.py`.

## Theorem 2: Posterior Log-Odds Decomposition

For every branch,

```text
logit P(z_b = 1 | x, evidence_b) - logit pi_b(x)
= n_plus_b log(p_h,b / p_l,b)
  + n_minus_b log((1 - p_h,b) / (1 - p_l,b)).
```

For soft conditions, replace `n_plus_b` with the sum of soft matches and
`n_minus_b` with the sum of soft mismatches.

Proof.  Taking the logit of the posterior in Theorem 1 cancels the
normalising denominator and leaves the prior log-odds plus the
log-likelihood ratio `log(L_b^+ / L_b^-)`.  Expanding this ratio over
matched and missed conditions gives the formula.

Interpretation.  Every condition has an exact signed contribution to the
branch posterior.  This is stronger than a post-hoc attribution: it is
an identity inside the model.

## Theorem 3: Soft-to-Hard Consistency

Assume an input is bounded away from every rule threshold.  If each
condition is evaluated with a sigmoid temperature `tau`, then as
`tau -> 0` the soft PPtheta-Post posterior converges to the hard-evidence
ProbLog posterior.

Proof.  Away from threshold hyperplanes, each sigmoid condition score
converges to its Boolean truth value.  The posterior formula in
Theorem 1 is continuous when priors are clipped away from `0` and `1`.
Therefore the soft posterior converges to the hard posterior.  Class
probabilities obtained by weighted-mean or noisy-or aggregation converge
by continuity of the aggregation map.

## Theorem 4: Depth-Normalised Positive Evidence

If all conditions in branch `b` match, the total positive evidence shift
is independent of branch depth:

```text
m_b log(p_high^(1/m_b) / p_low^(1/m_b))
= log(p_high / p_low).
```

Proof.  This follows directly from the logarithm power rule.  The
result means a long conjunction does not automatically dominate a short
conjunction merely because it contains more observed condition atoms.

## Theorem 5: Noisy-Or Expressivity Limit

For a fixed class `k`, the noisy-or head

```text
q_k(x) = 1 - product_b (1 - theta_bk z_b(x))
```

is monotone nondecreasing in every posterior rule activation `z_b(x)`.
Therefore a two-branch noisy-or head cannot represent XOR over binary
rule activations.

Proof.  The partial derivative is

```text
d q_k / d z_b = theta_bk product_{j != b} (1 - theta_jk z_j) >= 0.
```

XOR is not monotone: increasing one input from `0` to `1` can change the
output from `1` to `0`.  Hence no monotone noisy-or function can realise
the XOR truth table.

## How To Use This In The PPtheta-Post Paper

Use the PPtheta-Post theory block as:

1. Exact ProbLog equivalence.
2. Posterior log-odds decomposition.
3. Soft-to-hard convergence.
4. Depth-normalised evidence.
5. Noisy-or expressivity limits.

This keeps the PPtheta-Post claim focused: the contribution is not just
"symbolic rules", but an exact posterior symbolic layer with known
semantics, auditable condition-level contributions, and explicit
expressivity boundaries.
