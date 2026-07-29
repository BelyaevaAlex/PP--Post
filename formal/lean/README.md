# PPtheta-Post Lean verification

This is a self-contained Lean 4 project for the finite algebraic core of the
PPtheta-Post theory. It verifies:

1. exact equivalence between the branch-scoped analytic posterior masses and the
   two-state probabilistic-logic enumeration;
2. prediction sufficiency of the exported rule-evidence trace;
3. an elementary trace-score update lemma for adding one evidence item.

Build:

```bash
cd formal/lean
lake build
```

The LaTeX supplement contains the real-valued log-odds and soft-to-hard
arguments. Those are kept as mathematical proofs rather than Lean proofs because
this project intentionally avoids external math libraries.
