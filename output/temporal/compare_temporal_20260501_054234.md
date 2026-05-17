# PPθ-Post temporal comparison — 20260501_054234

> **Scope**: PPθ-Post L1 / L2 / L3 / L4 variants **plus external baselines**: lr, xgb, gru_d.  All baselines share the `(X_ts, mask, y)` interface; for InterpGN the entry is annotated with the average routing fraction `g(X)` — see §6.5 for the gate-opacity caveat.
>
> **Sanity-check disclaimer**: results on synthetic loaders (`p12` / `pam` / `mimic3`) are smoke-level only.  Paper-quality numbers require credentialed real benchmarks (PhysioNet/2012, PAMAP2, MIMIC-III/IV) — see `PAPER_LAYOUT.md`.

Levels: ['L1', 'L4'] | folds: 2 | epochs: 8 | baselines: ['lr', 'xgb', 'gru_d']

## synthetic_p12

| Variant | Acc | F1 | MCC | ROC AUC | PR AUC | fit (s) | pred (s) |
|---|---|---|---|---|---|---|---|
| L1_Neural | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.948±0.010 | 0.785±0.016 | 0.8 | 0.00 |
| L1_PL-fast | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.500±0.000 | 0.140±0.000 | 0.8 | 0.00 |
| L1_PL-full | 0.897±0.027 | 0.864±0.046 | 0.447±0.198 | 0.996±0.002 | 0.971±0.016 | 0.8 | 0.01 |
| L1_PL-wmean | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.999±0.000 | 0.996±0.001 | 0.8 | 0.00 |
| L4_PL-tMean | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.968±0.020 | 0.894±0.058 | 2.9 | 0.07 |
| L4_PL-tMax | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.995±0.004 | 0.976±0.019 | 2.9 | 0.04 |
| BL_LR-stats | 0.998±0.002 | 0.998±0.002 | 0.993±0.007 | 1.000±0.000 | 1.000±0.000 | 0.2 | 0.17 |
| BL_XGB-stats | 0.973±0.007 | 0.972±0.007 | 0.886±0.030 | 0.995±0.005 | 0.975±0.022 | 0.8 | 0.56 |
| BL_GRU-D | 0.917±0.007 | 0.901±0.010 | 0.607±0.038 | 0.945±0.008 | 0.842±0.028 | 1.3 | 0.03 |
