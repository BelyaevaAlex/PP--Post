# PPθ-Post temporal comparison — 20260501_071839

> **Scope**: PPθ-Post L1 / L2 / L3 / L4 variants **plus external baselines**: lr, xgb, transformer, sand, mtan, gru_d, raindrop, interp_gn, seft, camelot.  All baselines share the `(X_ts, mask, y)` interface; for InterpGN the entry is annotated with the average routing fraction `g(X)` — see §6.5 for the gate-opacity caveat.
>
> **Sanity-check disclaimer**: results on synthetic loaders (`p12` / `pam` / `mimic3`) are smoke-level only.  Paper-quality numbers require credentialed real benchmarks (PhysioNet/2012, PAMAP2, MIMIC-III/IV) — see `PAPER_LAYOUT.md`.

Levels: ['L1', 'L2'] | folds: 2 | epochs: 4 | baselines: ['lr', 'xgb', 'transformer', 'sand', 'mtan', 'gru_d', 'raindrop', 'interp_gn', 'seft', 'camelot']

## synthetic_p12

| Variant | Acc | F1 | MCC | ROC AUC | PR AUC | fit (s) | pred (s) |
|---|---|---|---|---|---|---|---|
| L1_Neural | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.595±0.090 | 0.223±0.064 | 0.9 | 0.03 |
| L1_PL-fast | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.500±0.000 | 0.140±0.000 | 0.9 | 0.00 |
| L1_PL-full | 0.870±0.007 | 0.818±0.015 | 0.233±0.090 | 0.993±0.004 | 0.943±0.037 | 0.9 | 0.01 |
| L1_PL-wmean | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.989±0.007 | 0.947±0.031 | 0.9 | 0.00 |
| L2_Neural | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.480±0.052 | 0.143±0.006 | 0.2 | 0.00 |
| L2_PL-fast | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.500±0.000 | 0.140±0.000 | 0.2 | 0.00 |
| L2_PL-full | 0.870±0.003 | 0.818±0.007 | 0.246±0.043 | 0.989±0.005 | 0.960±0.012 | 0.2 | 0.01 |
| L2_PL-wmean | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.887±0.003 | 0.621±0.090 | 0.2 | 0.00 |
| BL_LR-stats | 0.998±0.002 | 0.998±0.002 | 0.993±0.007 | 1.000±0.000 | 1.000±0.000 | 0.2 | 0.19 |
| BL_XGB-stats | 0.973±0.007 | 0.972±0.007 | 0.886±0.030 | 0.995±0.005 | 0.975±0.022 | 0.9 | 0.44 |
| BL_Transformer-IMTS | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.745±0.033 | 0.347±0.069 | 1.5 | 0.06 |
| BL_SAnD (vendored) | 0.830±0.013 | 0.808±0.006 | 0.139±0.011 | 0.722±0.007 | 0.297±0.001 | 2.6 | 0.26 |
| BL_mTAN (vendored) | 0.913±0.017 | 0.895±0.025 | 0.582±0.098 | 0.983±0.004 | 0.942±0.026 | 0.5 | 0.04 |
| BL_GRU-D (vendored) | 0.845±0.002 | 0.791±0.004 | -0.021±0.032 | 0.606±0.030 | 0.225±0.011 | 0.8 | 0.06 |
