# PPθ-Post temporal comparison — 20260501_054441

> **Scope**: PPθ-Post L1 / L2 / L3 / L4 variants **plus external baselines**: lr, xgb, gru_d, transformer, interp_gn.  All baselines share the `(X_ts, mask, y)` interface; for InterpGN the entry is annotated with the average routing fraction `g(X)` — see §6.5 for the gate-opacity caveat.
>
> **Sanity-check disclaimer**: results on synthetic loaders (`p12` / `pam` / `mimic3`) are smoke-level only.  Paper-quality numbers require credentialed real benchmarks (PhysioNet/2012, PAMAP2, MIMIC-III/IV) — see `PAPER_LAYOUT.md`.

Levels: ['L1'] | folds: 2 | epochs: 6 | baselines: ['lr', 'xgb', 'gru_d', 'transformer', 'interp_gn']

## synthetic_p12

| Variant | Acc | F1 | MCC | ROC AUC | PR AUC | fit (s) | pred (s) |
|---|---|---|---|---|---|---|---|
| L1_Neural | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.836±0.024 | 0.544±0.041 | 1.1 | 0.00 |
| L1_PL-fast | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.500±0.000 | 0.140±0.000 | 1.1 | 0.00 |
| L1_PL-full | 0.887±0.023 | 0.847±0.044 | 0.356±0.213 | 0.995±0.002 | 0.966±0.017 | 1.1 | 0.01 |
| L1_PL-wmean | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.997±0.002 | 0.983±0.010 | 1.1 | 0.00 |
| BL_LR-stats | 0.998±0.002 | 0.998±0.002 | 0.993±0.007 | 1.000±0.000 | 1.000±0.000 | 0.2 | 0.20 |
| BL_XGB-stats | 0.973±0.007 | 0.972±0.007 | 0.886±0.030 | 0.995±0.005 | 0.975±0.022 | 0.9 | 0.49 |
| BL_GRU-D | 0.872±0.002 | 0.822±0.004 | 0.269±0.020 | 0.942±0.007 | 0.844±0.025 | 1.0 | 0.03 |
| BL_Transformer-IMTS | 0.862±0.002 | 0.799±0.004 | 0.072±0.072 | 0.780±0.003 | 0.405±0.082 | 2.5 | 0.06 |
| BL_InterpGN | 0.948±0.008 | 0.946±0.010 | 0.771±0.041 | 0.984±0.000 | 0.918±0.017 | 0.5 | 0.21 |
