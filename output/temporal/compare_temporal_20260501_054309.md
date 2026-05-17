# PPθ-Post temporal comparison — 20260501_054309

> **Scope**: PPθ-Post L1 / L2 / L3 / L4 variants **plus external baselines**: lr, xgb, transformer, gru_d, sand, mtan, seft, strats, raindrop, camelot, interp_gn.  All baselines share the `(X_ts, mask, y)` interface; for InterpGN the entry is annotated with the average routing fraction `g(X)` — see §6.5 for the gate-opacity caveat.
>
> **Sanity-check disclaimer**: results on synthetic loaders (`p12` / `pam` / `mimic3`) are smoke-level only.  Paper-quality numbers require credentialed real benchmarks (PhysioNet/2012, PAMAP2, MIMIC-III/IV) — see `PAPER_LAYOUT.md`.

Levels: ['L1'] | folds: 2 | epochs: 6 | baselines: ['lr', 'xgb', 'transformer', 'gru_d', 'sand', 'mtan', 'seft', 'strats', 'raindrop', 'camelot', 'interp_gn']

## synthetic_p12

| Variant | Acc | F1 | MCC | ROC AUC | PR AUC | fit (s) | pred (s) |
|---|---|---|---|---|---|---|---|
| L1_Neural | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.831±0.029 | 0.539±0.046 | 0.7 | 0.00 |
| L1_PL-fast | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.500±0.000 | 0.140±0.000 | 0.7 | 0.00 |
| L1_PL-full | 0.887±0.023 | 0.847±0.044 | 0.356±0.213 | 0.995±0.002 | 0.966±0.017 | 0.7 | 0.01 |
| L1_PL-wmean | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.997±0.002 | 0.984±0.011 | 0.7 | 0.00 |
| BL_LR-stats | 0.998±0.002 | 0.998±0.002 | 0.993±0.007 | 1.000±0.000 | 1.000±0.000 | 0.2 | 0.19 |
| BL_XGB-stats | 0.973±0.007 | 0.972±0.007 | 0.886±0.030 | 0.995±0.005 | 0.975±0.022 | 0.8 | 0.49 |
| BL_Transformer-IMTS | 0.862±0.002 | 0.799±0.004 | 0.072±0.072 | 0.780±0.003 | 0.405±0.082 | 2.1 | 0.08 |
| BL_GRU-D | 0.872±0.002 | 0.822±0.004 | 0.269±0.020 | 0.942±0.007 | 0.844±0.025 | 1.1 | 0.04 |
| BL_SAnD | 0.857±0.003 | 0.797±0.001 | 0.018±0.018 | 0.810±0.031 | 0.476±0.105 | 1.8 | 0.06 |
| BL_mTAN | 0.857±0.003 | 0.799±0.004 | 0.040±0.040 | 0.680±0.038 | 0.288±0.044 | 0.3 | 0.01 |
| BL_SeFT | 0.862±0.002 | 0.799±0.004 | 0.072±0.072 | 0.946±0.036 | 0.785±0.106 | 2.3 | 0.11 |
| BL_STraTS | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.838±0.046 | 0.531±0.144 | 6.6 | 0.33 |
| BL_Raindrop | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.988±0.001 | 0.953±0.006 | 13.6 | 0.57 |
| BL_CAMELOT | 0.860±0.000 | 0.795±0.000 | 0.000±0.000 | 0.540±0.046 | 0.185±0.026 | 0.7 | 0.02 |
| BL_InterpGN (g≈0.54) | 0.957±0.000 | 0.956±0.000 | 0.813±0.000 | 0.985±0.000 | 0.934±0.000 | 0.5 | 0.22 |
| BL_InterpGN (g≈0.53) | 0.940±0.000 | 0.936±0.000 | 0.730±0.000 | 0.984±0.000 | 0.901±0.000 | 0.6 | 0.22 |
