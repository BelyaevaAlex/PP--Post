# Vendored baselines (`temporal/vendor/`)

This directory contains **seven git submodules** pinned to specific
upstream commits.  We never copy or fork the upstream code — the parent
repository only stores a `.gitmodules` reference and a commit SHA.

After the migration to the *only-vendored* SOTA strategy, these
adapters are the **only** source of truth for the seven IMTS-specific
SOTA baselines (GRU-D, SAnD, mTAN, SeFT, Raindrop, CAMELOT, InterpGN);
no re-implementation fallback exists.  The slim `temporal/baselines.py`
keeps only three baselines that have no dedicated upstream
(`LR-stats`, `XGB-stats`, `Transformer-IMTS`).

## PyTorch submodules — `temporal/baselines_vendored.py`

| Submodule  | Upstream                                                | License        | Pinned commit (date)     |
|------------|---------------------------------------------------------|----------------|--------------------------|
| `sand`     | https://github.com/khirotaka/SAnD                       | **MIT**        | `b5da888` (2022-08-23)   |
| `mtan`     | https://github.com/reml-lab/mTAN                        | **MIT**        | `7a3d536` (2021-06-07)   |
| `raindrop` | https://github.com/mims-harvard/Raindrop                | **MIT**        | `892eb57` (2022-10-04)   |
| `grud`     | https://github.com/zhiyongc/GRU-D                       | _no LICENSE_¹  | `d070b52` (2019-06-02)   |
| `interpgn` | https://github.com/YunshiWen/InterpretGatedNetwork      | _no LICENSE_¹  | `5ea6045` (2025-02-28)   |

## TensorFlow / Keras submodules — `temporal/baselines_vendored_tf.py`

| Submodule  | Upstream                                                | License        | Pinned commit (date)     |
|------------|---------------------------------------------------------|----------------|--------------------------|
| `seft`     | https://github.com/BorgwardtLab/Set_Functions_for_Time_Series | **BSD-3**| `6abd69c` (2020)         |
| `camelot`  | https://github.com/hrna-ox/camelot-icml                 | _no LICENSE_¹  | `6c493ba` (2022)         |

¹ The upstream repository ships **without an explicit licence file**;
under default copyright law we therefore cannot redistribute its
source.  Pinning via a git submodule is *not* redistribution — the
source is fetched at clone time directly from the original maintainer's
GitHub.  We rely on the fair-use carve-out for academic comparison; if
the authors object, remove the submodule.

## How to obtain the source

```bash
# from the PPθ-Post root
git submodule update --init --recursive
```

If you want to update a submodule to a newer upstream commit (re-pin):

```bash
cd temporal/vendor/<name>
git checkout <new-commit>
cd ../../..
git add temporal/vendor/<name>
git commit -m "vendor: bump <name> to <new-commit>"
```

## Adapter integration status

| Submodule  | Adapter                              | Required extras                                       |
|------------|--------------------------------------|-------------------------------------------------------|
| `sand`     | `VendoredSAnDBaseline`               | none — runs on CPU                                    |
| `mtan`     | `VendoredMTANBaseline`               | none — runs on CPU                                    |
| `grud`     | `VendoredGRUDBaseline`               | none — runs on CPU (`hidden_size = n_vars`)           |
| `raindrop` | `VendoredRaindropBaseline`           | `torch_geometric` + CUDA-capable GPU (A100 ready)     |
| `interpgn` | `VendoredInterpGNBaseline`           | `einops`, `reformer_pytorch` (transitive PatchTST/TimesNet deps); FCN backbone via `default_interpgn_configs()`. Top-level `models`/`utils`/`layers` packages are isolated via `_isolated_top_level_namespace` so they do not clash with mTAN's `vendor/mtan/src/models.py`. |
| `seft`     | `VendoredSeFTBaseline` (TF / Keras)  | TensorFlow ≥ 2.4 (upstream uses TF1-era APIs — may need patching on TF2-only environments) |
| `camelot`  | `VendoredCAMELOTBaseline` (TF / Keras) | TensorFlow ≥ 2.4                                    |

When an adapter cannot initialise (missing TensorFlow, missing CUDA,
missing `torch_geometric`, upstream import error, …), the comparison
driver in `compare_temporal.py` emits a one-line diagnostic
``[skipped] baseline 'X': <reason>`` and continues with the remaining
baselines — there is no re-implementation fallback for SOTA rows.

## Per-baseline virtualenvs (recommended)

Each vendored baseline pulls in different — and sometimes mutually
incompatible — Python dependencies (TF 2.x for SeFT/CAMELOT,
`torch_geometric` for Raindrop, `reformer_pytorch` + `einops` for
InterpGN, …).  To keep the main PPθ-Post environment clean, every
baseline can ship its dependencies in a *dedicated* virtualenv at
`temporal/vendor/.venvs/<name>/`.

Curated dependency manifests live in
`temporal/vendor_extras/<name>.txt` (we maintain them ourselves
because several upstream `requirements.txt` files are stale, broken
with merge-conflict markers, or pin Python 3.7-only TF 1.15).

Bootstrap one or more environments:

```bash
# All seven baselines (uses the current python interpreter).
python -m temporal.vendor_extras.setup_envs

# A single baseline (faster + lighter when iterating).
python -m temporal.vendor_extras.setup_envs interpgn

# Re-create from scratch.
python -m temporal.vendor_extras.setup_envs --force seft

# Use a specific Python (e.g. 3.10 for SeFT's TF 2.x compatibility).
python -m temporal.vendor_extras.setup_envs --python /opt/python3.10/bin/python3 seft
```

At runtime, `temporal.baselines_vendored._isolated_top_level_namespace`
detects each `.venvs/<name>/site-packages/`, prepends it to `sys.path`
*only for the duration* of the baseline's import, and restores the
prior state on exit.  This means:

* the main PPθ-Post environment never has to install heavy deps such
  as TF or `torch_geometric`;
* different baselines may pin incompatible versions of the same
  package without stepping on each other;
* if a venv is missing, the adapter falls back to the main
  environment, and `compare_temporal.py` logs `[skipped]` cleanly
  when a transitive dependency is unavailable there too.

## Why vendored only?

Running the *authors' original code* gives reviewers an
apples-to-apples comparison and removes any ambiguity about whose
implementation is being evaluated.  The driver wraps every adapter in
the same `StratifiedKFold` protocol and the same `(X_ts, mask, y)`
contract used by PPθ-Post itself, so cross-validation, metric
collection and reporting are uniform across rows — only the model
internals are the upstream black-box.
