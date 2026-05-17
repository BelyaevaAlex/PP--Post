"""Per-baseline dependency manifests for vendored IMTS SOTA models.

Each ``<name>.txt`` in this directory lists the *exact* set of Python
packages required by the adapter for one vendored baseline (mTAN,
SAnD, GRU-D, Raindrop, InterpGN, SeFT, CAMELOT).  These manifests
exist as our own curated lists rather than relying on upstream
``requirements.txt`` because:

* not every upstream ships one (SAnD, GRU-D, InterpGN do not);
* some upstream files are stale or broken (CAMELOT contains
  unresolved git-merge markers; SeFT pins TensorFlow 1.15 +
  Python 3.7);
* we want to keep the *adapter* surface installable on a modern
  Python 3.10+ stack, even when the original published setup is
  no longer reachable.

The companion script :mod:`temporal.vendor_extras.setup_envs` reads
each manifest and provisions an isolated virtualenv under
``temporal/vendor/.venvs/<name>/`` so a baseline's transitive
dependencies (e.g. ``reformer_pytorch`` for InterpGN, or TF 2.x for
SeFT) never enter the main PPθ-Post environment.

At runtime, :mod:`temporal.baselines_vendored` automatically detects
the per-baseline ``site-packages`` directory and prepends it to
``sys.path`` for the duration of the model's import — see
``_isolated_top_level_namespace`` for the cleanup logic.
"""

__all__: list[str] = []
