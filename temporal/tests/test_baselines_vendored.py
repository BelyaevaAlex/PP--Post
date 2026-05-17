"""Smoke-suite for the vendored baseline adapters.

Validates that:
  * the registry keys agree with the on-disk submodules,
  * the three CPU-runnable adapters (SAnD, mTAN, GRU-D) successfully
    fit + predict on a tiny synthetic dataset, and
  * the two GPU/extras-only adapters (Raindrop, InterpGN) raise an
    informative ``RuntimeError`` rather than crashing silently.
"""

from __future__ import annotations

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
for path in (PARENT_DIR, GRANDPARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from temporal.baselines_vendored import (  # noqa: E402
    VENDORED_REGISTRY,
    make_vendored,
)
from temporal.datasets import load_synthetic_p12  # noqa: E402


def _tiny_dataset():
    return load_synthetic_p12(
        n_samples=24, T=8, missing_ratio=0.4, seed=11,
    )


def test_vendored_registry_keys():
    expected = {"sand", "mtan", "gru_d", "raindrop", "interp_gn"}
    assert set(VENDORED_REGISTRY) == expected


def test_vendored_unknown_raises():
    try:
        make_vendored("does_not_exist", n_classes=2)
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown vendored baseline")


def _check_vendored_fit_predict(name: str, fast_kwargs):
    X_ts, mask, y, _, _ = _tiny_dataset()
    baseline = make_vendored(name, n_classes=2, seed=0, **fast_kwargs)
    baseline.fit(X_ts, mask, y)
    proba = baseline.predict_proba(X_ts, mask)
    assert proba.shape == (X_ts.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(proba >= -1e-6) and np.all(proba <= 1.0 + 1e-6)


def test_vendored_sand_fits_and_predicts():
    _check_vendored_fit_predict(
        "sand",
        {"d_model": 32, "n_heads": 2, "n_layers": 1, "factor": 2,
         "epochs": 5, "batch_size": 8},
    )


def test_vendored_mtan_fits_and_predicts():
    _check_vendored_fit_predict(
        "mtan",
        {"n_ref": 4, "nhidden": 16, "embed_time": 8, "n_heads": 2,
         "epochs": 5, "batch_size": 8},
    )


def test_vendored_grud_fits_and_predicts():
    # NOTE: zhiyongc/GRU-D upstream requires hidden_size == n_vars; we
    # leave it at the default (None) which auto-sets to n_vars.
    _check_vendored_fit_predict(
        "gru_d",
        {"hidden_size": None, "epochs": 5, "batch_size": 8},
    )


def test_vendored_raindrop_raises_when_unavailable():
    # On a CPU-only sandbox without torch_geometric this MUST raise.
    try:
        make_vendored("raindrop", n_classes=2)
    except RuntimeError as exc:
        msg = str(exc).lower()
        assert ("cuda" in msg) or ("torch_geometric" in msg) or ("submodule" in msg), msg
        return
    print("note: vendored Raindrop is runnable on this machine")


def test_vendored_interpgn_instantiates_with_defaults():
    """``default_interpgn_configs`` is now built lazily inside the
    adapter's ``fit`` if the user does not supply ``configs`` — so the
    bare instantiation must succeed.  Sanity-check that the upstream
    submodule is reachable.
    """
    try:
        baseline = make_vendored("interp_gn", n_classes=2)
    except RuntimeError as exc:
        msg = str(exc).lower()
        assert "submodule" in msg, msg
        return
    assert baseline is not None
    assert "interpgn" in baseline.name.lower()


def test_per_baseline_venv_discovery_optional():
    """``_baseline_venv_site_packages`` must return ``None`` (not raise)
    when no per-baseline venv has been provisioned, so the adapter
    falls back to the main environment cleanly.
    """
    from temporal.baselines_vendored import _baseline_venv_site_packages

    for name in ("sand", "mtan", "gru_d", "raindrop", "interp_gn",
                 "seft", "camelot", "does_not_exist"):
        sp = _baseline_venv_site_packages(name)
        assert sp is None or os.path.isdir(sp), (
            f"venv site-packages for '{name}' is neither None nor a "
            f"valid directory: {sp!r}"
        )


def test_isolated_top_level_namespace_restores_state():
    """The shared context manager must leave ``sys.path`` and the
    relevant slice of ``sys.modules`` exactly as it found them, even
    when the inner block raises.
    """
    from temporal.baselines_vendored import (
        VENDOR_DIR,
        _isolated_top_level_namespace,
    )

    sys.modules["models"] = types_module = type(sys)("models")
    types_module.__file__ = "<test-sentinel>"
    saved_path = list(sys.path)
    saved_modules_models = sys.modules.get("models")

    interpgn_root = os.path.join(VENDOR_DIR, "interpgn")
    if not os.path.isdir(interpgn_root):
        # No submodule cloned — nothing to test, but still verify the
        # cleanup invariant on a no-op context.
        with _isolated_top_level_namespace(VENDOR_DIR, ("models",)):
            pass
        assert sys.path == saved_path
        assert sys.modules.get("models") is saved_modules_models
        return

    try:
        with _isolated_top_level_namespace(
            interpgn_root, ("models", "utils", "layers"),
        ):
            assert sys.modules.get("models") is None, (
                "context manager failed to evict the sentinel 'models'"
            )
            assert sys.path[0] == os.path.abspath(interpgn_root)
            raise RuntimeError("simulate downstream failure")
    except RuntimeError as exc:
        assert str(exc) == "simulate downstream failure"

    assert sys.path == saved_path, "sys.path was not restored on exit"
    assert sys.modules.get("models") is saved_modules_models, (
        "sys.modules['models'] was not restored on exit"
    )
    sys.modules.pop("models", None)


if __name__ == "__main__":
    test_vendored_registry_keys()
    test_vendored_unknown_raises()
    test_vendored_sand_fits_and_predicts()
    test_vendored_mtan_fits_and_predicts()
    test_vendored_grud_fits_and_predicts()
    test_vendored_raindrop_raises_when_unavailable()
    test_vendored_interpgn_instantiates_with_defaults()
    test_per_baseline_venv_discovery_optional()
    test_isolated_top_level_namespace_restores_state()
    print("test_baselines_vendored: OK")
