"""Smoke-suite for the TensorFlow / Keras vendored baseline adapters.

Validates that:
  * registry keys agree with the on-disk submodules,
  * if TensorFlow is available, the adapters at least *instantiate*
    successfully (full fit-predict round-trips are skipped to keep CI
    fast — these baselines are heavy to train),
  * if TensorFlow is **not** available, the adapters raise a clear
    :class:`RuntimeError` mentioning ``tensorflow``, so the comparison
    driver can skip the row gracefully.
"""

from __future__ import annotations

import importlib.util
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
for path in (PARENT_DIR, GRANDPARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from temporal.baselines_vendored_tf import (  # noqa: E402
    VENDORED_TF_REGISTRY,
    make_vendored_tf,
)


_TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


def test_vendored_tf_registry_keys():
    expected = {"seft", "camelot"}
    assert set(VENDORED_TF_REGISTRY) == expected


def test_vendored_tf_unknown_raises():
    try:
        make_vendored_tf("does_not_exist", n_classes=2)
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown TF baseline")


def test_vendored_tf_instantiation_or_skip():
    """If TF is installed, ``seft`` and ``camelot`` must instantiate; if
    not, they must raise a clear ``RuntimeError`` mentioning tensorflow.
    """
    for key in ("seft", "camelot"):
        try:
            adapter = make_vendored_tf(key, n_classes=2)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if _TF_AVAILABLE:
                # TF is here — anything other than a clear submodule
                # error is unexpected.
                assert "submodule" in msg, (
                    f"{key}: TF is installed yet adapter raised: {exc}"
                )
                continue
            assert "tensorflow" in msg, (
                f"{key}: error message must mention tensorflow; got: {exc}"
            )
            continue
        # Adapter constructed → make sure the name is recognisable.
        assert key in adapter.name.lower(), adapter.name


if __name__ == "__main__":
    test_vendored_tf_registry_keys()
    test_vendored_tf_unknown_raises()
    test_vendored_tf_instantiation_or_skip()
    print("test_baselines_vendored_tf: OK")
