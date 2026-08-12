#!/usr/bin/env python3
"""Silence third-party import noise around dataset conversion.

A conversion run prints hundreds of lines that have nothing to do with the
conversion -- and with a multiprocess pool every worker repeats them, which
buries the messages that matter (a broken ffmpeg, a dropped trajectory).

Sources, and what actually silences each:

* ``Unable to register cuDNN/cuBLAS factory``, ``computation placer already
  registered``, ``absl::InitializeLog`` -- emitted from TensorFlow's C++ layer
  at import. Python warning filters and ``TF_CPP_MIN_LOG_LEVEL`` do **not**
  touch them; the only fix is not importing TensorFlow. Only the RLDS-based
  loaders (OXE, SOAR, AgiBotWorld) need it, so the import is opt-in.
* ``Using TRANSFORMERS_CACHE is deprecated`` -- set ``HF_HUB_CACHE`` instead,
  which is the direct replacement and keeps the same cache directory.
* ``google.api_core ... Python version`` and similar FutureWarnings from
  transitive dependencies.

Set ``RBM_VERBOSE=1`` to turn all of this off and see everything.
"""

from __future__ import annotations

import os
import warnings

__all__ = ["quiet_imports", "should_import_tensorflow"]

_NOISY_WARNING_PATTERNS = (
    r".*TRANSFORMERS_CACHE.*",
    r".*google\.api_core.*",
    r".*which Google will stop supporting.*",
    r".*`resume_download` is deprecated.*",
)


def _verbose() -> bool:
    return os.environ.get("RBM_VERBOSE", "").strip() not in ("", "0", "false", "False")


def quiet_imports() -> None:
    """Set env vars and warning filters. Call before importing heavy deps."""
    if _verbose():
        return

    # Must be set before the library that reads them is imported.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # TRANSFORMERS_CACHE was renamed; carry the value over so the existing cache
    # is still used, then drop the deprecated name to stop the warning.
    legacy_cache = os.environ.pop("TRANSFORMERS_CACHE", None)
    if legacy_cache and not os.environ.get("HF_HUB_CACHE"):
        os.environ["HF_HUB_CACHE"] = legacy_cache

    for pattern in _NOISY_WARNING_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern)
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\..*")


def should_import_tensorflow(dataset_name: str | None = None) -> bool:
    """Whether this run needs TensorFlow.

    Only the RLDS/TFDS-backed loaders do. Force it on with ``RBM_IMPORT_TF=1``.
    """
    if os.environ.get("RBM_IMPORT_TF", "").strip() not in ("", "0", "false", "False"):
        return True
    if not dataset_name:
        return False
    name = dataset_name.lower()
    return any(tag in name for tag in ("oxe", "soar", "agibot", "rlds", "roboreward", "bridge"))
