"""Tests for the import-noise suppression helpers."""

import importlib
import os
import warnings

import pytest

from dataset_upload import quiet


@pytest.fixture
def clean_env(monkeypatch):
    for key in (
        "RBM_VERBOSE",
        "RBM_IMPORT_TF",
        "TRANSFORMERS_CACHE",
        "HF_HUB_CACHE",
        "TF_CPP_MIN_LOG_LEVEL",
        "GRPC_VERBOSITY",
        "GLOG_minloglevel",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch


def test_sets_logging_env_vars(clean_env):
    quiet.quiet_imports()
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "3"
    assert os.environ["GRPC_VERBOSITY"] == "ERROR"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_existing_values_are_not_clobbered(clean_env):
    clean_env.setenv("TF_CPP_MIN_LOG_LEVEL", "0")
    quiet.quiet_imports()
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "0"


def test_legacy_cache_is_migrated_not_dropped(clean_env):
    # Losing the cache path would silently re-download every model.
    clean_env.setenv("TRANSFORMERS_CACHE", "/data/hf-cache")
    quiet.quiet_imports()
    assert os.environ["HF_HUB_CACHE"] == "/data/hf-cache"
    assert "TRANSFORMERS_CACHE" not in os.environ


def test_explicit_hub_cache_wins_over_legacy(clean_env):
    clean_env.setenv("TRANSFORMERS_CACHE", "/old")
    clean_env.setenv("HF_HUB_CACHE", "/new")
    quiet.quiet_imports()
    assert os.environ["HF_HUB_CACHE"] == "/new"


def test_verbose_disables_everything(clean_env):
    clean_env.setenv("RBM_VERBOSE", "1")
    clean_env.setenv("TRANSFORMERS_CACHE", "/data/hf-cache")
    quiet.quiet_imports()
    assert "TF_CPP_MIN_LOG_LEVEL" not in os.environ
    assert os.environ["TRANSFORMERS_CACHE"] == "/data/hf-cache"


def test_noisy_warnings_are_filtered(clean_env):
    quiet.quiet_imports()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet.quiet_imports()  # re-apply after simplefilter reset
        warnings.warn("Using `TRANSFORMERS_CACHE` is deprecated", FutureWarning)
        warnings.warn("something that actually matters", UserWarning)
    messages = [str(w.message) for w in caught]
    assert "something that actually matters" in messages
    assert not any("TRANSFORMERS_CACHE" in m for m in messages)


def test_tensorflow_is_not_imported_by_default(clean_env):
    assert quiet.should_import_tensorflow() is False
    assert quiet.should_import_tensorflow("b1k_skill_train") is False


def test_tfds_backed_datasets_are_detected(clean_env):
    assert quiet.should_import_tensorflow("oxe_bridge_v2") is True
    assert quiet.should_import_tensorflow("soar_rfm") is True


def test_force_flag(clean_env):
    clean_env.setenv("RBM_IMPORT_TF", "1")
    assert quiet.should_import_tensorflow("b1k_skill_train") is True


def test_converter_module_imports_without_tensorflow(clean_env):
    # The regression this guards: generate_hf_dataset used to import TensorFlow
    # unconditionally, which prints cuDNN/cuBLAS errors once per pool worker.
    module = importlib.import_module("dataset_upload.generate_hf_dataset")
    assert module._TF_IMPORTED is False
