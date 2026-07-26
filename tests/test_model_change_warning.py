"""Tests for the F16 model-change warning and F08 CLI credential warning."""
from __future__ import annotations

import argparse
import warnings

from skillopt_sleep.cycle import _check_model_change, _make_model_key
from skillopt_sleep.config import load_config
from skillopt_sleep.state import SleepState


def _cfg():
    cfg = load_config()
    return cfg


def test_last_model_key_roundtrips(tmp_path) -> None:
    state = SleepState.load(str(tmp_path / "state.json"))
    assert state.last_model_key == ""
    state.set_last_model_key("anthropic::claude")
    assert state.last_model_key == "anthropic::claude"


def test_warns_when_model_changed(tmp_path, capsys) -> None:
    cfg = _cfg()
    state = SleepState.load(str(tmp_path / "state.json"))
    state.set_last_model_key("some-other::model")
    _check_model_change(cfg, state)
    err = capsys.readouterr().err
    assert "model changed since last night" in err


def test_no_warning_on_first_night(tmp_path, capsys) -> None:
    cfg = _cfg()
    state = SleepState.load(str(tmp_path / "state.json"))  # last_model_key == ""
    _check_model_change(cfg, state)
    assert "model changed" not in capsys.readouterr().err


def test_no_warning_when_model_same(tmp_path, capsys) -> None:
    cfg = _cfg()
    state = SleepState.load(str(tmp_path / "state.json"))
    state.set_last_model_key(_make_model_key(cfg))
    _check_model_change(cfg, state)
    assert "model changed" not in capsys.readouterr().err


def test_cli_api_key_emits_deprecation_warning() -> None:
    from scripts.train import load_config as train_load_config

    args = argparse.Namespace(
        config="configs/_base_/default.yaml",
        cfg_options=None,
        azure_openai_api_key="sk-secret-value",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            train_load_config(args)
        except Exception:
            pass  # config loading may fail; we only assert the warning fired
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "azure_openai_api_key" in str(w.message)
        for w in caught
    )
