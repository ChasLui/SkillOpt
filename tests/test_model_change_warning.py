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
    path = str(tmp_path / "state.json")
    state = SleepState.load(path)
    assert state.last_model_key == ""
    state.set_last_model_key("anthropic::claude")
    state.save()
    assert SleepState.load(path).last_model_key == "anthropic::claude"


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


def test_model_key_tracks_effective_optimizer_and_target_roles() -> None:
    cfg = load_config(
        backend="mock",
        model="shared",
        optimizer_backend="claude",
        optimizer_model="opus",
        target_backend="codex",
        target_model="gpt",
    )
    assert _make_model_key(cfg) == (
        "optimizer=claude::opus;target=codex::gpt"
    )

    inherited = load_config(
        backend="mock",
        model="shared",
        optimizer_backend="claude",
    )
    assert _make_model_key(inherited) == (
        "optimizer=claude::shared;target=mock::shared"
    )


def test_warns_when_one_split_backend_role_changes(tmp_path, capsys) -> None:
    previous = load_config(
        optimizer_backend="claude",
        optimizer_model="opus",
        target_backend="codex",
        target_model="gpt",
    )
    current = load_config(
        optimizer_backend="claude",
        optimizer_model="opus",
        target_backend="cursor",
        target_model="composer",
    )
    state = SleepState.load(str(tmp_path / "state.json"))
    state.set_last_model_key(_make_model_key(previous))

    _check_model_change(current, state)

    assert "model changed since last night" in capsys.readouterr().err


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


def test_cli_key_warnings_name_the_correct_environment_variable() -> None:
    from scripts.train import load_config as train_load_config

    cases = {
        "optimizer_azure_openai_api_key": "OPTIMIZER_AZURE_OPENAI_API_KEY",
        "target_azure_openai_api_key": "TARGET_AZURE_OPENAI_API_KEY",
        "qwen_chat_api_key": "QWEN_CHAT_API_KEY",
        "optimizer_qwen_chat_api_key": "OPTIMIZER_QWEN_CHAT_API_KEY",
        "target_qwen_chat_api_key": "TARGET_QWEN_CHAT_API_KEY",
        "minimax_api_key": "MINIMAX_API_KEY",
    }
    for flag, environment_variable in cases.items():
        args = argparse.Namespace(
            config="configs/_base_/default.yaml",
            cfg_options=None,
            **{flag: "secret-value"},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                train_load_config(args)
            except Exception:
                pass
        messages = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DeprecationWarning)
        ]
        assert any(environment_variable in message for message in messages), flag
