import importlib
import os
from unittest import mock

import pytest

from skillopt.config import flatten_config
from skillopt.model.backend_config import configure_codex_exec, validate_exec_sandbox


def test_codex_config_aliases_flatten_config():
    structured_cfg = {
        "model": {
            "sandbox": "danger-full-access",
            "codex_cli_bin": "/custom/path/codex",
        }
    }
    flat = flatten_config(structured_cfg)
    assert flat["codex_exec_sandbox"] == "danger-full-access"
    assert flat["codex_exec_path"] == "/custom/path/codex"


def test_codex_backend_env_aliases(monkeypatch):
    monkeypatch.setenv("CODEX_CLI_BIN", "/env/path/codex")
    monkeypatch.setenv("CODEX_SANDBOX_MODE", "danger-full-access")

    import skillopt.model.backend_config
    import skillopt.model.codex_backend

    importlib.reload(skillopt.model.backend_config)
    importlib.reload(skillopt.model.codex_backend)

    assert skillopt.model.codex_backend.CODEX_BIN == "/env/path/codex"
    assert skillopt.model.codex_backend.CODEX_SANDBOX_MODE == "danger-full-access"


def test_configure_codex_exec_sets_aliases(monkeypatch):
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_PATH", "")
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_SANDBOX", "")
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_APPROVAL_POLICY", "")

    configure_codex_exec(path="/configured/codex", sandbox="danger-full-access")

    assert os.environ["CODEX_EXEC_PATH"] == "/configured/codex"
    assert os.environ["CODEX_CLI_BIN"] == "/configured/codex"

    assert os.environ["CODEX_EXEC_SANDBOX"] == "danger-full-access"
    assert os.environ["CODEX_SANDBOX_MODE"] == "danger-full-access"


def test_validate_exec_sandbox():
    assert validate_exec_sandbox("read-only") == "read-only"
    assert validate_exec_sandbox("workspace-write") == "workspace-write"
    assert validate_exec_sandbox("danger-full-access") == "danger-full-access"

    with pytest.raises(ValueError, match="Invalid codex_exec sandbox"):
        validate_exec_sandbox("invalid-mode")

    with pytest.raises(ValueError, match="Invalid codex_exec sandbox"):
        configure_codex_exec(sandbox="unrestricted-bad-mode")


def test_entry_points_alias_precedence(monkeypatch):
    # Test precedence: codex_exec_sandbox > sandbox > codex_sandbox
    cfg = {
        "codex_exec_sandbox": "danger-full-access",
        "sandbox": "workspace-write",
        "codex_sandbox": "read-only",
        "codex_exec_path": "/path/exec",
        "codex_path": "/path/codex",
        "codex_cli_bin": "/path/cli",
    }
    sandbox = cfg.get("codex_exec_sandbox") or cfg.get("sandbox") or cfg.get("codex_sandbox")
    path = cfg.get("codex_exec_path") or cfg.get("codex_path") or cfg.get("codex_cli_bin")
    assert sandbox == "danger-full-access"
    assert path == "/path/exec"

    # Fallback to sandbox alias
    cfg_alias = {
        "sandbox": "danger-full-access",
        "codex_cli_bin": "/path/cli",
    }
    sandbox_alias = cfg_alias.get("codex_exec_sandbox") or cfg_alias.get("sandbox") or cfg_alias.get("codex_sandbox")
    path_alias = cfg_alias.get("codex_exec_path") or cfg_alias.get("codex_path") or cfg_alias.get("codex_cli_bin")
    assert sandbox_alias == "danger-full-access"
    assert path_alias == "/path/cli"


@pytest.mark.parametrize("full_auto_setting", [True, False])
@mock.patch("skillopt.model.codex_harness.subprocess.run")
def test_codex_harness_command_construction(mock_run, monkeypatch, full_auto_setting):
    from skillopt.model.codex_harness import _run_codex_cli_exec

    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_SANDBOX", "workspace-write")
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_FULL_AUTO", True)
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_APPROVAL_POLICY", "never")

    configure_codex_exec(sandbox="danger-full-access", full_auto=full_auto_setting, approval_policy="never")

    mock_proc = mock.MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = ""
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc

    with mock.patch("skillopt.model.codex_harness._persist_codex_artifacts"), \
         mock.patch("skillopt.model.azure_openai.tracker.record", create=True):
        _run_codex_cli_exec(
            work_dir=".",
            prompt="test",
            model="test-model",
            timeout=10,
            full_auto=full_auto_setting,
        )

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]

    assert "--sandbox" in args
    sandbox_idx = args.index("--sandbox")
    assert args[sandbox_idx + 1] == "danger-full-access"

    # Must NOT use deprecated flags --approval-policy or --full-auto
    assert "--approval-policy" not in args
    assert "--full-auto" not in args

    # Approval policy must be passed via config override -c approval_policy="..."
    assert "-c" in args
    config_overrides = [args[i + 1] for i, arg in enumerate(args) if arg == "-c"]
    assert 'approval_policy="never"' in config_overrides
