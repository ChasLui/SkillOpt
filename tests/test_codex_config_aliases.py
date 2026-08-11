import os
from unittest import mock

from skillopt.config import flatten_config
from skillopt.model.backend_config import configure_codex_exec


def test_codex_config_aliases_flatten_config():
    structured_cfg = {
        "model": {
            "sandbox": "danger-full-access",
            "codex_cli_bin": "/custom/path/codex"
        }
    }
    flat = flatten_config(structured_cfg)
    assert flat["codex_exec_sandbox"] == "danger-full-access"
    assert flat["codex_exec_path"] == "/custom/path/codex"


def test_codex_backend_env_aliases(monkeypatch):
    monkeypatch.setenv("CODEX_CLI_BIN", "/env/path/codex")
    monkeypatch.setenv("CODEX_SANDBOX_MODE", "danger-full-access-env")

    import importlib

    import skillopt.model.backend_config
    import skillopt.model.codex_backend

    importlib.reload(skillopt.model.backend_config)
    importlib.reload(skillopt.model.codex_backend)

    assert skillopt.model.codex_backend.CODEX_BIN == "/env/path/codex"
    assert skillopt.model.codex_backend.CODEX_SANDBOX_MODE == "danger-full-access-env"


def test_configure_codex_exec_sets_aliases(monkeypatch):
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_PATH", "")
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_SANDBOX", "")
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_APPROVAL_POLICY", "")

    configure_codex_exec(path="/configured/codex", sandbox="configured-sandbox")

    assert os.environ["CODEX_EXEC_PATH"] == "/configured/codex"
    assert os.environ["CODEX_CLI_BIN"] == "/configured/codex"

    assert os.environ["CODEX_EXEC_SANDBOX"] == "configured-sandbox"
    assert os.environ["CODEX_SANDBOX_MODE"] == "configured-sandbox"


@mock.patch("skillopt.model.codex_harness.subprocess.run")
def test_codex_harness_sandbox_passed_with_full_auto(mock_run, monkeypatch):
    from skillopt.model.codex_harness import _run_codex_cli_exec

    # Isolate global config
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_SANDBOX", "")
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_FULL_AUTO", False)
    monkeypatch.setattr("skillopt.model.backend_config.CODEX_EXEC_APPROVAL_POLICY", "ask")

    configure_codex_exec(sandbox="danger-full-access", full_auto=True, approval_policy="never")

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
        )

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]

    assert "--sandbox" in args
    sandbox_idx = args.index("--sandbox")
    assert args[sandbox_idx + 1] == "danger-full-access"

    assert "--approval-policy" in args
    approval_idx = args.index("--approval-policy")
    assert args[approval_idx + 1] == "never"
