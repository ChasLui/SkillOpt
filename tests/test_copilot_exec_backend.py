from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

import skillopt.model as model
from skillopt.config import flatten_config
from skillopt.model import backend_config, copilot_backend
from skillopt.model import codex_harness as harness
from skillopt.model.common import normalize_backend_name


@pytest.fixture(autouse=True)
def restore_backend_state() -> Iterator[None]:
    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    copilot_path = backend_config.COPILOT_EXEC_PATH
    copilot_home = backend_config.COPILOT_EXEC_HOME
    copilot_tools = backend_config.COPILOT_EXEC_ALLOW_ALL_TOOLS
    chat_optimizer_model = backend_config.COPILOT_CHAT_OPTIMIZER_MODEL
    chat_target_model = backend_config.COPILOT_CHAT_TARGET_MODEL
    chat_timeout = backend_config.COPILOT_CHAT_TIMEOUT
    retries = backend_config.EXEC_EMPTY_RESPONSE_RETRIES
    env = {
        key: os.environ.get(key)
        for key in (
            "OPTIMIZER_BACKEND",
            "TARGET_BACKEND",
            "COPILOT_EXEC_PATH",
            "COPILOT_EXEC_HOME",
            "COPILOT_EXEC_ALLOW_ALL_TOOLS",
            "COPILOT_CHAT_OPTIMIZER_MODEL",
            "COPILOT_CHAT_TARGET_MODEL",
            "COPILOT_CHAT_TIMEOUT",
        )
    }
    yield
    backend_config.OPTIMIZER_BACKEND = optimizer_backend
    backend_config.TARGET_BACKEND = target_backend
    backend_config.COPILOT_EXEC_PATH = copilot_path
    backend_config.COPILOT_EXEC_HOME = copilot_home
    backend_config.COPILOT_EXEC_ALLOW_ALL_TOOLS = copilot_tools
    backend_config.COPILOT_CHAT_OPTIMIZER_MODEL = chat_optimizer_model
    backend_config.COPILOT_CHAT_TARGET_MODEL = chat_target_model
    backend_config.COPILOT_CHAT_TIMEOUT = chat_timeout
    backend_config.EXEC_EMPTY_RESPONSE_RETRIES = retries
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.mark.parametrize("alias", ["copilot_exec"])
def test_aliases_normalize_to_copilot_exec(alias: str) -> None:
    assert normalize_backend_name(alias) == "copilot_exec"


@pytest.mark.parametrize("alias", ["copilot", "copilot_cli", "github_copilot"])
def test_bare_copilot_aliases_normalize_to_the_fully_local_chat_backend(
    alias: str,
) -> None:
    assert normalize_backend_name(alias) == "copilot_chat"


def test_set_backend_routes_target_to_copilot_and_keeps_chat_optimizer() -> None:
    assert model.set_backend("copilot_exec") == "copilot_exec"
    # The CLI agent is the *target*; the optimizer must stay a chat model
    # because it has to emit structured skill edits.
    assert backend_config.get_target_backend() == "copilot_exec"
    assert backend_config.get_optimizer_backend() == "openai_chat"
    assert backend_config.is_target_exec_backend() is True
    assert model.get_backend_name() == "copilot_exec"


def test_chat_target_refuses_exec_backend() -> None:
    model.set_backend("copilot_exec")
    with pytest.raises(NotImplementedError):
        model.chat_target(system="s", user="u")


def test_configure_rejects_non_boolean_allow_all_tools() -> None:
    with pytest.raises(ValueError):
        model.configure_copilot_exec(allow_all_tools="sometimes")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", "1"), ("false", "0"), ("yes", "1"), ("off", "0"), ("1", "1"), ("weird", "0")],
)
def test_env_allow_all_tools_is_normalized_on_load(raw, expected, monkeypatch) -> None:
    # A user may set the env var to 'true'/'false'; loading it raw used to make
    # get_copilot_exec_config() raise. It is now normalized to '0'/'1' (unknown
    # values fall back to the safe '0'), matching the other boolean-ish flags.
    import importlib

    monkeypatch.setenv("COPILOT_EXEC_ALLOW_ALL_TOOLS", raw)
    importlib.reload(backend_config)
    try:
        assert backend_config.COPILOT_EXEC_ALLOW_ALL_TOOLS == expected
        assert backend_config.get_copilot_exec_config()["allow_all_tools"] == expected
    finally:
        monkeypatch.delenv("COPILOT_EXEC_ALLOW_ALL_TOOLS", raising=False)
        importlib.reload(backend_config)


def test_parse_jsonl_concatenates_assistant_messages_and_ignores_noise() -> None:
    raw = "\n".join(
        [
            "not json at all",
            json.dumps({"type": "tool.call", "data": {"content": "ignored"}}),
            json.dumps({"type": "assistant.message", "data": {"content": "first"}}),
            "{ broken json",
            json.dumps({"type": "assistant.message", "data": {"content": "second"}}),
            json.dumps({"type": "assistant.message", "data": {}}),
        ]
    )
    assert harness._parse_copilot_jsonl(raw) == "first\nsecond"


def test_parse_jsonl_returns_empty_for_no_assistant_messages() -> None:
    assert harness._parse_copilot_jsonl('{"type":"tool.call","data":{}}') == ""
    assert harness._parse_copilot_jsonl("") == ""


def _fake_run(captured: dict, stdout: str, returncode: int = 0):
    def _run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env") or {}
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return _run


def test_run_copilot_exec_builds_isolated_readonly_command(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot")
    model.configure_copilot_exec(path="copilot", home=str(tmp_path / "home"), allow_all_tools=False)
    captured: dict = {}
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "answer"}})
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, stdout))

    response, raw = harness.run_copilot_exec(work_dir=str(tmp_path), prompt="solve it", model="", timeout=30)

    assert response == "answer"
    assert "COPILOT CLI ATTEMPT 1" in raw
    cmd = captured["cmd"]
    # Prompt is a single argv element -- never interpolated into a shell string.
    assert "solve it" in " ".join(cmd)
    assert cmd[cmd.index("-p") + 1].endswith("solve it") or "solve it" in cmd[cmd.index("-p") + 1]
    assert "--output-format" in cmd and cmd[cmd.index("--output-format") + 1] == "json"
    # Startup isolation: no user MCP servers or custom instructions.
    assert "--disable-builtin-mcps" in cmd
    assert "--no-custom-instructions" in cmd
    # Read-only rollout must NOT bypass the CLI approval gate.
    assert "--allow-all-tools" not in cmd
    assert captured["env"]["COPILOT_HOME"] == str(tmp_path / "home")


def test_config_default_does_not_clobber_the_env_opt_in() -> None:
    # The shipped base config must not pin this to false: trainer.py and
    # eval_only.py pass cfg.get("copilot_exec_allow_all_tools") straight into
    # configure_copilot_exec(), and a non-None value overwrites the env var --
    # which would make the documented COPILOT_EXEC_ALLOW_ALL_TOOLS=1 opt-in
    # impossible. null leaves the env (default off) in charge.
    import yaml

    root = Path(__file__).resolve().parent.parent
    with open(root / "configs" / "_base_" / "default.yaml", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    assert base["model"]["copilot_exec_allow_all_tools"] is None

    # None must be a no-op, so a prior opt-in survives.
    model.configure_copilot_exec(allow_all_tools=True)
    model.configure_copilot_exec(allow_all_tools=None)
    assert backend_config.COPILOT_EXEC_ALLOW_ALL_TOOLS == "1"


def test_copilot_backends_keep_a_real_deployment_fallback() -> None:
    # This table also feeds the shared Azure deployment fallback in the entry
    # points (cfg.get("optimizer_model", default_model_for_backend(backend))),
    # and the shipped base config sets no optimizer_model/target_model -- so a
    # "" here left `--backend copilot_exec` with an EMPTY optimizer deployment
    # even though that role is still a real openai_chat model. The CLI's own
    # model comes from copilot_chat_optimizer_model / _target_model instead.
    from skillopt.model.common import default_model_for_backend

    assert default_model_for_backend("copilot_exec") == "gpt-4o"
    assert default_model_for_backend("copilot_chat") == "gpt-4o"


def test_no_response_error_includes_cli_output(monkeypatch, tmp_path) -> None:    # copilot_exec persists no artifacts, so a bare "returned no response"
    # would leave an empty/invalid JSONL stream undebuggable.
    model.set_backend("copilot")
    model.configure_copilot_exec(path="copilot", home="", allow_all_tools=False)
    captured: dict = {}
    noise = '{"type":"tool.call","data":{"content":"NOISE-MARKER"}}'
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, noise))

    with pytest.raises(RuntimeError) as exc:
        harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30)

    assert "returned no response" in str(exc.value)
    assert "NOISE-MARKER" in str(exc.value)


def test_allow_all_tools_requires_both_opt_in_and_file_edits(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot")
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "ok"}})

    # Opted in, but read-only rollout -> still gated.
    model.configure_copilot_exec(allow_all_tools=True)
    captured: dict = {}
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, stdout))
    harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30, allow_file_edits=False)
    assert "--allow-all-tools" not in captured["cmd"]

    # File edits requested but operator did not opt in -> still gated.
    model.configure_copilot_exec(allow_all_tools=False)
    captured = {}
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, stdout))
    harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30, allow_file_edits=True)
    assert "--allow-all-tools" not in captured["cmd"]

    # Both -> allowed.
    model.configure_copilot_exec(allow_all_tools=True)
    captured = {}
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, stdout))
    harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30, allow_file_edits=True)
    assert "--allow-all-tools" in captured["cmd"]


def test_nonzero_exit_raises_with_detail(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    monkeypatch.setattr(harness.subprocess, "run", _fake_run(captured, "", returncode=3))
    with pytest.raises(RuntimeError, match="exit code 3"):
        harness.run_copilot_exec(work_dir=str(tmp_path), prompt="p", model="", timeout=30)


def test_run_target_exec_dispatches_to_copilot(monkeypatch, tmp_path) -> None:
    model.set_backend("copilot_exec")
    called: dict = {}

    def _fake(**kwargs):
        called.update(kwargs)
        return "resp", "raw"

    monkeypatch.setattr(harness, "run_copilot_exec", _fake)
    response, raw = harness.run_target_exec(work_dir=str(tmp_path), prompt="p", model="m", timeout=15)
    assert (response, raw) == ("resp", "raw")
    assert called["model"] == "m"


# --- copilot_chat: the fully-local path (no cloud API key) -------------------


def test_copilot_alias_selects_fully_local_chat_backend() -> None:
    assert normalize_backend_name("copilot") == "copilot_chat"
    assert model.set_backend("copilot") == "copilot_chat"
    # Both halves run on the local CLI -- this is what makes a run key-free.
    assert backend_config.get_optimizer_backend() == "copilot_chat"
    assert backend_config.get_target_backend() == "copilot_chat"
    assert backend_config.is_optimizer_chat_backend() is True
    assert backend_config.is_target_chat_backend() is True
    assert backend_config.is_target_exec_backend() is False
    # Both roles unify to the canonical label, like claude_chat/qwen_chat, not
    # the generic "copilot_chat+copilot_chat".
    assert model.get_backend_name() == "copilot_chat"


def test_copilot_exec_still_pairs_with_a_chat_optimizer() -> None:
    # Guards the split: `copilot` is fully local, `copilot_exec` is not.
    assert model.set_backend("copilot_exec") == "copilot_exec"
    assert backend_config.get_optimizer_backend() == "openai_chat"
    assert backend_config.get_target_backend() == "copilot_exec"


def test_chat_backend_composes_system_and_user_into_one_prompt(monkeypatch) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "done"}})
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, stdout))

    text, usage = model.chat_target(system="SYS", user="USR", retries=1)

    assert text == "done"
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    cmd = captured["cmd"]
    prompt = cmd[cmd.index("-p") + 1]
    assert "SYS" in prompt and "USR" in prompt
    assert "--disable-builtin-mcps" in cmd and "--no-custom-instructions" in cmd
    # A chat call must never be granted unattended tool use.
    assert "--allow-all-tools" not in cmd


def test_chat_backend_routes_optimizer_and_target_models(monkeypatch) -> None:
    model.set_backend("copilot")
    model.configure_copilot_chat(optimizer_model="opt-m", target_model="tgt-m")
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "x"}})

    captured: dict = {}
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, stdout))
    model.chat_optimizer(system="s", user="u", retries=1)
    assert captured["cmd"][captured["cmd"].index("--model") + 1] == "opt-m"

    captured = {}
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, stdout))
    model.chat_target(system="s", user="u", retries=1)
    assert captured["cmd"][captured["cmd"].index("--model") + 1] == "tgt-m"


def test_chat_backend_raises_on_empty_response(monkeypatch) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, ""))
    with pytest.raises(RuntimeError, match="after 1 retries"):
        model.chat_target(system="s", user="u", retries=1)


def test_chat_backend_raises_on_nonzero_exit(monkeypatch) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, "", returncode=2))
    with pytest.raises(RuntimeError, match="exit code 2"):
        model.chat_target(system="s", user="u", retries=1)


def test_configure_copilot_chat_rejects_bad_timeout() -> None:
    with pytest.raises(ValueError):
        model.configure_copilot_chat(timeout="0")
    with pytest.raises(ValueError):
        model.configure_copilot_chat(timeout="soon")


def test_messages_variant_flattens_roles(monkeypatch) -> None:
    model.set_backend("copilot")
    captured: dict = {}
    stdout = json.dumps({"type": "assistant.message", "data": {"content": "ok"}})
    monkeypatch.setattr(copilot_backend.subprocess, "run", _fake_run(captured, stdout))

    text, _ = model.chat_target_messages(
        messages=[
            {"role": "system", "content": "SYSTEM_TEXT"},
            {"role": "user", "content": "USER_TEXT"},
        ],
        retries=1,
    )
    assert text == "ok"
    prompt = captured["cmd"][captured["cmd"].index("-p") + 1]
    assert "SYSTEM_TEXT" in prompt and "USER_TEXT" in prompt


def test_copilot_config_keys_survive_flattening() -> None:
    # Guards the 4-place wiring: config.py map, both scripts, and trainer.
    flat = flatten_config(
        {
            "model": {
                "copilot_exec_path": "/opt/copilot",
                "copilot_exec_home": "/tmp/cph",
                "copilot_exec_allow_all_tools": True,
                "copilot_chat_optimizer_model": "opt-m",
                "copilot_chat_target_model": "tgt-m",
                "copilot_chat_timeout": 900,
            }
        }
    )
    assert flat["copilot_exec_path"] == "/opt/copilot"
    assert flat["copilot_exec_home"] == "/tmp/cph"
    assert flat["copilot_exec_allow_all_tools"] is True
    assert flat["copilot_chat_optimizer_model"] == "opt-m"
    assert flat["copilot_chat_target_model"] == "tgt-m"
    assert flat["copilot_chat_timeout"] == 900


@pytest.mark.parametrize("script", ["train", "eval_only"])
def test_cli_exposes_copilot_backend_and_flags(script: str) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert '"copilot_chat"' in text
    assert '"copilot_exec"' in text
    for flag in (
        "--copilot_exec_path",
        "--copilot_chat_target_model",
        "--copilot_chat_timeout",
    ):
        assert flag in text
    # train.py delegates backend configuration to the trainer; eval_only wires
    # it directly. Assert whichever applies so the wiring can't silently drop.
    applier = (
        text if script == "eval_only" else (root / "skillopt" / "engine" / "trainer.py").read_text(encoding="utf-8")
    )
    assert "configure_copilot_chat" in applier
    assert "configure_copilot_exec" in applier
