"""Tests for shell-injection hardening in the ReAct agent's bash tool.

``_run_bash`` previously ran commands with ``shell=True``, allowing arbitrary
shell metacharacter injection. It now uses ``shlex.split`` + ``shell=False``
and restricts the executable to a small allow-list (python/python3). These
tests assert both the allow-list gate and that shell metacharacters are no
longer interpreted.
"""
from __future__ import annotations

from skillopt.envs.spreadsheetbench.react_agent import _run_bash


def test_disallowed_command_is_blocked(tmp_path) -> None:
    out = _run_bash("curl http://example.com/evil", str(tmp_path))
    assert "blocked" in out.lower()


def test_allowed_python_runs(tmp_path) -> None:
    # Bare 'python' resolves via PATH; a full Windows path would be mangled by
    # shlex.split (posix mode), which is expected agent-input behaviour here.
    out = _run_bash('python -c "print(42)"', str(tmp_path))
    assert "42" in out


def test_shell_metacharacters_not_interpreted(tmp_path) -> None:
    # With shell=False the ';' and following tokens become arguments to python,
    # not a second shell command, so the marker file must NOT be created.
    marker = tmp_path / "pwned.txt"
    cmd = "python -c \"print(1)\" ; python -c \"open('pwned.txt','w')\""
    _run_bash(cmd, str(tmp_path))
    assert not marker.exists()
