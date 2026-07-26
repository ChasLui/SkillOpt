"""Tests for subprocess environment isolation in the spreadsheet executor.

``run_generated_code`` runs LLM-generated Python in a child process. To avoid
leaking API keys / cloud credentials into untrusted generated code, the child
must run with a minimal, scrubbed environment rather than inheriting the
parent process environment. These tests assert that scrubbing behaviour.
"""
from __future__ import annotations

import os

from skillopt.envs.spreadsheetbench.executor import run_generated_code


# User code that records whether a given env var is visible to the child.
_PROBE = (
    "import os\n"
    "with open(OUTPUT_PATH, 'w', encoding='utf-8') as _f:\n"
    "    _f.write(os.environ.get('SUPER_SECRET_TOKEN', 'ABSENT'))\n"
)


def test_secret_env_not_visible_to_generated_code(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "leak-me-please")
    out = tmp_path / "out.txt"

    ok, err = run_generated_code(_PROBE, str(tmp_path / "in.xlsx"), str(out))

    assert ok, err
    assert out.read_text(encoding="utf-8") == "ABSENT"


def test_path_still_available_to_generated_code(tmp_path) -> None:
    # PATH must be preserved so the child can still locate the interpreter's
    # tooling; only sensitive vars are dropped.
    probe = (
        "import os\n"
        "with open(OUTPUT_PATH, 'w', encoding='utf-8') as _f:\n"
        "    _f.write('YES' if os.environ.get('PATH') else 'NO')\n"
    )
    out = tmp_path / "out.txt"

    ok, err = run_generated_code(probe, str(tmp_path / "in.xlsx"), str(out))

    assert ok, err
    assert out.read_text(encoding="utf-8") == "YES"
