"""Tests for subprocess environment isolation in the spreadsheet executor.

``run_generated_code`` runs LLM-generated Python in a child process. To avoid
leaking API keys / cloud credentials into untrusted generated code, the child
must run with a minimal, scrubbed environment rather than inheriting the
parent process environment. These tests assert that scrubbing behaviour.
"""
from __future__ import annotations

import subprocess
import sys

from skillopt.envs.spreadsheetbench.codegen_agent import _build_codex_driver
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


def test_codex_driver_scrubs_env_sets_tempdir_and_cleans_runner(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "do-not-inherit")
    (tmp_path / "solution.py").write_text(
        "import os\n"
        "with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:\n"
        "    f.write('|'.join([\n"
        "        os.environ.get('SUPER_SECRET_TOKEN', 'ABSENT'),\n"
        "        'TMPDIR' if os.environ.get('TMPDIR') else 'NO_TMPDIR',\n"
        "    ]))\n",
        encoding="utf-8",
    )
    driver = tmp_path / "run_solution.py"
    driver.write_text(_build_codex_driver(), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(driver)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "output.xlsx").read_text(encoding="utf-8") == "ABSENT|TMPDIR"
    assert not (tmp_path / "_driver_runner.py").exists()
