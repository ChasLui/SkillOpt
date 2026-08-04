from __future__ import annotations

import os
import sqlite3

import pytest

from skillopt_sleep.harvest_copilot_cli import default_session_store, harvest_copilot_cli

_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    cwd TEXT,
    repository TEXT,
    branch TEXT,
    summary TEXT,
    created_at TEXT,
    updated_at TEXT,
    host_type TEXT
);
CREATE TABLE turns (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    turn_index INTEGER,
    user_message TEXT,
    assistant_response TEXT,
    timestamp TEXT
);
CREATE TABLE session_files (
    session_id TEXT,
    file_path TEXT,
    tool_name TEXT
);
"""


def _store(tmp_path, sessions, turns, files=()):
    path = os.path.join(str(tmp_path), "session-store.db")
    con = sqlite3.connect(path)
    con.executescript(_SCHEMA)
    con.executemany(
        "INSERT INTO sessions (id, cwd, repository, branch, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        sessions,
    )
    con.executemany(
        "INSERT INTO turns (session_id, turn_index, user_message, assistant_response, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        turns,
    )
    if files:
        con.executemany(
            "INSERT INTO session_files (session_id, file_path, tool_name) VALUES (?, ?, ?)",
            files,
        )
    con.commit()
    con.close()
    return path


def test_missing_store_returns_empty(tmp_path) -> None:
    assert harvest_copilot_cli(os.path.join(str(tmp_path), "nope.db")) == []


def test_default_store_points_at_copilot_home() -> None:
    assert default_session_store().endswith(os.path.join(".copilot", "session-store.db"))


def test_maps_session_and_turn_fields(tmp_path) -> None:
    path = _store(
        tmp_path,
        [("s1", r"C:\proj", "repo", "main", "2026-01-01 10:00:00", "2026-01-01 10:30:00")],
        [
            ("s1", 0, "Find the Nille repo", "Found it at C:/e", "2026-01-01 10:00:00"),
            ("s1", 1, "that is still broken", "Fixed now", "2026-01-01 10:20:00"),
        ],
        [("s1", r"C:\proj\a.py", "edit")],
    )
    digests = harvest_copilot_cli(path, scope="all")

    assert len(digests) == 1
    d = digests[0]
    assert d.session_id == "s1"
    assert d.project == r"C:\proj"
    assert d.git_branch == "main"
    assert d.n_user_turns == 2
    assert d.n_assistant_turns == 2
    assert "Find the Nille repo" in d.user_prompts
    assert d.files_touched == [r"C:\proj\a.py"]
    assert d.tools_used == ["edit"]
    # "still broken" is a negative-feedback phrase and must survive as a label.
    assert any(s.startswith("neg:") for s in d.feedback_signals)
    assert d.raw_path.endswith("#s1")


def test_engine_self_calls_are_filtered(tmp_path) -> None:
    # SkillOpt's own Copilot backend writes to this same store; harvesting them
    # would train the engine on its own output.
    path = _store(
        tmp_path,
        [
            ("real", r"C:\proj", "", "main", "2026-01-01 10:00:00", "2026-01-01 10:30:00"),
            ("engine", r"C:\proj", "", "main", "2026-01-01 11:00:00", "2026-01-01 11:00:02"),
        ],
        [
            ("real", 0, "Review and rebase the PR", "Done", "2026-01-01 10:00:00"),
            (
                "engine",
                0,
                "You are an expert question answering agent.\n\n## Skill\n# QA Skill\n",
                "Oslo",
                "2026-01-01 11:00:00",
            ),
        ],
    )
    ids = [d.session_id for d in harvest_copilot_cli(path, scope="all")]
    assert ids == ["real"]


def test_scope_invoked_filters_by_project(tmp_path) -> None:
    path = _store(
        tmp_path,
        [
            ("a", r"C:\projA", "", "", "2026-01-01 10:00:00", "2026-01-01 10:30:00"),
            ("b", r"C:\projB", "", "", "2026-01-01 11:00:00", "2026-01-01 11:30:00"),
        ],
        [
            ("a", 0, "task in A", "ok", "2026-01-01 10:00:00"),
            ("b", 0, "task in B", "ok", "2026-01-01 11:00:00"),
        ],
    )
    ids = [d.session_id for d in harvest_copilot_cli(path, scope="invoked", invoked_project=r"C:\projA")]
    assert ids == ["a"]


def test_since_iso_compares_at_day_granularity(tmp_path) -> None:
    # Timestamps mix "YYYY-MM-DD HH:MM:SS" and ISO-8601, so only the date part
    # is safe to compare.
    path = _store(
        tmp_path,
        [
            ("old", r"C:\p", "", "", "2026-01-01 10:00:00", "2026-01-01 10:20:00"),
            ("new", r"C:\p", "", "", "2026-06-01T10:00:00.000Z", "2026-06-01T10:20:00.000Z"),
        ],
        [
            ("old", 0, "old task", "ok", "2026-01-01 10:00:00"),
            ("new", 0, "new task", "ok", "2026-06-01T10:00:00.000Z"),
        ],
    )
    ids = [d.session_id for d in harvest_copilot_cli(path, scope="all", since_iso="2026-05-01")]
    assert ids == ["new"]


def test_limit_caps_results(tmp_path) -> None:
    sessions = [(f"s{i}", r"C:\p", "", "", f"2026-01-0{i} 10:00:00", f"2026-01-0{i} 10:30:00") for i in range(1, 6)]
    turns = [(f"s{i}", 0, f"task {i}", "ok", f"2026-01-0{i} 10:00:00") for i in range(1, 6)]
    path = _store(tmp_path, sessions, turns)
    assert len(harvest_copilot_cli(path, scope="all", limit=2)) == 2


def test_sessions_without_usable_prompts_are_skipped(tmp_path) -> None:
    path = _store(
        tmp_path,
        [("empty", r"C:\p", "", "", "2026-01-01 10:00:00", "2026-01-01 10:00:00")],
        [("empty", 0, "", "orphan answer", "2026-01-01 10:00:00")],
    )
    assert harvest_copilot_cli(path, scope="all") == []


def test_short_programmatic_sessions_are_filtered(tmp_path) -> None:
    # A sub-3-second single-turn session with a short prompt is an engine call,
    # not interactive work.
    path = _store(
        tmp_path,
        [("quick", r"C:\p", "", "", "2026-01-01T10:00:00.000Z", "2026-01-01T10:00:01.000Z")],
        [("quick", 0, "ping", "pong", "2026-01-01T10:00:00.000Z")],
    )
    assert harvest_copilot_cli(path, scope="all") == []


@pytest.mark.parametrize("source", ["copilot_cli"])
def test_source_is_registered(source: str) -> None:
    from skillopt_sleep import harvest_sources

    with open(harvest_sources.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert f'source == "{source}"' in text
    assert "harvest_copilot_cli" in text


def test_session_without_cwd_is_skipped(tmp_path) -> None:
    # No stable cwd -> not scopable and would collide on project+intent hashing.
    path = _store(
        tmp_path,
        [("nocwd", None, "owner/repo", "main", "2026-01-01 10:00:00", "2026-01-01 10:30:00")],
        [("nocwd", 0, "do a real task here", "ok", "2026-01-01 10:00:00")],
    )
    assert harvest_copilot_cli(path, scope="all") == []


def test_short_session_with_space_timestamps_is_filtered(tmp_path) -> None:
    # The store uses a space separator; normalization must let the sub-3-second
    # replay heuristic fire just as it does for 'T'-separated ISO timestamps.
    path = _store(
        tmp_path,
        [("quick", r"C:\p", "", "", "2026-01-01 10:00:00", "2026-01-01 10:00:01")],
        [("quick", 0, "ping", "pong", "2026-01-01 10:00:00")],
    )
    assert harvest_copilot_cli(path, scope="all") == []


def test_connect_failure_fails_closed(tmp_path, monkeypatch) -> None:
    # A locked/unreadable store must yield nothing rather than abort the run.
    path = _store(
        tmp_path,
        [("s1", r"C:\p", "", "", "2026-01-01 10:00:00", "2026-01-01 10:30:00")],
        [("s1", 0, "a real task", "ok", "2026-01-01 10:00:00")],
    )

    def _boom(_store_path):
        raise OSError("permission denied")

    monkeypatch.setattr("skillopt_sleep.harvest_copilot_cli._connect", _boom)
    assert harvest_copilot_cli(path, scope="all") == []

