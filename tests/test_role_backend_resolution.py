"""``--backend`` must survive the role backends the base config sets."""

from __future__ import annotations

import pytest

from skillopt.engine.trainer import _resolve_role_backends

# What configs/_base_/default.yaml ships.
_BASE_CONFIG = ("openai_chat", "openai_chat")


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("cursor", ("openai_chat", "cursor_exec")),
        ("cursor_exec", ("openai_chat", "cursor_exec")),
        ("claude_code_exec", ("openai_chat", "claude_code_exec")),
        ("codex", ("codex_exec", "codex_exec")),
        ("codex_exec", ("codex_exec", "codex_exec")),
        ("qwen", ("openai_chat", "qwen_chat")),
        ("qwen_chat", ("openai_chat", "qwen_chat")),
    ],
)
def test_backend_flag_wins_over_base_config_defaults(backend, expected) -> None:
    # Regression: the base config sets both roles to openai_chat, which used to
    # skip resolution entirely and silently run --backend <x> on openai_chat.
    assert _resolve_role_backends(backend, *_BASE_CONFIG) == expected


def test_azure_openai_stays_on_openai_chat() -> None:
    assert _resolve_role_backends("azure_openai", *_BASE_CONFIG) == _BASE_CONFIG


def test_unset_roles_are_resolved() -> None:
    assert _resolve_role_backends("cursor", "", "") == ("openai_chat", "cursor_exec")
    assert _resolve_role_backends("cursor", None, None) == ("openai_chat", "cursor_exec")


def test_explicit_non_default_roles_are_preserved() -> None:
    # An operator who names a role backend outranks the high-level label.
    assert _resolve_role_backends("cursor", "qwen_chat", "minimax_chat") == (
        "qwen_chat",
        "minimax_chat",
    )


def test_explicit_target_is_preserved_when_optimizer_is_default() -> None:
    assert _resolve_role_backends("cursor", "openai_chat", "minimax_chat") == (
        "openai_chat",
        "minimax_chat",
    )


def test_copilot_maps_both_roles_to_the_local_cli() -> None:
    # `copilot` is the only fully local option: no cloud API key is needed
    # because the CLI carries its own sign-in.
    assert _resolve_role_backends("copilot", *_BASE_CONFIG) == ("copilot_chat", "copilot_chat")
    assert _resolve_role_backends("copilot_chat", *_BASE_CONFIG) == (
        "copilot_chat",
        "copilot_chat",
    )


def test_copilot_exec_keeps_a_chat_optimizer() -> None:
    assert _resolve_role_backends("copilot_exec", *_BASE_CONFIG) == (
        "openai_chat",
        "copilot_exec",
    )


def test_claude_maps_both_roles() -> None:
    assert _resolve_role_backends("claude", None, None) == ("claude_chat", "claude_chat")
