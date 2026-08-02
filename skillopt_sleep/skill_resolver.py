"""SkillOpt-Sleep — resolve a discovered skill name to a local ``SKILL.md``.

Skill names observed in transcripts are untrusted strings, so resolution is
deliberately narrow: a name is normalized, matched only inside documented local
skill roots, and reported. Nothing here writes, edits, or creates files.

Resolution outcomes are distinguishable on purpose — ``missing`` (no root has
the skill) is a different signal from ``ambiguous`` (several roots do) and from
``rejected`` (the name itself is unusable), so callers can fall back to the
existing managed-skill behavior instead of guessing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

SKILL_FILENAME = "SKILL.md"

FOUND = "found"
MISSING = "missing"
AMBIGUOUS = "ambiguous"
REJECTED = "rejected"


@dataclass(frozen=True)
class SkillResolution:
    """The outcome of resolving one skill name. Never a partial success."""

    name: str
    status: str
    path: str = ""
    candidates: Tuple[str, ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == FOUND


def normalize_skill_name(name: object) -> str:
    """Return a usable skill directory name, or "" when the name is unusable.

    Only a single path segment is accepted: no separators, no parent traversal,
    no absolute or home-relative paths, no control characters. The name is
    whitespace-trimmed but otherwise preserved, since skill directories are
    case- and punctuation-sensitive.
    """
    if not isinstance(name, str):
        return ""
    candidate = name.strip()
    if not candidate or candidate in {os.curdir, os.pardir}:
        return ""
    if candidate.startswith("~"):
        return ""
    if os.path.isabs(candidate) or os.path.splitdrive(candidate)[0]:
        return ""
    if "/" in candidate or "\\" in candidate or os.sep in candidate:
        return ""
    if os.altsep and os.altsep in candidate:
        return ""
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in candidate):
        return ""
    return candidate


def _listdir(path: str) -> List[str]:
    """Sorted directory entries, or [] when the directory is absent/unreadable."""
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def skill_search_roots(cfg: object) -> List[str]:
    """Documented local skill roots for a config: user skills, then plugin cache.

    ``<claude_home>/skills`` holds hand-written skills; installed Claude Code
    plugins expose theirs under ``<claude_home>/plugins/cache/*/*/skills``.
    Only existing directories are returned, in that fixed precedence order.
    """
    configured = str(getattr(cfg, "claude_home", "") or "").strip()
    if not configured:
        # Guard before abspath: os.path.abspath("") is the CWD, which would
        # silently search a tree well outside the documented ~/.claude root.
        return []
    claude_home = os.path.abspath(os.path.expanduser(configured))
    roots = [os.path.join(claude_home, "skills")]

    cache = os.path.join(claude_home, "plugins", "cache")
    for marketplace in _listdir(cache):
        plugins_dir = os.path.join(cache, marketplace)
        if not os.path.isdir(plugins_dir):
            continue
        for plugin in _listdir(plugins_dir):
            roots.append(os.path.join(plugins_dir, plugin, "skills"))
    return [r for r in roots if os.path.isdir(r)]


def _contained_skill_file(root: str, name: str) -> str:
    """Return the real ``SKILL.md`` path under ``root`` for ``name``, else "".

    Symlinks are followed and then re-checked against the real root, so a skill
    directory or file that points outside the root is refused rather than read.
    """
    try:
        real_root = os.path.realpath(root)
        skill_file = os.path.realpath(os.path.join(real_root, name, SKILL_FILENAME))
    except OSError:
        return ""
    if not os.path.isfile(skill_file):
        return ""
    try:
        contained = os.path.commonpath([real_root, skill_file]) == real_root
    except ValueError:
        # Different drives / mixed path flavours (Windows): by definition the
        # file is not inside this root, so refuse it rather than crash.
        return ""
    if not contained:
        return ""
    return skill_file


def resolve_skill(name: object, roots: Sequence[str]) -> SkillResolution:
    """Resolve ``name`` against ``roots`` without touching any skill content."""
    normalized = normalize_skill_name(name)
    if not normalized:
        return SkillResolution(
            name=name if isinstance(name, str) else "",
            status=REJECTED,
            reason="skill name is empty or not a single safe path segment",
        )

    matches: List[str] = []
    for root in roots:
        found = _contained_skill_file(root, normalized)
        if found and found not in matches:
            matches.append(found)

    if not matches:
        return SkillResolution(
            name=normalized,
            status=MISSING,
            reason=f"no {SKILL_FILENAME} for this skill in the configured skill roots",
        )
    if len(matches) > 1:
        return SkillResolution(
            name=normalized,
            status=AMBIGUOUS,
            candidates=tuple(matches),
            reason="several skill roots define this skill",
        )
    return SkillResolution(
        name=normalized, status=FOUND, path=matches[0], candidates=tuple(matches)
    )
