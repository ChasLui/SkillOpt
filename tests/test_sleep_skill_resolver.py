"""Tests for bounded skill-name resolution (issue #120).

Pure-stdlib (unittest), hermetic (tmpdir only), no API key, no network.
Run:  python -m pytest tests/test_sleep_skill_resolver.py
"""
from __future__ import annotations

import os
import tempfile
import unittest

from skillopt_sleep.config import load_config
from skillopt_sleep.skill_resolver import (
    AMBIGUOUS,
    FOUND,
    MISSING,
    REJECTED,
    normalize_skill_name,
    resolve_skill,
    skill_search_roots,
)


def _write_skill(root, name, body="# skill\n"):
    path = os.path.join(root, name, "SKILL.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


class TestNormalizeSkillName(unittest.TestCase):
    def test_trims_but_preserves_case_and_punctuation(self):
        self.assertEqual(normalize_skill_name("  Brand-Voice.v2  "), "Brand-Voice.v2")

    def test_rejects_unusable_names(self):
        for bad in ["", "   ", ".", "..", "../escape", "a/b", "a\\b", "/abs/skill",
                    "~/skill", "bad\nname", "bad\x00name", None, 3]:
            self.assertEqual(normalize_skill_name(bad), "", repr(bad))


class TestResolveSkill(unittest.TestCase):
    def test_resolves_a_single_local_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = _write_skill(tmp, "example-skill")
            res = resolve_skill("  example-skill  ", [tmp])
            self.assertEqual(res.status, FOUND)
            self.assertTrue(res.ok)
            self.assertEqual(res.path, os.path.realpath(expected))
            self.assertEqual(res.name, "example-skill")

    def test_missing_skill_is_distinct_from_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_skill(tmp, "other-skill")
            res = resolve_skill("example-skill", [tmp])
            self.assertEqual(res.status, MISSING)
            self.assertEqual(res.path, "")
            self.assertEqual(res.candidates, ())

    def test_directory_without_skill_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "example-skill"))
            self.assertEqual(resolve_skill("example-skill", [tmp]).status, MISSING)

    def test_same_skill_in_two_roots_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            local, cache = os.path.join(tmp, "local"), os.path.join(tmp, "cache")
            first = _write_skill(local, "example-skill")
            second = _write_skill(cache, "example-skill")
            res = resolve_skill("example-skill", [local, cache])
            self.assertEqual(res.status, AMBIGUOUS)
            self.assertEqual(res.path, "")
            self.assertEqual(res.candidates,
                             (os.path.realpath(first), os.path.realpath(second)))

    def test_repeated_root_is_not_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_skill(tmp, "example-skill")
            self.assertEqual(resolve_skill("example-skill", [tmp, tmp]).status, FOUND)

    def test_traversal_is_rejected_without_touching_the_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "roots")
            _write_skill(tmp, "outside-skill")
            os.makedirs(root, exist_ok=True)
            res = resolve_skill("../outside-skill", [root])
            self.assertEqual(res.status, REJECTED)
            self.assertEqual(res.path, "")

    def test_symlinked_skill_dir_escaping_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "roots")
            os.makedirs(root)
            outside = os.path.join(tmp, "outside")
            _write_skill(outside, "example-skill")
            os.symlink(os.path.join(outside, "example-skill"),
                       os.path.join(root, "example-skill"))
            self.assertEqual(resolve_skill("example-skill", [root]).status, MISSING)

    def test_symlinked_skill_file_escaping_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "roots")
            os.makedirs(os.path.join(root, "example-skill"))
            elsewhere = os.path.join(tmp, "elsewhere.md")
            with open(elsewhere, "w", encoding="utf-8") as f:
                f.write("# not in the root\n")
            os.symlink(elsewhere, os.path.join(root, "example-skill", "SKILL.md"))
            self.assertEqual(resolve_skill("example-skill", [root]).status, MISSING)

    def test_symlinked_root_itself_still_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = os.path.join(tmp, "real")
            _write_skill(real, "example-skill")
            link = os.path.join(tmp, "link")
            os.symlink(real, link)
            self.assertEqual(resolve_skill("example-skill", [link]).status, FOUND)

    def test_resolution_never_modifies_the_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_skill(tmp, "example-skill", "# original\n")
            with open(path, encoding="utf-8") as f:
                before = (os.stat(path).st_size, f.read())
            resolve_skill("example-skill", [tmp])
            with open(path, encoding="utf-8") as f:
                after = (os.stat(path).st_size, f.read())
            self.assertEqual(after, before)

    def test_no_roots_is_missing(self):
        self.assertEqual(resolve_skill("example-skill", []).status, MISSING)

    def test_candidates_are_an_immutable_tuple(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_skill(tmp, "example-skill")
            res = resolve_skill("example-skill", [tmp])
            self.assertIsInstance(res.candidates, tuple)
            with self.assertRaises(AttributeError):
                res.candidates.append("/injected")  # type: ignore[attr-defined]

    def test_skill_file_on_another_drive_is_refused_not_crashed(self):
        # os.path.commonpath raises ValueError for paths that share no root
        # (mixed drives on Windows). Resolution must treat that as "outside".
        with tempfile.TemporaryDirectory() as tmp:
            _write_skill(tmp, "example-skill")
            real_commonpath = os.path.commonpath

            def exploding_commonpath(paths):
                raise ValueError("paths don't have the same drive")

            os.path.commonpath = exploding_commonpath
            try:
                self.assertEqual(resolve_skill("example-skill", [tmp]).status, MISSING)
            finally:
                os.path.commonpath = real_commonpath


class TestSkillSearchRoots(unittest.TestCase):
    def test_user_skills_root_comes_first_then_plugin_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            skills = os.path.join(claude_home, "skills")
            os.makedirs(skills)
            plugin_skills = os.path.join(
                claude_home, "plugins", "cache", "marketplace", "plugin", "skills"
            )
            os.makedirs(plugin_skills)
            cfg = load_config(claude_home=claude_home)
            self.assertEqual(skill_search_roots(cfg), [skills, plugin_skills])

    def test_absent_roots_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(claude_home=os.path.join(tmp, ".claude"))
            self.assertEqual(skill_search_roots(cfg), [])

    def test_blank_claude_home_never_falls_back_to_the_cwd(self):
        # os.path.abspath("") is the CWD; a blank override must not turn the
        # working directory into a skill root.
        class _Cfg:
            def __init__(self, claude_home):
                self.claude_home = claude_home

        for blank in ["", "   ", None]:
            self.assertEqual(skill_search_roots(_Cfg(blank)), [], repr(blank))

    def test_unreadable_plugin_cache_does_not_break_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            skills = os.path.join(claude_home, "skills")
            os.makedirs(skills)
            cache = os.path.join(claude_home, "plugins", "cache")
            os.makedirs(cache)
            os.chmod(cache, 0o000)
            try:
                cfg = load_config(claude_home=claude_home)
                self.assertEqual(skill_search_roots(cfg), [skills])
            finally:
                os.chmod(cache, 0o700)

    def test_resolution_through_config_roots_prefers_the_user_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            claude_home = os.path.join(tmp, ".claude")
            expected = _write_skill(os.path.join(claude_home, "skills"), "example-skill")
            cfg = load_config(claude_home=claude_home)
            res = resolve_skill("example-skill", skill_search_roots(cfg))
            self.assertEqual(res.status, FOUND)
            self.assertEqual(res.path, os.path.realpath(expected))

    def test_legacy_target_skill_path_behavior_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "repo", "SKILL.md")
            cfg = load_config(claude_home=os.path.join(tmp, ".claude"),
                              target_skill_path=target)
            self.assertEqual(cfg.managed_skill_path(), os.path.abspath(target))


if __name__ == "__main__":
    unittest.main()
