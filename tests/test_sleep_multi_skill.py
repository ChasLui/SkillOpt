"""Tests for per-skill-group consolidation (issue #120).

Pure-stdlib (unittest), deterministic MockBackend, no API key, no network.
Run:  python -m pytest tests/test_sleep_multi_skill.py
"""
from __future__ import annotations

import unittest

from skillopt_sleep.backend import MockBackend
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.experiments.personas import programmer_persona, researcher_persona
from skillopt_sleep.memory import set_learned
from skillopt_sleep.mine import assign_splits
from skillopt_sleep.multi_skill import (
    CONSOLIDATED,
    FAILED,
    SKIPPED,
    GroupConsolidation,
    SkillGroup,
    accepted_group_skills,
    consolidate_groups,
)


def _tasks(persona, seed=42):
    return assign_splits(persona(), holdout_fraction=0.34, seed=seed)


class TestConsolidateGroups(unittest.TestCase):
    def test_no_groups_is_an_empty_mapping(self):
        self.assertEqual(consolidate_groups(MockBackend(), []), {})

    def test_single_group_matches_the_existing_single_skill_run(self):
        skill = set_learned("", [])
        direct = consolidate(MockBackend(), _tasks(researcher_persona), skill, "",
                             edit_budget=4, gate_metric="mixed", night=1,
                             evolve_memory=False)
        grouped = consolidate_groups(
            MockBackend(),
            [SkillGroup("research-skill", skill, _tasks(researcher_persona))],
            edit_budget=4, gate_metric="mixed", night=1,
        )
        self.assertEqual(list(grouped), ["research-skill"])
        outcome = grouped["research-skill"]
        self.assertEqual(outcome.status, CONSOLIDATED)
        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.result.new_skill, direct.new_skill)
        self.assertEqual(outcome.result.gate_action, direct.gate_action)
        self.assertEqual(outcome.result.candidate_score, direct.candidate_score)

    def test_each_group_gets_its_own_decision_and_document(self):
        groups = [
            SkillGroup("research-skill", set_learned("# research\n", []),
                       _tasks(researcher_persona)),
            SkillGroup("programming-skill", set_learned("# programming\n", []),
                       _tasks(programmer_persona, seed=1)),
        ]
        outcomes = consolidate_groups(MockBackend(), groups, edit_budget=4, night=1)
        self.assertEqual(list(outcomes), ["research-skill", "programming-skill"])
        self.assertIn("# research", outcomes["research-skill"].result.new_skill)
        self.assertIn("# programming", outcomes["programming-skill"].result.new_skill)
        self.assertNotIn("# programming", outcomes["research-skill"].result.new_skill)

    def test_group_without_tasks_is_skipped_not_fatal(self):
        outcomes = consolidate_groups(
            MockBackend(),
            [
                SkillGroup("empty-skill", set_learned("", []), []),
                SkillGroup("research-skill", set_learned("", []), _tasks(researcher_persona)),
            ],
            edit_budget=4, night=1,
        )
        self.assertEqual(outcomes["empty-skill"].status, SKIPPED)
        self.assertIsNone(outcomes["empty-skill"].result)
        self.assertIn("no mined tasks", outcomes["empty-skill"].reason)
        self.assertEqual(outcomes["research-skill"].status, CONSOLIDATED)

    def test_group_without_a_name_is_skipped_and_reported(self):
        outcomes = consolidate_groups(
            MockBackend(),
            [SkillGroup("  ", set_learned("", []), _tasks(researcher_persona))],
            edit_budget=4, night=1,
        )
        self.assertEqual(outcomes[""].status, SKIPPED)
        self.assertIn("no skill name", outcomes[""].reason)

    def test_repeated_group_name_runs_once(self):
        calls = []

        def _fake(backend, tasks, skill, memory, **kwargs):
            calls.append(skill)
            return consolidate(backend, tasks, skill, memory, **kwargs)

        outcomes = consolidate_groups(
            MockBackend(),
            [
                SkillGroup("research-skill", set_learned("# first\n", []),
                           _tasks(researcher_persona)),
                SkillGroup("research-skill", set_learned("# second\n", []),
                           _tasks(researcher_persona)),
            ],
            consolidate_fn=_fake, edit_budget=4, night=1,
        )
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("# first", calls[0])

    def test_one_failing_group_is_isolated_and_reported(self):
        def _fake(backend, tasks, skill, memory, **kwargs):
            if "boom" in skill:
                raise RuntimeError("backend exploded")
            return consolidate(backend, tasks, skill, memory, **kwargs)

        outcomes = consolidate_groups(
            MockBackend(),
            [
                SkillGroup("broken-skill", "boom", _tasks(researcher_persona)),
                SkillGroup("research-skill", set_learned("", []), _tasks(researcher_persona)),
            ],
            consolidate_fn=_fake, edit_budget=4, night=1,
        )
        self.assertEqual(outcomes["broken-skill"].status, FAILED)
        self.assertIsNone(outcomes["broken-skill"].result)
        self.assertIn("RuntimeError: backend exploded", outcomes["broken-skill"].reason)
        self.assertEqual(outcomes["research-skill"].status, CONSOLIDATED)
        self.assertTrue(outcomes["research-skill"].accepted)

    def test_group_runs_do_not_evolve_shared_memory(self):
        seen = {}

        def _fake(backend, tasks, skill, memory, **kwargs):
            seen.update(kwargs)
            seen["memory"] = memory
            return consolidate(backend, tasks, skill, memory, **kwargs)

        consolidate_groups(
            MockBackend(),
            [SkillGroup("research-skill", set_learned("", []), _tasks(researcher_persona))],
            "# shared memory\n", consolidate_fn=_fake, edit_budget=4, night=1,
        )
        self.assertEqual(seen["memory"], "# shared memory\n")
        self.assertFalse(seen["evolve_memory"])

    def test_accepted_group_skills_lists_only_accepted_updates(self):
        outcomes = {
            "kept": GroupConsolidation(
                "kept", CONSOLIDATED,
                result=consolidate(MockBackend(), _tasks(researcher_persona),
                                   set_learned("", []), "", edit_budget=4, night=1),
            ),
            "skipped": GroupConsolidation("skipped", SKIPPED, reason="no mined tasks"),
            "failed": GroupConsolidation("failed", FAILED, reason="RuntimeError: x"),
        }
        self.assertTrue(outcomes["kept"].accepted)
        self.assertEqual(list(accepted_group_skills(outcomes)), ["kept"])
        self.assertFalse(outcomes["skipped"].accepted)
        self.assertFalse(outcomes["failed"].accepted)

    def test_consolidating_groups_writes_no_files(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            before = sorted(os.listdir(tmp))
            consolidate_groups(
                MockBackend(),
                [SkillGroup("research-skill", set_learned("", []), _tasks(researcher_persona))],
                edit_budget=4, night=1,
            )
            self.assertEqual(sorted(os.listdir(tmp)), before)


if __name__ == "__main__":
    unittest.main()
