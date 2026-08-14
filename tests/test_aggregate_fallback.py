import json
from unittest.mock import patch

import pytest

from skillopt.gradient.aggregate import _merge_batch, merge_patches


def test_merge_batch_fallback_on_runtime_error():
    """Test that _merge_batch falls back to concatenation when optimizer raises RuntimeError."""
    patches = [
        {"reasoning": "r1", "edits": [{"id": "1", "content": "one"}]},
        {"reasoning": "r2", "edits": [{"id": "2", "content": "two"}]},
    ]

    with patch("skillopt.gradient.aggregate.chat_optimizer", side_effect=RuntimeError("API is down")):
        with pytest.warns(UserWarning, match="Optimizer call or parsing failed during batch merge"):
            result = _merge_batch(
                skill_content="skill content",
                patches=patches,
                system_prompt="merge this",
                update_mode="patch",
                level=2,
            )

    assert result["reasoning"] == "fallback concatenation"
    assert "edits" in result
    assert len(result["edits"]) == 2
    assert result["edits"][0]["id"] == "1"
    assert result["edits"][0]["merge_level"] == 2
    assert result["edits"][1]["id"] == "2"
    assert result["edits"][1]["merge_level"] == 2


def test_merge_batch_fallback_on_malformed_json():
    """Test that _merge_batch falls back when JSON extraction fails."""
    patches = [{"reasoning": "r1", "edits": [{"id": "1", "content": "one"}]}]

    with patch("skillopt.gradient.aggregate.chat_optimizer", return_value=("not json", None)):
        with patch("skillopt.gradient.aggregate.extract_json", side_effect=json.JSONDecodeError("Expecting value", "doc", 0)):
            with pytest.warns(UserWarning, match="Optimizer call or parsing failed during batch merge"):
                result = _merge_batch(
                    skill_content="skill content",
                    patches=patches,
                    system_prompt="merge this",
                    update_mode="patch",
                    level=1,
                )

    assert result["reasoning"] == "fallback concatenation"
    assert len(result["edits"]) == 1


def test_merge_patches_fallback_on_runtime_error():
    """Test that merge_patches falls back during the final merge if the optimizer fails."""
    failure_patches = [{"reasoning": "f1", "edits": [{"id": "1", "content": "f_one"}]}]
    success_patches = [{"reasoning": "s1", "edits": [{"id": "2", "content": "s_two"}]}]

    # We patch _hierarchical_merge to just return the single patches to skip the batch merges,
    # and then patch chat_optimizer to fail on the final merge.
    def mock_hierarchical(skill_content, patches, *args, **kwargs):
        return patches[0]

    with patch("skillopt.gradient.aggregate._hierarchical_merge", side_effect=mock_hierarchical):
        with patch("skillopt.gradient.aggregate.chat_optimizer", side_effect=RuntimeError("Timeout")):
            with pytest.warns(UserWarning, match="Optimizer call or parsing failed during final merge"):
                result = merge_patches(
                    skill_content="content",
                    failure_patches=failure_patches,
                    success_patches=success_patches,
                    batch_size=2,
                    verbose=False,
                )

    assert result["reasoning"] == "fallback: failure first, then success"
    assert len(result["edits"]) == 2
    assert result["edits"][0]["id"] == "1"
    assert result["edits"][1]["id"] == "2"
