from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

import scripts.train as train_script


def _argv(*extra: str) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    return [
        "skillopt-train",
        "--config",
        str(root / "configs" / "searchqa" / "default.yaml"),
        *extra,
    ]


# "0" is the case from the original report: the option has to be recognised as
# supplied even when it is falsy.
@pytest.mark.parametrize("value", ["5", "0"])
def test_retired_option_warns_and_stays_out_of_the_config(monkeypatch, value) -> None:
    monkeypatch.setattr(sys, "argv", _argv("--max_analyst_rounds", value))

    with pytest.warns(DeprecationWarning, match="max_analyst_rounds"):
        cfg = train_script.load_config(train_script.parse_args())

    # A retired option has no structured path, so without an explicit skip the
    # legacy CLI mapping files it under ``env.`` and it reaches the trainer.
    assert "max_analyst_rounds" not in cfg


def test_no_warning_when_the_option_is_absent(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", _argv())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_script.load_config(train_script.parse_args())

    assert [w for w in caught if "max_analyst_rounds" in str(w.message)] == []
