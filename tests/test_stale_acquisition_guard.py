"""A slow acquisition cannot overwrite a run the operator started after it.

THE PROBLEM. An Isaac render plus FoundationPose inference is about a minute of
wall clock; a physical capture plus five inference passes is tens of seconds. In
that window the operator can generate a preset, detect with the planar camera or
start any other run. A result that lands afterwards is not WRONG — it is a
correct measurement about a run that no longer exists, and applying it would
replace the objects of the run on screen with the objects of one that was left.

    acquisition A starts   ->   operator starts run B   ->   A finishes late
                                                             A is REFUSED
                                                             B stays current

THE EXISTING IDENTIFIERS, not a second scheme. `run_id` and `scenario_revision`
already stamp every run and every batch, and the approval gate already refuses a
decision taken against a superseded plan. This is the same problem one step
earlier and uses the same tokens.

Needs FastAPI, which this host provides only inside `.venv-perception` and the
container — so the module is skipped rather than failed where it is absent.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("fastapi", reason="web/app.py needs FastAPI")
sys.path.insert(0, os.path.join(REPO, "web"))
sys.path.insert(0, os.path.join(REPO, "wisepack_ws", "src", "wisepack_core"))

sys.argv = ["app.py"]
app = pytest.importorskip("app", reason="the dashboard module could not import")


class _Batch:
    """The minimum a batch needs to be to reach the guard. It never gets past."""

    batch_id = "late-1"

    def to_dict(self):                                           # pragma: no cover
        return {}


@pytest.fixture
def dashboard():
    """A dashboard on the local engine, with one ordinary preset run started."""
    app.STATE.engine = app.start_run(app.STATE.settings, acquire=False)
    yield app
    app.STATE.engine = None


def test_a_token_names_the_run_and_the_revision(dashboard):
    token = dashboard.run_token()
    assert token["run_id"] == dashboard.STATE.engine.run_id
    assert token["scenario_revision"] == dashboard.STATE.engine.scenario_revision


def test_a_fresh_token_is_not_superseded(dashboard):
    assert dashboard.superseded_reason(dashboard.run_token()) == ""


def test_a_result_from_a_replaced_run_is_refused(dashboard):
    """THE REGRESSION, in the order the brief specifies."""
    # 1. acquisition A starts, for the run currently on screen
    token_a = dashboard.run_token()
    items_before = len(dashboard.STATE.engine.scenario.items)

    # 2. the operator starts a NEWER run while A is still computing
    dashboard.STATE.engine = dashboard.start_run(dashboard.STATE.settings,
                                                 acquire=False)
    run_b = dashboard.STATE.engine.run_id
    assert run_b != token_a["run_id"]

    # 3. A finishes late
    outcome = dashboard._apply_physical_batch(_Batch(), token_a)

    # 4. A is refused, and B is untouched
    assert outcome["applied"] is False
    assert outcome.get("superseded") is True
    assert token_a["run_id"] in outcome["reason"] and run_b in outcome["reason"], (
        "the refusal must name both runs; 'stale result' alone tells an "
        "operator nothing about what happened")
    assert dashboard.STATE.engine.run_id == run_b
    assert len(dashboard.STATE.engine.scenario.items) == items_before, (
        "the late result changed the newer run's objects")


def test_a_result_from_a_superseded_revision_is_refused(dashboard):
    """The run can survive while its REVISION moves on — a re-plan, an injected
    item, another batch. That is equally superseding."""
    from wisepack_core.generator import build_scenario

    token = dashboard.run_token()
    engine = dashboard.STATE.engine
    engine.generate_or_load_scenario(build_scenario("mixed_pipes_dense", 7))
    assert engine.scenario_revision != token["scenario_revision"]

    outcome = dashboard._apply_physical_batch(_Batch(), token)
    assert outcome["applied"] is False and outcome.get("superseded") is True
    assert str(token["scenario_revision"]) in outcome["reason"]


def test_no_token_means_no_guard_and_that_is_deliberate(dashboard):
    """A caller that did not capture a token is not silently refused.

    The guard protects callers that OPT IN by capturing a token before their
    slow work. Refusing an untokened call would break every other path that
    applies a batch synchronously, none of which has a window to be stale in.
    """
    assert dashboard.superseded_reason(None) == ""
    assert dashboard.superseded_reason({}) == ""
