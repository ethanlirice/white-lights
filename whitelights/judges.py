"""Pick the right online judge for a (lift, mode) selection.

The live WebSocket swaps trackers when the client changes lift or mode.

Squat has both judges: a free-rep counter (training) and a referee-command
single-attempt judge (competition). Bench and deadlift currently have only the
referee-command judge, so ``TRAINING_CAPABLE`` lists squat alone and the UI hides
the training toggle for the other two — asking for training on bench or deadlift
returns the competition judge, which is a *different* interaction (it waits for a
setup hold, issues commands and ends after one attempt), not a degraded one.

``supports_training`` is the single source of truth for that; `web/live.html`
mirrors it in its ``LIFTS[...].modes`` config.
"""

from __future__ import annotations

from .bench import BenchTracker
from .deadlift import DeadliftTracker
from .live import CompetitionTracker, OnlineRepTracker

Tracker = OnlineRepTracker | CompetitionTracker | BenchTracker | DeadliftTracker

LIFTS = ("squat", "bench", "deadlift")
MODES = ("training", "competition")

#: Lifts with a real free-rep (training) judge. TODO(ethan): bench + deadlift
#: free-rep counters — cheap once the trackers share one engine.
TRAINING_CAPABLE = frozenset({"squat"})


def supports_training(lift: str | None) -> bool:
    """Does ``lift`` have a genuine free-rep judge (as opposed to only a
    competition judge)?"""
    return _normalise_lift(lift) in TRAINING_CAPABLE


def tracker_for(lift: str | None, mode: str | None) -> Tracker:
    """Return a fresh tracker for ``lift`` in {squat, bench, deadlift} and
    ``mode`` in {training, competition}. Unknown values fall back to squat /
    training.

    Bench and deadlift return their competition judge for either mode — see the
    module docstring; ``supports_training`` reports which lifts that applies to.
    """
    lift = _normalise_lift(lift)
    mode = (mode or "training").lower()
    if mode not in MODES:
        mode = "training"

    if lift == "bench":
        return BenchTracker()
    if lift == "deadlift":
        return DeadliftTracker()
    return CompetitionTracker() if mode == "competition" else OnlineRepTracker()


def _normalise_lift(lift: str | None) -> str:
    lift = (lift or "squat").lower()
    return lift if lift in LIFTS else "squat"
