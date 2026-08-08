"""Pick the right online judge for a (lift, mode) selection.

The live WebSocket swaps trackers when the client changes lift or mode.

Every lift now has both judges: a free-rep counter (training) and a
referee-command single-attempt judge (competition). The squat's free-rep judge is
`live.OnlineRepTracker`; bench and deadlift share `freereps.FreeRepTracker`,
since every free-rep cycle is the same out-and-back shape.

``supports_training`` remains the single source of truth for which lifts offer
training, and `web/live.html` mirrors it in its ``LIFTS[...].modes`` config —
they must not disagree, or the UI offers a mode the backend will not honour.
"""

from __future__ import annotations

from .bench import BenchTracker
from .deadlift import DeadliftTracker
from .freereps import FreeRepTracker, bench_rep_tracker, deadlift_rep_tracker
from .live import CompetitionTracker, OnlineRepTracker

Tracker = OnlineRepTracker | CompetitionTracker | BenchTracker | DeadliftTracker | FreeRepTracker

LIFTS = ("squat", "bench", "deadlift")
MODES = ("training", "competition")

#: Lifts with a real free-rep (training) judge — now all of them.
TRAINING_CAPABLE = frozenset(LIFTS)


def supports_training(lift: str | None) -> bool:
    """Does ``lift`` have a genuine free-rep judge (as opposed to only a
    competition judge)?"""
    return _normalise_lift(lift) in TRAINING_CAPABLE


def tracker_for(lift: str | None, mode: str | None) -> Tracker:
    """Return a fresh tracker for ``lift`` in {squat, bench, deadlift} and
    ``mode`` in {training, competition}. Unknown values fall back to squat /
    training.
    """
    lift = _normalise_lift(lift)
    mode = (mode or "training").lower()
    if mode not in MODES:
        mode = "training"
    competition = mode == "competition"

    if lift == "bench":
        return BenchTracker() if competition else bench_rep_tracker()
    if lift == "deadlift":
        return DeadliftTracker() if competition else deadlift_rep_tracker()
    return CompetitionTracker() if competition else OnlineRepTracker()


def _normalise_lift(lift: str | None) -> str:
    lift = (lift or "squat").lower()
    return lift if lift in LIFTS else "squat"
