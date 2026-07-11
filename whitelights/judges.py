"""Pick the right online judge for a (lift, mode) selection.

The live WebSocket swaps trackers when the client changes lift or mode. Squat has
both a free-reps (training) and a referee-command (competition) judge; bench and
deadlift currently have only their referee-command judge, so it is used for both
modes until dedicated free-reps judges exist (TODO(ethan)).
"""

from __future__ import annotations

from .bench import BenchTracker
from .deadlift import DeadliftTracker
from .live import CompetitionTracker, OnlineRepTracker

Tracker = OnlineRepTracker | CompetitionTracker | BenchTracker | DeadliftTracker


def tracker_for(lift: str | None, mode: str | None) -> Tracker:
    """Return a fresh tracker for ``lift`` in {squat, bench, deadlift} and
    ``mode`` in {training, competition}. Unknown values fall back to squat /
    training."""
    lift = (lift or "squat").lower()
    mode = (mode or "training").lower()
    if lift == "bench":
        return BenchTracker()
    if lift == "deadlift":
        return DeadliftTracker()
    # squat
    if mode == "competition":
        return CompetitionTracker()
    return OnlineRepTracker()
