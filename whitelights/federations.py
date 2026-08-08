"""Federation strictness profiles.

The rulebooks agree on *what* the faults are and differ on how tightly they are
enforced, so strictness is a per-federation profile applied to a lift's config
rather than a separate judge.

This lives in its own module because more than one lift needs it: `deadlift` used
to import ``Federation`` from `bench`, which made the two lifts depend on each
other for no reason.
"""

from __future__ import annotations

from enum import StrEnum


class Federation(StrEnum):
    IPF = "IPF"  # strict
    USAPL = "USAPL"  # more lenient
