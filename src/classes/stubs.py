"""Classes that are NOT yet implemented, declared so the router has no gaps.

One is left. Downtime used to live here and moved to classes/downtime.py;
afa_incomplete used to live here and moved to classes/afa_incomplete.py. What
remains is ``ambiguous_decline``, and it exists so the taxonomy is complete and
the router has no missing branch - a stub that says "not implemented" is safer
than a KeyError in production and far safer than a plausible-looking guess.

**This produces no results. Do not report numbers for it.** Its ``implemented``
flag is False and the multi-class evaluation excludes it from any claim.
"""

from __future__ import annotations

from datetime import date

from generator import rules
from policy.base import Action, PolicyState

from .base import FailureClassHandler


class AmbiguousHandler(FailureClassHandler):
    """"Do not honour", or a code nobody documented. NOT IMPLEMENTED.

    The issuer declined and said nothing useful. The defensible response is one
    cautious retry and then stop: a second identical decline carries no new
    information and the notice it requires is not free.
    """

    failure_class = rules.FAILURE_CLASS_AMBIGUOUS
    implemented = False
    rationale = "NOT IMPLEMENTED - one cautious retry, then stop"

    #: how many presentments a cautious response permits, including the original
    MAX_CAUTIOUS_ATTEMPTS = 2

    def success_probability(self, state: PolicyState, day: date) -> float:
        return 0.0

    def allowed_actions(self, state: PolicyState) -> list[Action]:
        if state is not None and state.attempts_used >= self.MAX_CAUTIOUS_ATTEMPTS:
            return ["HOLD", "CANCEL"]
        return ["RETRY", "WAIT", "HOLD", "CANCEL"]
