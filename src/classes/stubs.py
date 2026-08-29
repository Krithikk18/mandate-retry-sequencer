"""Classes that are NOT yet implemented, declared so the router has no gaps.

Each of these is a real failure class with a real response, and none of them is
modelled here. Downtime USED to live here and now has a real model in
classes/downtime.py - these are what is left. They exist so the taxonomy is complete and the router has no
missing branch - a stub that says "not implemented" is safer than a KeyError in
production and far safer than a plausible-looking guess.

**None of these produce results. Do not report numbers for them.** Their
``implemented`` flag is False and the multi-class evaluation excludes them from
any claim.
"""

from __future__ import annotations

from datetime import date

from generator import rules
from policy.base import Action, PolicyState

from .base import FailureClassHandler


class AfaIncompleteHandler(FailureClassHandler):
    """Above Rs 15,000 and the customer did not complete OTP. NOT IMPLEMENTED.

    Retrying is pointless until the OTP is done, so the action is a NUDGE to
    complete authentication. Modelling it needs an OTP-completion hazard the
    generator does not produce.
    """

    failure_class = rules.FAILURE_CLASS_AFA_INCOMPLETE
    implemented = False
    rationale = "NOT IMPLEMENTED - needs an OTP completion model"

    def success_probability(self, state: PolicyState, day: date) -> float:
        return 0.0

    def allowed_actions(self, state: PolicyState) -> list[Action]:
        return ["NUDGE", "WAIT", "HOLD", "CANCEL"]


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
