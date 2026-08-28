"""Customer withdrew consent at the pre-debit notification.

Under RBI Sec 6(c) the pre-debit notice must carry an AFA-validated opt-out.
When the customer uses it, the mandate is legally revoked - not paused, not
failing, revoked.

Presenting against a revoked mandate is a COMPLIANCE BREACH, not an
inefficiency. That is why this class is stricter than expired_card: there is no
NUDGE either, because there is no longer a relationship in which to send one.
The customer must give fresh consent through a new mandate, which is a sales
problem rather than a collections one.

The simulator raises ComplianceViolation on any RETRY against this class. That
is deliberate. A business rule can be traded off against revenue; this one
cannot, and a harness that let it slide would be reporting a number obtained by
breaking the law.
"""

from __future__ import annotations

from datetime import date

from generator import rules
from policy.base import Action, PolicyState

from .base import NeverRetryHandler


class OptedOutHandler(NeverRetryHandler):
    failure_class = rules.FAILURE_CLASS_OPTED_OUT
    rationale = "consent withdrawn under RBI Sec 6(c); the mandate is revoked"

    def allowed_actions(self, state: PolicyState) -> list[Action]:
        """CANCEL only. Presenting again would be a regulatory breach."""
        return ["CANCEL"]

    def success_probability(self, state: PolicyState, day: date) -> float:
        return 0.0
