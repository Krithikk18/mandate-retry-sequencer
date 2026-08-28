"""Expired or reissued card: the stored credential is dead.

The card behind the mandate has expired or been reissued with a new number.
Every presentment sends the same dead credential and gets the same answer, on
every date, forever. No amount of timing helps, because nothing about the
customer's balance is the problem.

Recovery requires the CUSTOMER to update the card. That is a NUDGE, not a
retry - and the mandate is worth keeping alive while they do it, so HOLD stays
on the table alongside CANCEL.

A fixed schedule presents four times and recovers nothing. That is pure waste:
four presentments charged, and under RBI Sec 6(a) four pre-debit notices sent
to somebody whose only possible action is one the notices do not ask for.
"""

from __future__ import annotations

from datetime import date

from generator import rules
from policy.base import Action, PolicyState

from .base import NeverRetryHandler


class ExpiredCardHandler(NeverRetryHandler):
    failure_class = rules.FAILURE_CLASS_EXPIRED_CARD
    rationale = "stored credential is dead; only the customer can replace it"

    def allowed_actions(self, state: PolicyState) -> list[Action]:
        """NUDGE the customer to update the card; hold or cancel meanwhile.

        No RETRY: the same dead credential would be presented again.
        """
        return ["NUDGE", "HOLD", "CANCEL"]

    def success_probability(self, state: PolicyState, day: date) -> float:
        return 0.0
