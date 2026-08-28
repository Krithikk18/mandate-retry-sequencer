"""Insufficient funds: the only class with a real model behind it.

The account was short. Whether a later date succeeds depends on when the
customer's salary lands, which is not observable and has to be inferred from
attempt history - see agent/payday.py. This handler is a thin adapter that
puts the tier-2 model behind the shared two-function interface.

This is the class the rest of the project is about: 60-70% of NACH returns,
and the only one of the six where WHEN you retry changes the answer.
"""

from __future__ import annotations

from datetime import date

from agent.payday import PaydayModel
from generator import rules
from policy.base import Action, PolicyState

from .base import FailureClassHandler


class InsufficientFundsHandler(FailureClassHandler):
    failure_class = rules.FAILURE_CLASS_INSUFFICIENT_FUNDS
    rationale = "account was short; a later date may clear once salary lands"

    def __init__(self, tier: int = 2, prior: dict[int, float] | None = None) -> None:
        self.tier = tier
        self.prior = prior

    def success_probability(self, state: PolicyState, day: date) -> float:
        model = PaydayModel(tier=self.tier, prior=self.prior)
        model.fit(state.payment_history)
        curve = model.liquidity_curve(day, day, state.customer.subscription_amount)
        return curve.get(day, 0.0)

    def allowed_actions(self, state: PolicyState) -> list[Action]:
        """The full set. This is the class where timing actually matters."""
        return ["RETRY", "NUDGE", "WAIT", "HOLD", "CANCEL"]
