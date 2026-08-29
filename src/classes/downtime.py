"""Bank or PSP downtime: the mirror image of insufficient funds.

    insufficient funds -> recovery takes DAYS.  Wait for payday. Patience wins.
    downtime           -> recovery takes HOURS. Present at the first clear
                          hour. Patience loses.

Same engine, opposite policy - and neither shape is hardcoded. Both fall out of
the same money equation being fed a different P(clear) curve. For insufficient
funds that curve rises days later, when the salary lands. For downtime it is
flat except for a dip while the bank is unavailable, so the equation naturally
prefers the earliest reachable date rather than a distant one.

The success model is a product of two independent conditions, exactly mirroring
the truth in ``generator/oracle.would_succeed_at``:

    P(clear on d) = P(bank up at the presentment hour) x P(balance covers it)

The balance term is NOT reimplemented here. It reuses ``agent/payday.py``'s
PaydayModel, the same way ``classes/insufficient_funds.py`` does - a debit that
fails during an outage tells you nothing about the account, so the customer's
liquidity still has to be modelled from their own history.

The bank term comes from ``agent/bank_health.py``, which pools evidence across
every customer on the issuer. That is the part no per-customer model could
supply.

No NUDGE. There is nothing the customer can do about their bank being down, and
a message telling them so spends goodwill to deliver an apology.
"""

from __future__ import annotations

from datetime import date, datetime

from agent.bank_health import BankHealthModel
from agent.payday import PaydayModel
from generator import rules
from policy.base import Action, PolicyState

from .base import FailureClassHandler


class DowntimeHandler(FailureClassHandler):
    """Prices dates on bank availability as well as customer liquidity."""

    failure_class = rules.FAILURE_CLASS_DOWNTIME
    rationale = "the bank was unavailable; recovery is hours away, not days"

    def __init__(
        self,
        bank_health: BankHealthModel | None = None,
        payday_tier: int = 2,
        prior: dict[int, float] | None = None,
    ) -> None:
        # A shared model, injected: the whole point is that it has seen other
        # customers' failures on this bank. A per-handler private instance
        # would defeat the pooling.
        self.bank_health = bank_health or BankHealthModel()
        self.payday_tier = payday_tier
        self.prior = prior

    def uptime_probability(self, state: PolicyState, day: date) -> float:
        """P(the bank can process a presentment at ``day``'s presentment hour)."""
        at = datetime.combine(day, rules.DEFAULT_PRESENTMENT_TIME)
        return self.bank_health.uptime_probability(state.customer.bank_id, at)

    def success_probability(self, state: PolicyState, day: date) -> float:
        """P(bank up) x P(balance sufficient). Both must hold.

        The conjunction matters: a customer can be broke AND their bank down.
        Waiting out the outage does not put money in the account, so treating a
        clear bank as sufficient would over-price every date.
        """
        model = PaydayModel(tier=self.payday_tier, prior=self.prior)
        model.fit(state.payment_history)
        liquidity = model.liquidity_curve(
            day, day, state.customer.subscription_amount
        ).get(day, 0.0)
        return self.uptime_probability(state, day) * liquidity

    def allowed_actions(self, state: PolicyState) -> list[Action]:
        """No NUDGE: the customer cannot fix their bank."""
        return ["RETRY", "WAIT", "HOLD", "CANCEL"]
