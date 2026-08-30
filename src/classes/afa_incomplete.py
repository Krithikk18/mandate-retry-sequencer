"""AFA incomplete: the first class where the agent can change the answer.

RBI Sec 8(a) exempts recurring debits from Additional Factor of Authentication
only up to Rs 15,000 per transaction. Above that ceiling every debit needs its
own OTP, so a presentment is refused with code 1A unless the customer completed
authentication for that debit.

**Why this class is not like the four before it.** In insufficient funds,
downtime, expired cards and revoked consent the agent's actions are purely
SELECTIONAL: it picks a date, or declines to pick one. The success probability
is a fact about the world - a balance, an outage, a dead credential - that the
agent can estimate but never move. Its whole skill is choosing well on a curve
somebody else drew.

Here the curve is movable. A NUDGE is an out-of-band message asking the
customer to authorise the mandate, and it causally raises the chance the gate
is open at the next presentment. The agent is no longer only scheduling; it can
intervene. That is a real architectural distinction, and it is why this class
needed simulator support (``evaluation/simulator.py`` had treated NUDGE as a
no-op) that no earlier class did.

The contrast with ``expired_card.py`` is the sharp one, because the two look
similar and are not. An expired credential is DEAD: no date succeeds, and the
only fix is the customer replacing it, which the merchant cannot cause. An
AFA-gated mandate is ALIVE and merely GATED, the gate reopens on every
presentment, and asking is a lever the merchant actually holds.

The success model mirrors the truth in ``generator.oracle.would_succeed_at``:

    P(clear on d) = P(authentication complete) x P(balance covers it)

Both must hold, and neither implies the other. A customer can authenticate
faithfully and still be broke; a wealthy one can ignore every OTP prompt. The
balance term is NOT reimplemented here - it reuses ``agent/payday.py``, exactly
as ``downtime.py`` does, because a debit refused at the authentication gate
never reached the account and says nothing about the money in it.

Reported by VALUE first, not by count. These are the only mandates on the book
above Rs 15,000, so the class is thin by episode count and heavy by rupees. A
recovery-rate table alone would make it look ignorable.
"""

from __future__ import annotations

from datetime import date

from agent.otp_completion import MAX_NUDGES_PER_CYCLE, OtpCompletionModel
from agent.payday import PaydayModel
from generator import rules
from policy.base import Action, PolicyState

from .base import FailureClassHandler


class AfaIncompleteHandler(FailureClassHandler):
    """Prices dates on authentication as well as liquidity, and can nudge."""

    failure_class = rules.FAILURE_CLASS_AFA_INCOMPLETE
    rationale = (
        "the debit was refused before it reached the account; the customer "
        "must authenticate, and asking them makes that more likely"
    )

    #: merchant policy on dunning, not a regulation - see MAX_NUDGES_PER_CYCLE
    max_nudges = MAX_NUDGES_PER_CYCLE

    def __init__(
        self, payday_tier: int = 2, prior: dict[int, float] | None = None
    ) -> None:
        self.payday_tier = payday_tier
        self.prior = prior

    def authentication_model(self, state: PolicyState) -> OtpCompletionModel:
        """Fit the completion model to this customer's own record.

        Per-customer, unlike ``BankHealthModel``: whether you authenticate says
        nothing about whether anyone else does, so there is nothing to pool.
        """
        return OtpCompletionModel.from_history(state.payment_history)

    def completion_probability(self, state: PolicyState) -> float:
        """P(the gate is open at the next presentment), given nudges so far."""
        return self.authentication_model(state).completion_probability(
            state.nudges_sent_this_cycle
        )

    def success_probability(self, state: PolicyState, day: date) -> float:
        """P(authenticated) x P(balance sufficient). Both must hold.

        The conjunction matters in both directions. Treating a completed OTP as
        sufficient would over-price every date for a customer who is also
        broke; treating a funded account as sufficient would over-price every
        date for one who never authenticates. The generator refuses on
        whichever gate closes first, and this mirrors it.
        """
        model = PaydayModel(tier=self.payday_tier, prior=self.prior)
        model.fit(state.payment_history)
        liquidity = model.liquidity_curve(
            day, day, state.customer.subscription_amount
        ).get(day, 0.0)
        return self.completion_probability(state) * liquidity

    def allowed_actions(self, state: PolicyState) -> list[Action]:
        """The full set, and NUDGE is the one that matters.

        RETRY stays available - unlike the never-retry classes, a later
        presentment genuinely can clear, because the customer may authenticate
        in the meantime. But presenting into a shut gate burns an NPCI attempt
        and fires a cancel-capable pre-debit notice to learn nothing, so the
        money equation should usually ask first and present second. That
        ordering is left to the arithmetic rather than hardcoded here.

        NUDGE is withdrawn once the cycle's dunning budget is spent, so the
        action set itself stops the agent talking its way to certainty even if
        some future scoring bug said it should.
        """
        if state is not None and state.nudges_sent_this_cycle >= self.max_nudges:
            return ["RETRY", "WAIT", "HOLD", "CANCEL"]
        return ["NUDGE", "RETRY", "WAIT", "HOLD", "CANCEL"]
