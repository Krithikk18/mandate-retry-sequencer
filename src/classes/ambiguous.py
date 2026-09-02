"""Ambiguous decline: a posterior over the other five classes.

**Why this class is not like the five before it.** Classes 1, 2 and 5 hide a
PARAMETER behind a return code that at least says which parameter to estimate.
A 04 says "find the payday". A 91 says "find the outage". A 1A says "find out
whether they will authenticate". The agent knows what question it is answering
and goes and answers it.

A ``05`` says nothing. It hides the QUESTION.

So this handler is not a sixth success model sitting alongside the others. It
is a MIXTURE over them:

    P(clear on d) = SUM over causes c of  P(true class = c | evidence)
                                          x  handler_c.success_probability(d)

The sixth class is the other five, weighted by a belief about which one is
really in play. **The taxonomy closes on itself** - and it does so literally in
the code below, which imports the four implemented component handlers and calls
their ``success_probability`` rather than reimplementing any of them, exactly as
``downtime.py`` and ``afa_incomplete.py`` both reuse ``PaydayModel`` instead of
rebuilding a liquidity curve.

Two causes have no component handler because they exist only behind a 05:

    permanent_block   an issuer risk block. NO date ever clears, and unlike an
                      expired card nobody will ever send a code that says so.
                      Contributes exactly 0.0 to the mixture - the same
                      structural zero as ``NeverRetryHandler``, arrived at
                      through the posterior rather than through the taxonomy.
    transient         a glitch measured in hours. Contributes a term that RISES
                      with days waited, which is the only reason a cautious
                      retry is worth anything on this class at all.

**The weights are fitted, never assumed.** ``agent/masking.py`` estimates them
from the resolution history of past 05s across the whole book. Nothing here or
there reads the generator's mixture constant; a test perturbs the generator and
asserts the fitted posterior moves with it. Without that discipline every
class-6 result would be synthetic-data self-fulfilment.

**No NUDGE.** You cannot ask a customer to fix a decline nobody explained. A
nudge on this class would be a message saying "something went wrong, please do
something" - it spends goodwill to deliver confusion. RETRY, WAIT, HOLD and
CANCEL only.

**And no hardcoded attempt cap.** The stub this replaces allowed "one cautious
retry, then stop" as a rule. It is not a rule here: as unexplained declines
accumulate on a mandate, the fitted posterior shifts mass onto
``permanent_block``, the mixture probability falls, and the money equation
stops presenting on its own. The cautious shape is DERIVED, the same way
class 2's "present immediately" and class 1's "wait for payday" both fall out
of the same equation being handed different curves.
"""

from __future__ import annotations

from datetime import date

from agent.masking import (
    CAUSE_PERMANENT_BLOCK,
    CAUSE_TRANSIENT,
    MaskingModel,
)
from agent.payday import PaydayModel
from generator import rules
from policy.base import Action, PolicyState

from .afa_incomplete import AfaIncompleteHandler
from .base import FailureClassHandler
from .downtime import DowntimeHandler
from .expired_card import ExpiredCardHandler
from .insufficient_funds import InsufficientFundsHandler

# How fast a transient block is assumed to clear, as a per-day hazard. A glitch
# is hours-to-days, so most of the mass is gone within three days. ESTIMATE -
# and note it is an estimate about the AGENT's belief, not a readback of the
# generator's glitch duration, which this module never sees.
TRANSIENT_CLEAR_PER_DAY: float = 0.55


class AmbiguousHandler(FailureClassHandler):
    """A fitted posterior over the other five classes, and their mixture."""

    failure_class = rules.FAILURE_CLASS_AMBIGUOUS
    implemented = True
    rationale = (
        "the issuer declined without saying why; price every cause it could "
        "have been, weighted by what past unexplained declines turned out to be"
    )

    def __init__(
        self,
        masking: MaskingModel | None = None,
        payday_tier: int = 2,
        prior: dict[int, float] | None = None,
    ) -> None:
        # Injected for the same reason DowntimeHandler injects BankHealthModel:
        # the whole value is that it has seen other customers' 05s on this
        # issuer. A private instance per handler would defeat the pooling.
        self.masking = masking or MaskingModel()
        self.payday_tier = payday_tier
        self.prior = prior
        # The component handlers, composed rather than reimplemented.
        self._components: dict[str, FailureClassHandler] = {
            # NB: this one names the parameter `tier`, not `payday_tier`.
            rules.FAILURE_CLASS_INSUFFICIENT_FUNDS: InsufficientFundsHandler(
                tier=payday_tier, prior=prior
            ),
            rules.FAILURE_CLASS_EXPIRED_CARD: ExpiredCardHandler(),
            rules.FAILURE_CLASS_DOWNTIME: DowntimeHandler(
                payday_tier=payday_tier, prior=prior
            ),
            rules.FAILURE_CLASS_AFA_INCOMPLETE: AfaIncompleteHandler(
                payday_tier=payday_tier, prior=prior
            ),
        }

    # --- belief --------------------------------------------------------------

    def posterior(self, state: PolicyState) -> dict[str, float]:
        """P(true cause | this issuer's history of unexplained declines)."""
        return self.masking.posterior(state.customer.bank_id)

    def _liquidity(self, state: PolicyState, day: date) -> float:
        """The balance term, from the customer's own history."""
        model = PaydayModel(tier=self.payday_tier, prior=self.prior)
        model.fit(state.payment_history)
        return model.liquidity_curve(
            day, day, state.customer.subscription_amount
        ).get(day, 0.0)

    def _transient_probability(self, state: PolicyState, day: date) -> float:
        """P(a transient block has cleared by ``day``) x P(balance covers it).

        Rises with the wait, which is what makes one cautious retry rational
        and a same-day retry not. Still multiplied by liquidity: a glitch
        clearing does not put money in the account, and treating it as
        sufficient would over-price every date for a customer who is also
        broke. Same conjunction discipline as downtime.py and afa_incomplete.py.
        """
        waited = max(0, (day - state.failure_event.original_debit_date).days)
        cleared = 1.0 - (1.0 - TRANSIENT_CLEAR_PER_DAY) ** waited
        return cleared * self._liquidity(state, day)

    # --- the mixture ---------------------------------------------------------

    def success_probability(self, state: PolicyState, day: date) -> float:
        """The weighted sum over every cause the decline could have been.

        A mixture, not a maximum and not a guess at the single most likely
        cause. Committing to one cause and being wrong is confidently wrong;
        spreading over the posterior is merely uncertain, and uncertainty is
        what the money equation can price - the same argument ``payday.py``
        makes for marginalising over the payday posterior instead of taking its
        mode.
        """
        posterior = self.posterior(state)
        total = 0.0
        for cause, weight in posterior.items():
            if weight <= 0.0:
                continue
            total += weight * self._cause_probability(cause, state, day)
        return min(1.0, max(0.0, total))

    def _cause_probability(
        self, cause: str, state: PolicyState, day: date
    ) -> float:
        """P(clear on ``day``) supposing the true cause were ``cause``."""
        if cause == CAUSE_PERMANENT_BLOCK:
            # Structurally dead, and never announced as dead. The same zero
            # NeverRetryHandler returns, reached through evidence instead of
            # through the return code.
            return 0.0
        if cause == CAUSE_TRANSIENT:
            return self._transient_probability(state, day)
        component = self._components.get(cause)
        if component is None:
            return 0.0
        return component.success_probability(state, day)

    # --- actions -------------------------------------------------------------

    def allowed_actions(self, state: PolicyState) -> list[Action]:
        """No NUDGE. There is nothing to ask the customer to do.

        RETRY stays available even though a quarter of the mass sits on a
        permanent block, because the other three quarters are recoverable and
        the money equation is the right place to weigh that - not a rule here.
        """
        return ["RETRY", "WAIT", "HOLD", "CANCEL"]
