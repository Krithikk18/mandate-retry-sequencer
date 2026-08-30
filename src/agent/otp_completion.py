"""Will this customer complete the authentication step, and can we move that?

**This is the third distinct SHAPE of inference in the project, and the three
together are the architectural point rather than three copies of one idea.**

    agent/payday.py         one customer, in isolation. Another subscriber's
                            salary date tells you nothing about yours, so
                            pooling would be meaningless.
    agent/bank_health.py    pooled ACROSS customers. An outage is one event
                            that fails every mandate on the issuer at once, so
                            a stranger's decline is evidence about you.
    agent/otp_completion.py back to one customer - but with a LEVER the agent
                            controls.

That last difference is the one that matters. A salary date is a fact. A bank
outage is a fact. Both can be estimated and neither can be moved: the agent's
whole job in classes 1 to 4 is to pick a date on a curve it did not draw.

Authentication is not a fact, it is a customer ACTION, and an out-of-band nudge
raises the chance it happens. So this module estimates a curve the agent can
then shift - and the money equation, unchanged, decides whether shifting it is
worth what it costs. That is why class 5 is the first INTERVENTIONAL class and
why it needed simulator support none of the others did.

Agent-safe by the same rule as the rest of ``agent/``: it may not import
``generator.oracle``, ``generator.afa``, ``generator.banks``, or read any
hidden field. It never sees ``otp_responsiveness``. It sees
``(amount, return_code, succeeded)`` per attempt - all three of which a real
PSP has on every presentment it made - and infers the rest.

The discriminating property, and why the class is separable at all:

    1A on a mandate over Rs 15,000  -> the customer did not authenticate
    04 on the same mandate          -> they DID authenticate, and were broke

Both are declines on the same large mandate. Only the first says anything about
authentication, and the second is positive evidence the customer authenticates
- which is what stops the model reading a run of insufficient-funds bounces as
an authentication problem. The same censoring discipline ``payday.py`` learned
from code 91: a decline that never reached the gate you are modelling is not
evidence about that gate.

Caveat worth stating in any write-up, the same one ``churn.py`` carries: the
functional form of the nudge lift here matches the generator's. On real data
the merchant would fit it from its own dunning history and the shape would be
somewhat wrong. Getting it right for free flatters the agent, which is why
``scripts/run_afa.py`` reports the ``no_afa_nudge`` ablation - what the action
is worth at all is the question that survives the caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from generator import rules

# Book-level completion rate on OTP-gated mandates, used for cold start and as
# the mean a thin history is smoothed toward. A merchant reads this off its own
# 1A telemetry; it cannot read any individual's willingness to authenticate.
# ESTIMATE, and the single most important number in this model.
BOOK_COMPLETION_RATE: float = 0.75

# Strength of that prior, in pseudo-observations. Four is deliberately weak:
# a customer with a dozen large debits behind them should be modelled on their
# own record, while one with two should not be. It also keeps the estimate off
# 0.0 and 1.0, which the money equation would price as certainties.
PRIOR_STRENGTH: float = 4.0

# The agent's own estimate of what a nudge is worth. Mirrors the shape in
# generator/afa.py rather than importing it - importing would be reading the
# truth's parameters, which is exactly the leak this package is built to
# prevent. Both are ESTIMATES; that they agree is a convenience the write-up
# has to declare, not a result.
NUDGE_GAP_CLOSED: float = 0.45
MAX_NUDGED_COMPLETION: float = 0.95

# How many nudges the merchant is willing to send in one billing cycle.
#
# NOT a regulation - nothing in RBI or NPCI limits an out-of-band message - so
# it does not belong in rules.py. It is a merchant policy about dunning
# fatigue, and it is needed because the money equation alone would not stop:
# the lift decays geometrically while the goodwill cost per nudge is flat, so
# on the largest mandates "one more message" stays marginally positive for
# eight or nine messages. Nobody sends nine.
#
# It deliberately equals ``generator.afa.MAX_USEFUL_NUDGES``, so the agent and
# the oracle ceiling have the SAME action set. A ceiling allowed more messages
# than the agent would not be an upper bound on the agent's problem.
MAX_NUDGES_PER_CYCLE: int = 2

# Below this the mandate is not worth presenting on at all until something
# changes - the presentment would almost certainly return 1A again, burning an
# NPCI attempt and a cancel-capable notice to learn nothing new.
MIN_WORTH_PRESENTING: float = 0.20


@dataclass(frozen=True)
class AuthOutcome:
    """One resolved presentment, as a PSP sees it.

    Three agent-visible fields and nothing else. ``amount`` is here rather than
    a precomputed "needed OTP" flag so the Sec 8(a) threshold is applied in one
    place, through ``rules.requires_otp`` - a call site that hardcoded 15,000
    would silently survive the rule changing.
    """

    at: date
    amount: float
    succeeded: bool
    return_code: str | None

    @property
    def needed_authentication(self) -> bool:
        return rules.requires_otp(self.amount)

    @property
    def is_afa_coded(self) -> bool:
        return rules.classify(self.return_code) == rules.FAILURE_CLASS_AFA_INCOMPLETE


@dataclass
class OtpCompletionModel:
    """P(this customer authenticates the next debit), from their own record.

    Online by construction, like ``BankHealthModel``: ``observe`` takes
    outcomes one at a time, so the same object serves a batch backfill and a
    live stream.
    """

    book_rate: float = BOOK_COMPLETION_RATE
    prior_strength: float = PRIOR_STRENGTH
    _completed: int = field(default=0, init=False)
    _refused: int = field(default=0, init=False)
    _seen: int = field(default=0, init=False)

    # --- ingestion -----------------------------------------------------------

    @staticmethod
    def _tells_us_about_authentication(outcome: AuthOutcome) -> bool:
        """Is this attempt evidence about whether the customer authenticates?

        Only presentments that actually reached the authentication gate are.
        Three kinds do not, and counting them would be the class-2 bug again:

        * anything below the Sec 8(a) ceiling - there was no gate to pass;
        * a 91 - the debit never reached the issuer, so the gate was never
          consulted and the customer was never asked;
        * a 54 or 22 - the mandate was dead before authentication mattered.

        What IS evidence: a 1A (asked, did not authenticate) and any 04 or
        success on a large mandate (asked, did authenticate). That second case
        is the important one - without it a customer who authenticates reliably
        but keeps running out of money would look like an authentication
        problem, and the agent would nudge somebody whose issue is their
        balance.
        """
        if not outcome.needed_authentication:
            return False
        if outcome.succeeded:
            return True
        return rules.classify(outcome.return_code) in (
            rules.FAILURE_CLASS_AFA_INCOMPLETE,
            rules.FAILURE_CLASS_INSUFFICIENT_FUNDS,
        )

    def observe(self, outcome: AuthOutcome) -> None:
        """Record one resolved presentment."""
        self._seen += 1
        if not self._tells_us_about_authentication(outcome):
            return
        if outcome.is_afa_coded:
            self._refused += 1
        else:
            self._completed += 1

    def observe_many(self, outcomes: Iterable[AuthOutcome]) -> None:
        for outcome in outcomes:
            self.observe(outcome)

    @classmethod
    def from_history(cls, records: Iterable[Any]) -> "OtpCompletionModel":
        """Fit from agent-visible payment history.

        ``records`` are ``AgentPaymentHistoryRecord``. Nothing else is needed:
        the amount decides whether the gate applied and the return code says
        whether it opened.
        """
        model = cls()
        model.observe_many(
            AuthOutcome(
                at=record.attempt_date,
                amount=record.amount,
                succeeded=record.succeeded,
                return_code=record.return_code,
            )
            for record in records
        )
        return model

    # --- inference -----------------------------------------------------------

    @property
    def evidence_count(self) -> int:
        """Presentments that actually reached the authentication gate."""
        return self._completed + self._refused

    @property
    def cold_start(self) -> bool:
        """True when this customer has never faced an authentication step.

        Common and not a defect: a mandate crosses the Sec 8(a) ceiling only if
        it is large, and a customer who has just been upgraded onto one has no
        record at all. The prior answers rather than a crash or a 0/1.
        """
        return self.evidence_count == 0

    def responsiveness(self) -> float:
        """Posterior mean of P(authenticates when asked), unprompted.

        A Beta posterior over the customer's own record, smoothed toward the
        book rate. Never 0 and never 1: no finite run of refusals proves a
        customer will never authenticate, and the money equation would price
        either endpoint as a certainty it has not earned.
        """
        alpha = self._completed + self.prior_strength * self.book_rate
        beta = self._refused + self.prior_strength * (1.0 - self.book_rate)
        return float(alpha / (alpha + beta))

    def completion_probability(self, nudges: int = 0) -> float:
        """P(authentication is complete for a presentment, after ``nudges``.

        Same shape as the generator's hazard: responsiveness is both the
        unprompted rate AND the rate at which asking works, so a nudge cannot
        rescue a customer who ignores authentication prompts - they ignore the
        reminder too. Diminishing by construction, and converging below
        certainty, so the agent can never buy its way to a sure thing.
        """
        if nudges < 0:
            raise ValueError("nudges must be >= 0")
        base = self.responsiveness()
        gap = max(0.0, MAX_NUDGED_COMPLETION - base)
        if gap == 0.0 or nudges == 0:
            return base
        per_nudge = NUDGE_GAP_CLOSED * base
        return float(base + gap * (1.0 - (1.0 - per_nudge) ** nudges))

    def nudge_lift(self, nudges_already_sent: int = 0) -> float:
        """How much the NEXT nudge would add to the completion probability.

        The quantity the money equation actually needs: not what nudging is
        worth in total, but what one more message buys on top of what has
        already been spent.
        """
        return self.completion_probability(
            nudges_already_sent + 1
        ) - self.completion_probability(nudges_already_sent)

    def worth_presenting(self, nudges: int = 0) -> bool:
        """Is the gate open enough that a presentment is worth its attempt?"""
        return self.completion_probability(nudges) >= MIN_WORTH_PRESENTING

    # --- audit ---------------------------------------------------------------

    def explain(self, nudges: int = 0) -> dict:
        """What the model believes, and on what evidence."""
        return {
            "presentments_observed": self._seen,
            "authentication_evidence": self.evidence_count,
            "authenticated": self._completed,
            "refused_1a": self._refused,
            "cold_start": self.cold_start,
            "book_rate": round(self.book_rate, 4),
            "responsiveness": round(self.responsiveness(), 4),
            "nudges_sent": nudges,
            "completion_probability": round(self.completion_probability(nudges), 4),
            "next_nudge_lift": round(self.nudge_lift(nudges), 4),
        }
