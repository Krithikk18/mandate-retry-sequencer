"""The sequencer: decides WHEN and WHETHER to retry.

Three layers, in order:

1. ``PaydayModel`` (tier 2) turns attempt history into a liquidity curve -
   P(clear) for every date in the remaining cycle.
2. ``DecisionEngine`` prices each of those dates in rupees, netting expected
   recovery against processing, bounce charges and churn risk.
3. This module picks an action from the five available, and writes down why.

The decision order:

    shock?          strong evidence the income has stopped -> stop presenting,
                    and compare HOLD against CANCEL explicitly rather than
                    assuming one dominates.
    score           price every legal date.
    best > 0        RETRY on it (deferring the commitment to the last
                    compliant moment, so the option to change is kept).
    best <= 0       is a later cycle worth reaching? HOLD if so, CANCEL if the
                    mandate is worth less than the cost of carrying it.

Every branch populates ``reasoning``. Razorpay's brief asks for an audit trail,
and it is also the only way to tell a good decision from a lucky one: the log
records what the agent believed, not merely what it did.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from generator import rules
from policy.base import Policy, PolicyState, RetryDecision
from router import route

from .bank_health import BASE_OUTAGE_RATE, BankHealthModel, PresentmentOutcome
from .churn import DEFAULT_IRRITATION_DIRECTION, IrritationDirection
from .decision_engine import DecisionEngine, ScoredDate
from .otp_completion import MAX_NUDGES_PER_CYCLE, OtpCompletionModel
from .payday import PaydayModel

# Above this posterior probability of a stopped income, presenting again is
# throwing an attempt and a cancel-capable notice at somebody who has no money.
# ESTIMATE - swept in the ablations.
SHOCK_STOP_THRESHOLD: float = 0.60

# How many candidates to record in the audit trail.
AUDIT_TOP_N: int = 5

# Days of the next cycle to look at when deciding whether holding is worth it.
NEXT_CYCLE_LOOKAHEAD_DAYS: int = 31


@dataclass
class SequencerConfig:
    """Every ablation switch in one place."""

    payday_tier: int = 2
    payday_prior: dict[int, float] | None = None
    flat_liquidity: bool = False  # no_payday: replace the model with a base rate
    include_optout_cost: bool = True  # no_optout_cost
    use_hold: bool = True  # no_hold
    # OFF BY DEFAULT, on the evidence. The ablation measured NUDGE costing 3.1
    # points of recovery on INSUFFICIENT FUNDS: nudging spends a day without
    # presenting, and the modelled lift does not pay for the delay. Kept behind
    # a switch so the finding stays reproducible rather than deleted.
    #
    # This flag is now explicitly the CLASS-1 switch. Class 5 has its own,
    # below, because the two questions are different ones - see there.
    use_nudge: bool = False  # with_nudge

    # no_afa_nudge: the AFA class may only choose DATES, not ask for
    # authentication. This is the ablation that isolates the interventional
    # action, which is the novel claim of class 5.
    #
    # ON by default, and that is not a quiet reversal of the class-1 finding.
    # The two are different actions on different curves. Nudging an
    # insufficient-funds customer asks somebody with no money to find some, and
    # it measurably loses: it burns a day and the modelled lift does not cover
    # it. Nudging an AFA customer asks them to press a button, and presenting
    # before they do is GUARANTEED to fail - the gate is shut, so the
    # presentment cannot clear whatever the balance is. Same action, opposite
    # arithmetic, and the money equation reaches both conclusions unaided.
    #
    # Making it class-dependent rather than flipping ``use_nudge`` globally is
    # what keeps the class-1 result intact and still reported.
    use_afa_nudge: bool = True  # no_afa_nudge
    # no_bank_health: strip the cross-customer signal. The agent still knows a
    # failure was downtime-coded, but has to price every date at the book-wide
    # base rate instead of pooling evidence from other customers on the same
    # bank. This isolates what the shared signal is worth.
    use_bank_health: bool = True
    # Time preference on recovered money. 0.0 reproduces the equation as
    # originally specified; see DecisionEngine.daily_discount.
    daily_discount: float = 0.0
    shock_threshold: float = SHOCK_STOP_THRESHOLD
    lead_time_hours: int = rules.PRE_DEBIT_NOTICE_HOURS
    irritation_direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION
    defer_commitment: bool = True  # WAIT rather than committing early


class SequencerPolicy(Policy):
    """Payday-aware, cost-aware retry sequencing."""

    def __init__(
        self,
        config: SequencerConfig | None = None,
        name: str = "sequencer",
        bank_health: BankHealthModel | None = None,
    ) -> None:
        self.config = config or SequencerConfig()
        self.name = name
        # Shared across every customer this policy decides for. That is the
        # point: a downtime-coded failure on one mandate is evidence about
        # every other mandate on the same bank. Injected when the caller has
        # already primed it from the whole book; otherwise it accumulates as
        # customers are seen, which is how a live system would learn.
        self.bank_health = bank_health or BankHealthModel()
        self._ingested: set[str] = set()
        self.engine = DecisionEngine(
            lead_time_hours=self.config.lead_time_hours,
            irritation_direction=self.config.irritation_direction,
            include_optout_cost=self.config.include_optout_cost,
            daily_discount=self.config.daily_discount,
        )

    # --- liquidity -----------------------------------------------------------

    def _ingest_bank_evidence(self, state: PolicyState) -> None:
        """Fold this customer's declines into the shared bank-health model.

        Deduplicated by customer id, so replaying a customer's history on every
        day of their episode does not count the same failure twenty times.
        """
        customer_id = state.customer.customer_id
        if customer_id in self._ingested:
            return
        self._ingested.add(customer_id)
        for record in state.payment_history:
            if record.succeeded:
                continue
            self.bank_health.observe(
                PresentmentOutcome(
                    bank_id=state.customer.bank_id,
                    at=datetime.combine(
                        record.attempt_date, rules.DEFAULT_PRESENTMENT_TIME
                    ),
                    return_code=record.return_code,
                )
            )

    def _uptime_curve(
        self, state: PolicyState, start: date, end: date
    ) -> dict[date, float]:
        """P(the bank can process a presentment), per candidate date."""
        curve: dict[date, float] = {}
        day = start
        while day <= end:
            if self.config.use_bank_health:
                at = datetime.combine(day, rules.DEFAULT_PRESENTMENT_TIME)
                curve[day] = self.bank_health.uptime_probability(
                    state.customer.bank_id, at
                )
            else:
                # Ablation: no pooling. Every date gets the book-wide base rate,
                # so the agent knows outages exist but never which hours.
                curve[day] = 1.0 - BASE_OUTAGE_RATE
            day += timedelta(days=1)
        return curve

    def _fit_model(self, state: PolicyState) -> PaydayModel:
        model = PaydayModel(tier=self.config.payday_tier, prior=self.config.payday_prior)
        model.fit(state.payment_history)
        return model

    def _liquidity(
        self, model: PaydayModel, state: PolicyState, start: date, end: date
    ) -> dict[date, float]:
        if start > end:
            # No window left - the notice period already runs past the end of
            # the cycle. The caller sees an empty curve and no candidates.
            return {}

        amount = state.customer.subscription_amount
        if not self.config.flat_liquidity:
            return model.liquidity_curve(start, end, amount)

        # no_payday ablation: every date gets the same probability, the
        # customer's own historical success rate. This is what a cost model
        # without a timing model looks like - it can still decide WHETHER to
        # retry, but it has nothing to say about WHEN.
        history = state.payment_history
        rate = (sum(1 for r in history if r.succeeded) + 1) / (len(history) + 2)
        rate *= 1.0 - model.shock_probability()
        day, curve = start, {}
        while day <= end:
            curve[day] = rate
            day += timedelta(days=1)
        return curve

    def _apply_bank_term(
        self, state: PolicyState, curve: dict[date, float], start: date, end: date
    ) -> dict[date, float]:
        """Multiply in P(bank up) for a downtime failure.

        Deliberately a multiplication into the SAME curve the money equation
        already consumes, rather than a new branch in the decision engine. A
        bank that is probably down simply makes those dates worth less, and
        WAIT wins over RETRY for the ordinary reason - the arithmetic - not
        because anything says "wait during an outage".
        """
        uptime = self._uptime_curve(state, start, end)
        return {day: value * uptime.get(day, 1.0) for day, value in curve.items()}

    def _apply_afa_term(
        self, curve: dict[date, float], completion: float
    ) -> dict[date, float]:
        """Multiply in P(authentication complete) for an AFA failure.

        The same device as ``_apply_bank_term`` and for the same reason: one
        multiplication into the curve the money equation already consumes,
        never a new branch in the decision engine. A shut gate simply makes
        every date worth less, and NUDGE beats RETRY when it does - because of
        the arithmetic, not because anything says "authenticate first".

        Flat across dates, unlike the bank term. An outage is an event at a
        particular hour, so P(bank up) varies by date. Authentication is a
        state of the customer, so until they act it is the same on every date -
        which is exactly why waiting cannot fix this class and asking can. The
        policy shape falls out of that difference in the curve's SHAPE rather
        than out of a rule.
        """
        return {day: value * completion for day, value in curve.items()}

    # --- stopping ------------------------------------------------------------

    def _hold_versus_cancel(
        self, state: PolicyState, per_cycle_clear: float
    ) -> tuple[float, float]:
        """Value of carrying the mandate versus writing it off, in rupees.

        Carrying is not free: every cycle costs the carrying charge plus a
        fresh presentment that may bounce, and the customer may drift away
        anyway. Cancelling forfeits the whole annual LTV immediately. The agent
        computes both from ``rules`` and picks - it does not assume either
        dominates, because which one wins depends on parameters the merchant
        sets rather than on anything about payments.
        """
        customer = state.customer
        amount = customer.subscription_amount
        carrying = rules.hold_carrying_cost(amount)
        processing = rules.processing_cost(customer.rail)
        bounce = rules.bounce_charge(customer.rail)
        attrition = rules.NATURAL_ATTRITION_PER_CYCLE

        # Walk the horizon forward. Each cycle the mandate either recovers (and
        # the episode ends well), churns (and the LTV is lost anyway, just
        # later and after paying to get there), or is carried again.
        hold_value = 0.0
        still_carrying = 1.0
        for _ in range(rules.MAX_EPISODE_CYCLES):
            cycle_value = (
                per_cycle_clear * amount
                - processing
                - (1.0 - per_cycle_clear) * bounce
                - carrying
            )
            hold_value += still_carrying * cycle_value
            not_recovered = still_carrying * (1.0 - per_cycle_clear)
            hold_value -= not_recovered * attrition * customer.annual_ltv
            still_carrying = not_recovered * (1.0 - attrition)

        return hold_value, -customer.annual_ltv

    # --- the decision --------------------------------------------------------

    def decide(self, state: PolicyState) -> RetryDecision:
        self._ingest_bank_evidence(state)
        handler = route(state.failure_event)
        if not handler.permits_retry(state):
            # Two of the six classes are structurally dead: an expired
            # credential and a revoked mandate cannot clear on ANY date. The
            # router says so from the return code alone, before any modelling,
            # and no presentment is scheduled. A fixed schedule retries them
            # four times.
            return self._class_stop(state, handler)

        model = self._fit_model(state)
        shock = model.shock_probability()
        posterior = model.payday_posterior()
        mode = max(posterior, key=posterior.get) if posterior else None
        confidence = posterior.get(mode, 0.0) if mode is not None else 0.0

        base = {
            "policy": self.name,
            "shock_probability": round(shock, 4),
            "payday_mode": mode,
            "payday_confidence": round(confidence, 4),
            "attempts_remaining": state.attempts_remaining,
            "notifications_sent": state.notifications_sent_this_cycle,
            "days_since_failure": state.days_since_failure,
            "payday_tier": self.config.payday_tier,
        }

        is_afa = (
            state.failure_event.failure_class == rules.FAILURE_CLASS_AFA_INCOMPLETE
        )
        otp: OtpCompletionModel | None = None
        completion = 1.0
        if is_afa:
            otp = OtpCompletionModel.from_history(state.payment_history)
            completion = otp.completion_probability(state.nudges_sent_this_cycle)
            base["afa"] = otp.explain(state.nudges_sent_this_cycle)
            base["nudges_sent"] = state.nudges_sent_this_cycle

        if state.failure_event.failure_class == rules.FAILURE_CLASS_DOWNTIME:
            at = datetime.combine(
                self.engine.earliest_presentment_date(state.today),
                rules.DEFAULT_PRESENTMENT_TIME,
            )
            base["bank_id"] = state.customer.bank_id
            base["bank_outage_probability"] = round(
                self.bank_health.outage_probability(state.customer.bank_id, at)
                if self.config.use_bank_health
                else BASE_OUTAGE_RATE,
                4,
            )
            base["expected_recovery_hours"] = round(
                self.bank_health.expected_recovery_hours(state.customer.bank_id), 2
            )

        earliest = self.engine.earliest_presentment_date(state.today)
        curve = self._liquidity(model, state, earliest, state.cycle_end_date)
        if state.failure_event.failure_class == rules.FAILURE_CLASS_DOWNTIME:
            curve = self._apply_bank_term(state, curve, earliest, state.cycle_end_date)
        if is_afa:
            curve = self._apply_afa_term(curve, completion)
        scored = self.engine.score(state, curve)
        base["candidates"] = [s.as_dict() for s in scored[:AUDIT_TOP_N]]
        base["binding_constraints"] = self.engine.binding_constraints()

        # 0. Authentication is the binding gate, and asking is cheaper than
        # presenting into a shut one. Checked BEFORE the shock rule and before
        # scoring, because on this class no date is worth anything until the
        # gate opens - the presentment does not fail on the balance, it is
        # refused before the balance is read.
        if is_afa and self.config.use_afa_nudge and otp is not None:
            nudge = self._afa_nudge_decision(state, base, otp, model, completion)
            if nudge is not None:
                return nudge

        # 1. The income appears to have stopped.
        if shock >= self.config.shock_threshold:
            return self._stop(state, base, per_cycle_clear=0.0, rule="income shock inferred")

        # 2. Nothing legal to present on this cycle.
        if not scored:
            return self._stop(
                state,
                base,
                per_cycle_clear=self._next_cycle_clear(model, state),
                rule="no schedulable date left in this cycle",
            )

        best = scored[0]
        base["chosen_p_clear"] = round(best.p_clear, 4)
        base["chosen_p_optout"] = round(best.p_optout, 4)

        # 3. A date is worth presenting on.
        if best.value > 0:
            return self._present(state, base, best, model)

        # 4/5. Nothing is worth presenting this cycle. Carry or write off.
        return self._stop(
            state,
            base,
            per_cycle_clear=self._next_cycle_clear(model, state),
            rule=f"best date scores {best.value:,.0f}, not worth presenting",
        )

    def _afa_nudge_decision(
        self,
        state: PolicyState,
        base: dict,
        otp: OtpCompletionModel,
        model: PaydayModel,
        completion: float,
    ) -> RetryDecision | None:
        """NUDGE when asking beats presenting. Priced, not asserted.

        The comparison is between two versions of the SAME scored curve - the
        one the money equation would use now, and the one it would use after
        one more nudge - so nothing new enters the objective. What a nudge buys
        is the lift in P(authenticated); what it costs is goodwill, and a day
        if the presentment was ready to go.

        Returns None when nudging is not worth it, and the ordinary date-
        pricing path continues. That is the whole point: on this class the
        agent nudges because of the arithmetic, and on class 1 the identical
        arithmetic says do not.
        """
        sent = state.nudges_sent_this_cycle
        if sent >= MAX_NUDGES_PER_CYCLE:
            return None

        earliest = self.engine.earliest_presentment_date(state.today)
        liquidity = self._liquidity(model, state, earliest, state.cycle_end_date)
        if not liquidity:
            return None

        now = self.engine.score(state, self._apply_afa_term(liquidity, completion))
        after = self.engine.score(
            state,
            self._apply_afa_term(liquidity, otp.completion_probability(sent + 1)),
        )
        if not now or not after:
            return None

        cost = rules.NUDGE_GOODWILL_COST_FRACTION * state.customer.annual_ltv
        gain = after[0].value - now[0].value - cost

        reasoning = dict(base)
        reasoning["nudge_value"] = round(gain, 2)
        reasoning["completion_now"] = round(completion, 4)
        reasoning["completion_after_nudge"] = round(
            otp.completion_probability(sent + 1), 4
        )
        reasoning["nudge_goodwill_cost"] = round(cost, 2)

        if gain <= 0:
            return None

        reasoning["rule"] = (
            "authentication is the binding gate: asking is worth "
            f"Rs {gain:,.0f} more than presenting into it"
        )
        return RetryDecision(action="NUDGE", reasoning=reasoning)

    def _class_stop(self, state: PolicyState, handler) -> RetryDecision:
        """Respond to a class where no date can succeed.

        The handler decides what is even available. For an expired card the
        customer can fix it, so a nudge is worth one message and the mandate is
        worth carrying while they do. For a revoked mandate there is nothing
        left to carry: consent is gone and CANCEL is the only lawful action.
        """
        allowed = set(handler.allowed_actions(state))
        reasoning = {
            "policy": self.name,
            "failure_class": handler.failure_class,
            "class_rationale": handler.rationale,
            "allowed_actions": sorted(allowed),
            "retry_permitted": False,
            "attempts_remaining": state.attempts_remaining,
            "notifications_sent": state.notifications_sent_this_cycle,
            "candidates": [],
            "binding_constraints": [
                {"reason": "failure_class", "detail": handler.rationale}
            ],
            "shock_probability": 0.0,
            "payday_mode": None,
            "payday_confidence": 0.0,
        }

        # One nudge, on the day the failure landed. Repeating it would just be
        # noise, and the customer only needs telling once.
        first_day_of_cycle = state.today == state.failure_event.original_debit_date
        if "NUDGE" in allowed and first_day_of_cycle:
            reasoning["rule"] = (
                f"{handler.failure_class}: no retry can help, but the customer can - "
                "nudging them to update the credential"
            )
            return RetryDecision(action="NUDGE", reasoning=reasoning)

        if "HOLD" in allowed and self.config.use_hold:
            hold_value, cancel_value = self._hold_versus_cancel(state, 0.0)
            reasoning["hold_value"] = round(hold_value, 2)
            reasoning["cancel_value"] = round(cancel_value, 2)
            if hold_value >= cancel_value:
                reasoning["rule"] = (
                    f"{handler.failure_class}: carrying the mandate while the "
                    "customer fixes it beats writing off the LTV"
                )
                return RetryDecision(action="HOLD", reasoning=reasoning)

        reasoning["rule"] = f"{handler.failure_class}: {handler.rationale}"
        return RetryDecision(action="CANCEL", reasoning=reasoning)

    def _next_cycle_clear(self, model: PaydayModel, state: PolicyState) -> float:
        """Best P(clear) reachable in the next cycle, for the hold decision."""
        start = state.cycle_end_date + timedelta(days=1)
        end = start + timedelta(days=NEXT_CYCLE_LOOKAHEAD_DAYS)
        curve = self._liquidity(model, state, start, end)
        return max(curve.values()) if curve else 0.0

    def _present(
        self, state: PolicyState, base: dict, best: ScoredDate, model: PaydayModel
    ) -> RetryDecision:
        reasoning = dict(base)

        if self.config.use_nudge:
            earliest = self.engine.earliest_presentment_date(state.today)
            curve = self._liquidity(model, state, earliest, state.cycle_end_date)
            lift = self.engine.nudge_value(state, curve)
            reasoning["nudge_value"] = round(lift, 2)
            if lift > 0 and state.notifications_sent_this_cycle == 0:
                reasoning["rule"] = "nudge first: the date is plausible but not certain"
                return RetryDecision(action="NUDGE", reasoning=reasoning)

        # Defer the commitment to the last compliant moment. In this model the
        # notice fires a fixed lead before the debit either way, so deferring is
        # value-neutral here; it is kept because in production the option to
        # change your mind on fresher evidence is worth something, and the
        # no-defer ablation measures whether it costs anything.
        commit_on = best.day - timedelta(days=self.engine.lead_time_days)
        if self.config.defer_commitment and state.today < commit_on:
            reasoning["rule"] = "best date is not yet due for commitment"
            reasoning["would_present_on"] = best.day.isoformat()
            reasoning["commit_on"] = commit_on.isoformat()
            return RetryDecision(action="WAIT", reasoning=reasoning)

        reasoning["rule"] = "highest positive expected value in the cycle"
        reasoning["chosen_value"] = round(best.value, 2)
        return RetryDecision(action="RETRY", retry_date=best.day, reasoning=reasoning)

    def _stop(
        self, state: PolicyState, base: dict, per_cycle_clear: float, rule: str
    ) -> RetryDecision:
        """HOLD or CANCEL, whichever the money equation prefers."""
        hold_value, cancel_value = self._hold_versus_cancel(state, per_cycle_clear)
        reasoning = dict(base)
        reasoning.update(
            {
                "rule": rule,
                "next_cycle_p_clear": round(per_cycle_clear, 4),
                "hold_value": round(hold_value, 2),
                "cancel_value": round(cancel_value, 2),
            }
        )

        if not self.config.use_hold:
            reasoning["rule"] = f"{rule}; HOLD disabled by ablation"
            return RetryDecision(action="CANCEL", reasoning=reasoning)

        if hold_value >= cancel_value:
            reasoning["chosen"] = "HOLD"
            return RetryDecision(action="HOLD", reasoning=reasoning)
        reasoning["chosen"] = "CANCEL"
        return RetryDecision(action="CANCEL", reasoning=reasoning)
