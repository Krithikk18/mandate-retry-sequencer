"""Layer 1 of the agent: when is money likely to be in the account?

This module DECIDES NOTHING. It estimates, per candidate date, the probability
that a debit of a given size would clear. The decision engine multiplies that
probability by real rupees, so being *calibrated* matters more than being
confident: a model that says 70% must be right about 70% of the time, or the
money equation is being fed a number that only looks like a probability.

Everything here is inferred from what a real PSP can see - attempt dates and
whether each cleared. No balance, no salary date, no hidden field. The only
outside knowledge is a documented prior over Indian payroll dates, which is
public statistical fact rather than anything about a particular customer.

Three tiers, each strictly more structured than the last:

    tier 1  day-of-month frequency with Beta shrinkage.
            Scores every day independently. Crude, ships, works.

    tier 2  infer WHICH day is payday, then derive liquidity from it.
            The India-specific novelty - see the class docstring.

    tier 3  tier 2 plus a fitted liquidity depth: how long the money lasts,
            learned per customer, and shortened for larger mandates.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable, Sequence

from generator import rules
from generator.models import AgentPaymentHistoryRecord

# --- the documented prior ----------------------------------------------------
# Indian payroll clusters at month end and in the first week. The Payment of
# Wages Act requires wages by the 7th (the 10th for larger establishments), and
# a large share of employers simply pay on the 1st or the last working day.
#
# HARDCODED ON PURPOSE. It is not learned from the customer's history - twelve
# sparse cycles cannot identify a payday on their own, and this is exactly the
# kind of population-level structure a domestic payments company already knows.
# Caveat worth stating in any write-up: the synthetic population is drawn from
# this same shape, so the agent starts with the correct prior FAMILY. On real
# data the prior would be estimated from payroll statistics and would be
# somewhat wrong, which would cost tier 2 some of its advantage.
INDIAN_PAYDAY_PRIOR: dict[int, float] = {
    1: 0.30,
    2: 0.075,
    3: 0.075,
    5: 0.12,
    7: 0.10,
    8: 0.08 / 3,
    9: 0.08 / 3,
    10: 0.08 / 3,
    30: 0.125,
    31: 0.125,
}

# Weight spread uniformly over every day of the month, so the posterior is not
# structurally incapable of locating an unusual payday. Without it a customer
# paid on the 18th could never be found however clear the evidence, which is
# the classic way a strong prior turns into a bug.
PRIOR_FLOOR_MASS: float = 0.08


def payday_prior() -> dict[int, float]:
    """The documented Indian payroll prior, with a floor over all 31 days."""
    floor = PRIOR_FLOOR_MASS / 31.0
    prior = {day: floor for day in range(1, 32)}
    for day, mass in INDIAN_PAYDAY_PRIOR.items():
        prior[day] += mass * (1.0 - PRIOR_FLOOR_MASS)
    total = sum(prior.values())
    return {day: mass / total for day, mass in prior.items()}

# --- liquidity shape ---------------------------------------------------------
# Probability a debit clears, as a function of days since the salary landed.
# High just after payday, falling away as the month is spent. The S-curve
# rather than a straight line because accounts do not drain linearly: they hold
# up, then run out.
LIQUIDITY_MAX: float = 0.95
LIQUIDITY_MIN: float = 0.03
DEFAULT_DEPTH_DAYS: float = 17.0  # phase at which the curve is half-way down
LIQUIDITY_WIDTH_DAYS: float = 4.0  # how sharply it falls

# Tier 3 searches over these depths when fitting how long a customer's money
# lasts. A short depth means somebody who is broke a week after payday.
DEPTH_GRID: tuple[float, ...] = (4.0, 7.0, 10.0, 13.0, 17.0, 21.0, 25.0)

# How hard a bigger mandate is to clear. Doubling the amount shortens the
# effective liquidity window by this power of the ratio.
AMOUNT_ELASTICITY: float = 0.6

# --- tier 1 smoothing --------------------------------------------------------
BETA_PRIOR_STRENGTH: float = 4.0  # pseudo-observations pulling toward the base rate
BASE_RATE_PRIOR_ALPHA: float = 1.0
BASE_RATE_PRIOR_BETA: float = 1.0

# --- shock detection ---------------------------------------------------------
# Prior odds that a failing customer has actually lost their income, and the
# likelihood ratio contributed by each additional cycle that was properly tried
# and failed outright. Both are judgement calls, stated rather than buried.
SHOCK_PRIOR: float = 0.05
SHOCK_LIKELIHOOD_RATIO: float = 5.0
MIN_ATTEMPTS_FOR_DEAD_CYCLE: int = 2

CYCLE_DAYS: int = 30


@lru_cache(maxsize=None)
def _days_in_month(year: int, month: int) -> int:
    """``calendar.monthrange``'s second element, memoised.

    Pure in (year, month) and the calendar is not going to change, so this is a
    cache with no invalidation story to get wrong. It exists because
    ``_last_payday_on_or_before`` is the innermost call in the liquidity curve:
    the curve is rebuilt for every candidate date on every decision day, and
    each rebuild marginalises over all 31 candidate paydays, so this ran three
    million times in a two-seed downtime run and ``monthrange`` re-derived the
    weekday every time.
    """
    return calendar.monthrange(year, month)[1]


@lru_cache(maxsize=100_000)
def _last_payday_on_or_before(day: date, payday: int) -> date:
    """The most recent occurrence of day-of-month ``payday`` at or before ``day``.

    Month-end paydays clamp to the last day of short months, exactly as real
    payroll does: a 31st salary lands on the 28th in February.

    Memoised for the same reason as ``_days_in_month``, and bounded because the
    key space is small in practice: one entry per (date, payday) pair, and a run
    only ever asks about the dates inside its own window.
    """
    this_month = min(payday, _days_in_month(day.year, day.month))
    if this_month <= day.day:
        return date(day.year, day.month, this_month)
    year, month = (day.year - 1, 12) if day.month == 1 else (day.year, day.month - 1)
    return date(year, month, min(payday, _days_in_month(year, month)))


def days_since_payday(day: date, payday: int) -> int:
    """How long ``day`` sits after the most recent payday. Always >= 0."""
    return (day - _last_payday_on_or_before(day, payday)).days


def liquidity_at(phase: float, depth: float = DEFAULT_DEPTH_DAYS) -> float:
    """P(a debit clears | it is ``phase`` days since the salary landed)."""
    z = (depth - phase) / LIQUIDITY_WIDTH_DAYS
    s = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z))))
    return LIQUIDITY_MIN + (LIQUIDITY_MAX - LIQUIDITY_MIN) * s


def effective_depth(depth: float, fitted_amount: float, amount: float) -> float:
    """Shorten the liquidity window for a mandate larger than the fitted one.

    A Rs 200 subscription clears out of loose change three weeks into the
    month; a Rs 12,000 one needs a fresh salary. Same account, same date,
    different answer - which is why the curve has to take an amount at all.
    """
    if amount <= 0 or fitted_amount <= 0:
        return depth
    return depth * (fitted_amount / amount) ** AMOUNT_ELASTICITY


@dataclass
class PaydayModel:
    """Estimates when a customer is liquid, from attempt history alone.

    ``tier`` selects how much structure is imposed:

    1. day-of-month frequencies, smoothed. No notion of payday at all.
    2. a posterior over WHICH day-of-month is payday, from the Indian payroll
       prior and a likelihood that says successes cluster after payday and
       failures cluster before it. The liquidity curve is then the posterior
       marginalised over that unknown. This is the India-specific novelty:
       the informative variable is time-to-liquidity, not retry index, and the
       thing worth inferring is a calendar event the merchant cannot see.
    3. tier 2 plus a per-customer liquidity depth fitted jointly with payday,
       so a customer who is broke by the 10th is modelled differently from one
       whose money lasts the month.
    """

    tier: int = 2
    # Override the payroll prior. Used by scripts/prior_robustness.py to ask
    # how much of the win comes from the prior being right versus the method
    # being sound. None means the documented Indian distribution.
    prior: dict[int, float] | None = None

    # Fitted state
    _fitted: bool = field(default=False, init=False)
    _records: list[AgentPaymentHistoryRecord] = field(default_factory=list, init=False)
    _amount: float = field(default=0.0, init=False)
    _base_rate: float = field(default=0.5, init=False)
    _by_day: dict[int, tuple[int, int]] = field(default_factory=dict, init=False)
    _posterior: dict[int, float] = field(default_factory=dict, init=False)
    # Joint posterior over (payday, liquidity depth). Tier 2 pins the depth at
    # its default and infers only the payday; tier 3 infers both, which is the
    # whole difference between them.
    _joint: dict[tuple[int, float], float] = field(default_factory=dict, init=False)
    _depth: float = field(default=DEFAULT_DEPTH_DAYS, init=False)
    _dead_cycles: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3):
            raise ValueError(f"tier must be 1, 2 or 3, got {self.tier}")

    # --- fitting -------------------------------------------------------------

    @staticmethod
    def _tells_us_about_the_balance(record: AgentPaymentHistoryRecord) -> bool:
        """Is this attempt evidence about the customer's liquidity?

        A success is: the money was there. A 04 is: it was not. But a debit
        that failed because the BANK was down, or because the credential was
        dead, never reached the account - the balance is unknown, not short.

        Counting those as evidence of an empty account is a real error, not a
        rounding one. It drags the inferred payday around and makes the model
        wait for a salary that was never the problem, which showed up as the
        sequencer recovering downtime failures weeks after the bank came back.
        Censored observations are dropped, not treated as failures.
        """
        if record.succeeded:
            return True
        return rules.classify(record.return_code) == rules.FAILURE_CLASS_INSUFFICIENT_FUNDS

    def fit(self, payment_history: Iterable[AgentPaymentHistoryRecord]) -> None:
        """Learn from attempt dates and outcomes. Nothing else is available.

        Only attempts that actually tested the balance are used; see
        ``_tells_us_about_the_balance``.
        """
        records = sorted(
            (r for r in payment_history if self._tells_us_about_the_balance(r)),
            key=lambda r: r.attempt_date,
        )
        self._records = records
        self._amount = records[-1].amount if records else 0.0

        successes = sum(1 for r in records if r.succeeded)
        total = len(records)
        self._base_rate = (successes + BASE_RATE_PRIOR_ALPHA) / (
            total + BASE_RATE_PRIOR_ALPHA + BASE_RATE_PRIOR_BETA
        )

        by_day: dict[int, tuple[int, int]] = {}
        for record in records:
            hit, miss = by_day.get(record.attempt_date.day, (0, 0))
            by_day[record.attempt_date.day] = (
                hit + (1 if record.succeeded else 0),
                miss + (0 if record.succeeded else 1),
            )
        self._by_day = by_day
        self._dead_cycles = self._count_trailing_dead_cycles(records)
        if self.tier >= 2:
            self._fit_posterior(records)
        self._fitted = True

    def _fit_posterior(self, records: Sequence[AgentPaymentHistoryRecord]) -> None:
        """Bayes over (payday, depth), given every attempt we have seen.

        Likelihood: for a candidate payday d, an attempt on date t sits
        ``phase`` days after the most recent d. A success is likely when phase
        is small and unlikely when the month has been spent; a failure says the
        opposite. Multiply across attempts, weight by the payroll prior,
        normalise.

        The evidence is sparse and biased - the legacy schedule only ever
        probed the debit day and +2/+4/+7 after it - which is precisely why the
        prior has to carry weight rather than being estimated away.
        """
        prior = self.prior if self.prior is not None else payday_prior()
        depths = DEPTH_GRID if self.tier >= 3 else (DEFAULT_DEPTH_DAYS,)

        log_scores: dict[tuple[int, float], float] = {}
        for day, prior_mass in prior.items():
            for depth in depths:
                total = math.log(prior_mass) - math.log(len(depths))
                for entry in records:
                    phase = days_since_payday(entry.attempt_date, day)
                    p = liquidity_at(phase, depth)
                    total += math.log(p if entry.succeeded else 1.0 - p)
                log_scores[(day, depth)] = total

        peak = max(log_scores.values())
        weights = {key: math.exp(value - peak) for key, value in log_scores.items()}
        mass = sum(weights.values())
        self._joint = {key: value / mass for key, value in weights.items()}

        posterior: dict[int, float] = {}
        for (day, _), value in self._joint.items():
            posterior[day] = posterior.get(day, 0.0) + value
        self._posterior = posterior
        self._depth = sum(depth * value for (_, depth), value in self._joint.items())

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() before querying the model")

    # --- shock detection -----------------------------------------------------

    @staticmethod
    def _cycles(records: Sequence[AgentPaymentHistoryRecord]) -> list[list]:
        """Group attempts into billing cycles.

        Attempts inside a cycle are days apart; cycles are a month apart. A gap
        of more than half a cycle starts a new one, which needs no knowledge of
        the anchor date.
        """
        cycles: list[list] = []
        current: list = []
        previous: date | None = None
        for record in records:
            if previous is not None and (record.attempt_date - previous).days > CYCLE_DAYS // 2:
                cycles.append(current)
                current = []
            current.append(record)
            previous = record.attempt_date
        if current:
            cycles.append(current)
        return cycles

    def _count_trailing_dead_cycles(
        self, records: Sequence[AgentPaymentHistoryRecord]
    ) -> int:
        """Recent cycles that were properly tried and failed outright.

        A cycle counts only if it saw at least two attempts: the current cycle,
        which has only its original bounce so far, is not yet evidence of
        anything. This is the entire observable signature of a lost income -
        the flag itself is hidden and must never be read.
        """
        dead = 0
        for cycle in reversed(self._cycles(records)):
            if len(cycle) < MIN_ATTEMPTS_FOR_DEAD_CYCLE:
                continue
            if any(r.succeeded for r in cycle):
                break
            dead += 1
        return dead

    def shock_probability(self) -> float:
        """P(this customer's income has stopped), from history alone."""
        self._require_fit()
        odds = (SHOCK_PRIOR / (1.0 - SHOCK_PRIOR)) * (
            SHOCK_LIKELIHOOD_RATIO**self._dead_cycles
        )
        return odds / (1.0 + odds)

    # --- tier 1 --------------------------------------------------------------

    def _tier1_probability(self, day: date) -> float:
        """Smoothed success frequency for this day of month.

        Beta-binomial shrinkage toward the customer's own base rate, so a day
        seen once does not claim certainty and an unseen day falls back to what
        we know overall. No notion of payday, no notion of adjacency - day 6
        tells day 7 nothing. That is the ceiling on this tier.
        """
        hit, miss = self._by_day.get(day.day, (0, 0))
        prior = BETA_PRIOR_STRENGTH * self._base_rate
        return (hit + prior) / (hit + miss + BETA_PRIOR_STRENGTH)

    # --- public queries ------------------------------------------------------

    def payday_posterior(self) -> dict[int, float]:
        """Posterior over which day of the month the salary lands.

        Tier 1 has no such concept and returns the untouched prior; that gap is
        the point of the tiering.
        """
        self._require_fit()
        if self.tier == 1:
            return dict(self.prior) if self.prior is not None else payday_prior()
        return dict(self._posterior)

    def liquidity_curve(
        self, from_date: date, to_date: date, amount: float
    ) -> dict[date, float]:
        """P(a debit of ``amount`` clears), for every date in the window."""
        self._require_fit()
        if to_date < from_date:
            raise ValueError("to_date must not precede from_date")

        survives_shock = 1.0 - self.shock_probability()
        curve: dict[date, float] = {}
        day = from_date
        while day <= to_date:
            raw = self._raw_probability(day, amount)
            curve[day] = max(0.0, min(1.0, raw * survives_shock))
            day += timedelta(days=1)
        return curve

    def _raw_probability(self, day: date, amount: float) -> float:
        if self.tier == 1:
            base = self._tier1_probability(day)
            # Crude amount handling: no structure to hang it on, so damp
            # toward zero as the ask grows relative to what history was fitted
            # on. Tier 2 and 3 do this properly, through the depth.
            if self._amount > 0 and amount > self._amount:
                base *= (self._amount / amount) ** AMOUNT_ELASTICITY
            return base
        # Tiers 2 and 3: marginalise the liquidity curve over everything still
        # uncertain, rather than committing to a single guess at the payday. A
        # model that picks one payday and is wrong is confidently wrong; one
        # that spreads over its posterior is merely uncertain, and uncertainty
        # is what the money equation can actually price.
        total = 0.0
        for (payday, depth), weight in self._joint.items():
            phase = days_since_payday(day, payday)
            total += weight * liquidity_at(
                phase, effective_depth(depth, self._amount, amount)
            )
        return total

    def explain(self) -> dict:
        """Audit trail: what the model believes and why."""
        self._require_fit()
        top = sorted(self.payday_posterior().items(), key=lambda kv: -kv[1])[:3]
        return {
            "tier": self.tier,
            "attempts_observed": len(self._records),
            "successes_observed": sum(1 for r in self._records if r.succeeded),
            "base_success_rate": round(self._base_rate, 4),
            "fitted_amount": self._amount,
            "likely_paydays": [{"day": d, "probability": round(p, 4)} for d, p in top],
            "liquidity_depth_days": round(self._depth, 2),
            "payday_confidence": round(max(top, key=lambda kv: kv[1])[1] if top else 0.0, 4),
            "trailing_dead_cycles": self._dead_cycles,
            "shock_probability": round(self.shock_probability(), 4),
        }
