"""Inferring which banks are down, from return codes alone.

**This is the first genuinely cross-customer signal in the project, and that is
an architectural difference rather than more code.**

``agent/payday.py`` reasons about ONE customer in isolation: their salary date
is theirs, and another subscriber's history says nothing about it. Pooling
would be meaningless there.

Here it is the opposite. A bank outage is a single event that fails every
mandate on that issuer at once. Five downtime-coded failures on the same bank
inside an hour are strong evidence of an outage, and that evidence is worth
most to the customers who have NOT failed yet - the ones whose debit is
scheduled for tomorrow morning, on the same bank, into the same outage. No
amount of per-customer history could produce that inference.

Agent-safe by the same rule as the rest of ``agent/``: it may not import
``generator.oracle``, ``generator.banks`` or any hidden field. It never sees an
outage interval. It sees ``(bank_id, when, return_code)`` - all three of which a
real PSP has on every declined presentment - and infers the rest.

The discriminating property, which is the whole reason the class is separable:

    a cluster of 91s on one bank   -> that bank is down
    a cluster of 04s on one bank   -> those customers are broke

Both cluster in time. Only the first says anything about the bank. This falls
out of counting nothing but downtime-coded evidence, and needs no cleverness -
which is exactly why the return code is worth routing on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

from generator import rules

# Book-wide prior probability that any given bank is unavailable at any given
# instant. Used for cold start, and as the floor a bank with a clean record
# decays back toward. ESTIMATE - a merchant would read it off its own decline
# telemetry.
BASE_OUTAGE_RATE: float = 0.015

# Evidence older than this tells you nothing about whether the bank is up NOW.
# Outages last hours, so the window is hours, not days.
EVIDENCE_WINDOW_HOURS: float = 12.0

# How sharply confidence rises with corroborating failures. Each additional
# downtime-coded failure on the same bank in the window multiplies the odds.
# One failure could be anything; three on one bank in an hour is an outage.
EVIDENCE_LIKELIHOOD_RATIO: float = 4.0

# Beyond this many corroborating failures the evidence saturates - a hundred
# failures is not meaningfully more conclusive than ten.
MAX_CORROBORATION: int = 6

# Recency: evidence decays with a half-life, because a bank that failed six
# hours ago has probably recovered. Outage medians sit near three hours.
EVIDENCE_HALF_LIFE_HOURS: float = 3.0

# What to assume about recovery time before there is evidence. Matches the
# generator's median outage duration, which a merchant would likewise estimate
# from its own incident history rather than know.
DEFAULT_RECOVERY_HOURS: float = 3.0
MIN_RECOVERY_HOURS: float = 0.5
MAX_RECOVERY_HOURS: float = 18.0


@dataclass(frozen=True)
class PresentmentOutcome:
    """One resolved presentment, as a PSP sees it.

    Deliberately just three fields, all agent-visible. ``PaymentHistoryRecord``
    carries no bank_id - it is a property of the customer, not the attempt - so
    the caller joins the two before handing anything here. Keeping the shape
    this narrow is what makes it obvious nothing hidden can be smuggled in.
    """

    bank_id: str
    at: datetime
    return_code: str | None

    @property
    def is_downtime_coded(self) -> bool:
        return rules.classify(self.return_code) == rules.FAILURE_CLASS_DOWNTIME


@dataclass
class BankHealthModel:
    """Pools decline evidence across every customer sharing an issuer.

    Online by construction: ``observe`` takes outcomes one at a time as they
    resolve, so the same object serves a batch backfill and a live stream.
    """

    base_rate: float = BASE_OUTAGE_RATE
    _downtime: dict[str, list[datetime]] = field(
        default_factory=lambda: defaultdict(list), init=False
    )
    _observations: dict[str, int] = field(
        default_factory=lambda: defaultdict(int), init=False
    )

    # --- ingestion -----------------------------------------------------------

    def observe(self, outcome: PresentmentOutcome) -> None:
        """Record one resolved presentment. Only 91s count as bank evidence."""
        self._observations[outcome.bank_id] += 1
        if outcome.is_downtime_coded:
            self._downtime[outcome.bank_id].append(outcome.at)
            self._downtime[outcome.bank_id].sort()

    def observe_many(self, outcomes: Iterable[PresentmentOutcome]) -> None:
        for outcome in outcomes:
            self.observe(outcome)

    @classmethod
    def from_history(
        cls, records: Iterable[Any], bank_of: dict[str, str]
    ) -> "BankHealthModel":
        """Backfill from agent-visible payment history.

        ``records`` are ``AgentPaymentHistoryRecord``; ``bank_of`` maps
        customer_id to bank_id, which the agent reads off the public customer
        field. Attempt dates are dates, so they are placed at the presentment
        hour - the same hour the generator resolved them at.
        """
        model = cls()
        for record in records:
            bank_id = bank_of.get(record.customer_id)
            if bank_id is None or record.succeeded:
                continue
            model.observe(
                PresentmentOutcome(
                    bank_id=bank_id,
                    at=datetime.combine(
                        record.attempt_date, rules.DEFAULT_PRESENTMENT_TIME
                    ),
                    return_code=record.return_code,
                )
            )
        return model

    # --- inference -----------------------------------------------------------

    def _corroboration(self, bank_id: str, at_time: datetime) -> float:
        """Recency-weighted count of downtime-coded failures near ``at_time``."""
        weight = 0.0
        for when in self._downtime.get(bank_id, ()):
            gap_hours = abs((at_time - when).total_seconds()) / 3600.0
            if gap_hours > EVIDENCE_WINDOW_HOURS:
                continue
            weight += 0.5 ** (gap_hours / EVIDENCE_HALF_LIFE_HOURS)
        return min(weight, float(MAX_CORROBORATION))

    def outage_probability(self, bank_id: str, at_time: datetime) -> float:
        """P(this bank is unavailable at this hour).

        A Bayes factor over corroborating downtime-coded failures. With no
        evidence it returns the book-wide base rate - never 0, because absence
        of evidence on a quiet bank is not evidence of health, and never 1,
        because no finite pile of declines makes an outage certain.
        """
        corroboration = self._corroboration(bank_id, at_time)
        prior_odds = self.base_rate / (1.0 - self.base_rate)
        odds = prior_odds * (EVIDENCE_LIKELIHOOD_RATIO**corroboration)
        return odds / (1.0 + odds)

    def uptime_probability(self, bank_id: str, at_time: datetime) -> float:
        """P(a presentment can reach the account at this hour)."""
        return 1.0 - self.outage_probability(bank_id, at_time)

    def expected_recovery_hours(self, bank_id: str) -> float:
        """How long this bank's outages have been running lately.

        Measured as the span from the first to the last downtime-coded failure
        in each observed burst.

        That span is a CENSORED LOWER BOUND, not the duration. The agent only
        learns a bank was down at hours it happened to present in, and mandates
        present once a day at one hour - so a three-hour outage typically shows
        up as a single observation with a span of zero. Treating the observed
        span as the answer would systematically under-estimate recovery time and
        push the policy into retrying too early.

        So the book-wide default acts as a floor rather than only a fallback:
        the estimate is the larger of what was observed and what outages
        generally run. Only a burst that visibly spans longer than the default
        moves the number up.
        """
        stamps = self._downtime.get(bank_id, [])
        if len(stamps) < 2:
            return DEFAULT_RECOVERY_HOURS

        bursts: list[list[datetime]] = [[stamps[0]]]
        for when in stamps[1:]:
            if (when - bursts[-1][-1]).total_seconds() / 3600.0 <= EVIDENCE_WINDOW_HOURS:
                bursts[-1].append(when)
            else:
                bursts.append([when])

        spans = [
            (burst[-1] - burst[0]).total_seconds() / 3600.0
            for burst in bursts
            if len(burst) >= 2
        ]
        if not spans:
            return DEFAULT_RECOVERY_HOURS
        observed = sum(spans) / len(spans)
        return float(min(max(observed, DEFAULT_RECOVERY_HOURS), MAX_RECOVERY_HOURS))

    def likely_clear_by(self, bank_id: str, at_time: datetime) -> datetime:
        """When this bank is expected to be back, if it is down now."""
        return at_time + timedelta(hours=self.expected_recovery_hours(bank_id))

    # --- audit ---------------------------------------------------------------

    def explain(self, bank_id: str | None = None, at_time: datetime | None = None) -> dict:
        """What the model believes, and on what evidence."""
        if bank_id is None:
            return {
                "banks_observed": len(self._observations),
                "base_rate": round(self.base_rate, 4),
                "downtime_evidence": {
                    bank: len(stamps) for bank, stamps in self._downtime.items() if stamps
                },
            }
        at_time = at_time or datetime.now()
        return {
            "bank_id": bank_id,
            "presentments_observed": self._observations.get(bank_id, 0),
            "downtime_coded_failures": len(self._downtime.get(bank_id, ())),
            "corroboration": round(self._corroboration(bank_id, at_time), 3),
            "outage_probability": round(self.outage_probability(bank_id, at_time), 4),
            "expected_recovery_hours": round(self.expected_recovery_hours(bank_id), 2),
            "cold_start": bank_id not in self._observations,
        }
