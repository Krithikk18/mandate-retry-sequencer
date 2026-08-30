"""Scoring policies across a whole book, and comparing them across seeds.

``evaluate`` runs one policy over every failure event in one dataset.
``compare`` runs several policies across several seeds and reports mean plus
min-max range for every metric, because a single seed is not evidence.

Some metrics are computable only here, never inside a policy:

``wasted_attempts``
    presentments made when the TRUE success probability was below 20%. A
    policy cannot know this; the oracle can. It is the cleanest measure of
    whether a schedule is firing blind.

``wrongly_cancelled`` / ``correctly_cancelled``
    whether a CANCEL killed a mandate the oracle says was recoverable, or
    correctly gave up on one whose income had genuinely stopped. Without the
    second number a stopping rule cannot be credited for being right.

    Recoverability is judged with nudging ALLOWED (``afa.MAX_USEFUL_NUDGES``).
    The question is whether any compliant policy could have saved the mandate,
    and since class 5 the answer sometimes depends on an action rather than a
    date: an AFA-gated customer one message away from authenticating is not
    unrecoverable, and scoring a CANCEL on them as correct would let the agent
    be credited for giving up on money that was there to take.

Both call into ``generator.oracle``. Nothing in ``policy`` imports it, and a
test enforces that.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Callable, Iterable, Sequence

from generator.afa import MAX_USEFUL_NUDGES
from generator.generate import Dataset, generate_dataset
from generator.models import Customer
from generator.optout import DEFAULT_IRRITATION_DIRECTION, IrritationDirection
from generator.oracle import first_success_date, success_probability_true
from policy.base import Policy

from . import economics
from .simulator import EpisodeResult, run_episode

# A presentment with less than this true chance of clearing is a wasted attempt:
# it burns an attempt from the NPCI budget and fires a cancel-capable notice at
# the customer, in exchange for almost no chance of the money.
WASTED_ATTEMPT_THRESHOLD: float = 0.20

# Monte-Carlo depth for the calibration oracle. 1000 is the generator default;
# 200 is plenty to classify against a 0.20 threshold and is five times cheaper
# across twenty seeds.
ORACLE_SIMS: int = 200

RECOVERY_HORIZON_DAYS: int = 30


@lru_cache(maxsize=200_000)
def _true_probability(customer: Customer, day: date, amount: float, n_sims: int) -> float:
    """Cached because policies attempt the same dates as each other."""
    return success_probability_true(customer, day, amount, n_sims=n_sims)


@dataclass
class PolicyReport:
    """One policy, one dataset, one seed."""

    policy_name: str
    seed: int
    n_failures: int

    money_recovered: float
    recovery_rate: float
    attempts_used_total: int
    cost_per_rupee_recovered: float | None
    wasted_attempts: int
    wrongly_cancelled: int
    correctly_cancelled: int
    customers_retained: int
    optouts_induced: int
    churned: int
    compliance_breaches: int
    cycles_held_total: int
    carrying_cost_total: float
    net_value_total: float
    mean_days_to_resolution: float
    outcomes: dict[str, int] = field(default_factory=dict)

    # Filled in by compare(), which is the only place that knows the baseline
    # and the ceiling. (mine - legacy) / (oracle - legacy).
    headroom_captured: float | None = None

    episodes: list[EpisodeResult] = field(default_factory=list, repr=False)


def evaluate(
    policy: Policy,
    dataset: Dataset,
    seed: int,
    irritation_direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
    oracle_metrics: bool = True,
    n_sims: int = ORACLE_SIMS,
    lead_time_days: int = 1,
    notifications_can_cancel: bool = True,
    strict_compliance: bool = True,
) -> PolicyReport:
    """Run ``policy`` over every failure event in ``dataset``."""
    episodes = [
        run_episode(
            policy,
            dataset.customer(event.customer_id),
            event,
            dataset.balance_of(event.customer_id),
            dataset.payment_history,
            seed=seed,
            irritation_direction=irritation_direction,
            lead_time_days=lead_time_days,
            notifications_can_cancel=notifications_can_cancel,
            strict_compliance=strict_compliance,
        )
        for event in dataset.failure_events
    ]
    n = len(episodes)
    if n == 0:
        raise ValueError("dataset contains no failure events to evaluate")

    money = sum(e.recovered_amount for e in episodes)
    attempts = sum(e.attempts_used for e in episodes)
    total_cost = sum(
        economics.processing_cost_total(
            dataset.customer(e.customer_id).rail, e.attempts_used
        )
        + e.bounce_charges_paid
        + e.carrying_cost_paid
        for e in episodes
    )

    wasted = 0
    wrongly_cancelled = 0
    correctly_cancelled = 0
    if oracle_metrics:
        for episode in episodes:
            customer = dataset.customer(episode.customer_id)
            amount = customer.subscription_amount
            for retry_date in episode.retry_dates:
                if _true_probability(customer, retry_date, amount, n_sims) < (
                    WASTED_ATTEMPT_THRESHOLD
                ):
                    wasted += 1
            if episode.outcome == "CANCELLED":
                event = next(
                    e for e in dataset.failure_events if e.customer_id == episode.customer_id
                )
                salvageable = first_success_date(
                    customer,
                    dataset.balance_of(episode.customer_id),
                    amount,
                    after=event.original_debit_date,
                    horizon_days=RECOVERY_HORIZON_DAYS,
                    nudges=MAX_USEFUL_NUDGES,
                )
                if salvageable is None:
                    correctly_cancelled += 1
                else:
                    wrongly_cancelled += 1

    outcomes: dict[str, int] = {}
    for episode in episodes:
        outcomes[episode.outcome] = outcomes.get(episode.outcome, 0) + 1

    return PolicyReport(
        policy_name=policy.name,
        seed=seed,
        n_failures=n,
        money_recovered=money,
        recovery_rate=sum(e.recovered for e in episodes) / n,
        attempts_used_total=attempts,
        cost_per_rupee_recovered=economics.cost_per_rupee_recovered(total_cost, money),
        wasted_attempts=wasted,
        wrongly_cancelled=wrongly_cancelled,
        correctly_cancelled=correctly_cancelled,
        customers_retained=sum(e.mandate_survives for e in episodes),
        optouts_induced=sum(e.outcome == "OPTED_OUT" for e in episodes),
        churned=sum(e.outcome == "CHURNED" for e in episodes),
        compliance_breaches=sum(e.compliance_breaches for e in episodes),
        cycles_held_total=sum(e.cycles_held for e in episodes),
        carrying_cost_total=sum(e.carrying_cost_paid for e in episodes),
        net_value_total=sum(e.net_value for e in episodes),
        mean_days_to_resolution=statistics.fmean(e.days_to_resolution for e in episodes),
        outcomes=outcomes,
        episodes=episodes,
    )


@dataclass
class MetricSummary:
    """Mean and range of one metric across seeds. Never quote a single seed."""

    name: str
    values: list[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def low(self) -> float:
        return min(self.values)

    @property
    def high(self) -> float:
        return max(self.values)

    def render(self) -> str:
        return f"{self.mean:>12,.3f}  [{self.low:>10,.3f} .. {self.high:>10,.3f}]"


SUMMARISED_METRICS: tuple[str, ...] = (
    "money_recovered",
    "recovery_rate",
    "headroom_captured",
    "attempts_used_total",
    "cost_per_rupee_recovered",
    "wasted_attempts",
    "wrongly_cancelled",
    "correctly_cancelled",
    "customers_retained",
    "optouts_induced",
    "churned",
    "compliance_breaches",
    "cycles_held_total",
    "carrying_cost_total",
    "net_value_total",
    "mean_days_to_resolution",
)


@dataclass
class ComparisonReport:
    """Several policies, several seeds."""

    seeds: list[int]
    n_customers: int
    per_seed: dict[str, list[PolicyReport]]
    summary: dict[str, dict[str, MetricSummary]]

    def policy_names(self) -> list[str]:
        return list(self.per_seed)

    def render(self, metrics: Sequence[str] = SUMMARISED_METRICS) -> str:
        lines = [
            f"{len(self.seeds)} seeds x {self.n_customers} customers   "
            f"seeds={self.seeds[0]}..{self.seeds[-1]}",
            "",
        ]
        for metric in metrics:
            lines.append(metric)
            for name in self.policy_names():
                summary = self.summary[name].get(metric)
                if summary is None or not summary.values:
                    lines.append(f"  {name:<16} {'n/a':>12}")
                else:
                    lines.append(f"  {name:<16} {summary.render()}")
            lines.append("")
        return "\n".join(lines)


PolicySpec = Policy | Callable[[Dataset], Policy]


def _materialise(spec: PolicySpec, dataset: Dataset) -> Policy:
    """Resolve a policy spec against one seed's dataset.

    A plain Policy is reused as-is. A callable is invoked per seed, which is
    what any policy that must be bound to that seed's data needs - the oracle
    ceiling above all, since customer ids repeat across seeds and a shared
    truth table would silently be the wrong one.
    """
    return spec if isinstance(spec, Policy) else spec(dataset)


def compare(
    policies: Iterable[PolicySpec],
    seeds: Sequence[int],
    n_customers: int = 500,
    baseline_name: str | None = None,
    ceiling_name: str | None = None,
    irritation_direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
    oracle_metrics: bool = True,
    n_sims: int = ORACLE_SIMS,
    lead_time_days: int = 1,
    strict_compliance: bool = True,
    datasets: dict[int, Dataset] | None = None,
) -> ComparisonReport:
    """Run every policy on every seed and summarise.

    ``policies`` may contain Policy instances or callables taking the seed's
    Dataset and returning one; see ``_materialise``.

    ``headroom_captured`` needs both a floor and a ceiling, so it is filled in
    here: (policy - baseline) / (ceiling - baseline), measured on recovery rate
    within each seed before averaging.
    """
    specs = list(policies)
    per_seed: dict[str, list[PolicyReport]] = {}

    for seed in seeds:
        dataset = (datasets or {}).get(seed) or generate_dataset(
            n_customers=n_customers, seed=seed
        )
        bound = [_materialise(spec, dataset) for spec in specs]
        reports = {
            p.name: evaluate(
                p,
                dataset,
                seed=seed,
                irritation_direction=irritation_direction,
                oracle_metrics=oracle_metrics,
                n_sims=n_sims,
                lead_time_days=lead_time_days,
                strict_compliance=strict_compliance,
            )
            for p in bound
        }

        if baseline_name in reports and ceiling_name in reports:
            floor = reports[baseline_name].recovery_rate
            ceiling = reports[ceiling_name].recovery_rate
            span = ceiling - floor
            for report in reports.values():
                report.headroom_captured = (
                    (report.recovery_rate - floor) / span if span > 1e-12 else None
                )

        for name, report in reports.items():
            per_seed.setdefault(name, []).append(report)

    summary: dict[str, dict[str, MetricSummary]] = {}
    for name, reports in per_seed.items():
        summary[name] = {}
        for metric in SUMMARISED_METRICS:
            values = [
                float(getattr(r, metric))
                for r in reports
                if getattr(r, metric) is not None
            ]
            summary[name][metric] = MetricSummary(metric, values)

    return ComparisonReport(
        seeds=list(seeds),
        n_customers=n_customers,
        per_seed=per_seed,
        summary=summary,
    )
