"""Main entry point: build, save, load and de-classify the dataset.

    dataset = generate_dataset(n_customers=500, seed=42)
    save_dataset(dataset, "data/run_42")
    for_the_agent = agent_view(dataset)   # hidden truth stripped

Timeline. Everything is anchored on ``as_of``, the moment the retry decision is
being made. The observed history runs for the 365 days before it; the balance
series runs on for another 60 days past it, so the oracle can score any retry
date the agent chooses.

History is billed on the legacy fixed schedule (attempt, then day +2, +4, +7)
so the agent has realistic prior behaviour to learn from. The current cycle
contains only its original failed attempt: deciding what happens after that is
the agent's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import rules
from .balance import daily_balance
from .models import (
    AgentCustomer,
    AgentFailureEvent,
    Customer,
    FailureEvent,
    PaymentHistoryRecord,
)
from .oracle import (
    days_to_next_salary,
    first_success_date,
    presentment_succeeds,
    would_succeed,
)
from .population import make_population

DEFAULT_AS_OF: date = date(2026, 8, 1)
HISTORY_DAYS: int = 365

# Long enough for an episode to run the full MAX_EPISODE_CYCLES horizon after a
# failure near `as_of`. A HOLD is not a terminal action - it carries the mandate
# into the next cycle - so the oracle has to be able to answer questions about
# dates a year out, not just the 60 days the first build allowed.
FUTURE_HORIZON_DAYS: int = 400

# The dumb schedule the merchant used historically: retry on day +2, +4, +7.
# Exactly 3 retries, which is the NPCI ceiling of 4 attempts per cycle.
LEGACY_RETRY_OFFSETS: tuple[int, ...] = (2, 4, 7)
assert 1 + len(LEGACY_RETRY_OFFSETS) <= rules.MAX_ATTEMPTS_PER_CYCLE


@dataclass
class Dataset:
    """Everything the generator produced, truth included."""

    customers: list[Customer]
    failure_events: list[FailureEvent]
    payment_history: list[PaymentHistoryRecord]
    balance_series: dict[str, pd.Series]
    as_of: date
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def natural_failure_rate(self) -> float:
        """Share of customers whose current-cycle debit actually bounced.

        Not a knob: it falls out of the balance process.
        """
        return len(self.failure_events) / len(self.customers)

    def customer(self, customer_id: str) -> Customer:
        return {c.customer_id: c for c in self.customers}[customer_id]

    def balance_of(self, customer_id: str) -> pd.Series:
        return self.balance_series[customer_id]


@dataclass
class AgentDataset:
    """The same world with every hidden field removed.

    No salary_day, no burn_rate, no buffer_ratio, no optout_sensitivity, no
    balance_seed, no balance series - and no balance_at_failure either, since a
    real PSP learns only that the account was short, never by how much.
    """

    customers: list[AgentCustomer]
    failure_events: list[AgentFailureEvent]
    payment_history: list[PaymentHistoryRecord]
    as_of: date

    def customers_frame(self) -> pd.DataFrame:
        return _to_frame(self.customers)

    def payment_history_frame(self) -> pd.DataFrame:
        return _to_frame(self.payment_history)

    def failure_events_frame(self) -> pd.DataFrame:
        return _to_frame(self.failure_events)


def _to_frame(records: Iterable[Any]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in records])


def _cycle_debit_dates(
    customer: Customer, as_of: date, history_start: date
) -> list[date]:
    """Original debit dates inside the observed window, oldest first.

    Truncated to the customer's tenure: someone who signed up three months ago
    has three cycles of history, not twelve.
    """
    out: list[date] = []
    year, month = history_start.year, history_start.month
    while True:
        attempt = date(year, month, customer.debit_day_of_month)
        if attempt >= as_of:
            break
        if attempt >= history_start:
            out.append(attempt)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out[-customer.tenure_months:]


CLASS_RETURN_CODES: dict[str, str] = {
    rules.FAILURE_CLASS_EXPIRED_CARD: rules.RETURN_CODE_EXPIRED_CARD,
    rules.FAILURE_CLASS_OPTED_OUT: rules.RETURN_CODE_MANDATE_REVOKED,
}


def _resolve_attempt(
    customer: Customer, day: date, balance_series: pd.Series
) -> tuple[bool, str | None, str]:
    """Present the debit and report (succeeded, return_code, failure_class).

    A dead mandate fails on every date regardless of the balance, and says WHY
    through the return code. That is the only channel the agent gets - the
    fields behind it are hidden.
    """
    if customer.mandate_dead_from is not None and day >= customer.mandate_dead_from:
        dead_class = customer.mandate_dead_class or rules.FAILURE_CLASS_AMBIGUOUS
        return False, CLASS_RETURN_CODES[dead_class], dead_class

    cleared = presentment_succeeds(
        customer, day, customer.subscription_amount, balance_series
    )
    return (
        cleared,
        None if cleared else rules.RETURN_CODE_INSUFFICIENT_FUNDS,
        rules.FAILURE_CLASS_INSUFFICIENT_FUNDS,
    )


def _record(
    customer: Customer, day: date, succeeded: bool, return_code: str | None = None
) -> PaymentHistoryRecord:
    if not succeeded and return_code is None:
        return_code = rules.RETURN_CODE_INSUFFICIENT_FUNDS
    return PaymentHistoryRecord(
        customer_id=customer.customer_id,
        attempt_date=day,
        amount=customer.subscription_amount,
        succeeded=succeeded,
        return_code=None if succeeded else return_code,
    )


RECOVERY_HORIZON_DAYS: int = 30


def _failure_diagnostics(
    customers: list[Customer],
    failure_events: list[FailureEvent],
    balance_series: dict[str, pd.Series],
    legacy_offsets: tuple[int, ...],
) -> dict[str, Any]:
    """Dataset-quality numbers, computed with the oracle.

    These are properties of the generated world, not results of a policy: they
    say how hard the problem is, and they are what the realism tests assert.
    Two of them decide whether the dataset is worth evaluating on at all:

    ``legacy_recovery_rate``
        What the fixed day-2/4/7 schedule recovers. Published fixed-retry
        recovery runs 35-45%; a synthetic world where the dumb baseline scores
        far above that has made the problem too easy and will overstate any
        agent compared against it.

    ``unrecoverable_share``
        Failures where NO date inside the horizon clears. If this is zero,
        CANCEL is never the correct action and a stopping rule can never be
        wrong - so it can never be tested either.
    """
    if not failure_events:
        return {
            "legacy_recovery_rate": 0.0,
            "unrecoverable_share": 0.0,
            "median_days_to_payday": None,
            "share_failing_beyond_legacy_reach": 0.0,
        }

    by_id = {c.customer_id: c for c in customers}
    legacy_wins = 0
    unrecoverable = 0
    gaps: list[int] = []

    for event in failure_events:
        customer = by_id[event.customer_id]
        series = balance_series[event.customer_id]
        amount = customer.subscription_amount

        if event.failure_class in rules.NEVER_RETRY_CLASSES:
            # The credential is dead or consent is gone. The balance is
            # irrelevant: no date clears, and the fixed schedule recovers
            # nothing however many times it presents.
            unrecoverable += 1
            gaps.append(days_to_next_salary(customer, event.original_debit_date))
            continue

        if any(
            would_succeed(
                customer, event.original_debit_date + timedelta(days=offset), amount, series
            )
            for offset in legacy_offsets
        ):
            legacy_wins += 1

        if first_success_date(
            customer,
            series,
            amount,
            after=event.original_debit_date,
            horizon_days=RECOVERY_HORIZON_DAYS,
        ) is None:
            unrecoverable += 1

        gaps.append(days_to_next_salary(customer, event.original_debit_date))

    total = len(failure_events)
    return {
        "legacy_recovery_rate": legacy_wins / total,
        "unrecoverable_share": unrecoverable / total,
        "median_days_to_payday": float(np.median(gaps)),
        "share_failing_beyond_legacy_reach": float(
            np.mean(np.asarray(gaps) > max(legacy_offsets))
        ),
    }


def generate_dataset(
    n_customers: int = 500,
    seed: int = 42,
    as_of: date = DEFAULT_AS_OF,
) -> Dataset:
    """Generate a full synthetic book of subscription mandates.

    The failure rate is not imposed. Every attempt is resolved against the
    customer's simulated balance, so whatever rate comes out is what the
    income and spending process actually produces.
    """
    if n_customers < 1:
        raise ValueError("n_customers must be >= 1")

    rng = np.random.default_rng(seed)
    customers = make_population(n_customers, rng, as_of=as_of)

    series_start = as_of - timedelta(days=HISTORY_DAYS)
    series_days = HISTORY_DAYS + FUTURE_HORIZON_DAYS

    balance_series: dict[str, pd.Series] = {}
    payment_history: list[PaymentHistoryRecord] = []
    failure_events: list[FailureEvent] = []

    historical_originals = 0
    historical_original_failures = 0

    for customer in customers:
        series = daily_balance(customer, series_start, series_days)
        balance_series[customer.customer_id] = series
        amount = customer.subscription_amount

        debit_dates = _cycle_debit_dates(customer, as_of, series_start)
        if not debit_dates:
            continue
        past_cycles, current_cycle = debit_dates[:-1], debit_dates[-1]

        # closed cycles: original attempt plus the legacy +2/+4/+7 retries
        for debit_date in past_cycles:
            ok, code, _ = _resolve_attempt(customer, debit_date, series)
            payment_history.append(_record(customer, debit_date, ok, code))
            historical_originals += 1
            if ok:
                continue
            historical_original_failures += 1
            for offset in LEGACY_RETRY_OFFSETS:
                retry_date = debit_date + timedelta(days=offset)
                retry_ok, retry_code, _ = _resolve_attempt(customer, retry_date, series)
                payment_history.append(_record(customer, retry_date, retry_ok, retry_code))
                if retry_ok:
                    break

        # the open cycle: the original attempt only
        ok, code, failure_class = _resolve_attempt(customer, current_cycle, series)
        payment_history.append(_record(customer, current_cycle, ok, code))
        if not ok:
            failure_events.append(
                FailureEvent(
                    customer_id=customer.customer_id,
                    cycle_start_date=current_cycle,
                    original_debit_date=current_cycle,
                    return_code=code or rules.RETURN_CODE_INSUFFICIENT_FUNDS,
                    failure_class=failure_class,
                    balance_at_failure=float(series.loc[pd.Timestamp(current_cycle)]),
                )
            )

    diagnostics = _failure_diagnostics(
        customers, failure_events, balance_series, LEGACY_RETRY_OFFSETS
    )

    meta: dict[str, Any] = {
        "n_customers": n_customers,
        "seed": seed,
        "as_of": as_of.isoformat(),
        "history_days": HISTORY_DAYS,
        "future_horizon_days": FUTURE_HORIZON_DAYS,
        "series_start": series_start.isoformat(),
        "series_days": series_days,
        "n_failure_events": len(failure_events),
        "natural_failure_rate": len(failure_events) / n_customers,
        "historical_original_attempts": historical_originals,
        "historical_first_attempt_failure_rate": (
            historical_original_failures / historical_originals
            if historical_originals
            else 0.0
        ),
        "legacy_retry_offsets": list(LEGACY_RETRY_OFFSETS),
        "n_income_shocked": sum(
            c.income_shock_start is not None for c in customers
        ),
        **diagnostics,
    }

    return Dataset(
        customers=customers,
        failure_events=failure_events,
        payment_history=payment_history,
        balance_series=balance_series,
        as_of=as_of,
        seed=seed,
        meta=meta,
    )


def agent_view(dataset: Dataset) -> AgentDataset:
    """Strip every hidden field. This is the only thing the agent may read."""
    return AgentDataset(
        customers=[AgentCustomer.from_customer(c) for c in dataset.customers],
        failure_events=[
            AgentFailureEvent.from_failure(e) for e in dataset.failure_events
        ],
        payment_history=list(dataset.payment_history),
        as_of=dataset.as_of,
    )


# --- persistence -------------------------------------------------------------

_TABLES = ("customers", "failure_events", "payment_history", "balance_series")


def _balance_frame(dataset: Dataset) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                "customer_id": customer_id,
                "date": series.index,
                "balance": series.to_numpy(),
            }
        )
        for customer_id, series in dataset.balance_series.items()
    ]
    return pd.concat(frames, ignore_index=True)


def save_dataset(dataset: Dataset, path: str | Path, fmt: str = "parquet") -> Path:
    """Write the dataset to ``path`` (a directory). Returns that directory."""
    if fmt not in ("parquet", "csv"):
        raise ValueError("fmt must be 'parquet' or 'csv'")
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    frames = {
        "customers": _to_frame(dataset.customers),
        "failure_events": _to_frame(dataset.failure_events),
        "payment_history": _to_frame(dataset.payment_history),
        "balance_series": _balance_frame(dataset),
    }
    for name, frame in frames.items():
        target = out / f"{name}.{fmt}"
        if fmt == "parquet":
            frame.to_parquet(target, index=False)
        else:
            frame.to_csv(target, index=False)

    (out / "meta.json").write_text(json.dumps(dataset.meta, indent=2), encoding="utf-8")
    return out


def load_dataset(path: str | Path) -> Dataset:
    """Read back a dataset written by :func:`save_dataset`."""
    src = Path(path)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))

    fmt = "parquet" if (src / "customers.parquet").exists() else "csv"

    def read(path: Path) -> pd.DataFrame:
        if fmt == "parquet":
            return pd.read_parquet(path)
        # Two CSV traps worth naming: return codes are strings with a
        # meaningful leading zero ("04") and must not be inferred as integers,
        # and pandas' default float parser is not round-trip exact, which would
        # silently perturb the hidden parameters the oracle depends on.
        return pd.read_csv(
            path,
            dtype={"return_code": "string"},
            float_precision="round_trip",
        )

    frames = {name: read(src / f"{name}.{fmt}") for name in _TABLES}

    history = frames["payment_history"]
    history["attempt_date"] = pd.to_datetime(history["attempt_date"]).dt.date
    history["succeeded"] = history["succeeded"].astype(bool)
    history["return_code"] = [
        None if pd.isna(code) else str(code) for code in history["return_code"]
    ]

    events = frames["failure_events"]
    for column in ("cycle_start_date", "original_debit_date"):
        if not events.empty:
            events[column] = pd.to_datetime(events[column]).dt.date
        events[column] = events[column].astype(object)

    # income_shock_start is a nullable date: None for ~96% of customers, and
    # both parquet and CSV round-trip it as NaT/NaN rather than None.
    people = frames["customers"]
    for column in ("income_shock_start", "mandate_dead_from"):
        parsed = pd.to_datetime(people[column], errors="coerce")
        people[column] = [None if pd.isna(value) else value.date() for value in parsed]
    people["mandate_dead_class"] = [
        None if pd.isna(value) else str(value) for value in people["mandate_dead_class"]
    ]

    balances = frames["balance_series"]
    balances["date"] = pd.to_datetime(balances["date"])
    balance_series = {
        str(customer_id): pd.Series(
            group["balance"].to_numpy(),
            index=pd.DatetimeIndex(group["date"], freq="D"),
            name="balance",
        )
        for customer_id, group in balances.groupby("customer_id", sort=False)
    }

    return Dataset(
        customers=[Customer(**row) for row in frames["customers"].to_dict("records")],
        failure_events=[
            FailureEvent(**row) for row in events.to_dict("records")
        ],
        payment_history=[
            PaymentHistoryRecord(**row) for row in history.to_dict("records")
        ],
        balance_series=balance_series,
        as_of=date.fromisoformat(meta["as_of"]),
        seed=int(meta["seed"]),
        meta=meta,
    )
