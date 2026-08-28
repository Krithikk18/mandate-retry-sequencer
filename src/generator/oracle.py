"""Ground-truth oracle over the hidden balance process.

This is the reason the dataset is synthetic. Because we invented every
customer's salary date and spending pattern, we can answer the question no
real payments company can answer:

    "Would a debit on <any date> have succeeded?"

That makes counterfactual evaluation of a retry policy exact rather than
estimated. A live PSP only observes the dates it actually attempted; here we
observe the whole surface.

Two flavours of truth are provided:

``would_succeed``
    The realised truth on the customer's actual path. Deterministic.

``success_probability_true``
    The marginal probability over fresh draws of the same generative process.
    This is the calibration target for an agent that outputs probabilities.
    It is *unconditional*: it does not condition on the failures already
    observed on the customer's realised path, so it answers "how often does a
    customer like this have the money on this date", not "given this customer
    just bounced, how often...".

Note on bounce charges: the balance process deliberately does NOT subtract the
Rs 500 NACH bounce charge that a failed debit levies. If it did, the balance
path would depend on the retry policy being evaluated and the oracle would
stop being policy-independent. The charge is a cost the evaluator applies on
top (see ``rules.bounce_charge``), not a feedback loop inside the truth.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .balance import counterfactual_paths
from .models import Customer

# How much settled history to simulate before the date being asked about, on
# top of the balance module's own burn-in.
ORACLE_LOOKBACK_DAYS: int = 90


def _lookup(balance_series: pd.Series, attempt_date: date) -> float:
    key = pd.Timestamp(attempt_date)
    try:
        return float(balance_series.loc[key])
    except KeyError as exc:
        raise ValueError(
            f"{attempt_date} is outside the simulated window "
            f"({balance_series.index[0].date()} .. {balance_series.index[-1].date()})"
        ) from exc


def would_succeed(
    customer: Customer,
    attempt_date: date,
    amount: float,
    balance_series: pd.Series,
) -> bool:
    """Ground truth: did the customer have the money on ``attempt_date``?

    Deterministic given the seed that produced ``balance_series``.
    """
    return _lookup(balance_series, attempt_date) >= amount


def success_probability_true(
    customer: Customer,
    attempt_date: date,
    amount: float,
    n_sims: int = 1000,
) -> float:
    """Monte-Carlo probability that a debit on ``attempt_date`` clears.

    Averages over fresh draws of the customer's own generative process
    (different salary delays, spend noise and shocks), so it measures the true
    uncertainty an agent could not have resolved. Deterministic in the
    customer's seed.
    """
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    window_start = attempt_date - timedelta(days=ORACLE_LOOKBACK_DAYS)
    sim = counterfactual_paths(
        customer, window_start, days=ORACLE_LOOKBACK_DAYS + 1, n_paths=n_sims
    )
    balances_on_day = sim.balances[:, -1]
    return float(np.mean(balances_on_day >= amount))


def mandate_is_alive(customer: Customer, on: date) -> bool:
    """False once the credential has expired or consent has been withdrawn.

    Separate from the balance entirely: a dead mandate fails on every date, and
    no amount of money in the account changes that.
    """
    return customer.mandate_dead_from is None or on < customer.mandate_dead_from


def presentment_succeeds(
    customer: Customer,
    attempt_date: date,
    amount: float,
    balance_series: pd.Series,
) -> bool:
    """Ground truth for a real presentment: the mandate must be alive AND funded.

    ``would_succeed`` answers only the funding half, which is what the payday
    model is graded against. This is the one every simulated presentment goes
    through, so a dead mandate can never be "recovered" by a later cycle
    happening to find money in the account.
    """
    return mandate_is_alive(customer, attempt_date) and would_succeed(
        customer, attempt_date, amount, balance_series
    )


def success_curve(
    customer: Customer,
    balance_series: pd.Series,
    amount: float,
    start: date | None = None,
    end: date | None = None,
) -> pd.Series:
    """The full counterfactual surface: would a debit have cleared on each day?

    This is the object a retry policy is graded against - every date it could
    have chosen, with the true answer attached.
    """
    window = balance_series
    if start is not None:
        window = window.loc[pd.Timestamp(start) :]
    if end is not None:
        window = window.loc[: pd.Timestamp(end)]
    curve = window >= amount
    if customer.mandate_dead_from is not None:
        curve &= window.index < pd.Timestamp(customer.mandate_dead_from)
    return curve.rename("would_succeed")


def first_success_date(
    customer: Customer,
    balance_series: pd.Series,
    amount: float,
    after: date,
    horizon_days: int = 60,
) -> date | None:
    """Earliest date strictly after ``after`` on which a debit would clear.

    The oracle's answer to "what was the best possible retry date". Returns
    None if the customer never has the money inside the horizon.
    """
    start = after + timedelta(days=1)
    end = min(
        after + timedelta(days=horizon_days), balance_series.index[-1].date()
    )
    if start > end:
        return None
    curve = success_curve(customer, balance_series, amount, start=start, end=end)
    hits = curve[curve]
    if hits.empty:
        return None
    return hits.index[0].date()


def next_salary_date(customer: Customer, after: date) -> date:
    """The customer's next *scheduled* payday strictly after ``after``.

    Nominal, not realised: it ignores the delay draw, so it is the date a
    perfectly-informed sequencer would aim at. Month-end salary days clamp to
    the last day of short months.
    """
    year, month = after.year, after.month
    for _ in range(3):
        day = min(customer.salary_day, calendar.monthrange(year, month)[1])
        candidate = date(year, month, day)
        if candidate > after:
            return candidate
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    raise AssertionError("unreachable: a payday occurs every month")


def days_to_next_salary(customer: Customer, from_date: date) -> int:
    """How long the customer must wait for money to land."""
    return (next_salary_date(customer, from_date) - from_date).days
