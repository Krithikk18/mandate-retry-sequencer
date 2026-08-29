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
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from . import rules
from .balance import counterfactual_paths
from .banks import BankOutageBook
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


# --- hour-resolution truth, for the downtime class ---------------------------
#
# The functions above are BALANCE-ONLY and stay that way: the payday model is
# graded against them, and folding a second failure mode into them would make
# that grading meaningless. What follows is the truth for a presentment, which
# needs both conditions and needs the hour rather than the day.


def would_succeed_at(
    customer: Customer,
    at: datetime,
    amount: float,
    balance_series: pd.Series,
    outages: BankOutageBook,
) -> bool:
    """Ground truth for a presentment at a specific HOUR.

    An AND of two independent conditions, and the conjunction is the point:

    * the bank must be up at ``at`` (hour resolution), and
    * the balance must cover ``amount`` on ``at.date()`` (daily resolution).

    A customer can be broke AND their bank down at the same moment. Waiting for
    the outage to clear does not put money in the account, and money in the
    account does not bring the bank back. Reporting a bank-clear window as
    recoverable when the balance would have failed anyway is exactly the error
    this signature exists to prevent.
    """
    if not outages.is_up(customer.bank_id, at):
        return False
    return presentment_succeeds(customer, at.date(), amount, balance_series)


def _window_slots(day: date) -> list[datetime]:
    """Presentment hours available on one day, inside the NPCI window."""
    start = rules.UPI_AUTOPAY_WINDOW_START.hour
    end = rules.UPI_AUTOPAY_WINDOW_END.hour
    return [datetime.combine(day, time(hour)) for hour in range(start, end)]


def first_success_datetime(
    customer: Customer,
    after: datetime,
    amount: float,
    balance_series: pd.Series,
    outages: BankOutageBook,
    horizon_days: int = 30,
) -> datetime | None:
    """Earliest legal presentment hour strictly after ``after`` that clears.

    Restricted to the NPCI 00:00-07:00 execution window, so this is the best a
    compliant policy could ever have done - not merely the first hour the bank
    happened to be up.

    The search is bounded rather than hour-by-hour across a year. Two shortcuts,
    both from the structure of the problem:

    * if the bank is down, jump straight to the end of that outage using the
      book rather than probing each hour inside it;
    * if the bank is up but the BALANCE is short, no other hour that day can
      help - the balance series is daily - so skip to the next day's window.
    """
    limit = datetime.combine(after.date() + timedelta(days=horizon_days), time(0, 0))
    series_end = balance_series.index[-1].date()

    day = after.date()
    while day <= series_end:
        if datetime.combine(day, time(0, 0)) > limit:
            return None

        # The balance is a daily fact: if it cannot cover the amount today,
        # no hour of today will, so do not scan hours.
        if not presentment_succeeds(customer, day, amount, balance_series):
            day += timedelta(days=1)
            continue

        for slot in _window_slots(day):
            if slot <= after:
                continue
            if outages.is_up(customer.bank_id, slot):
                return slot
            # Jump over the outage instead of probing every hour inside it.
            resume = outages.next_up_at_or_after(customer.bank_id, slot)
            if resume.date() == day:
                # Round up to the next whole presentment hour once the bank is
                # back, then re-check the window. The rounding can push past
                # 07:00, and the window end is EXCLUSIVE - returning 07:00
                # would be an illegal presentment dressed up as the answer.
                candidate = resume.replace(minute=0, second=0, microsecond=0)
                if candidate < resume:
                    candidate += timedelta(hours=1)
                if (
                    candidate > after
                    and rules.in_execution_window(candidate.time())
                    and outages.is_up(customer.bank_id, candidate)
                ):
                    return candidate
            break  # rest of today's window is inside the outage

        day += timedelta(days=1)

    return None
