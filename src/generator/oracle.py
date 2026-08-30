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

Note on AFA, which is the one place that principle needed rethinking
-------------------------------------------------------------------
Class 5 breaks the assumption every earlier class relied on: that the truth is
fixed and a policy can only estimate it. Above the RBI Sec 8(a) ceiling a debit
clears only if the customer completed authentication, and a NUDGE causally
raises the chance they did. So "would a debit on date D have cleared?" no
longer has one answer - it has one answer per number of nudges.

Pretending otherwise fails in both directions. Fix the oracle at zero nudges
and the ceiling drops below what a real compliant policy can reach, so the
agent could BEAT it and the scale stops meaning anything. Let it assume
unlimited nudging and the ceiling becomes unattainable, so every policy is
scored against a fiction. Class 2 already produced the first of those failures
- OraclePolicy scheduled into live outages because it used a balance-only
lookup - which is why this one is settled explicitly rather than left to fall
out.

The resolution, stated rather than derived:

* every truth function takes ``nudges`` and answers **"would this have cleared
  given the nudges that actually happened"**. It defaults to 0, which is the
  world the dataset was BILLED in: the merchant legacy schedule nudged nobody;
* the CEILING is separately allowed to nudge, up to ``afa.MAX_USEFUL_NUDGES``,
  because nudging is a lawful action available to every policy and a ceiling
  computed over a smaller action set than the agent has is not an upper bound;
* ``success_probability_true`` stays BALANCE-ONLY, like ``would_succeed``. It is
  the calibration target for the payday model, and folding an authentication
  hazard into it would grade that model on something it does not claim to
  predict.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from . import rules
from .afa import MAX_USEFUL_NUDGES, otp_completed
from .balance import counterfactual_paths
from .banks import BankOutageBook
from .models import Customer

# How much settled history to simulate before the date being asked about, on
# top of the balance module's own burn-in.
ORACLE_LOOKBACK_DAYS: int = 90

# Return code for each way a mandate can be structurally dead. Only the two
# NEVER_RETRY_CLASSES can appear here; ``Customer._known_dead_class`` enforces
# that at construction, so a missing key would be a taxonomy bug rather than a
# data problem and should raise.
_DEAD_CLASS_RETURN_CODES: dict[str, str] = {
    rules.FAILURE_CLASS_EXPIRED_CARD: rules.RETURN_CODE_EXPIRED_CARD,
    rules.FAILURE_CLASS_OPTED_OUT: rules.RETURN_CODE_MANDATE_REVOKED,
}


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


def resolve_presentment(
    customer: Customer,
    attempt_date: date,
    amount: float,
    balance_series: pd.Series,
    outages: BankOutageBook | None = None,
    nudges: int = 0,
) -> tuple[bool, str | None, str]:
    """Present the debit and report (succeeded, return_code, failure_class).

    THE canonical resolution order. Every simulated presentment - historical
    billing in the generator, and every retry the evaluator plays - goes
    through this one function, so the four gates cannot drift apart between
    the world the data was built in and the world a policy is scored in.

    The order is by WHAT ACTUALLY BLOCKED the presentment, not by what is
    cheapest to check:

    1. A DEAD MANDATE wins first. There is no credential to present, so the
       bank, the authentication and the balance are all irrelevant.
    2. A BANK OUTAGE at the presentment hour comes next. The debit never
       reaches the issuer, so nothing downstream of the issuer is even
       consulted - the acquirer returns 91, not 04 and not 1A.
    3. AFA comes before the balance. Above the RBI Sec 8(a) ceiling the
       authentication gate sits IN FRONT OF the account: an unauthenticated
       debit is refused before anybody reads the balance, so it returns 1A
       whatever the balance happened to be. Resolving the balance first would
       leak the account state into a decline that never touched the account,
       and would let a funded customer look like an authentication problem and
       a broke one look solvent.
    4. Otherwise the BALANCE decides: 04, or success.

    The return code is the only channel the agent gets. Everything behind it -
    the outage book, the balance series, the dead-mandate fields, the
    responsiveness trait - is hidden.

    ``nudges`` is how many out-of-band nudges preceded this presentment. It is
    the ONLY policy-dependent input, and it only ever moves gate 3.
    """
    if customer.mandate_dead_from is not None and attempt_date >= customer.mandate_dead_from:
        dead_class = customer.mandate_dead_class or rules.FAILURE_CLASS_AMBIGUOUS
        return False, _DEAD_CLASS_RETURN_CODES[dead_class], dead_class

    if outages is not None and outages.is_down(
        customer.bank_id, datetime.combine(attempt_date, rules.DEFAULT_PRESENTMENT_TIME)
    ):
        return False, rules.RETURN_CODE_DOWNTIME, rules.FAILURE_CLASS_DOWNTIME

    if not otp_completed(customer, attempt_date, nudges):
        return (
            False,
            rules.RETURN_CODE_AFA_INCOMPLETE,
            rules.FAILURE_CLASS_AFA_INCOMPLETE,
        )

    cleared = would_succeed(customer, attempt_date, amount, balance_series)
    return (
        cleared,
        None if cleared else rules.RETURN_CODE_INSUFFICIENT_FUNDS,
        rules.FAILURE_CLASS_INSUFFICIENT_FUNDS,
    )


def presentment_succeeds(
    customer: Customer,
    attempt_date: date,
    amount: float,
    balance_series: pd.Series,
    nudges: int = 0,
) -> bool:
    """Ground truth for a real presentment: every gate must be open.

    ``would_succeed`` answers only the funding half, which is what the payday
    model is graded against. This is the one every simulated presentment goes
    through, so a dead mandate can never be "recovered" by a later cycle
    happening to find money in the account, and an unauthenticated debit on a
    large mandate cannot be recovered by the salary landing.

    No outage book: the evaluator does not model hour-resolution outages inside
    an episode (see the class-2 finding that no compliant policy can present
    into the outage it is reacting to). ``resolve_presentment`` takes one when
    the caller has it.
    """
    return resolve_presentment(
        customer, attempt_date, amount, balance_series, nudges=nudges
    )[0]


def success_curve(
    customer: Customer,
    balance_series: pd.Series,
    amount: float,
    start: date | None = None,
    end: date | None = None,
    nudges: int = 0,
) -> pd.Series:
    """The full counterfactual surface: would a debit have cleared on each day?

    This is the object a retry policy is graded against - every date it could
    have chosen, with the true answer attached.

    Every gate is ANDed in, not just the balance. A date on which the money was
    there but the mandate was dead, or the authentication was never completed,
    is not a date on which the debit would have cleared, and reporting it as
    one would credit a policy for money it could never have collected.

    ``nudges`` is the count assumed for every date in the window. See the
    module docstring: 0 is the world as billed, ``afa.MAX_USEFUL_NUDGES`` is
    the ceiling action set.
    """
    window = balance_series
    if start is not None:
        window = window.loc[pd.Timestamp(start) :]
    if end is not None:
        window = window.loc[: pd.Timestamp(end)]
    curve = window >= amount
    if customer.mandate_dead_from is not None:
        curve &= window.index < pd.Timestamp(customer.mandate_dead_from)
    if customer.requires_otp:
        # Only walked for OTP-gated mandates, which are ~3% of the book. Below
        # the Sec 8(a) ceiling there is nothing to complete and the term is
        # identically True, so paying for the loop everywhere would be waste.
        curve &= pd.Series(
            [otp_completed(customer, stamp.date(), nudges) for stamp in window.index],
            index=window.index,
        )
    return curve.rename("would_succeed")


def first_success_date(
    customer: Customer,
    balance_series: pd.Series,
    amount: float,
    after: date,
    horizon_days: int = 60,
    nudges: int = 0,
) -> date | None:
    """Earliest date strictly after ``after`` on which a debit would clear.

    The oracle answer to "what was the best possible retry date". Returns None
    if no date inside the horizon clears every gate.

    Callers asking the RECOVERABILITY question - could any compliant policy
    have saved this mandate? - should pass ``nudges=afa.MAX_USEFUL_NUDGES``.
    Leaving it at 0 there would mark an AFA customer unrecoverable whom a
    single message would have rescued, and then score the agent CANCEL as
    correct.
    """
    start = after + timedelta(days=1)
    end = min(
        after + timedelta(days=horizon_days), balance_series.index[-1].date()
    )
    if start > end:
        return None
    curve = success_curve(
        customer, balance_series, amount, start=start, end=end, nudges=nudges
    )
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
    nudges: int = 0,
) -> bool:
    """Ground truth for a presentment at a specific HOUR.

    An AND of every independent condition, and the conjunction is the point:

    * the mandate must be alive on ``at.date()``, and
    * the bank must be up at ``at`` (hour resolution), and
    * AFA must be complete for that presentment, if the amount needs it, and
    * the balance must cover ``amount`` on ``at.date()`` (daily resolution).

    None of these implies any other, and every pair can disagree. A customer
    can be broke AND their bank down. They can have authenticated and still be
    broke, or be flush and never have authenticated. Waiting out an outage does
    not put money in the account; money in the account does not authenticate a
    debit; and an OTP completed today does nothing for next month, because
    Sec 8(a) exempts AFA per TRANSACTION.

    Reporting any one clear condition as "recoverable" while another was
    blocking is exactly the error this signature exists to prevent, and it is
    an error the class-5 tests probe as an explicit three-way overlap rather
    than trusting the AND to fall out.
    """
    if not outages.is_up(customer.bank_id, at):
        return False
    return presentment_succeeds(
        customer, at.date(), amount, balance_series, nudges=nudges
    )


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
    nudges: int = 0,
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
      help - the balance series is daily - so skip to the next day window. The
      AFA gate is a daily fact too (one draw per presentment date), so it is
      checked in the same place and skipped over the same way.
    """
    limit = datetime.combine(after.date() + timedelta(days=horizon_days), time(0, 0))
    series_end = balance_series.index[-1].date()

    day = after.date()
    while day <= series_end:
        if datetime.combine(day, time(0, 0)) > limit:
            return None

        # The balance is a daily fact: if it cannot cover the amount today,
        # no hour of today will, so do not scan hours.
        if not presentment_succeeds(
            customer, day, amount, balance_series, nudges=nudges
        ):
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
