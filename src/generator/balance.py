"""The hidden daily bank-balance process.

Every customer's current account is simulated day by day:

    salary lands (sometimes late) -> daily spend erodes it -> occasional
    shocks knock chunks out -> a per-customer floor stops it at their cushion

A small minority carry ``income_shock_start``: from that date the salary stops
and the cushion goes with it, so the balance decays to zero and stays there.
For those customers no retry date can ever succeed.

Everything is driven by ``customer.balance_seed``, so a given customer always
gets the same path. The same routine is vectorised over ``n_paths`` so the
oracle can Monte-Carlo the counterfactual distribution of that path.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .models import Customer

# --- process constants -------------------------------------------------------
# Days simulated before ``start_date`` and thrown away, so the returned window
# starts from a settled balance rather than an arbitrary initial value.
BURN_IN_DAYS: int = 60

SPEND_NOISE_SIGMA: float = 0.3  # lognormal sigma; mean of the noise is 1.0

SHOCK_PROB: float = 0.05  # chance of a large unexpected expense on any day
SHOCK_MIN_FRAC: float = 0.10  # as a fraction of monthly salary
SHOCK_MAX_FRAC: float = 0.30

# The floor is salary * buffer_ratio * U(FLOOR_JITTER_LOW, FLOOR_JITTER_HIGH):
# customers with buffer_ratio near zero genuinely run their account to empty.
FLOOR_JITTER_LOW: float = 0.1
FLOOR_JITTER_HIGH: float = 1.0

# Surplus above this multiple of salary is swept to savings when the next
# salary lands. Without it, low-burn customers accumulate without bound.
SWEEP_MULTIPLE: float = 2.5

INITIAL_BALANCE_FRAC: float = 0.5  # of salary, before burn-in settles it

# Offset added to balance_seed so oracle Monte-Carlo draws form an independent
# stream from the customer's own realised path.
COUNTERFACTUAL_SEED_SALT: int = 1_000_003


@dataclass(frozen=True)
class BalanceSimulation:
    """Result of simulating one or more paths for a single customer."""

    dates: pd.DatetimeIndex
    balances: np.ndarray  # shape (n_paths, n_days)
    salary_offsets: np.ndarray  # shape (n_paths, n_months), offsets into `dates`

    @property
    def n_paths(self) -> int:
        return int(self.balances.shape[0])

    def path(self, i: int = 0) -> pd.Series:
        """Path ``i`` as a date-indexed Series."""
        return pd.Series(self.balances[i], index=self.dates, name="balance")

    def salary_dates(self, i: int = 0) -> list[date]:
        """Realised (post-delay) salary landing dates on path ``i``."""
        offsets = self.salary_offsets[i]
        offsets = offsets[(offsets >= 0) & (offsets < len(self.dates))]
        return [self.dates[int(o)].date() for o in np.unique(offsets)]


def _months_spanned(start: date, n_days: int) -> list[tuple[int, int]]:
    """(year, month) pairs touched by the window, plus one month either side so
    salary credits near the edges are not lost."""
    end = start + timedelta(days=n_days - 1)
    y, m = (start.year - 1, 12) if start.month == 1 else (start.year, start.month - 1)
    stop_y, stop_m = (end.year + 1, 1) if end.month == 12 else (end.year, end.month + 1)
    out: list[tuple[int, int]] = []
    while (y, m) <= (stop_y, stop_m):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _nominal_salary_offsets(customer: Customer, sim_start: date, n_days: int) -> np.ndarray:
    """Day offsets of the scheduled salary date in each month spanned.

    A salary_day of 30 or 31 is clamped to the last day of short months, which
    is exactly how month-end Indian payroll behaves. Paydays on or after
    ``income_shock_start`` are dropped: the income has stopped.
    """
    offsets = []
    for year, month in _months_spanned(sim_start, n_days):
        last = calendar.monthrange(year, month)[1]
        day = min(customer.salary_day, last)
        payday = date(year, month, day)
        if customer.income_shock_start is not None and payday >= customer.income_shock_start:
            continue
        offsets.append((payday - sim_start).days)
    return np.asarray(offsets, dtype=np.int64)


def simulate(
    customer: Customer,
    start_date: date,
    days: int,
    n_paths: int = 1,
    seed: int | None = None,
) -> BalanceSimulation:
    """Simulate ``n_paths`` daily balance paths for ``customer``.

    ``seed`` defaults to ``customer.balance_seed``, which makes path 0 the
    customer's canonical, reproducible truth.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    rng = np.random.default_rng(customer.balance_seed if seed is None else seed)

    sim_start = start_date - timedelta(days=BURN_IN_DAYS)
    n_sim = BURN_IN_DAYS + days
    salary = customer.salary_amount

    # --- salary credits, with per-path delay draws ---------------------------
    nominal = _nominal_salary_offsets(customer, sim_start, n_sim)
    n_months = nominal.size
    is_late = rng.random((n_paths, n_months)) < customer.salary_delay_prob
    if customer.salary_delay_days_max > 0:
        delay = rng.integers(1, customer.salary_delay_days_max + 1, size=(n_paths, n_months))
    else:
        delay = np.zeros((n_paths, n_months), dtype=np.int64)
    offsets = nominal[None, :] + np.where(is_late, delay, 0)

    credits = np.zeros((n_paths, n_sim), dtype=np.float64)
    valid = (offsets >= 0) & (offsets < n_sim)
    path_idx = np.repeat(np.arange(n_paths)[:, None], n_months, axis=1)
    np.add.at(credits, (path_idx[valid], offsets[valid]), salary)

    # --- daily spend: lognormal noise with mean exactly 1.0 ------------------
    mu = -0.5 * SPEND_NOISE_SIGMA**2
    noise = rng.lognormal(mean=mu, sigma=SPEND_NOISE_SIGMA, size=(n_paths, n_sim))
    spend = salary * customer.burn_rate * noise

    # --- shocks --------------------------------------------------------------
    shock_hit = rng.random((n_paths, n_sim)) < SHOCK_PROB
    shock_size = rng.uniform(SHOCK_MIN_FRAC, SHOCK_MAX_FRAC, size=(n_paths, n_sim)) * salary
    shocks = np.where(shock_hit, shock_size, 0.0)

    # --- floor ---------------------------------------------------------------
    jitter = rng.uniform(FLOOR_JITTER_LOW, FLOOR_JITTER_HIGH, size=(n_paths, n_sim))
    floor = salary * customer.buffer_ratio * jitter
    if customer.income_shock_start is not None:
        # No income, no cushion. Without this the floor would keep an
        # unemployed customer solvent forever and no retry would ever be
        # genuinely hopeless.
        day_offsets = np.arange(n_sim)
        shock_offset = (customer.income_shock_start - sim_start).days
        floor[:, day_offsets >= shock_offset] = 0.0

    # --- roll it forward -----------------------------------------------------
    sweep_cap = salary * SWEEP_MULTIPLE
    balances = np.empty((n_paths, n_sim), dtype=np.float64)
    bal = np.full(n_paths, salary * INITIAL_BALANCE_FRAC, dtype=np.float64)
    for t in range(n_sim):
        credit_t = credits[:, t]
        bal = bal + credit_t
        # surplus swept to savings when salary lands
        bal = np.where(credit_t > 0, np.minimum(bal, sweep_cap), bal)
        bal = bal - spend[:, t] - shocks[:, t]
        bal = np.maximum(bal, floor[:, t])
        balances[:, t] = bal

    dates = pd.date_range(start=start_date, periods=days, freq="D")
    return BalanceSimulation(
        dates=dates,
        balances=balances[:, BURN_IN_DAYS:].copy(),
        salary_offsets=offsets - BURN_IN_DAYS,
    )


def daily_balance(customer: Customer, start_date: date, days: int) -> pd.Series:
    """The customer's canonical daily balance, as a date-indexed Series.

    Deterministic: same customer, same window, same numbers, every time.
    """
    return simulate(customer, start_date, days, n_paths=1).path(0)


def counterfactual_paths(
    customer: Customer, start_date: date, days: int, n_paths: int
) -> BalanceSimulation:
    """``n_paths`` alternative draws of the same process, for the oracle."""
    return simulate(
        customer,
        start_date,
        days,
        n_paths=n_paths,
        seed=customer.balance_seed + COUNTERFACTUAL_SEED_SALT,
    )
