"""Draws a realistic Indian subscriber population.

The India-specific signal that makes this problem interesting lives here: Indian
payroll clusters hard at month end and in the first week (the Payment of Wages
Act requires wages by the 7th, the 10th for larger establishments), while
merchants set mandate debit days arbitrarily across the month. That mismatch -
not randomness - is what produces most insufficient-funds failures, and it is
what a smart sequencer can exploit.

Every distribution below is a tunable knob; the regulatory constants live in
rules.py and are not.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from . import banks, rules
from .models import Customer

# --- salary day --------------------------------------------------------------
# 30% day 1 | 25% month end (30/31) | 15% days 2-3 | 12% day 5 | 10% day 7 |
# 8% days 8-10. Days 30/31 clamp to the last day of short months at simulation
# time, so "31" really means "month end".
SALARY_DAY_PMF: dict[int, float] = {
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
MONTH_END_SALARY_DAYS: frozenset[int] = frozenset({1, 30, 31})

# --- debit day ---------------------------------------------------------------
# Uniform over 1-28: merchants pick the signup date, which carries no
# information about payday. This is the mismatch that causes failures.
DEBIT_DAY_MIN: int = 1
DEBIT_DAY_MAX: int = 28

# --- subscription amount tiers (probability, low, high) ----------------------
AMOUNT_TIERS: tuple[tuple[float, float, float], ...] = (
    (0.40, 99.0, 299.0),        # streaming, apps
    (0.30, 300.0, 999.0),       # SaaS, fitness
    (0.20, 1_000.0, 4_999.0),   # larger services
    (0.10, 5_000.0, 20_000.0),  # some above the Rs 15,000 OTP threshold
)

# --- rail --------------------------------------------------------------------
RAIL_PMF: dict[str, float] = {"upi_autopay": 0.70, "nach": 0.30}

# --- tenure ------------------------------------------------------------------
TENURE_MEAN_MONTHS: float = 8.0
TENURE_MIN: int = 1
TENURE_MAX: int = 48

# --- income ------------------------------------------------------------------
SALARY_MEDIAN_INR: float = 35_000.0
SALARY_LOG_SIGMA: float = 0.6
SALARY_MIN_INR: float = 15_000.0
SALARY_MAX_INR: float = 600_000.0
# Nobody puts a fifth of their salary on a single subscription mandate.
MIN_SALARY_TO_SUBSCRIPTION_MULTIPLE: float = 8.0

SALARY_DELAY_PROB_BETA: tuple[float, float] = (2.0, 11.0)  # mean ~0.15
SALARY_DELAY_DAYS_PMF: dict[int, float] = {
    1: 0.30, 2: 0.25, 3: 0.18, 4: 0.12, 5: 0.08, 6: 0.04, 7: 0.03,
}

# --- spending ----------------------------------------------------------------
# burn_rate is the fraction of salary spent per day. Shocks add roughly another
# 1% of salary per day on average, so a burn_rate near 0.023 is a break-even
# household; above it they run dry before the next payday.
#
# The SPREAD matters more than the median, and is the parameter that controls
# how hard the retry problem is. A customer burning 0.030/day exhausts their
# salary around day 25 and is dry for the last ~5 days of the cycle; one
# burning 0.090/day is dry from about day 11. Failures can only occur inside
# that dry window, so the width of this distribution sets the distribution of
# "days from failure to next payday" - and therefore how often a fixed
# day-2/4/7 schedule can reach the money at all. A narrow spread clusters every
# failure a few days before payday, where day 7 catches it by luck.
BURN_RATE_MEDIAN: float = 0.026
BURN_RATE_LOG_SIGMA: float = 0.55
BURN_RATE_MIN: float = 0.010
BURN_RATE_MAX: float = 0.120

# --- income interruption -----------------------------------------------------
# A minority genuinely lose their income inside the observed window. For them
# no retry date succeeds, so CANCEL/HOLD is the only correct action. Without
# this population a stopping rule can never be wrong, and therefore can never
# be tested. The shock is placed far enough before `as_of` that the customer is
# already dry at the decision point and the run of failures is visible in
# payment history - which is the only evidence the agent gets.
INCOME_SHOCK_SHARE: float = 0.04
INCOME_SHOCK_EARLIEST_DAYS_BEFORE_AS_OF: int = 240
INCOME_SHOCK_LATEST_DAYS_BEFORE_AS_OF: int = 40

# --- mandates that die for non-financial reasons ----------------------------
# A card expires, or a customer revokes consent at a pre-debit notice. From
# that date the mandate fails on EVERY date regardless of the balance, with a
# return code that says so.
#
# These shares are small because they are shares of the WHOLE BOOK, while the
# target is a share of FAILURES. Affected customers fail every cycle, so a
# ~1% book share becomes a ~6% share of failures against a ~14% natural
# insufficient-funds rate. scripts/run_multiclass.py reports the realised mix.
CARD_EXPIRY_SHARE: float = 0.010
CONSENT_REVOKED_SHARE: float = 0.005
MANDATE_DEATH_EARLIEST_DAYS_BEFORE_AS_OF: int = 200
MANDATE_DEATH_LATEST_DAYS_BEFORE_AS_OF: int = 35

# Beta with alpha < 1 puts real mass at zero: plenty of customers genuinely run
# their account to empty, while others keep a cushion.
BUFFER_RATIO_BETA: tuple[float, float] = (0.5, 14.0)  # mean ~0.034
BUFFER_RATIO_MAX: float = 0.40

# --- behaviour ---------------------------------------------------------------
OPTOUT_SENSITIVITY_BETA: tuple[float, float] = (1.5, 48.5)  # mean ~0.03

_SEED_SPACE: int = 2**31 - 1


def _pmf_choice(rng: np.random.Generator, pmf: dict, size: int) -> np.ndarray:
    keys = list(pmf.keys())
    probs = np.asarray(list(pmf.values()), dtype=float)
    return rng.choice(keys, size=size, p=probs / probs.sum())


def draw_salary_day(rng: np.random.Generator, n: int) -> np.ndarray:
    return _pmf_choice(rng, SALARY_DAY_PMF, n).astype(int)


def draw_debit_day(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(DEBIT_DAY_MIN, DEBIT_DAY_MAX + 1, size=n)


def draw_subscription_amount(rng: np.random.Generator, n: int) -> np.ndarray:
    probs = np.asarray([t[0] for t in AMOUNT_TIERS])
    tier_idx = rng.choice(len(AMOUNT_TIERS), size=n, p=probs / probs.sum())
    lows = np.asarray([t[1] for t in AMOUNT_TIERS])[tier_idx]
    highs = np.asarray([t[2] for t in AMOUNT_TIERS])[tier_idx]
    return np.round(rng.uniform(lows, highs))  # subscription prices are whole rupees


def draw_rail(rng: np.random.Generator, n: int) -> np.ndarray:
    return _pmf_choice(rng, RAIL_PMF, n)


def draw_bank_id(rng: np.random.Generator, n: int) -> np.ndarray:
    """Assign each customer an issuing bank from the book-share distribution.

    Concentration matters here, not just variety: most customers sit on a
    handful of large issuers, so an outage at one of those hits many mandates
    at once - which is what makes the shared signal worth learning.
    """
    return _pmf_choice(rng, banks.BANK_SHARES, n)


def draw_tenure(rng: np.random.Generator, n: int) -> np.ndarray:
    tenure = rng.exponential(scale=TENURE_MEAN_MONTHS, size=n) + 1.0
    return np.clip(np.floor(tenure), TENURE_MIN, TENURE_MAX).astype(int)


def draw_salary_amount(
    rng: np.random.Generator, n: int, subscription_amounts: np.ndarray
) -> np.ndarray:
    salary = rng.lognormal(
        mean=np.log(SALARY_MEDIAN_INR), sigma=SALARY_LOG_SIGMA, size=n
    )
    salary = np.clip(salary, SALARY_MIN_INR, SALARY_MAX_INR)
    floor = subscription_amounts * MIN_SALARY_TO_SUBSCRIPTION_MULTIPLE
    return np.round(np.maximum(salary, floor), 2)


def draw_burn_rate(rng: np.random.Generator, n: int) -> np.ndarray:
    burn = rng.lognormal(
        mean=np.log(BURN_RATE_MEDIAN), sigma=BURN_RATE_LOG_SIGMA, size=n
    )
    return np.clip(burn, BURN_RATE_MIN, BURN_RATE_MAX)


def draw_buffer_ratio(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.clip(rng.beta(*BUFFER_RATIO_BETA, size=n), 0.0, BUFFER_RATIO_MAX)


def draw_optout_sensitivity(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.beta(*OPTOUT_SENSITIVITY_BETA, size=n)


def draw_income_shock_start(
    rng: np.random.Generator, n: int, as_of: date
) -> list[date | None]:
    """Dates on which income ceases, or None for the vast majority."""
    hit = rng.random(n) < INCOME_SHOCK_SHARE
    days_before = rng.integers(
        INCOME_SHOCK_LATEST_DAYS_BEFORE_AS_OF,
        INCOME_SHOCK_EARLIEST_DAYS_BEFORE_AS_OF + 1,
        size=n,
    )
    return [
        as_of - timedelta(days=int(days_before[i])) if hit[i] else None
        for i in range(n)
    ]


def draw_mandate_death(
    rng: np.random.Generator, n: int, as_of: date
) -> list[tuple[date | None, str | None]]:
    """When (and how) each mandate stopped being presentable at all.

    Returns (date, class) pairs, both None for the overwhelming majority.
    """
    roll = rng.random(n)
    days_before = rng.integers(
        MANDATE_DEATH_LATEST_DAYS_BEFORE_AS_OF,
        MANDATE_DEATH_EARLIEST_DAYS_BEFORE_AS_OF + 1,
        size=n,
    )
    out: list[tuple[date | None, str | None]] = []
    for i in range(n):
        when = as_of - timedelta(days=int(days_before[i]))
        if roll[i] < CARD_EXPIRY_SHARE:
            out.append((when, rules.FAILURE_CLASS_EXPIRED_CARD))
        elif roll[i] < CARD_EXPIRY_SHARE + CONSENT_REVOKED_SHARE:
            out.append((when, rules.FAILURE_CLASS_OPTED_OUT))
        else:
            out.append((None, None))
    return out


def make_population(
    n_customers: int, rng: np.random.Generator, as_of: date
) -> list[Customer]:
    """Draw ``n_customers`` subscribers, hidden parameters and all.

    ``as_of`` is the decision date the dataset is anchored on. It is needed
    here only to place income shocks on absolute dates, so that a customer's
    shock lands at the same real date no matter which window is simulated -
    otherwise ``would_succeed`` and ``success_probability_true`` would disagree.
    """
    if n_customers < 1:
        raise ValueError("n_customers must be >= 1")

    amounts = draw_subscription_amount(rng, n_customers)
    rails = draw_rail(rng, n_customers)
    bank_ids = draw_bank_id(rng, n_customers)
    debit_days = draw_debit_day(rng, n_customers)
    tenures = draw_tenure(rng, n_customers)
    salary_days = draw_salary_day(rng, n_customers)
    salaries = draw_salary_amount(rng, n_customers, amounts)
    delay_probs = rng.beta(*SALARY_DELAY_PROB_BETA, size=n_customers)
    delay_days = _pmf_choice(rng, SALARY_DELAY_DAYS_PMF, n_customers).astype(int)
    burn_rates = draw_burn_rate(rng, n_customers)
    buffer_ratios = draw_buffer_ratio(rng, n_customers)
    optout = draw_optout_sensitivity(rng, n_customers)
    shock_starts = draw_income_shock_start(rng, n_customers, as_of)
    dead_mandates = draw_mandate_death(rng, n_customers, as_of)
    seeds = rng.integers(0, _SEED_SPACE, size=n_customers)

    width = max(4, len(str(n_customers)))
    return [
        Customer(
            customer_id=f"CUST_{i + 1:0{width}d}",
            subscription_amount=float(amounts[i]),
            rail=str(rails[i]),
            bank_id=str(bank_ids[i]),
            debit_day_of_month=int(debit_days[i]),
            tenure_months=int(tenures[i]),
            monthly_ltv=float(amounts[i]) * 12.0,
            salary_day=int(salary_days[i]),
            salary_amount=float(salaries[i]),
            salary_delay_prob=float(delay_probs[i]),
            salary_delay_days_max=int(delay_days[i]),
            income_shock_start=shock_starts[i],
            mandate_dead_from=dead_mandates[i][0],
            mandate_dead_class=dead_mandates[i][1],
            burn_rate=float(burn_rates[i]),
            buffer_ratio=float(buffer_ratios[i]),
            optout_sensitivity=float(optout[i]),
            balance_seed=int(seeds[i]),
        )
        for i in range(n_customers)
    ]
