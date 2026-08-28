"""Tests for the synthetic failure-data generator.

Grouped by the property under test: reproducibility, population realism,
balance-process sanity, oracle correctness, natural failure rate, agent/truth
separation, and opt-out monotonicity.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from generator.balance import daily_balance, simulate
from generator.models import Customer

AS_OF = date(2026, 8, 1)


def make_customer(**overrides) -> Customer:
    """A fixed customer, so balance tests do not depend on the population."""
    base = dict(
        customer_id="CUST_0001",
        subscription_amount=499.0,
        rail="nach",
        debit_day_of_month=12,
        tenure_months=9,
        monthly_ltv=None,
        salary_day=1,
        salary_amount=40_000.0,
        salary_delay_prob=0.0,
        salary_delay_days_max=0,
        burn_rate=0.023,
        buffer_ratio=0.04,
        optout_sensitivity=0.03,
        balance_seed=7,
    )
    base.update(overrides)
    return Customer(**base)


# --- 3. balance sanity -------------------------------------------------------


def test_balance_peaks_within_one_day_of_salary_and_then_declines():
    """Money arrives on payday and leaves afterwards."""
    start = date(2025, 8, 1)
    for seed in range(12):
        customer = make_customer(balance_seed=seed)
        sim = simulate(customer, start, days=365)
        series = sim.path(0)

        for pay_day in sim.salary_dates(0):
            window_start = pay_day - timedelta(days=3)
            window_end = pay_day + timedelta(days=6)
            if window_start < start or window_end > series.index[-1].date():
                continue
            window = series.loc[str(window_start) : str(window_end)]
            peak = window.idxmax().date()
            assert abs((peak - pay_day).days) <= 1, (
                f"seed {seed}: peak {peak} is not within a day of payday {pay_day}"
            )
            # A week later they are strictly poorer (only salary adds money).
            assert series[str(pay_day + timedelta(days=6))] < series[str(pay_day)]


def test_balance_respects_the_customer_floor():
    customer = make_customer(buffer_ratio=0.10)
    series = daily_balance(customer, date(2025, 8, 1), 365)
    lowest_possible = customer.salary_amount * customer.buffer_ratio * 0.3
    assert series.min() >= lowest_possible - 1e-9


def test_zero_buffer_customer_runs_the_account_to_empty():
    customer = make_customer(buffer_ratio=0.0, burn_rate=0.04)
    series = daily_balance(customer, date(2025, 8, 1), 365)
    assert series.min() < 1.0


def test_salary_delay_shifts_the_landing_date():
    always_late = make_customer(salary_delay_prob=1.0, salary_delay_days_max=5)
    sim = simulate(always_late, date(2025, 8, 1), days=365)
    for pay_day in sim.salary_dates(0):
        assert pay_day.day != always_late.salary_day


# --- 1. reproducibility (balance layer) --------------------------------------


def test_daily_balance_is_deterministic():
    customer = make_customer()
    first = daily_balance(customer, date(2025, 8, 1), 365)
    second = daily_balance(customer, date(2025, 8, 1), 365)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


def test_different_seeds_give_different_paths():
    a = daily_balance(make_customer(balance_seed=1), date(2025, 8, 1), 90)
    b = daily_balance(make_customer(balance_seed=2), date(2025, 8, 1), 90)
    assert not np.array_equal(a.to_numpy(), b.to_numpy())


def test_spend_noise_has_unit_mean():
    """burn_rate must mean what it says: the average daily spend is
    salary * burn_rate, so the noise multiplier cannot be biased."""
    customer = make_customer(
        buffer_ratio=0.0, salary_delay_prob=0.0, salary_delay_days_max=0
    )
    sim = simulate(customer, date(2025, 8, 1), days=365, n_paths=4000)
    # Reconstruct spend on a day with no salary credit and no floor binding.
    # Easier: check the process drains at roughly the expected rate over a
    # month, allowing for shocks (mean 0.05 * 0.20 = 1% of salary per day).
    expected_daily_drain = customer.salary_amount * (customer.burn_rate + 0.05 * 0.20)
    day10 = sim.balances[:, 9].mean()
    day20 = sim.balances[:, 19].mean()
    observed = (day10 - day20) / 10.0
    assert observed == pytest.approx(expected_daily_drain, rel=0.10)


# --- 4. oracle consistency ---------------------------------------------------


def test_would_succeed_agrees_with_the_balance_series():
    from generator.oracle import would_succeed

    customer = make_customer()
    start = date(2025, 8, 1)
    series = daily_balance(customer, start, 365)
    amount = customer.subscription_amount

    for offset in range(0, 365, 7):
        day = start + timedelta(days=offset)
        assert would_succeed(customer, day, amount, series) == (
            series[str(day)] >= amount
        )


def test_would_succeed_is_deterministic_across_regenerated_series():
    from generator.oracle import would_succeed

    customer = make_customer()
    start = date(2025, 8, 1)
    day = start + timedelta(days=200)
    a = daily_balance(customer, start, 365)
    b = daily_balance(customer, start, 365)
    assert would_succeed(customer, day, 5_000.0, a) is would_succeed(
        customer, day, 5_000.0, b
    )


def test_would_succeed_rejects_dates_outside_the_window():
    from generator.oracle import would_succeed

    customer = make_customer()
    series = daily_balance(customer, date(2025, 8, 1), 30)
    with pytest.raises(ValueError, match="outside the simulated window"):
        would_succeed(customer, date(2026, 1, 1), 100.0, series)


def test_success_probability_is_deterministic_and_ordered_by_amount():
    from generator.oracle import success_probability_true

    customer = make_customer()
    day = date(2026, 1, 20)  # deep in the cycle, money is tight
    p_small = success_probability_true(customer, day, 199.0, n_sims=400)
    assert p_small == success_probability_true(customer, day, 199.0, n_sims=400)
    p_large = success_probability_true(customer, day, 12_000.0, n_sims=400)
    assert 0.0 <= p_large <= p_small <= 1.0


def test_success_probability_is_higher_just_after_payday():
    from generator.oracle import success_probability_true

    customer = make_customer(salary_day=1)
    amount = 15_000.0
    after_payday = success_probability_true(customer, date(2026, 1, 2), amount, n_sims=600)
    before_payday = success_probability_true(customer, date(2025, 12, 30), amount, n_sims=600)
    assert after_payday > before_payday


def test_success_curve_and_first_success_date_agree():
    from generator.oracle import first_success_date, success_curve

    amount = 8_000.0
    customer = make_customer(subscription_amount=amount, monthly_ltv=None)
    start = date(2025, 8, 1)
    series = daily_balance(customer, start, 365)

    shortfalls = series[series < amount]
    assert not shortfalls.empty, "expected this customer to run short at least once"
    failed_on = shortfalls.index[0].date()

    best = first_success_date(customer, series, amount, after=failed_on)
    assert best is not None
    curve = success_curve(
        customer, series, amount, start=failed_on + timedelta(days=1), end=best
    )
    assert curve.iloc[-1]
    assert not curve.iloc[:-1].any(), "an earlier date would also have cleared"


def test_next_salary_date_clamps_month_end():
    from generator.oracle import days_to_next_salary, next_salary_date

    month_end = make_customer(salary_day=31)
    assert next_salary_date(month_end, date(2026, 2, 10)) == date(2026, 2, 28)
    assert next_salary_date(month_end, date(2026, 2, 28)) == date(2026, 3, 31)
    assert days_to_next_salary(month_end, date(2026, 2, 10)) == 18


# --- 2. salary distribution --------------------------------------------------


def test_salary_days_cluster_at_month_end():
    """The core India-specific signal: Indian payroll clusters at month end.

    ~55% of customers should be paid on day 1, 30 or 31.
    """
    import numpy as np_

    from generator.population import MONTH_END_SALARY_DAYS, make_population

    population = make_population(4000, np_.random.default_rng(42), as_of=AS_OF)
    share = np_.mean([c.salary_day in MONTH_END_SALARY_DAYS for c in population])
    assert 0.50 <= share <= 0.60, f"month-end share was {share:.3f}"


def test_salary_days_stay_inside_the_indian_payroll_bands():
    import numpy as np_

    from generator.population import make_population

    for customer in make_population(1000, np_.random.default_rng(7), as_of=AS_OF):
        assert 1 <= customer.salary_day <= 10 or 25 <= customer.salary_day <= 31


def test_population_matches_the_specified_mix():
    import numpy as np_

    from generator.population import make_population

    population = make_population(4000, np_.random.default_rng(1), as_of=AS_OF)
    rail_share = np_.mean([c.rail == "upi_autopay" for c in population])
    assert rail_share == pytest.approx(0.70, abs=0.03)

    debit_days = np_.array([c.debit_day_of_month for c in population])
    assert debit_days.min() >= 1 and debit_days.max() <= 28

    amounts = np_.array([c.subscription_amount for c in population])
    assert np_.mean(amounts < 300) == pytest.approx(0.40, abs=0.03)
    assert np_.mean((amounts >= 300) & (amounts < 1000)) == pytest.approx(0.30, abs=0.03)
    assert np_.mean((amounts >= 1000) & (amounts < 5000)) == pytest.approx(0.20, abs=0.03)
    assert np_.mean(amounts >= 5000) == pytest.approx(0.10, abs=0.03)

    tenures = np_.array([c.tenure_months for c in population])
    assert 6.5 <= tenures.mean() <= 9.5
    assert tenures.min() >= 1 and tenures.max() <= 48

    sensitivity = np_.array([c.optout_sensitivity for c in population])
    assert sensitivity.mean() == pytest.approx(0.03, abs=0.01)


def test_population_is_reproducible():
    import numpy as np_

    from generator.population import make_population

    a = make_population(200, np_.random.default_rng(99), as_of=AS_OF)
    b = make_population(200, np_.random.default_rng(99), as_of=AS_OF)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]


# --- 7. opt-out monotonicity -------------------------------------------------


def test_optout_probability_rises_with_each_extra_notification():
    from generator.optout import optout_probability

    customer = make_customer(optout_sensitivity=0.03, tenure_months=8)
    probs = [
        optout_probability(customer, n, days_since_first_failure=3.0)
        for n in range(1, 5)
    ]
    assert probs == sorted(probs)
    assert all(b > a for a, b in zip(probs, probs[1:])), probs


def test_no_notification_means_no_cancel_opportunity():
    from generator.optout import optout_probability

    assert optout_probability(make_customer(), 0, 0.0) == 0.0


def test_longer_tenure_customers_are_stickier():
    from generator.optout import optout_probability

    new = make_customer(tenure_months=1)
    loyal = make_customer(tenure_months=48)
    assert optout_probability(loyal, 2, 3.0) < optout_probability(new, 2, 3.0)


def test_waiting_longer_after_the_failure_is_less_irritating():
    from generator.optout import optout_probability

    customer = make_customer()
    immediate = optout_probability(customer, 2, days_since_first_failure=0.0)
    patient = optout_probability(customer, 2, days_since_first_failure=21.0)
    assert immediate > patient


def test_optout_probability_is_capped():
    from generator.optout import OPTOUT_CEILING, optout_probability

    twitchy = make_customer(optout_sensitivity=0.9, tenure_months=1)
    assert optout_probability(twitchy, 4, 0.0) == OPTOUT_CEILING


def test_draw_optout_matches_its_probability():
    from generator.optout import draw_optout, optout_probability

    customer = make_customer(optout_sensitivity=0.10)
    rng = np.random.default_rng(0)
    p = optout_probability(customer, 3, 1.0)
    draws = [draw_optout(customer, 3, 1.0, rng) for _ in range(20_000)]
    assert np.mean(draws) == pytest.approx(p, abs=0.01)


def test_survival_probability_falls_as_the_schedule_gets_denser():
    from generator.optout import survival_probability

    customer = make_customer()
    patient = survival_probability(customer, [14.0])
    aggressive = survival_probability(customer, [1.0, 2.0, 3.0])
    assert aggressive < patient < 1.0


# --- income shock: the population that makes a stopping rule testable --------


def test_income_shock_stops_salary_and_the_cushion():
    shocked = make_customer(
        buffer_ratio=0.15,  # a real cushion, which must NOT survive the shock
        income_shock_start=date(2025, 11, 1),
    )
    sim = simulate(shocked, date(2025, 8, 1), days=365)
    series = sim.path(0)

    assert all(day < date(2025, 11, 1) for day in sim.salary_dates(0))
    assert series[str(date(2025, 10, 15))] > 1_000.0
    # Given no income and no floor, the account decays to nothing and stays.
    assert series[str(date(2026, 1, 1))] < 1.0
    assert series.loc["2026-01-01":].max() < 1.0


def test_no_date_clears_for_an_income_shocked_customer():
    """This is the whole point of the shock population: CANCEL becomes correct."""
    from generator.oracle import first_success_date

    shocked = make_customer(
        subscription_amount=499.0, monthly_ltv=None, income_shock_start=date(2025, 10, 1)
    )
    series = daily_balance(shocked, date(2025, 8, 1), 365)
    failed_on = date(2025, 12, 12)
    assert not (series[str(failed_on)] >= 499.0)
    assert first_success_date(shocked, series, 499.0, after=failed_on, horizon_days=60) is None


def test_unshocked_customers_are_unaffected():
    plain = make_customer(income_shock_start=None)
    sim = simulate(plain, date(2025, 8, 1), days=365)
    assert len(sim.salary_dates(0)) >= 11


def test_income_shock_lands_on_the_same_real_date_in_any_window():
    """The shock is an absolute date, not a window offset.

    would_succeed reads the dataset window while success_probability_true
    simulates its own 90-day window. If the shock were window-relative the two
    would disagree about whether this customer still had a job.
    """
    from generator.balance import counterfactual_paths

    shock = date(2025, 12, 1)
    shocked = make_customer(income_shock_start=shock)

    wide = simulate(shocked, date(2025, 8, 1), days=365)
    narrow = simulate(shocked, date(2025, 11, 1), days=120)
    assert all(day < shock for day in wide.salary_dates(0))
    assert all(day < shock for day in narrow.salary_dates(0))
    assert any(day < shock for day in wide.salary_dates(0)), "income before the shock"

    # Every Monte-Carlo path agrees too, whatever the delay draws do.
    paths = counterfactual_paths(shocked, date(2025, 11, 1), 120, n_paths=200)
    for i in range(paths.n_paths):
        assert all(day < shock for day in paths.salary_dates(i))


# --- the irritation-direction switch -----------------------------------------


def test_both_irritation_directions_are_monotone_in_notifications():
    from generator.optout import optout_probability

    customer = make_customer(optout_sensitivity=0.03, tenure_months=8)
    for direction in ("decay", "build"):
        probs = [
            optout_probability(customer, n, 3.0, direction) for n in range(1, 5)
        ]
        assert all(b > a for a, b in zip(probs, probs[1:])), (direction, probs)


def test_flipping_the_irritation_direction_inverts_the_time_preference():
    """The assumption that makes waiting cheap must be a switch, because
    flipping it inverts the central trade-off of the project."""
    from generator.optout import optout_probability

    customer = make_customer()
    soon, late = 0.0, 21.0

    decay_soon = optout_probability(customer, 2, soon, "decay")
    decay_late = optout_probability(customer, 2, late, "decay")
    build_soon = optout_probability(customer, 2, soon, "build")
    build_late = optout_probability(customer, 2, late, "build")

    assert decay_soon > decay_late, "under 'decay', waiting should be cheaper"
    assert build_late > build_soon, "under 'build', waiting should be dearer"
    # The two directions are exact reflections: at any lag their recency
    # multipliers sum to a constant, so the ablation changes which end of the
    # time axis is expensive and nothing else.
    from generator.optout import RECENCY_WEIGHT, recency_multiplier

    for days in (0.0, 3.0, 7.0, 21.0):
        total = recency_multiplier(days, "decay") + recency_multiplier(days, "build")
        assert total == pytest.approx(2.0 + RECENCY_WEIGHT, rel=1e-12)


def test_recency_multiplier_is_bounded():
    from generator.optout import RECENCY_WEIGHT, recency_multiplier

    for direction in ("decay", "build"):
        for days in (0.0, 1.0, 7.0, 30.0, 365.0):
            m = recency_multiplier(days, direction)
            assert 1.0 <= m <= 1.0 + RECENCY_WEIGHT + 1e-12


def test_unknown_irritation_direction_is_rejected():
    from generator.optout import recency_multiplier

    with pytest.raises(ValueError, match="decay"):
        recency_multiplier(1.0, "sideways")


# --- regulatory rules --------------------------------------------------------


def test_otp_thresholds_follow_rbi_sec_8():
    from generator import rules

    assert not rules.requires_otp(15_000.0)
    assert rules.requires_otp(15_000.01)
    # Sec 8(b): the ceiling rises to Rs 1,00,000 for the named categories.
    assert not rules.requires_otp(50_000.0, category="insurance")
    assert not rules.requires_otp(1_00_000.0, category="mutual_fund")
    assert rules.requires_otp(1_00_000.01, category="credit_card_bill")


def test_upi_execution_window_is_enforced():
    from datetime import time

    from generator import rules

    assert rules.in_execution_window(time(2, 30))
    assert rules.in_execution_window(time(0, 0))
    assert not rules.in_execution_window(time(7, 0))
    assert not rules.in_execution_window(time(13, 0))
    assert rules.in_execution_window(time(13, 0), rail="nach")


def test_two_failure_classes_can_never_be_retried():
    from generator import rules

    assert not rules.can_retry(rules.FAILURE_CLASS_EXPIRED_CARD)
    assert not rules.can_retry(rules.FAILURE_CLASS_OPTED_OUT)
    assert rules.can_retry(rules.FAILURE_CLASS_INSUFFICIENT_FUNDS)
    assert len(rules.FAILURE_CLASSES) == 6
    with pytest.raises(ValueError, match="unknown failure class"):
        rules.can_retry("made_up")


def test_bounce_charge_is_never_blended_across_rails():
    from generator import rules

    assert rules.bounce_charge("nach") == 500.0
    assert rules.bounce_charge("upi_autopay") == 0.0
