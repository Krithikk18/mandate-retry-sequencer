"""Tests for the payday inference model (agent layer 1)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from agent.payday import (
    INDIAN_PAYDAY_PRIOR,
    PaydayModel,
    days_since_payday,
    liquidity_at,
)
from generator.models import AgentPaymentHistoryRecord

CUSTOMER = "CUST_0001"


def record(day: date, succeeded: bool, amount: float = 500.0):
    return AgentPaymentHistoryRecord(
        customer_id=CUSTOMER,
        attempt_date=day,
        amount=amount,
        succeeded=succeeded,
        return_code=None if succeeded else "04",
    )


def history_for_payday(
    payday: int,
    debit_day: int,
    months: int = 12,
    amount: float = 500.0,
    start: date = date(2025, 8, 1),
    dead_from: int | None = None,
) -> list[AgentPaymentHistoryRecord]:
    """Synthesise a clean history for a customer paid on ``payday``.

    A debit clears if it lands within the two weeks after payday. Retries walk
    the legacy +2/+4/+7 ladder, exactly as the real dataset does, so the model
    sees the same sparse and biased evidence it will see in production.
    """
    out: list[AgentPaymentHistoryRecord] = []
    year, month = start.year, start.month
    for index in range(months):
        attempt = date(year, month, debit_day)
        dead = dead_from is not None and index >= dead_from
        for offset in (0, 2, 4, 7):
            day = attempt + timedelta(days=offset)
            cleared = (not dead) and 0 <= days_since_payday(day, payday) <= 14
            out.append(record(day, cleared, amount))
            if cleared:
                break
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


# --- shared machinery --------------------------------------------------------


def test_days_since_payday_clamps_month_end():
    assert days_since_payday(date(2026, 3, 5), 31) == 5  # Feb 28 + 5
    assert days_since_payday(date(2026, 2, 28), 31) == 0
    assert days_since_payday(date(2026, 7, 9), 7) == 2
    assert days_since_payday(date(2026, 7, 3), 7) == 26  # last month's payday


def test_liquidity_falls_as_the_month_is_spent():
    values = [liquidity_at(phase) for phase in range(0, 30, 3)]
    assert all(b <= a for a, b in zip(values, values[1:])), values
    assert values[0] > 0.9
    assert values[-1] < 0.1


def test_model_rejects_unknown_tiers():
    with pytest.raises(ValueError, match="tier must be"):
        PaydayModel(tier=4)


def test_querying_before_fitting_raises():
    with pytest.raises(RuntimeError, match="call fit"):
        PaydayModel(tier=1).payday_posterior()


# --- the agent boundary ------------------------------------------------------


def test_model_only_ever_receives_agent_records():
    """The model is fed payment history and nothing else.

    AgentPaymentHistoryRecord has four fields and none of them is hidden, so
    there is no salary date, no balance and no shock flag to read even by
    mistake.
    """
    history = history_for_payday(payday=5, debit_day=20)
    fields = set()
    for entry in history:
        fields |= set(entry.model_dump())
    assert fields == {"customer_id", "attempt_date", "amount", "succeeded", "return_code"}

    model = PaydayModel(tier=1)
    model.fit(history)
    assert "salary_day" not in model.explain()


def test_model_source_cannot_import_the_truth():
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "agent"
    forbidden = {"generator.oracle", "generator.balance", "generator.population"}
    for path in src.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not (imported & forbidden), f"{path.name} imports the truth"


# --- tier 1 ------------------------------------------------------------------


def test_tier1_learns_which_days_of_month_worked():
    model = PaydayModel(tier=1)
    model.fit(history_for_payday(payday=5, debit_day=8))
    curve = model.liquidity_curve(date(2026, 9, 1), date(2026, 9, 30), 500.0)

    # Day 8 is two days after payday and always cleared; day 1 is four days
    # before it and never did.
    assert curve[date(2026, 9, 8)] > curve[date(2026, 9, 1)]


def test_tier1_falls_back_to_the_base_rate_on_unseen_days():
    model = PaydayModel(tier=1)
    model.fit(history_for_payday(payday=5, debit_day=8))
    curve = model.liquidity_curve(date(2026, 9, 1), date(2026, 9, 30), 500.0)
    unseen = curve[date(2026, 9, 23)]
    assert 0.0 < unseen < 1.0


def test_tier1_has_no_concept_of_payday():
    """It returns the untouched prior, which is exactly its limitation."""
    from agent.payday import payday_prior

    model = PaydayModel(tier=1)
    model.fit(history_for_payday(payday=5, debit_day=8))
    assert model.payday_posterior() == payday_prior()


def test_the_prior_covers_every_day_but_still_favours_month_end():
    from agent.payday import payday_prior

    prior = payday_prior()
    assert set(prior) == set(range(1, 32))
    assert sum(prior.values()) == pytest.approx(1.0)
    assert all(mass > 0 for mass in prior.values()), "an unusual payday must stay findable"
    assert max(prior, key=prior.get) == 1
    assert sum(prior[d] for d in (1, 30, 31)) > 0.5


def test_larger_amounts_are_less_likely_to_clear():
    model = PaydayModel(tier=1)
    model.fit(history_for_payday(payday=5, debit_day=8, amount=500.0))
    window = (date(2026, 9, 1), date(2026, 9, 30))
    small = model.liquidity_curve(*window, 500.0)
    large = model.liquidity_curve(*window, 8_000.0)
    assert all(large[d] <= small[d] for d in small)
    assert large[date(2026, 9, 8)] < small[date(2026, 9, 8)]


def test_model_output_is_deterministic():
    history = history_for_payday(payday=5, debit_day=8)
    window = (date(2026, 9, 1), date(2026, 9, 30))
    for tier in (1,):
        first, second = PaydayModel(tier=tier), PaydayModel(tier=tier)
        first.fit(history)
        second.fit(list(history))
        assert first.liquidity_curve(*window, 500.0) == second.liquidity_curve(
            *window, 500.0
        )
        assert first.explain() == second.explain()


# --- income shock detection --------------------------------------------------


def test_shock_probability_rises_with_consecutive_dead_cycles():
    solvent = PaydayModel(tier=1)
    solvent.fit(history_for_payday(payday=5, debit_day=8))

    shocked = PaydayModel(tier=1)
    shocked.fit(history_for_payday(payday=5, debit_day=8, dead_from=8))

    assert shocked.shock_probability() > solvent.shock_probability()
    assert shocked.shock_probability() > 0.9
    assert solvent.shock_probability() < 0.2


def test_shock_collapses_the_liquidity_curve():
    shocked = PaydayModel(tier=1)
    shocked.fit(history_for_payday(payday=5, debit_day=8, dead_from=6))
    curve = shocked.liquidity_curve(date(2026, 9, 1), date(2026, 9, 30), 500.0)
    assert max(curve.values()) < 0.10


def test_the_current_cycles_single_bounce_is_not_yet_evidence():
    """One failed original presentment is why we are here; it must not on its
    own look like a lost income."""
    history = history_for_payday(payday=5, debit_day=8, months=6)
    history.append(record(date(2026, 2, 8), False))
    model = PaydayModel(tier=1)
    model.fit(history)
    assert model.shock_probability() < 0.2


# --- tier 2: the payday posterior --------------------------------------------


def straddling_debit_day(payday: int) -> int:
    """A debit day three days BEFORE payday.

    This is the case CONTEXT describes - "a customer paid on the 7th and
    debited on the 2nd" - and it is also the most informative one: the original
    presentment fails, the ladder walks forward across payday, and one of the
    retries clears. Successes and failures at different phases are what
    identifies the salary date. A history of nothing but failures cannot.
    """
    return (payday - 3 - 1) % 28 + 1


@pytest.mark.parametrize("payday", [1, 5, 7, 30])
def test_tier2_posterior_peaks_at_the_true_payday(payday):
    """The core claim: the salary date is recoverable from attempt outcomes."""
    model = PaydayModel(tier=2)
    model.fit(history_for_payday(payday=payday, debit_day=straddling_debit_day(payday)))
    posterior = model.payday_posterior()
    best = max(posterior, key=posterior.get)
    assert abs(best - payday) <= 2 or {best, payday} <= {1, 30, 31}, (
        f"true payday {payday}, model said {best} "
        f"(top: {sorted(posterior.items(), key=lambda kv: -kv[1])[:3]})"
    )


def test_tier2_liquidity_is_higher_just_after_payday_than_just_before():
    model = PaydayModel(tier=2)
    model.fit(history_for_payday(payday=5, debit_day=straddling_debit_day(5)))
    curve = model.liquidity_curve(date(2026, 9, 1), date(2026, 9, 30), 500.0)
    just_after = curve[date(2026, 9, 7)]
    just_before = curve[date(2026, 9, 3)]
    assert just_after > just_before, f"after={just_after:.3f} before={just_before:.3f}"


def test_a_history_of_pure_failure_cannot_identify_a_payday():
    """Worth pinning: with no successes anywhere, there is nothing to locate.

    The posterior should fall back toward the prior rather than inventing
    confidence it has not earned.
    """
    model = PaydayModel(tier=2)
    model.fit(history_for_payday(payday=5, debit_day=20))
    assert all(not r.succeeded for r in history_for_payday(payday=5, debit_day=20))
    assert model.explain()["payday_confidence"] < 0.7


def test_tier2_generalises_to_days_it_never_observed():
    """Tier 1 scores each day independently, so an unprobed day tells it
    nothing. Tier 2 knows day 6 and day 7 are both just after payday."""
    history = history_for_payday(payday=5, debit_day=straddling_debit_day(5))
    observed = {r.attempt_date.day for r in history}
    assert 25 not in observed

    tier2 = PaydayModel(tier=2)
    tier2.fit(history)
    curve = tier2.liquidity_curve(date(2026, 9, 1), date(2026, 9, 30), 500.0)
    assert curve[date(2026, 9, 6)] > curve[date(2026, 9, 25)]


@pytest.mark.parametrize("tier", [2, 3])
def test_larger_amounts_are_less_likely_to_clear_in_every_tier(tier):
    model = PaydayModel(tier=tier)
    model.fit(history_for_payday(payday=5, debit_day=8, amount=500.0))
    window = (date(2026, 9, 1), date(2026, 9, 30))
    small = model.liquidity_curve(*window, 500.0)
    large = model.liquidity_curve(*window, 8_000.0)
    assert all(large[d] <= small[d] + 1e-12 for d in small)
    assert large[date(2026, 9, 7)] < small[date(2026, 9, 7)]


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_every_tier_is_deterministic(tier):
    history = history_for_payday(payday=5, debit_day=8)
    window = (date(2026, 9, 1), date(2026, 9, 30))
    first, second = PaydayModel(tier=tier), PaydayModel(tier=tier)
    first.fit(history)
    second.fit(list(history))
    assert first.liquidity_curve(*window, 500.0) == second.liquidity_curve(*window, 500.0)
    assert first.explain() == second.explain()


@pytest.mark.parametrize("tier", [2, 3])
def test_probabilities_stay_in_range(tier):
    model = PaydayModel(tier=tier)
    model.fit(history_for_payday(payday=30, debit_day=14))
    curve = model.liquidity_curve(date(2026, 9, 1), date(2026, 12, 31), 500.0)
    assert all(0.0 <= p <= 1.0 for p in curve.values())
    posterior = model.payday_posterior()
    assert sum(posterior.values()) == pytest.approx(1.0)


# --- tier 3: fitted liquidity depth ------------------------------------------


def test_tier3_infers_a_shorter_window_for_a_customer_who_runs_dry_early():
    """Two customers, same payday, different staying power.

    The one who only ever clears in the first few days after payday should be
    fitted with a shorter liquidity depth than the one who clears all month.
    """
    def history(window_days: int):
        out = []
        year, month = 2025, 8
        for _ in range(12):
            for offset in (0, 2, 4, 7):
                day = date(year, month, 12) + timedelta(days=offset)
                cleared = days_since_payday(day, 5) <= window_days
                out.append(record(day, cleared))
                if cleared:
                    break
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return out

    broke_early = PaydayModel(tier=3)
    broke_early.fit(history(window_days=4))
    lasts_longer = PaydayModel(tier=3)
    lasts_longer.fit(history(window_days=20))

    assert broke_early.explain()["liquidity_depth_days"] < (
        lasts_longer.explain()["liquidity_depth_days"]
    )


# --- calibration: the property the money equation actually depends on --------


def _calibration(tier: int, datasets, horizon: int = 30):
    """Brier and ECE for one tier against the realised truth.

    Tests may read the oracle; the model may not. It is fitted only on history
    up to the failure date, so nothing from the scored window leaks in.
    """
    from generator.oracle import would_succeed

    predicted: list[float] = []
    actual: list[int] = []
    for dataset in datasets:
        by_customer: dict[str, list] = {}
        for entry in dataset.payment_history:
            by_customer.setdefault(entry.customer_id, []).append(entry)

        for event in dataset.failure_events:
            # The payday model answers one question: when will there be money?
            # A dead credential or a revoked mandate is a different class with
            # a different return code, routed elsewhere, and scoring the
            # liquidity model on them would measure the wrong thing.
            if event.failure_class != "insufficient_funds":
                continue
            customer = dataset.customer(event.customer_id)
            series = dataset.balance_of(event.customer_id)
            amount = customer.subscription_amount
            history = [
                r
                for r in by_customer[event.customer_id]
                if r.attempt_date <= event.original_debit_date
            ]
            start = event.original_debit_date + timedelta(days=1)
            end = event.original_debit_date + timedelta(days=horizon)
            if not history or end > series.index[-1].date():
                continue
            model = PaydayModel(tier=tier)
            model.fit(history)
            for day, p in model.liquidity_curve(start, end, amount).items():
                predicted.append(p)
                actual.append(int(would_succeed(customer, day, amount, series)))

    n = len(predicted)
    brier = sum((p - a) ** 2 for p, a in zip(predicted, actual)) / n

    buckets: dict[int, list] = {}
    for p, a in zip(predicted, actual):
        buckets.setdefault(min(9, int(p * 10)), []).append((p, a))
    ece = sum(
        len(b) / n * abs(sum(p for p, _ in b) / len(b) - sum(a for _, a in b) / len(b))
        for b in buckets.values()
    )
    base = sum(actual) / n
    skill = 1.0 - brier / (sum((base - a) ** 2 for a in actual) / n)
    return brier, ece, skill


@pytest.fixture(scope="module")
def calibration_dataset():
    """Several seeds: a single seed gives only ~30 failures, and the skill
    estimate off that is too noisy to assert on."""
    from generator.generate import generate_dataset

    return [generate_dataset(n_customers=250, seed=s) for s in (11, 12, 13)]


def test_tier2_calibration_is_better_than_tier1(calibration_dataset):
    """The whole thesis in one assertion.

    Inferring WHICH day is payday and deriving liquidity from it must beat
    scoring days of the month independently. If this ever fails, the prior or
    the likelihood is wrong and the novelty claim does not stand.
    """
    b1, e1, _ = _calibration(1, calibration_dataset)
    b2, e2, _ = _calibration(2, calibration_dataset)
    assert b2 < b1, f"tier 2 Brier {b2:.4f} is not better than tier 1 {b1:.4f}"
    assert e2 < e1, f"tier 2 ECE {e2:.4f} is not better than tier 1 {e1:.4f}"


def test_tier2_beats_predicting_the_base_rate(calibration_dataset):
    """Positive skill means it has learned about timing, not just prevalence."""
    _, _, skill = _calibration(2, calibration_dataset)
    assert skill > 0.05, f"tier 2 skill is only {skill:+.1%}"


def test_tier1_is_the_weak_baseline_it_claims_to_be(calibration_dataset):
    """Pins the finding that tier 1 is worse than ignoring the date entirely.

    Scoring each day of the month independently cannot generalise from the
    four days the legacy ladder happened to probe, so it is confidently wrong
    on the other twenty-six. Documented so it cannot silently drift.
    """
    _, _, skill = _calibration(1, calibration_dataset)
    assert skill < 0.05, f"tier 1 skill improved to {skill:+.1%} - update the README"
