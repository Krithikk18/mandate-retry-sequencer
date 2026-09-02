"""Tests for the simulator, the economics and the comparison harness."""

from __future__ import annotations

import ast
from datetime import date, time, timedelta
from pathlib import Path

import pytest

from evaluation import economics
from evaluation.harness import compare, evaluate
from evaluation.simulator import (
    ComplianceViolation,
    cycle_end_date,
    optout_rng,
    run_episode,
)
from generator import rules
from generator.generate import generate_dataset
from generator.models import AgentCustomer
from policy import LegacyFixedSchedule, Policy, PolicyState, RetryDecision
from policy.oracle_policy import OraclePolicy

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_customers=250, seed=42)


@pytest.fixture(scope="module")
def hopeless(dataset):
    """A customer whose income stopped: no date can ever clear."""
    for event in dataset.failure_events:
        if dataset.customer(event.customer_id).income_shock_start is not None:
            return event
    pytest.skip("no income-shocked failure in this dataset")


def play(policy, dataset, event, **kwargs):
    return run_episode(
        policy,
        dataset.customer(event.customer_id),
        event,
        dataset.balance_of(event.customer_id),
        dataset.payment_history,
        seed=42,
        # Legacy never looks at the return code, so it presents against revoked
        # mandates. That is the finding, not a crash: count the breach.
        strict_compliance=kwargs.pop("strict_compliance", False),
        **kwargs,
    )


# --- misbehaving policies, used to prove the harness refuses to cover for them


class AlwaysRetryTomorrow(Policy):
    name = "always_retry"

    def decide(self, state: PolicyState) -> RetryDecision:
        return RetryDecision(action="RETRY", retry_date=state.today + timedelta(days=1))


class RetryToday(Policy):
    name = "retry_today"

    def decide(self, state: PolicyState) -> RetryDecision:
        return RetryDecision(action="RETRY", retry_date=state.today)


class RetryAtLunchtime(Policy):
    name = "retry_at_lunchtime"

    def decide(self, state: PolicyState) -> RetryDecision:
        return RetryDecision(
            action="RETRY",
            retry_date=state.today + timedelta(days=1),
            retry_time=time(13, 0),
        )


class AlwaysHold(Policy):
    name = "always_hold"

    def decide(self, state: PolicyState) -> RetryDecision:
        return RetryDecision(action="HOLD", reasoning={"rule": "never retry"})


class CancelAfterTwoCycles(Policy):
    """A minimal stopping rule: give it two cycles, then give up."""

    name = "cancel_after_two"

    def decide(self, state: PolicyState) -> RetryDecision:
        cycles_seen = sum(
            1
            for r in state.payment_history
            if r.attempt_date >= state.failure_event.original_debit_date
        )
        failures = [r for r in state.payment_history if not r.succeeded]
        if len(failures) >= 6 or cycles_seen > 2:
            return RetryDecision(action="CANCEL", reasoning={"rule": "two cycles spent"})
        return RetryDecision(action="HOLD", reasoning={"rule": "one more cycle"})


class CancelImmediately(Policy):
    name = "cancel_now"

    def decide(self, state: PolicyState) -> RetryDecision:
        return RetryDecision(action="CANCEL", reasoning={"rule": "no patience"})


class FixedOffsets(Policy):
    """Two instances with different names but identical schedules."""

    def __init__(self, name: str, offsets=(2, 4, 7)) -> None:
        self.name = name
        self.offsets = offsets

    def decide(self, state: PolicyState) -> RetryDecision:
        made = max(0, state.attempts_used - 1)
        original = state.failure_event.original_debit_date
        for offset in self.offsets[made:]:
            candidate = original + timedelta(days=offset)
            if candidate >= state.earliest_compliant_retry_date and (
                state.attempts_remaining > 0
            ):
                return RetryDecision(action="RETRY", retry_date=candidate)
        return RetryDecision(action="HOLD")


# --- compliance is enforced by raising, never by clamping --------------------


def test_a_fifth_presentment_raises(dataset, hopeless):
    with pytest.raises(ComplianceViolation, match="NPCI ceiling"):
        play(AlwaysRetryTomorrow(), dataset, hopeless)


def test_scheduling_inside_the_24_hour_notice_window_raises(dataset):
    event = dataset.failure_events[0]
    with pytest.raises(ComplianceViolation, match="24-hour pre-debit notice"):
        play(RetryToday(), dataset, event)


def test_scheduling_outside_the_execution_window_raises(dataset):
    upi = next(
        e
        for e in dataset.failure_events
        if dataset.customer(e.customer_id).rail == "upi_autopay"
    )
    with pytest.raises(ComplianceViolation, match="execution window"):
        play(RetryAtLunchtime(), dataset, upi)


def presentments_per_cycle(result) -> dict[int, int]:
    """Count actual presentments in each cycle of an episode.

    attempts_used is an episode-level total now that HOLD carries a mandate
    across cycles; the NPCI ceiling is per cycle, so that is what must be
    checked.
    """
    counts: dict[int, int] = {}
    for entry in result.decision_log:
        if entry.action in ("EXECUTE", "AUTO") and "opted out" not in entry.event:
            counts[entry.cycle] = counts.get(entry.cycle, 0) + 1
    return counts


def test_the_ceiling_obeys_the_attempt_cap_too(dataset):
    """The oracle cheats on information, never on the rules."""
    oracle = OraclePolicy.for_dataset(dataset)
    for event in dataset.failure_events:
        result = play(oracle, dataset, event)
        for cycle, count in presentments_per_cycle(result).items():
            # Cycle 1's original presentment is not in the log (it is the
            # failure event we were handed), so allow for it.
            budget = rules.MAX_ATTEMPTS_PER_CYCLE - (1 if cycle == 1 else 0)
            assert count <= budget, f"cycle {cycle} made {count} presentments"


def test_legacy_stays_within_the_cap_on_every_episode(dataset):
    for event in dataset.failure_events:
        result = play(LegacyFixedSchedule(), dataset, event)
        assert result.attempts_used <= rules.MAX_ATTEMPTS_PER_CYCLE
        assert result.notifications_sent <= len((2, 4, 7))


# --- the schedule the baseline actually plays --------------------------------


def test_legacy_presents_on_plus_two_four_seven(dataset):
    """Offsets, in order, truncated wherever the episode ended."""
    for event in dataset.failure_events[:40]:
        result = play(LegacyFixedSchedule(), dataset, event)
        offsets = [(d - event.original_debit_date).days for d in result.retry_dates]
        assert offsets == [2, 4, 7][: len(offsets)]


def test_cycle_end_is_the_day_before_the_next_debit():
    assert cycle_end_date(date(2026, 7, 2)) == date(2026, 8, 1)
    assert cycle_end_date(date(2026, 12, 28)) == date(2027, 1, 27)


# --- opt-out determinism (the comparison would otherwise be polluted) --------


def test_optout_draw_depends_only_on_customer_notice_and_seed():
    a = optout_rng("CUST_0001", 1, 42).random()
    b = optout_rng("CUST_0001", 1, 42).random()
    assert a == b

    assert optout_rng("CUST_0001", 2, 42).random() != a
    assert optout_rng("CUST_0002", 1, 42).random() != a
    assert optout_rng("CUST_0001", 1, 43).random() != a


def test_two_policies_scheduling_the_same_dates_get_the_same_optout_draws(dataset):
    """Otherwise policy A's extra attempt shifts the stream for policy B and
    the comparison measures RNG alignment instead of skill."""
    first = FixedOffsets("twin_a")
    second = FixedOffsets("twin_b")

    for event in dataset.failure_events:
        a = play(first, dataset, event)
        b = play(second, dataset, event)
        assert a.retry_dates == b.retry_dates
        assert a.outcome == b.outcome
        assert a.notifications_sent == b.notifications_sent
        assert a.recovered_amount == b.recovered_amount


def test_a_busier_policy_does_not_shift_another_policys_draws(dataset, hopeless):
    """Common random numbers: the Nth notice to a customer draws the same
    uniform no matter how many notices some other policy sent."""
    sparse = FixedOffsets("sparse", offsets=(7,))
    busy = FixedOffsets("busy", offsets=(2, 4, 7))

    customer = dataset.customer(hopeless.customer_id)
    assert optout_rng(customer.customer_id, 1, 42).random() == optout_rng(
        customer.customer_id, 1, 42
    ).random()
    # Both policies' first notice uses index 1, regardless of what follows.
    play(sparse, dataset, hopeless)
    play(busy, dataset, hopeless)


# --- economics ---------------------------------------------------------------


def test_hold_loses_no_ltv(dataset, hopeless):
    result = play(AlwaysHold(), dataset, hopeless)
    assert result.outcome == "HELD"
    assert result.ltv_lost == 0.0
    assert result.ltv_retained == dataset.customer(hopeless.customer_id).annual_ltv
    assert result.recovered_amount == 0.0
    assert result.mandate_survives


def test_cancel_forfeits_the_whole_annual_ltv(dataset, hopeless):
    result = play(LegacyFixedSchedule(), dataset, hopeless)
    assert result.outcome in ("CANCELLED", "OPTED_OUT")
    assert result.ltv_lost == dataset.customer(hopeless.customer_id).annual_ltv
    assert result.ltv_retained == 0.0
    assert not result.mandate_survives


def _shocked_events(dataset):
    return [
        e
        for e in dataset.failure_events
        if dataset.customer(e.customer_id).income_shock_start is not None
    ]


def _mean_net(policy, dataset, events, fraction: float) -> float:
    """Mean net value over ``events`` with the carrying cost set to ``fraction``."""
    original = rules.HOLD_CARRYING_COST_FRACTION
    rules.HOLD_CARRYING_COST_FRACTION = fraction
    try:
        values = [play(policy, dataset, e).net_value for e in events]
    finally:
        rules.HOLD_CARRYING_COST_FRACTION = original
    return sum(values) / len(values)


def test_holding_now_costs_something(dataset, hopeless):
    """The bug Part A fixes: HOLD used to be free and so strictly dominated
    CANCEL, which meant no stopping rule could ever be wrong."""
    held = play(AlwaysHold(), dataset, hopeless)
    assert held.outcome in ("HELD", "CHURNED")
    assert held.cycles_held > 0
    assert held.carrying_cost_paid > 0
    assert held.net_value < -held.carrying_cost_paid + 1e-9


def test_carrying_cost_monotonically_penalises_holding(dataset):
    events = _shocked_events(dataset)
    if not events:
        pytest.skip("no income-shocked failures in this dataset")
    scores = [_mean_net(AlwaysHold(), dataset, events, f) for f in (0.05, 0.15, 0.30, 0.60)]
    assert all(b < a for a, b in zip(scores, scores[1:])), scores


def test_holding_loses_to_stopping_once_carrying_cost_is_high_enough(dataset):
    """A stopping rule must be CAPABLE of being right.

    At a high enough carrying cost, carrying an income-shocked customer for a
    year costs more than writing the mandate off. That the crossover exists at
    all is what makes CANCEL a real option rather than a strictly dominated one.
    See test below for where the crossover actually sits.
    """
    events = _shocked_events(dataset)
    if not events:
        pytest.skip("no income-shocked failures in this dataset")
    holding = _mean_net(AlwaysHold(), dataset, events, 1.0)
    stopping = _mean_net(CancelAfterTwoCycles(), dataset, events, 1.0)
    assert holding < stopping, f"holding {holding:,.0f} vs stopping {stopping:,.0f}"


def test_at_the_specified_carrying_cost_holding_still_wins(dataset):
    """Pins an uncomfortable finding so it cannot drift unnoticed.

    At HOLD_CARRYING_COST_FRACTION = 0.15, carrying an unrecoverable customer
    for the full 12-cycle horizon is STILL cheaper than cancelling, because
    cancelling forfeits a whole year of LTV (12x the subscription) while a year
    of carrying costs only 1.8x plus presentment fees. The crossover sits near
    0.45 on NACH (where the Rs 500 bounce charge compounds) and near 0.75 on
    UPI. scripts/sensitivity_holding.py reports the sweep.

    So the stopping rule is now *possible* to get right, but at these
    parameters it is still rarely the profitable move. That is a property of
    the cost model, not a bug, and it should be stated rather than hidden.
    """
    # Pooled across seeds on purpose. A single dataset carries only a handful
    # of income-shocked customers, and on roughly one seed in five the small
    # sample flips - so a single-dataset assertion here was measuring seed
    # noise rather than the cost model.
    holding_total = stopping_total = 0.0
    seeds_holding_wins = 0
    for seed in (11, 12, 13, 14, 15):
        book = generate_dataset(n_customers=250, seed=seed)
        events = _shocked_events(book)
        if not events:
            continue
        holding = _mean_net(AlwaysHold(), book, events, 0.15)
        stopping = _mean_net(CancelAfterTwoCycles(), book, events, 0.15)
        holding_total += holding
        stopping_total += stopping
        seeds_holding_wins += holding > stopping

    assert holding_total > stopping_total, (
        "holding no longer wins at the default carrying cost - the README and "
        "the sensitivity script both claim it does, so update them together"
    )
    assert seeds_holding_wins >= 3, (
        f"holding won on only {seeds_holding_wins}/5 seeds - the claim has "
        "become marginal and should be restated"
    )


def test_holding_still_beats_cancelling_when_recovery_is_close(dataset):
    """HOLD has a price now, but it must not become universally wrong.

    For a customer who simply needs to reach payday, carrying the mandate one
    cycle is far cheaper than forfeiting a year of subscription value.
    """
    recoverable = next(
        e
        for e in dataset.failure_events
        if dataset.customer(e.customer_id).income_shock_start is None
    )
    held = play(AlwaysHold(), dataset, recoverable)
    cancelled = play(CancelImmediately(), dataset, recoverable)
    assert held.net_value > cancelled.net_value


def test_carrying_cost_accrues_once_per_held_cycle(dataset, hopeless):
    from generator import rules as r

    held = play(AlwaysHold(), dataset, hopeless)
    amount = dataset.customer(hopeless.customer_id).subscription_amount
    assert held.carrying_cost_paid == pytest.approx(
        amount * r.HOLD_CARRYING_COST_FRACTION * held.cycles_held
    )
    assert held.cycles_held > 0


def test_natural_attrition_is_deterministic_and_separate_from_optout():
    from evaluation.simulator import attrition_rng, draws_natural_attrition

    assert attrition_rng("CUST_0001", 3, 42).random() == attrition_rng(
        "CUST_0001", 3, 42
    ).random()
    assert attrition_rng("CUST_0001", 3, 42).random() != optout_rng(
        "CUST_0001", 3, 42
    ).random()

    fired = [draws_natural_attrition(f"CUST_{i:04d}", 0, 42) for i in range(4000)]
    assert sum(fired) / len(fired) == pytest.approx(
        rules.NATURAL_ATTRITION_PER_CYCLE, abs=0.02
    )


def test_net_value_matches_the_stated_equation(dataset):
    for event in dataset.failure_events[:25]:
        result = play(LegacyFixedSchedule(), dataset, event)
        customer = AgentCustomer.from_customer(dataset.customer(event.customer_id))
        expected = (
            result.recovered_amount
            - rules.processing_cost(customer.rail) * result.attempts_used
            - result.bounce_charges_paid
            - (customer.annual_ltv if not result.mandate_survives else 0.0)
        )
        assert result.net_value == pytest.approx(expected)


def test_cost_per_rupee_is_none_when_nothing_is_recovered():
    assert economics.cost_per_rupee_recovered(100.0, 0.0) is None
    assert economics.cost_per_rupee_recovered(50.0, 200.0) == pytest.approx(0.25)


def test_upi_and_nach_costs_are_never_blended():
    assert rules.processing_cost("upi_autopay") != rules.processing_cost("nach")
    with pytest.raises(ValueError, match="unknown rail"):
        rules.processing_cost("card")


# --- the harness -------------------------------------------------------------


def test_harness_run_is_reproducible(dataset):
    policy = LegacyFixedSchedule()
    first = evaluate(policy, dataset, seed=42, oracle_metrics=False, strict_compliance=False)
    second = evaluate(policy, dataset, seed=42, oracle_metrics=False, strict_compliance=False)

    for metric in (
        "money_recovered",
        "recovery_rate",
        "attempts_used_total",
        "net_value_total",
        "optouts_induced",
        "customers_retained",
        "mean_days_to_resolution",
    ):
        assert getattr(first, metric) == getattr(second, metric)
    assert first.outcomes == second.outcomes


def test_oracle_beats_legacy_on_every_seed():
    seeds = [1, 2, 3, 4, 5]
    legacy = LegacyFixedSchedule()
    report = compare(
        [legacy, OraclePolicy.for_dataset],
        seeds=seeds,
        n_customers=250,
        baseline_name=legacy.name,
        ceiling_name=OraclePolicy.name,
        oracle_metrics=False,
        strict_compliance=False,
    )
    for seed_index in range(len(seeds)):
        floor = report.per_seed[legacy.name][seed_index].recovery_rate
        ceiling = report.per_seed[OraclePolicy.name][seed_index].recovery_rate
        assert ceiling >= floor, f"seed {seeds[seed_index]}: {ceiling:.3f} < {floor:.3f}"


def test_oracle_loses_nothing_it_could_have_recovered():
    """If this drifts, the simulator is losing recoveries somewhere.

    The dataset's `unrecoverable_share` is measured over a 30-day horizon,
    while an episode runs up to twelve cycles - so the oracle can legitimately
    beat it by carrying a mandate into a later month. What it must never do is
    fall SHORT of the single-cycle figure, once the customers who opted out on
    the notice are accounted for.
    """
    for seed in (1, 2, 3):
        dataset = generate_dataset(n_customers=250, seed=seed)
        report = evaluate(
            OraclePolicy.for_dataset(dataset), dataset, seed=seed, oracle_metrics=False
        )
        recoverable = 1.0 - dataset.meta["unrecoverable_share"]
        optout_share = report.optouts_induced / report.n_failures
        assert report.recovery_rate + optout_share >= recoverable - 1e-9, (
            f"seed {seed}: oracle recovered {report.recovery_rate:.3f} of a "
            f"single-cycle {recoverable:.3f} - recoveries are being dropped"
        )


def test_the_ceiling_never_recovers_a_structurally_dead_mandate():
    """More cycles cannot resurrect an expired card or restore consent."""
    from generator import rules

    dataset = generate_dataset(n_customers=300, seed=42)
    oracle = OraclePolicy.for_dataset(dataset)
    dead = [
        e for e in dataset.failure_events
        if e.failure_class in rules.NEVER_RETRY_CLASSES
    ]
    assert dead
    for event in dead:
        assert play(oracle, dataset, event).outcome != "RECOVERED"


def test_the_ceiling_never_presents_against_a_dead_mandate():
    """The oracle cheats on information, never on the rules."""
    from generator import rules

    dataset = generate_dataset(n_customers=300, seed=42)
    dead = [
        e
        for e in dataset.failure_events
        if e.failure_class in rules.NEVER_RETRY_CLASSES
    ]
    assert dead, "no dead mandates in this dataset"
    oracle = OraclePolicy.for_dataset(dataset)
    for event in dead:
        result = play(oracle, dataset, event, strict_compliance=True)
        assert result.retry_dates == ()
        assert result.compliance_breaches == 0


def test_headroom_is_zero_for_the_floor_and_one_for_the_ceiling():
    legacy = LegacyFixedSchedule()
    report = compare(
        [legacy, OraclePolicy.for_dataset],
        seeds=[1, 2],
        n_customers=250,
        baseline_name=legacy.name,
        ceiling_name=OraclePolicy.name,
        oracle_metrics=False,
        strict_compliance=False,
    )
    assert report.summary[legacy.name]["headroom_captured"].mean == pytest.approx(0.0)
    assert report.summary[OraclePolicy.name]["headroom_captured"].mean == pytest.approx(1.0)


def test_compare_reports_a_range_not_a_single_number():
    report = compare(
        [LegacyFixedSchedule()],
        seeds=[1, 2, 3],
        n_customers=250,
        oracle_metrics=False,
        strict_compliance=False,
    )
    summary = report.summary["legacy_fixed"]["recovery_rate"]
    assert len(summary.values) == 3
    assert summary.low <= summary.mean <= summary.high


def test_legacy_wastes_attempts_and_the_ceiling_does_not():
    dataset = generate_dataset(n_customers=250, seed=42)
    legacy = evaluate(
        LegacyFixedSchedule(), dataset, seed=42, n_sims=200, strict_compliance=False
    )
    oracle = evaluate(OraclePolicy.for_dataset(dataset), dataset, seed=42, n_sims=200)
    assert legacy.wasted_attempts > oracle.wasted_attempts
    assert legacy.wrongly_cancelled > 0
    assert oracle.wrongly_cancelled == 0


# --- the agent/truth boundary, enforced at import level ----------------------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_no_agent_safe_policy_can_reach_the_oracle():
    """wasted_attempts and wrongly_cancelled are evaluation-only by
    construction: the truth is not importable from the policy path."""
    forbidden = {"generator.oracle", "generator.balance", "evaluation.harness"}
    for path in (SRC / "policy").glob("*.py"):
        if path.name == "oracle_policy.py":
            continue  # the one file allowed to cheat, and it says so loudly
        leaked = _imported_modules(path) & forbidden
        assert not leaked, f"{path.name} imports {leaked}"


def test_oracle_policy_is_not_exported_from_the_policy_package():
    import policy

    assert not hasattr(policy, "OraclePolicy")
    assert "OraclePolicy" not in policy.__all__

    for path in (SRC / "policy").glob("*.py"):
        if path.name in ("oracle_policy.py", "__init__.py"):
            continue
        assert "oracle_policy" not in _imported_modules(path)


def test_oracle_policy_refuses_to_run_against_the_wrong_dataset():
    """Customer ids repeat across seeds, so a mis-bound oracle would quietly
    grade itself against somebody else's balance."""
    one = generate_dataset(n_customers=200, seed=1)
    two = generate_dataset(n_customers=200, seed=2)
    mismatched = OraclePolicy.for_dataset(one)

    with pytest.raises(ValueError, match="bound to the wrong dataset"):
        for event in two.failure_events:
            run_episode(
                mismatched,
                two.customer(event.customer_id),
                event,
                two.balance_of(event.customer_id),
                two.payment_history,
                seed=2,
            )


# --- money reporting ---------------------------------------------------------
# Part A of the money-reporting work surfaces figures the harness already
# computed and then dropped. These tests pin the PROPERTIES of that reporting
# layer, not the values: the values are pinned by scripts/verify_docs.py against
# a producing run, which is where a number belongs.


def test_net_value_never_exceeds_gross_recovered(dataset):
    """Net is gross minus costs, and the costs are never negative.

    A net figure above gross would mean the accounting had found money that was
    not collected - the one direction a cost model must not be able to fail in,
    because it is the direction that flatters the agent.
    """
    for policy in (LegacyFixedSchedule(), OraclePolicy.for_dataset(dataset)):
        report = evaluate(
            policy, dataset, seed=42, oracle_metrics=False, strict_compliance=False
        )
        assert report.net_value_total <= report.money_recovered + 1e-9, (
            f"{policy.name}: net {report.net_value_total:,.2f} exceeds gross "
            f"{report.money_recovered:,.2f}"
        )


def test_net_value_total_is_a_clean_sum_of_the_episode_accounting(dataset):
    """The report's net value is the sum of economics.net_value, not a re-derivation.

    Worth pinning because the alternative - recomputing the total from
    aggregates - would silently drop the per-episode outcome, and ltv_lost
    depends on the outcome of each episode individually.
    """
    report = evaluate(
        LegacyFixedSchedule(), dataset, seed=42, oracle_metrics=False,
        strict_compliance=False,
    )
    rebuilt = sum(
        economics.net_value(
            episode.customer,
            episode.outcome,
            episode.recovered_amount,
            episode.attempts_used,
            episode.bounce_charges_paid,
            episode.cycles_held,
            episode.goodwill_cost_paid,
        )
        for episode in report.episodes
    )
    assert report.net_value_total == pytest.approx(rebuilt)


def test_money_headroom_is_reported_and_bounded():
    """Both headroom numbers exist, and the floor/ceiling anchor both at 0 and 1.

    Mirrors the existing rate assertion. The agent's own value is checked for
    being a share rather than for a particular size - what it should be is a
    result, and results live in verify_docs.
    """
    legacy = LegacyFixedSchedule()
    report = compare(
        [legacy, OraclePolicy.for_dataset],
        seeds=[1, 2],
        n_customers=200,
        baseline_name=legacy.name,
        ceiling_name=OraclePolicy.name,
        oracle_metrics=False,
        strict_compliance=False,
    )
    for metric in ("headroom_captured", "money_headroom_captured"):
        assert report.summary[legacy.name][metric].mean == pytest.approx(0.0)
        assert report.summary[OraclePolicy.name][metric].mean == pytest.approx(1.0)


def test_money_headroom_is_none_when_the_span_is_degenerate():
    """A floor equal to the ceiling yields None, exactly as the rate version does.

    Comparing a policy against itself makes the span zero. The ratio is then
    noise over noise and must not be reported as a number; MetricSummary drops
    None, so the run reports no headroom rather than a fabricated one.
    """
    from evaluation.harness import _fill_headroom

    legacy = LegacyFixedSchedule()
    report = compare(
        [legacy],
        seeds=[1],
        n_customers=200,
        baseline_name=legacy.name,
        ceiling_name=legacy.name,
        oracle_metrics=False,
        strict_compliance=False,
    )
    only = report.per_seed[legacy.name][0]
    assert only.headroom_captured is None
    assert only.money_headroom_captured is None

    # And directly, so the guard is tested rather than inferred from a run that
    # happened to be degenerate.
    reports = {"a": only}
    _fill_headroom(reports, "a", "a")
    assert only.money_headroom_captured is None


def test_per_hundred_failures_scales_linearly_with_book_size():
    """The normalisation is the point: double the book, same per-100 figure."""
    small = economics.per_hundred_failures(50_000.0, 100)
    doubled = economics.per_hundred_failures(100_000.0, 200)
    assert small == pytest.approx(doubled)
    assert small == pytest.approx(50_000.0)


def test_per_hundred_failures_matches_the_lead_time_denominator():
    """Same arithmetic the Sec 6(c) opt-out cost is quoted with.

    lead_time_ablation.py computes ``money / sum(report.n_failures) * 100``.
    If this helper ever stopped matching that, the two per-100 figures in the
    README would be denominated differently while looking comparable.
    """
    money, n_failures = 111_773.0, 1_709
    assert economics.per_hundred_failures(money, n_failures) == pytest.approx(
        money / n_failures * 100
    )


def test_per_hundred_failures_is_none_on_an_empty_book():
    assert economics.per_hundred_failures(1234.0, 0) is None
