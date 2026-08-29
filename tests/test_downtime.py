"""Tests for failure class 2: bank and PSP downtime."""

from __future__ import annotations

import ast
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import router
from agent.bank_health import (
    BASE_OUTAGE_RATE,
    BankHealthModel,
    PresentmentOutcome,
)
from agent.sequencer import SequencerConfig, SequencerPolicy
from classes.downtime import DowntimeHandler
from evaluation.simulator import run_episode
from generator import rules
from generator.banks import BANK_SHARES, BankOutageBook, outage_book
from generator.generate import generate_dataset, outage_book_for
from generator.models import HIDDEN_CUSTOMER_FIELDS
from generator.oracle import first_success_datetime, would_succeed, would_succeed_at
from policy import LegacyFixedSchedule
from policy.oracle_policy import OraclePolicy

SRC = Path(__file__).resolve().parents[1] / "src"
NOON = datetime(2026, 7, 10, 2, 0)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_customers=500, seed=1)


def downtime_events(dataset):
    return [
        e
        for e in dataset.failure_events
        if e.failure_class == rules.FAILURE_CLASS_DOWNTIME
    ]


def play(policy, dataset, event, seed=1, **kwargs):
    return run_episode(
        policy,
        dataset.customer(event.customer_id),
        event,
        dataset.balance_of(event.customer_id),
        dataset.payment_history,
        seed=seed,
        strict_compliance=False,
        **kwargs,
    )


# --- generation --------------------------------------------------------------


def test_downtime_failures_are_coded_and_distinguishable(dataset):
    events = downtime_events(dataset)
    assert events, "no downtime failures in this dataset"
    for event in events:
        assert event.return_code == rules.RETURN_CODE_DOWNTIME == "91"
        assert event.failure_class == rules.FAILURE_CLASS_DOWNTIME
        # Distinguishable from a balance failure at generation time, which is
        # the whole basis for routing.
        assert event.return_code != rules.RETURN_CODE_INSUFFICIENT_FUNDS


def test_a_downtime_failure_means_the_bank_was_down_not_the_account_empty(dataset):
    outages = outage_book_for(dataset)
    balance_was_fine = 0
    for event in downtime_events(dataset):
        customer = dataset.customer(event.customer_id)
        at = datetime.combine(event.original_debit_date, rules.DEFAULT_PRESENTMENT_TIME)
        assert outages.is_down(customer.bank_id, at)
        if would_succeed(
            customer,
            event.original_debit_date,
            customer.subscription_amount,
            dataset.balance_of(event.customer_id),
        ):
            balance_was_fine += 1
    assert balance_was_fine > 0, (
        "every downtime failure also had an empty account - the two mechanisms "
        "are not separable in this dataset"
    )


def test_bank_id_is_public_and_reaches_the_agent(dataset):
    assert "bank_id" not in HIDDEN_CUSTOMER_FIELDS
    from generator.generate import agent_view

    view = agent_view(dataset)
    assert all(c.bank_id in BANK_SHARES for c in view.customers)


def test_the_outage_book_is_a_pure_function_of_seed_and_window():
    """It is never persisted, so it must rebuild identically every time."""
    window = (datetime(2025, 8, 1), datetime(2026, 2, 1))
    first = outage_book(7, *window)
    assert outage_book(7, *window) is first  # cached
    assert outage_book(8, *window).intervals != first.intervals


def test_outage_book_survives_a_save_load_round_trip(tmp_path, dataset):
    from generator.generate import load_dataset, save_dataset

    save_dataset(dataset, tmp_path / "run")
    restored = load_dataset(tmp_path / "run")
    original = outage_book_for(dataset)
    rebuilt = outage_book_for(restored)
    assert rebuilt.intervals == original.intervals


def test_larger_banks_have_better_uptime():
    """One knob, not a hardcoded two-tier table."""
    book = outage_book(3, datetime(2025, 8, 1), datetime(2027, 8, 1))
    big = np.mean([book.down_fraction(b) for b in ("SBI", "HDFC", "ICICI")])
    small = np.mean([book.down_fraction(b) for b in ("UJJIVAN", "JANA_SFB", "SURYODAY_SFB")])
    assert small > big * 2


# --- the oracle: an AND of two conditions ------------------------------------


class _FixedBook(BankOutageBook):
    """A book with one known outage, for the overlap case."""


def _book_with_outage(bank_id: str, start: datetime, end: datetime) -> BankOutageBook:
    return BankOutageBook(
        seed=0,
        window_start=start - timedelta(days=30),
        window_end=end + timedelta(days=30),
        intervals={bank_id: [(start, end)]},
    )


def test_would_succeed_at_requires_both_bank_and_balance(dataset):
    """The overlap case: broke AND down are independent blockers.

    Waiting out an outage does not put money in the account, and money in the
    account does not bring the bank back. Reporting a clear bank as recoverable
    when the balance would have failed anyway is the error this guards.
    """
    customer = dataset.customers[0]
    series = dataset.balance_of(customer.customer_id)
    amount = customer.subscription_amount

    rich_day = next(
        (
            stamp.date()
            for stamp, value in series.items()
            if value >= amount * 3 and stamp.year == 2026
        ),
        None,
    )
    broke_day = next(
        (
            stamp.date()
            for stamp, value in series.items()
            if value < amount and stamp.year == 2026
        ),
        None,
    )
    assert rich_day and broke_day, "need one funded and one short day"

    rich_at = datetime.combine(rich_day, rules.DEFAULT_PRESENTMENT_TIME)
    broke_at = datetime.combine(broke_day, rules.DEFAULT_PRESENTMENT_TIME)

    up = _book_with_outage(customer.bank_id, datetime(2000, 1, 1), datetime(2000, 1, 2))
    down_rich = _book_with_outage(
        customer.bank_id, rich_at - timedelta(hours=1), rich_at + timedelta(hours=1)
    )
    down_broke = _book_with_outage(
        customer.bank_id, broke_at - timedelta(hours=1), broke_at + timedelta(hours=1)
    )

    # bank up + balance fine  -> True
    assert would_succeed_at(customer, rich_at, amount, series, up)
    # bank down + balance fine -> False
    assert not would_succeed_at(customer, rich_at, amount, series, down_rich)
    # bank up + balance short  -> False
    assert not would_succeed_at(customer, broke_at, amount, series, up)
    # bank down + balance short -> False
    assert not would_succeed_at(customer, broke_at, amount, series, down_broke)


def test_first_success_datetime_stays_inside_the_execution_window(dataset):
    outages = outage_book_for(dataset)
    checked = 0
    for event in downtime_events(dataset)[:10]:
        customer = dataset.customer(event.customer_id)
        at = datetime.combine(event.original_debit_date, rules.DEFAULT_PRESENTMENT_TIME)
        found = first_success_datetime(
            customer,
            at,
            customer.subscription_amount,
            dataset.balance_of(event.customer_id),
            outages,
        )
        if found is None:
            continue
        assert found > at
        assert rules.in_execution_window(found.time())
        assert would_succeed_at(
            customer,
            found,
            customer.subscription_amount,
            dataset.balance_of(event.customer_id),
            outages,
        )
        checked += 1
    assert checked > 0


# --- the bank health model ---------------------------------------------------


def test_bank_health_is_agent_safe():
    forbidden = {"generator.oracle", "generator.balance", "generator.banks"}
    tree = ast.parse((SRC / "agent" / "bank_health.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not (imported & forbidden)


def test_observe_accepts_only_agent_visible_fields():
    """Three fields, all of which a real PSP has on every decline."""
    fields = set(PresentmentOutcome.__dataclass_fields__)
    assert fields == {"bank_id", "at", "return_code"}
    assert not (fields & HIDDEN_CUSTOMER_FIELDS)


def test_outage_probability_rises_on_a_cluster_of_downtime_codes():
    model = BankHealthModel()
    before = model.outage_probability("HDFC", NOON)
    for minute in range(0, 25, 5):
        model.observe(
            PresentmentOutcome("HDFC", NOON + timedelta(minutes=minute), "91")
        )
    after = model.outage_probability("HDFC", NOON)
    assert after > before
    assert after > 0.8


def test_a_cluster_of_insufficient_funds_does_not_implicate_the_bank():
    """The single most important test here.

    Both clusters are same-bank and tight in time. Only one says anything about
    the bank. If this ever fails, the class has stopped being separable from
    insufficient funds and every downtime result is suspect.
    """
    model = BankHealthModel()
    for minute in range(0, 60, 5):
        model.observe(
            PresentmentOutcome("ICICI", NOON + timedelta(minutes=minute), "04")
        )
    assert model.outage_probability("ICICI", NOON) == pytest.approx(BASE_OUTAGE_RATE)


def test_dead_mandate_codes_are_not_bank_evidence_either():
    model = BankHealthModel()
    for code in ("54", "22", None):
        for minute in range(0, 30, 5):
            model.observe(
                PresentmentOutcome("AXIS", NOON + timedelta(minutes=minute), code)
            )
    assert model.outage_probability("AXIS", NOON) == pytest.approx(BASE_OUTAGE_RATE)


def test_cold_start_returns_a_prior_not_a_crash_and_not_a_certainty():
    model = BankHealthModel()
    probability = model.outage_probability("KOTAK", NOON)
    assert 0.0 < probability < 1.0
    assert probability == pytest.approx(BASE_OUTAGE_RATE)
    assert model.explain("KOTAK", NOON)["cold_start"] is True


def test_evidence_decays_so_an_old_outage_does_not_condemn_a_bank():
    model = BankHealthModel()
    for minute in range(0, 25, 5):
        model.observe(PresentmentOutcome("PNB", NOON + timedelta(minutes=minute), "91"))
    assert model.outage_probability("PNB", NOON) > 0.8
    assert model.outage_probability("PNB", NOON + timedelta(days=2)) < 0.05


def test_the_model_is_online():
    """Fed one at a time, not batch-fitted once."""
    model = BankHealthModel()
    trail = []
    for minute in range(0, 25, 5):
        model.observe(PresentmentOutcome("YES", NOON + timedelta(minutes=minute), "91"))
        trail.append(model.outage_probability("YES", NOON))
    assert trail == sorted(trail)
    assert trail[-1] > trail[0]


def test_expected_recovery_hours_is_bounded_and_defaults_sensibly():
    from agent.bank_health import DEFAULT_RECOVERY_HOURS, MAX_RECOVERY_HOURS

    model = BankHealthModel()
    assert model.expected_recovery_hours("IDFC") == DEFAULT_RECOVERY_HOURS

    for hour in (0, 4, 9):
        model.observe(PresentmentOutcome("IDFC", NOON + timedelta(hours=hour), "91"))
    observed = model.expected_recovery_hours("IDFC")
    assert DEFAULT_RECOVERY_HOURS <= observed <= MAX_RECOVERY_HOURS


# --- the handler and the router ----------------------------------------------


def test_router_sends_91_to_the_real_handler():
    handler = router.route_return_code("91")
    assert isinstance(handler, DowntimeHandler)
    assert handler.implemented is True


def test_downtime_offers_no_nudge():
    """There is nothing the customer can do about their bank."""
    actions = DowntimeHandler().allowed_actions(None)
    assert actions == ["RETRY", "WAIT", "HOLD", "CANCEL"]


def test_success_probability_is_the_product_of_both_terms(dataset):
    from policy.base import PolicyState
    from generator.models import AgentCustomer, AgentFailureEvent

    event = downtime_events(dataset)[0]
    customer = dataset.customer(event.customer_id)
    history = tuple(
        r for r in dataset.payment_history if r.customer_id == event.customer_id
    )
    state = PolicyState(
        customer=AgentCustomer.from_customer(customer),
        payment_history=history,
        failure_event=AgentFailureEvent.from_failure(event),
        today=event.original_debit_date,
        attempts_used=1,
        notifications_sent_this_cycle=0,
        cycle_end_date=event.original_debit_date + timedelta(days=30),
    )

    health = BankHealthModel()
    handler = DowntimeHandler(bank_health=health)
    day = event.original_debit_date + timedelta(days=3)

    combined = handler.success_probability(state, day)
    uptime = handler.uptime_probability(state, day)
    assert 0.0 <= combined <= uptime + 1e-9, "the bank term must bound the product"

    # A bank believed down drags the whole product down.
    for minute in range(0, 30, 5):
        health.observe(
            PresentmentOutcome(
                customer.bank_id,
                datetime.combine(day, rules.DEFAULT_PRESENTMENT_TIME)
                + timedelta(minutes=minute),
                "91",
            )
        )
    assert handler.success_probability(state, day) < combined


# --- policy behaviour --------------------------------------------------------


def test_the_sequencer_avoids_a_long_known_outage():
    """A constructed scenario, not a statistical check.

    The bank is unmistakably down across a run of days. The agent should not
    name one of those days for its presentment.
    """
    from generator.models import AgentCustomer, AgentFailureEvent
    from policy.base import PolicyState

    failed_on = date(2026, 7, 12)
    customer = AgentCustomer(
        customer_id="CUST_0001",
        subscription_amount=500.0,
        rail="upi_autopay",
        debit_day_of_month=12,
        tenure_months=9,
        monthly_ltv=6000.0,
        bank_id="HDFC",
    )
    history = tuple()
    state = PolicyState(
        customer=customer,
        payment_history=history,
        failure_event=AgentFailureEvent(
            customer_id="CUST_0001",
            cycle_start_date=failed_on,
            original_debit_date=failed_on,
            return_code="91",
            failure_class=rules.FAILURE_CLASS_DOWNTIME,
        ),
        today=failed_on,
        attempts_used=1,
        notifications_sent_this_cycle=0,
        cycle_end_date=failed_on + timedelta(days=30),
    )

    health = BankHealthModel()
    # Overwhelming evidence that HDFC is down for the next several days.
    for day_offset in range(0, 5):
        for minute in range(0, 60, 10):
            health.observe(
                PresentmentOutcome(
                    "HDFC",
                    datetime.combine(
                        failed_on + timedelta(days=day_offset),
                        rules.DEFAULT_PRESENTMENT_TIME,
                    )
                    + timedelta(minutes=minute),
                    "91",
                )
            )

    policy = SequencerPolicy(bank_health=health)
    decision = policy.decide(state)
    blackout = {failed_on + timedelta(days=d) for d in range(0, 5)}
    if decision.action == "RETRY":
        assert decision.retry_date not in blackout, (
            f"scheduled into a known outage on {decision.retry_date}"
        )
    assert decision.reasoning.get("bank_outage_probability", 0) > 0.5


def test_sequencer_beats_legacy_on_downtime_across_seeds():
    """Pooled: a single seed carries only a handful of downtime failures."""
    legacy_recovered = agent_recovered = total = 0
    for seed in (1, 2, 3, 4, 5):
        data = generate_dataset(n_customers=500, seed=seed)
        events = downtime_events(data)
        if not events:
            continue
        total += len(events)
        legacy_recovered += sum(
            play(LegacyFixedSchedule(), data, e, seed).outcome == "RECOVERED"
            for e in events
        )
        agent_recovered += sum(
            play(SequencerPolicy(), data, e, seed).outcome == "RECOVERED"
            for e in events
        )
    assert total >= 20, f"only {total} downtime failures to judge on"
    assert agent_recovered >= legacy_recovered, (
        f"agent {agent_recovered}/{total} vs legacy {legacy_recovered}/{total}"
    )


def test_the_agent_never_recovers_what_no_date_could_have():
    """The real leakage guard.

    Realised recovery RATE is not a clean upper bound: the oracle optimises
    timing, not churn, so it can lose an episode to an opt-out that the agent
    happened to dodge - which is luck on an independent draw, not the agent
    knowing something. What cannot happen is the agent collecting money on a
    date the truth says could never have cleared.
    """
    for seed in (1, 2, 3):
        data = generate_dataset(n_customers=500, seed=seed)
        outages = outage_book_for(data)
        for event in downtime_events(data):
            customer = data.customer(event.customer_id)
            series = data.balance_of(event.customer_id)
            result = play(SequencerPolicy(), data, event, seed)
            if result.outcome != "RECOVERED":
                continue
            reachable = first_success_datetime(
                customer,
                datetime.combine(
                    event.original_debit_date, rules.DEFAULT_PRESENTMENT_TIME
                ),
                customer.subscription_amount,
                series,
                outages,
                horizon_days=400,
            )
            assert reachable is not None, (
                f"{event.customer_id} recovered on a path the truth says has "
                "no clearing date at all"
            )


def test_no_policy_presents_into_the_outage_it_is_reacting_to(dataset):
    """Structural, not statistical.

    Outages cap at 18h and the notice period is 24h, so an outage that blocked
    today's presentment is always over before the earliest hour a compliant
    policy may present again.
    """
    outages = outage_book_for(dataset)
    for event in downtime_events(dataset):
        customer = dataset.customer(event.customer_id)
        at = datetime.combine(event.original_debit_date, rules.DEFAULT_PRESENTMENT_TIME)
        span = outages.outage_covering(customer.bank_id, at)
        assert span is not None
        earliest = datetime.combine(
            event.original_debit_date + timedelta(days=1),
            rules.DEFAULT_PRESENTMENT_TIME,
        )
        assert span[1] <= earliest, (
            "an outage survived the notice period - the structural claim in "
            "scripts/run_downtime.py no longer holds and must be restated"
        )


def test_the_no_bank_health_ablation_changes_what_the_agent_believes():
    """The switch must actually disconnect the signal."""
    from generator.models import AgentCustomer, AgentFailureEvent
    from policy.base import PolicyState

    failed_on = date(2026, 7, 12)
    state = PolicyState(
        customer=AgentCustomer(
            customer_id="CUST_0001",
            subscription_amount=500.0,
            rail="upi_autopay",
            debit_day_of_month=12,
            tenure_months=9,
            monthly_ltv=6000.0,
            bank_id="HDFC",
        ),
        payment_history=tuple(),
        failure_event=AgentFailureEvent(
            customer_id="CUST_0001",
            cycle_start_date=failed_on,
            original_debit_date=failed_on,
            return_code="91",
            failure_class=rules.FAILURE_CLASS_DOWNTIME,
        ),
        today=failed_on,
        attempts_used=1,
        notifications_sent_this_cycle=0,
        cycle_end_date=failed_on + timedelta(days=30),
    )
    health = BankHealthModel()
    for minute in range(0, 60, 10):
        health.observe(
            PresentmentOutcome(
                "HDFC",
                datetime.combine(
                    failed_on + timedelta(days=1), rules.DEFAULT_PRESENTMENT_TIME
                )
                + timedelta(minutes=minute),
                "91",
            )
        )

    informed = SequencerPolicy(bank_health=health).decide(state)
    blind = SequencerPolicy(
        SequencerConfig(use_bank_health=False), bank_health=health
    ).decide(state)
    assert informed.reasoning["bank_outage_probability"] > (
        blind.reasoning["bank_outage_probability"]
    )


def test_a_downtime_failure_does_not_teach_the_payday_model_the_account_was_empty():
    """A 91 never reached the account, so it says nothing about liquidity."""
    from agent.payday import PaydayModel
    from generator.models import AgentPaymentHistoryRecord

    def record(day: int, code: str | None, ok: bool = False):
        return AgentPaymentHistoryRecord(
            customer_id="CUST_0001",
            attempt_date=date(2026, 7, day),
            amount=500.0,
            succeeded=ok,
            return_code=None if ok else code,
        )

    balance_only = [record(2, "04"), record(9, None, ok=True)]
    with_noise = balance_only + [record(4, "91"), record(6, "54")]

    clean, noisy = PaydayModel(tier=2), PaydayModel(tier=2)
    clean.fit(balance_only)
    noisy.fit(with_noise)
    assert clean.payday_posterior() == noisy.payday_posterior()
    assert noisy.explain()["attempts_observed"] == 2


# --- reproducibility ---------------------------------------------------------


def test_a_downtime_run_is_reproducible():
    def run(seed: int):
        data = generate_dataset(n_customers=300, seed=seed)
        policy = SequencerPolicy()
        return [
            (e.customer_id, play(policy, data, e, seed).net_value)
            for e in downtime_events(data)
        ]

    assert run(4) == run(4)
