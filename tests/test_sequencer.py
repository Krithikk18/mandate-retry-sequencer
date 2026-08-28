"""Tests for the sequencer policy.

The two guardrails at the bottom of this file matter most: below legacy means
the agent is broken, above the oracle means it is cheating.
"""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path

import pytest

from agent.sequencer import SequencerConfig, SequencerPolicy
from evaluation.harness import evaluate
from evaluation.simulator import run_episode
from generator import rules
from generator.generate import generate_dataset
from generator.models import (
    HIDDEN_CUSTOMER_FIELDS,
    HIDDEN_FAILURE_FIELDS,
    AgentCustomer,
    AgentFailureEvent,
    AgentPaymentHistoryRecord,
)
from policy import LegacyFixedSchedule
from policy.base import PolicyState
from policy.oracle_policy import OraclePolicy

SRC = Path(__file__).resolve().parents[1] / "src"
FAILED_ON = date(2026, 7, 12)
CYCLE_END = date(2026, 8, 11)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_customers=250, seed=42)


def play(policy, dataset, event, **kwargs):
    return run_episode(
        policy,
        dataset.customer(event.customer_id),
        event,
        dataset.balance_of(event.customer_id),
        dataset.payment_history,
        seed=42,
        strict_compliance=kwargs.pop("strict_compliance", False),
        **kwargs,
    )


def make_state(history, today=FAILED_ON, attempts_used=1, **overrides) -> PolicyState:
    base = dict(
        customer=AgentCustomer(
            customer_id="CUST_0001",
            subscription_amount=500.0,
            rail="upi_autopay",
            debit_day_of_month=12,
            tenure_months=9,
            monthly_ltv=6000.0,
        ),
        payment_history=tuple(history),
        failure_event=AgentFailureEvent(
            customer_id="CUST_0001",
            cycle_start_date=FAILED_ON,
            original_debit_date=FAILED_ON,
            return_code="04",
            failure_class="insufficient_funds",
        ),
        today=today,
        attempts_used=attempts_used,
        notifications_sent_this_cycle=0,
        cycle_end_date=CYCLE_END,
    )
    base.update(overrides)
    return PolicyState(**base)


def attempt(day: date, ok: bool, amount: float = 500.0):
    return AgentPaymentHistoryRecord(
        customer_id="CUST_0001",
        attempt_date=day,
        amount=amount,
        succeeded=ok,
        return_code=None if ok else "04",
    )


def cycles(pattern, months=12, debit_day=12, start=(2025, 8)):
    """Build a history from a per-cycle callable ``pattern(day) -> cleared``."""
    out = []
    year, month = start
    for _ in range(months):
        anchor = date(year, month, debit_day)
        for offset in (0, 2, 4, 7):
            day = anchor + timedelta(days=offset)
            cleared = pattern(day)
            out.append(attempt(day, cleared))
            if cleared:
                break
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


# --- the agent boundary ------------------------------------------------------


def test_sequencer_sees_no_hidden_fields():
    history = cycles(lambda d: d.day >= 14)
    state = make_state(history)

    names: set[str] = set()

    def walk(value, depth=0):
        if depth > 6:
            return
        if hasattr(value, "model_dump"):
            for key, sub in value.model_dump().items():
                names.add(key)
                walk(sub, depth + 1)
        elif isinstance(value, (list, tuple)):
            for sub in value:
                walk(sub, depth + 1)

    walk(state)
    assert not (names & (HIDDEN_CUSTOMER_FIELDS | HIDDEN_FAILURE_FIELDS))
    SequencerPolicy().decide(state)  # must not need anything else


def test_no_agent_module_imports_the_truth():
    """generator.optout is on the list too: the real hazard reads a hidden
    per-customer sensitivity, so the agent has its own estimator in churn.py."""
    forbidden = {
        "generator.oracle",
        "generator.balance",
        "generator.population",
        "generator.optout",
        "evaluation.harness",
        "policy.oracle_policy",
    }
    for path in (SRC / "agent").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not (imported & forbidden), f"{path.name} imports {imported & forbidden}"


# --- reasoning ---------------------------------------------------------------


REQUIRED_REASONING_KEYS = {
    "policy",
    "rule",
    "shock_probability",
    "payday_mode",
    "payday_confidence",
    "attempts_remaining",
    "notifications_sent",
    "candidates",
    "binding_constraints",
}


def test_every_decision_is_audited(dataset):
    """Razorpay's brief asks for an audit trail; a decision without one is not
    a decision anybody can check."""
    policy = SequencerPolicy()
    seen_actions = set()
    for event in dataset.failure_events[:40]:
        result = play(policy, dataset, event)
        for entry in result.decision_log:
            if entry.action in ("EXECUTE", "AUTO", "CHURN"):
                continue
            seen_actions.add(entry.action)
            missing = REQUIRED_REASONING_KEYS - set(entry.reasoning)
            assert not missing, f"{entry.action} on {entry.day} is missing {missing}"
    assert seen_actions, "no policy decisions were logged at all"


def test_the_audit_trail_carries_the_runners_up():
    history = cycles(lambda d: d.day >= 14)
    decision = SequencerPolicy().decide(make_state(history))
    candidates = decision.reasoning["candidates"]
    assert 1 < len(candidates) <= 5
    assert [c["value"] for c in candidates] == sorted(
        (c["value"] for c in candidates), reverse=True
    )
    assert {"date", "p_clear", "p_optout", "value"} <= set(candidates[0])


def test_binding_constraint_is_recorded_when_the_budget_is_spent():
    history = cycles(lambda d: d.day >= 14)
    state = make_state(history, attempts_used=rules.MAX_ATTEMPTS_PER_CYCLE)
    decision = SequencerPolicy().decide(state)
    assert decision.action in ("HOLD", "CANCEL")
    assert decision.reasoning["binding_constraints"][0]["reason"] == "attempt_budget"


# --- the decision rules ------------------------------------------------------


def dry_rest_of_cycle_state(**overrides) -> PolicyState:
    """A NACH mandate with no good date left in the cycle.

    The customer clears only in the few days after a payday around the 19th.
    By 2 August that window has passed and the next one falls after the cycle
    closes on the 11th. On NACH every failed presentment also puts Rs 500 on
    the customer's bank statement, so a date has to be better than about a 70%
    chance before it is worth presenting on at all - and none of these are.
    """
    history = cycles(lambda d: 19 <= d.day <= 21, debit_day=12)
    base = dict(
        customer=AgentCustomer(
            customer_id="CUST_0001",
            subscription_amount=500.0,
            rail="nach",
            debit_day_of_month=12,
            tenure_months=9,
            monthly_ltv=6000.0,
        ),
        today=date(2026, 8, 2),
    )
    base.update(overrides)
    return make_state(history, **base)


def test_it_holds_when_no_date_left_in_the_cycle_is_worth_presenting_on():
    decision = SequencerPolicy().decide(dry_rest_of_cycle_state())
    assert decision.action == "HOLD"
    assert decision.reasoning["hold_value"] > decision.reasoning["cancel_value"]
    assert decision.reasoning["candidates"][0]["value"] <= 0


def test_the_same_dates_are_worth_presenting_on_over_upi():
    """The rail changes the answer: without the Rs 500 bounce charge the
    break-even probability is far lower, so the same dates are worth taking."""
    over_upi = dry_rest_of_cycle_state(
        customer=AgentCustomer(
            customer_id="CUST_0001",
            subscription_amount=500.0,
            rail="upi_autopay",
            debit_day_of_month=12,
            tenure_months=9,
            monthly_ltv=6000.0,
        )
    )
    assert SequencerPolicy().decide(over_upi).action in ("RETRY", "WAIT", "NUDGE")


def test_it_does_not_burn_attempts_on_a_confidently_shocked_customer(dataset):
    """The income has stopped. Every further presentment is an attempt and a
    cancel-capable notice spent on somebody with no money."""
    shocked = [
        e
        for e in dataset.failure_events
        if dataset.customer(e.customer_id).income_shock_start is not None
    ]
    assert shocked, "no income-shocked failures in this dataset"

    sequencer = SequencerPolicy()
    legacy = LegacyFixedSchedule()
    agent_notices = sum(play(sequencer, dataset, e).notifications_sent for e in shocked)
    legacy_notices = sum(play(legacy, dataset, e).notifications_sent for e in shocked)
    assert agent_notices < legacy_notices

    for event in shocked:
        result = play(sequencer, dataset, event)
        first_cycle = [e for e in result.decision_log if e.cycle == 1]
        presentments = sum(1 for e in first_cycle if e.action in ("EXECUTE", "AUTO"))
        assert presentments <= 2, f"{event.customer_id} spent {presentments} in cycle 1"


def test_dropping_hold_forces_a_cancel():
    no_hold = SequencerPolicy(SequencerConfig(use_hold=False))
    assert no_hold.decide(dry_rest_of_cycle_state()).action == "CANCEL"


def test_it_defers_commitment_rather_than_scheduling_far_ahead():
    history = cycles(lambda d: d.day >= 14)
    decision = SequencerPolicy().decide(make_state(history))
    if decision.action == "WAIT":
        assert "commit_on" in decision.reasoning
        assert decision.retry_date is None


def test_turning_off_deferral_makes_it_commit_immediately():
    history = cycles(lambda d: d.day >= 14)
    state = make_state(history)
    eager = SequencerPolicy(SequencerConfig(defer_commitment=False))
    decision = eager.decide(state)
    assert decision.action in ("RETRY", "NUDGE", "HOLD", "CANCEL")
    if decision.action == "RETRY":
        assert decision.retry_date >= state.earliest_compliant_retry_date


# --- compliance and determinism ----------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_sequencer_never_violates_a_compliance_constraint(seed):
    """A ComplianceViolation would raise out of run_episode; reaching the end
    of the book without one is the assertion."""
    data = generate_dataset(n_customers=200, seed=seed)
    report = evaluate(SequencerPolicy(), data, seed=seed, oracle_metrics=False)
    assert report.n_failures > 0
    for episode in report.episodes:
        per_cycle: dict[int, int] = {}
        for entry in episode.decision_log:
            if entry.action in ("EXECUTE", "AUTO") and "opted out" not in entry.event:
                per_cycle[entry.cycle] = per_cycle.get(entry.cycle, 0) + 1
        for cycle, count in per_cycle.items():
            budget = rules.MAX_ATTEMPTS_PER_CYCLE - (1 if cycle == 1 else 0)
            assert count <= budget


def test_sequencer_is_deterministic(dataset):
    first = evaluate(SequencerPolicy(), dataset, seed=42, oracle_metrics=False)
    second = evaluate(SequencerPolicy(), dataset, seed=42, oracle_metrics=False)
    for metric in (
        "money_recovered",
        "recovery_rate",
        "attempts_used_total",
        "net_value_total",
        "optouts_induced",
        "cycles_held_total",
    ):
        assert getattr(first, metric) == getattr(second, metric)
    assert first.outcomes == second.outcomes


# --- the guardrails ----------------------------------------------------------


@pytest.fixture(scope="module")
def three_way():
    """Legacy, sequencer and oracle on the same seeds."""
    rows = []
    for seed in range(1, 9):
        data = generate_dataset(n_customers=200, seed=seed)
        rows.append(
            {
                "seed": seed,
                "legacy": evaluate(
                    LegacyFixedSchedule(),
                    data,
                    seed=seed,
                    oracle_metrics=False,
                    strict_compliance=False,
                ).recovery_rate,
                "sequencer": evaluate(
                    SequencerPolicy(), data, seed=seed, oracle_metrics=False
                ).recovery_rate,
                "oracle": evaluate(
                    OraclePolicy.for_dataset(data), data, seed=seed, oracle_metrics=False
                ).recovery_rate,
            }
        )
    return rows


def test_sequencer_beats_legacy_on_mean_recovery(three_way):
    legacy = sum(r["legacy"] for r in three_way) / len(three_way)
    agent = sum(r["sequencer"] for r in three_way) / len(three_way)
    assert agent > legacy, f"sequencer {agent:.3f} does not beat legacy {legacy:.3f}"


def test_sequencer_never_exceeds_the_oracle(three_way):
    """Above the ceiling means information is leaking, not that the agent is
    clever. This is the more important of the two guardrails."""
    for row in three_way:
        assert row["sequencer"] <= row["oracle"] + 1e-9, (
            f"seed {row['seed']}: sequencer {row['sequencer']:.3f} beat the "
            f"oracle {row['oracle']:.3f} - something is leaking"
        )
