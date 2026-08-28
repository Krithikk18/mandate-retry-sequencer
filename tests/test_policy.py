"""Tests for the policy contract and the fixed-schedule baseline."""

from __future__ import annotations

from datetime import date, time, timedelta

import pytest

from generator.models import (
    HIDDEN_CUSTOMER_FIELDS,
    HIDDEN_FAILURE_FIELDS,
    AgentCustomer,
    AgentFailureEvent,
    AgentPaymentHistoryRecord,
    Customer,
)
from policy import LegacyFixedSchedule, PolicyState, RetryDecision

FAILED_ON = date(2026, 7, 2)
CYCLE_END = date(2026, 8, 1)


def agent_customer(**overrides) -> AgentCustomer:
    base = dict(
        customer_id="CUST_0001",
        subscription_amount=499.0,
        rail="upi_autopay",
        debit_day_of_month=2,
        tenure_months=9,
        monthly_ltv=5988.0,
    )
    base.update(overrides)
    return AgentCustomer(**base)


def make_state(today: date, attempts_used: int = 1, **overrides) -> PolicyState:
    base = dict(
        customer=agent_customer(),
        payment_history=(
            AgentPaymentHistoryRecord(
                customer_id="CUST_0001",
                attempt_date=FAILED_ON,
                amount=499.0,
                succeeded=False,
                return_code="04",
            ),
        ),
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


# --- the agent/truth boundary ------------------------------------------------


def _walk(value, names: set[str], depth: int = 0) -> None:
    """Collect every attribute and key name reachable from ``value``."""
    if depth > 6:
        return
    if hasattr(value, "model_dump"):
        for key, sub in value.model_dump().items():
            names.add(key)
            _walk(sub, names, depth + 1)
    elif isinstance(value, dict):
        for key, sub in value.items():
            names.add(str(key))
            _walk(sub, names, depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for sub in value:
            _walk(sub, names, depth + 1)


def test_policy_state_carries_no_hidden_fields():
    state = make_state(FAILED_ON)
    names: set[str] = set()
    _walk(state, names)
    names.update(dir(state))

    leaked = names & (HIDDEN_CUSTOMER_FIELDS | HIDDEN_FAILURE_FIELDS)
    assert not leaked, f"hidden fields reachable from PolicyState: {leaked}"
    assert not hasattr(state.customer, "salary_day")
    assert not hasattr(state.failure_event, "balance_at_failure")


def test_policy_state_refuses_a_truth_family_customer():
    """The split is enforced by the type system, not by remembering to strip."""
    truth = Customer(
        customer_id="CUST_0001",
        subscription_amount=499.0,
        rail="upi_autopay",
        debit_day_of_month=2,
        tenure_months=9,
        monthly_ltv=5988.0,
        salary_day=7,
        salary_amount=40_000.0,
        salary_delay_prob=0.1,
        salary_delay_days_max=3,
        burn_rate=0.03,
        buffer_ratio=0.02,
        optout_sensitivity=0.03,
        balance_seed=1,
    )
    with pytest.raises(Exception):
        make_state(FAILED_ON, customer=truth)


# --- the decision object -----------------------------------------------------


def test_retry_requires_a_date_and_others_forbid_one():
    with pytest.raises(Exception, match="RETRY requires a retry_date"):
        RetryDecision(action="RETRY")
    with pytest.raises(Exception, match="must not carry a retry_date"):
        RetryDecision(action="HOLD", retry_date=date(2026, 7, 5))
    assert RetryDecision(action="HOLD").is_terminal
    assert RetryDecision(action="CANCEL").is_terminal
    assert not RetryDecision(action="WAIT").is_terminal


def test_default_presentment_time_is_inside_the_execution_window():
    from generator import rules

    assert rules.in_execution_window(RetryDecision(action="WAIT").retry_time)


# --- the baseline ------------------------------------------------------------


def test_legacy_schedules_exactly_plus_two_four_seven_then_cancels():
    policy = LegacyFixedSchedule()
    scheduled = []
    for retries_made in range(3):
        decision = policy.decide(make_state(FAILED_ON, attempts_used=1 + retries_made))
        assert decision.action == "RETRY"
        scheduled.append((decision.retry_date - FAILED_ON).days)
    assert scheduled == [2, 4, 7]

    spent = policy.decide(make_state(FAILED_ON, attempts_used=4))
    assert spent.action == "CANCEL"
    assert spent.retry_date is None


def test_legacy_never_schedules_inside_the_24_hour_notice_window():
    policy = LegacyFixedSchedule()
    for offset in range(0, 8):
        today = FAILED_ON + timedelta(days=offset)
        decision = policy.decide(make_state(today, attempts_used=1))
        if decision.action == "RETRY":
            assert decision.retry_date >= today + timedelta(days=1)


def test_legacy_skips_offsets_whose_notice_window_has_closed():
    """Deciding late must skip +2, not clamp it to today."""
    policy = LegacyFixedSchedule()
    decision = policy.decide(make_state(FAILED_ON + timedelta(days=3)))
    assert decision.action == "RETRY"
    assert (decision.retry_date - FAILED_ON).days == 4
    assert decision.reasoning["skipped_offsets"] == [
        {"offset": 2, "reason": "lead_time_missed"}
    ]


def test_legacy_skips_offsets_that_fall_outside_the_cycle():
    policy = LegacyFixedSchedule()
    decision = policy.decide(
        make_state(FAILED_ON, cycle_end_date=FAILED_ON + timedelta(days=3))
    )
    assert decision.action == "RETRY"
    assert (decision.retry_date - FAILED_ON).days == 2

    later = policy.decide(
        make_state(
            FAILED_ON, attempts_used=2, cycle_end_date=FAILED_ON + timedelta(days=3)
        )
    )
    assert later.action == "CANCEL"
    assert {s["reason"] for s in later.reasoning["skipped_offsets"]} == {"outside_cycle"}


def test_legacy_cannot_be_built_above_the_npci_ceiling():
    with pytest.raises(ValueError, match="NPCI ceiling"):
        LegacyFixedSchedule(offsets=(1, 2, 3, 4))


def test_legacy_records_that_it_used_no_payday_or_cost_reasoning():
    """The audit trail must say what the baseline did NOT consider."""
    decision = LegacyFixedSchedule().decide(make_state(FAILED_ON))
    assert decision.reasoning["payday_considered"] is False
    assert decision.reasoning["optout_cost_considered"] is False
