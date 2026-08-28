"""Tests for the money equation and the agent-side churn estimator."""

from __future__ import annotations

from datetime import date, time, timedelta

import pytest

from agent.churn import ASSUMED_OPTOUT_SENSITIVITY, estimated_optout_probability
from agent.decision_engine import DecisionEngine
from generator import rules
from generator.models import (
    HIDDEN_CUSTOMER_FIELDS,
    AgentCustomer,
    AgentFailureEvent,
    AgentPaymentHistoryRecord,
)
from policy.base import PolicyState

FAILED_ON = date(2026, 7, 2)
CYCLE_END = date(2026, 8, 1)


def agent_customer(**overrides) -> AgentCustomer:
    base = dict(
        customer_id="CUST_0001",
        subscription_amount=500.0,
        rail="upi_autopay",
        debit_day_of_month=2,
        tenure_months=9,
        monthly_ltv=6000.0,
    )
    base.update(overrides)
    return AgentCustomer(**base)


def make_state(today: date = FAILED_ON, attempts_used: int = 1, **overrides) -> PolicyState:
    base = dict(
        customer=agent_customer(),
        payment_history=(
            AgentPaymentHistoryRecord(
                customer_id="CUST_0001",
                attempt_date=FAILED_ON,
                amount=500.0,
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


def flat_curve(value: float, start: date = FAILED_ON, days: int = 40) -> dict[date, float]:
    return {start + timedelta(days=i): value for i in range(days)}


# --- the agent-side churn estimator -----------------------------------------


def test_churn_estimator_uses_no_hidden_fields():
    """The true hazard reads optout_sensitivity, which the agent cannot see."""
    customer = agent_customer()
    assert set(customer.model_dump()).isdisjoint(HIDDEN_CUSTOMER_FIELDS)
    p = estimated_optout_probability(customer, 1, 0.0)
    assert 0.0 < p < 1.0


def test_churn_estimator_matches_the_book_average_on_a_fresh_notice():
    """With no annoyance and no tenure discount the estimate is the assumed
    population rate, scaled only by recency."""
    brand_new = agent_customer(tenure_months=1)
    p = estimated_optout_probability(brand_new, 1, days_since_failure=100.0)
    expected = ASSUMED_OPTOUT_SENSITIVITY / (1.0 + 1 / 12.0)
    assert p == pytest.approx(expected, rel=0.02)


def test_churn_rises_with_notices_and_falls_with_tenure():
    customer = agent_customer()
    probs = [estimated_optout_probability(customer, n, 3.0) for n in range(1, 5)]
    assert all(b > a for a, b in zip(probs, probs[1:]))

    loyal = agent_customer(tenure_months=48)
    assert estimated_optout_probability(loyal, 2, 3.0) < estimated_optout_probability(
        customer, 2, 3.0
    )


def test_churn_estimator_honours_both_irritation_directions():
    customer = agent_customer()
    assert estimated_optout_probability(customer, 2, 0.0, "decay") > (
        estimated_optout_probability(customer, 2, 21.0, "decay")
    )
    assert estimated_optout_probability(customer, 2, 21.0, "build") > (
        estimated_optout_probability(customer, 2, 0.0, "build")
    )


# --- candidate generation ----------------------------------------------------


def test_candidates_respect_the_notice_period():
    engine = DecisionEngine()
    assert engine.lead_time_days == 1
    days = engine.candidate_dates(make_state())
    assert days[0] == FAILED_ON + timedelta(days=1)
    assert days[-1] == CYCLE_END


@pytest.mark.parametrize(
    "hours,expected_days", [(0, 0), (6, 1), (12, 1), (24, 1), (48, 2)]
)
def test_lead_time_converts_to_whole_days(hours, expected_days):
    """The simulator runs at daily resolution, so 6h, 12h and 24h collapse to
    the same constraint. Reported rather than papered over."""
    engine = DecisionEngine(lead_time_hours=hours)
    assert engine.lead_time_days == expected_days
    assert engine.candidate_dates(make_state())[0] == FAILED_ON + timedelta(
        days=expected_days
    )


def test_no_candidates_once_the_attempt_budget_is_spent():
    engine = DecisionEngine()
    state = make_state(attempts_used=rules.MAX_ATTEMPTS_PER_CYCLE)
    assert engine.candidate_dates(state) == []
    assert engine.binding_constraints()[0]["reason"] == "attempt_budget"


def test_no_candidates_past_the_end_of_the_cycle():
    engine = DecisionEngine()
    assert engine.candidate_dates(make_state(today=CYCLE_END)) == []
    assert engine.binding_constraints()[0]["reason"] == "cycle_end"


def test_no_candidates_outside_the_execution_window():
    engine = DecisionEngine(presentment_time=time(13, 0))
    assert engine.candidate_dates(make_state()) == []
    assert engine.binding_constraints()[0]["reason"] == "execution_window"


# --- the money equation ------------------------------------------------------


def test_value_follows_the_stated_equation():
    engine = DecisionEngine()
    state = make_state()
    scored = engine.score(state, flat_curve(0.6))
    best = scored[0]
    customer = state.customer

    expected = (
        0.6 * customer.subscription_amount
        - rules.processing_cost(customer.rail)
        - 0.4 * rules.bounce_charge(customer.rail)
        - best.p_optout * customer.annual_ltv
    )
    assert best.value == pytest.approx(expected)


def test_the_whole_scored_list_is_returned_best_first():
    engine = DecisionEngine()
    curve = flat_curve(0.2)
    curve[date(2026, 7, 10)] = 0.9
    scored = engine.score(make_state(), curve)

    # 3 July .. 1 August inclusive
    assert len(scored) == 30
    assert scored[0].day == date(2026, 7, 10)
    assert [s.value for s in scored] == sorted((s.value for s in scored), reverse=True)


def test_a_nach_retry_is_worth_less_than_the_same_upi_retry():
    """The Rs 500 bounce charge only exists on one rail, so the same date is
    worth presenting on UPI and not on NACH."""
    engine = DecisionEngine()
    curve = flat_curve(0.3)
    upi = engine.score(make_state(), curve)[0]
    nach = engine.score(make_state(customer=agent_customer(rail="nach")), curve)[0]
    assert nach.value < upi.value
    assert nach.expected_bounce == pytest.approx(0.7 * rules.NACH_BOUNCE_CHARGE_INR)
    assert upi.expected_bounce == 0.0


def test_churn_cost_scales_with_lifetime_value_not_the_amount_at_stake():
    engine = DecisionEngine()
    curve = flat_curve(0.5)
    cheap = engine.score(make_state(), curve)[0]
    assert cheap.expected_churn == pytest.approx(cheap.p_optout * 6000.0)
    # Chasing Rs 500 risks Rs 6,000 - an order of magnitude more than the ask.
    assert cheap.expected_churn > 0.1 * cheap.expected_recovery


def test_more_notifications_already_sent_make_every_date_worth_less():
    engine = DecisionEngine()
    curve = flat_curve(0.5)
    first = engine.score(make_state(notifications_sent_this_cycle=0), curve)[0]
    fourth = engine.score(make_state(notifications_sent_this_cycle=3), curve)[0]
    assert fourth.value < first.value
    assert fourth.p_optout > first.p_optout


def test_dropping_the_optout_term_raises_every_value():
    """The ablation switch has to actually change the arithmetic."""
    curve = flat_curve(0.5)
    with_cost = DecisionEngine().score(make_state(), curve)[0]
    without = DecisionEngine(include_optout_cost=False).score(make_state(), curve)[0]
    assert without.p_optout == 0.0
    assert without.value > with_cost.value


def test_a_longer_lead_time_lands_the_notice_closer_to_the_bounce():
    """The customer reacts to the notice, not the debit. A 48-hour notice on
    the same date is sent a day earlier, when irritation is fresher."""
    curve = flat_curve(0.5)
    day = date(2026, 7, 12)
    short = {s.day: s for s in DecisionEngine(lead_time_hours=24).score(make_state(), curve)}
    long = {s.day: s for s in DecisionEngine(lead_time_hours=48).score(make_state(), curve)}
    assert long[day].p_optout > short[day].p_optout


def test_scored_dates_serialise_for_the_audit_trail():
    scored = DecisionEngine().score(make_state(), flat_curve(0.4))[0]
    row = scored.as_dict()
    assert set(row) >= {"date", "p_clear", "p_optout", "value", "days_after_failure"}
    assert isinstance(row["date"], str)


# --- nudge -------------------------------------------------------------------


def test_nudge_is_worthless_when_the_date_is_already_good_or_already_hopeless():
    engine = DecisionEngine()
    assert engine.nudge_value(make_state(), flat_curve(0.95)) == 0.0
    assert engine.nudge_value(make_state(), flat_curve(0.05)) == 0.0


def test_nudge_has_some_value_in_the_middle_band():
    engine = DecisionEngine()
    value = engine.nudge_value(make_state(), flat_curve(0.5))
    assert value != 0.0
