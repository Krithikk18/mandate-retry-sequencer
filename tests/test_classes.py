"""Tests for the failure-class handlers, the router, and the audit trail."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

import pytest

import router
from agent.sequencer import SequencerPolicy
from audit.trail import render_episode
from classes.expired_card import ExpiredCardHandler
from classes.opted_out import OptedOutHandler
from evaluation.simulator import ComplianceViolation, run_episode
from generator import rules
from generator.generate import generate_dataset
from generator.models import AgentCustomer, AgentFailureEvent, AgentPaymentHistoryRecord
from policy import LegacyFixedSchedule
from policy.base import Policy, PolicyState, RetryDecision

FAILED_ON = date(2026, 7, 12)


def make_state(failure_class: str, return_code: str) -> PolicyState:
    return PolicyState(
        customer=AgentCustomer(
            customer_id="CUST_0001",
            subscription_amount=500.0,
            rail="upi_autopay",
            debit_day_of_month=12,
            tenure_months=9,
            monthly_ltv=6000.0,
        ),
        payment_history=(
            AgentPaymentHistoryRecord(
                customer_id="CUST_0001",
                attempt_date=FAILED_ON,
                amount=500.0,
                succeeded=False,
                return_code=return_code,
            ),
        ),
        failure_event=AgentFailureEvent(
            customer_id="CUST_0001",
            cycle_start_date=FAILED_ON,
            original_debit_date=FAILED_ON,
            return_code=return_code,
            failure_class=failure_class,
        ),
        today=FAILED_ON,
        attempts_used=1,
        notifications_sent_this_cycle=0,
        cycle_end_date=FAILED_ON + timedelta(days=30),
    )


# --- the router --------------------------------------------------------------


def test_every_return_code_dispatches_to_a_handler():
    for code, expected in rules.RETURN_CODE_TO_CLASS.items():
        handler = router.route_return_code(code)
        assert handler.failure_class == expected, f"{code} routed to {handler}"


def test_no_return_code_falls_through():
    """An acquirer emits codes nobody documented. None may crash the router."""
    for code in ["ZZ", "", "99", "AB12", None]:
        handler = router.route_return_code(code)
        assert handler.failure_class == rules.FAILURE_CLASS_AMBIGUOUS


def test_every_declared_class_has_a_handler():
    for failure_class in rules.FAILURE_CLASSES:
        handler = router.handler_for_class(failure_class)
        assert handler.failure_class == failure_class


def test_an_unregistered_class_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="no handler registered"):
        router.handler_for_class("invented_class")


def test_never_retry_classes_have_handlers_that_exclude_retry():
    """The invariant the router asserts at import, tested explicitly."""
    assert rules.NEVER_RETRY_CLASSES
    for failure_class in rules.NEVER_RETRY_CLASSES:
        handler = router.handler_for_class(failure_class)
        state = make_state(failure_class, "22")
        assert "RETRY" not in handler.allowed_actions(state)
        assert not handler.permits_retry(state)


def test_the_router_checks_its_own_consistency():
    """Adding a class must not be able to silently make a dead mandate
    retryable, so the check runs at import and is callable on demand."""
    router._check_registry_is_consistent()


def test_only_insufficient_funds_and_the_dead_classes_are_implemented():
    """The stubs must not be mistaken for models."""
    assert router.implemented_classes() == {
        rules.FAILURE_CLASS_INSUFFICIENT_FUNDS,
        rules.FAILURE_CLASS_EXPIRED_CARD,
        rules.FAILURE_CLASS_OPTED_OUT,
    }
    for failure_class in (
        rules.FAILURE_CLASS_DOWNTIME,
        rules.FAILURE_CLASS_AFA_INCOMPLETE,
        rules.FAILURE_CLASS_AMBIGUOUS,
    ):
        handler = router.handler_for_class(failure_class)
        assert handler.implemented is False
        assert "NOT IMPLEMENTED" in handler.rationale


# --- the two never-retry classes ---------------------------------------------


@pytest.mark.parametrize("handler", [ExpiredCardHandler(), OptedOutHandler()])
def test_no_date_can_ever_succeed(handler):
    state = make_state(handler.failure_class, "54")
    day = FAILED_ON
    for _ in range(400):
        assert handler.success_probability(state, day) == 0.0
        day += timedelta(days=1)


def test_expired_card_allows_a_nudge_but_never_a_retry():
    handler = ExpiredCardHandler()
    actions = handler.allowed_actions(make_state(handler.failure_class, "54"))
    assert "RETRY" not in actions
    assert "NUDGE" in actions, "only the customer can replace the card"


def test_a_revoked_mandate_allows_nothing_but_cancel():
    handler = OptedOutHandler()
    actions = handler.allowed_actions(make_state(handler.failure_class, "22"))
    assert actions == ["CANCEL"]


# --- the regulatory assertion ------------------------------------------------


class RetryAlways(Policy):
    """Retries whenever the NPCI budget allows, ignoring the return code.

    Deliberately blind to failure class - that is what makes it useful for
    testing the class-level guards, without also tripping the attempt ceiling
    and masking which rule actually fired.
    """

    name = "retry_always"

    def decide(self, state: PolicyState) -> RetryDecision:
        if state.attempts_remaining <= 0:
            return RetryDecision(action="CANCEL", reasoning={"rule": "budget spent"})
        return RetryDecision(
            action="RETRY", retry_date=state.today + timedelta(days=1)
        )


def _dead_events(dataset, failure_class):
    return [e for e in dataset.failure_events if e.failure_class == failure_class]


@pytest.fixture(scope="module")
def mixed_dataset():
    return generate_dataset(n_customers=400, seed=42)


def test_retrying_a_revoked_mandate_raises(mixed_dataset):
    """Not a business rule. Consent was withdrawn under RBI Sec 6(c)."""
    revoked = _dead_events(mixed_dataset, rules.FAILURE_CLASS_OPTED_OUT)
    assert revoked, "no revoked mandates in this dataset"
    event = revoked[0]
    with pytest.raises(ComplianceViolation, match="Sec 6\\(c\\)"):
        run_episode(
            RetryAlways(),
            mixed_dataset.customer(event.customer_id),
            event,
            mixed_dataset.balance_of(event.customer_id),
            mixed_dataset.payment_history,
            seed=42,
        )


def test_retrying_an_expired_card_is_futile_but_lawful(mixed_dataset):
    """The distinction matters: waste is not the same as a breach."""
    expired = _dead_events(mixed_dataset, rules.FAILURE_CLASS_EXPIRED_CARD)
    assert expired
    event = expired[0]
    result = run_episode(
        RetryAlways(),
        mixed_dataset.customer(event.customer_id),
        event,
        mixed_dataset.balance_of(event.customer_id),
        mixed_dataset.payment_history,
        seed=42,
    )
    assert result.outcome != "RECOVERED"
    assert result.compliance_breaches == 0


def test_the_sequencer_never_presents_against_a_dead_mandate(mixed_dataset):
    policy = SequencerPolicy()
    for failure_class in rules.NEVER_RETRY_CLASSES:
        for event in _dead_events(mixed_dataset, failure_class):
            result = run_episode(
                policy,
                mixed_dataset.customer(event.customer_id),
                event,
                mixed_dataset.balance_of(event.customer_id),
                mixed_dataset.payment_history,
                seed=42,
            )
            assert result.retry_dates == ()
            assert result.notifications_sent == 0
            assert result.compliance_breaches == 0


def test_legacy_does_breach_because_it_never_reads_the_return_code(mixed_dataset):
    """The finding, pinned. Legacy presents against revoked mandates."""
    revoked = _dead_events(mixed_dataset, rules.FAILURE_CLASS_OPTED_OUT)
    breaches = sum(
        run_episode(
            LegacyFixedSchedule(),
            mixed_dataset.customer(e.customer_id),
            e,
            mixed_dataset.balance_of(e.customer_id),
            mixed_dataset.payment_history,
            seed=42,
            strict_compliance=False,
        ).compliance_breaches
        for e in revoked
    )
    assert breaches == len(revoked)


# --- the audit trail ---------------------------------------------------------


@pytest.fixture(scope="module")
def episodes(mixed_dataset):
    """One episode per outcome type, plus both policies."""
    out = {}
    for policy in (SequencerPolicy(), LegacyFixedSchedule()):
        for event in mixed_dataset.failure_events:
            result = run_episode(
                policy,
                mixed_dataset.customer(event.customer_id),
                event,
                mixed_dataset.balance_of(event.customer_id),
                mixed_dataset.payment_history,
                seed=42,
                strict_compliance=False,
            )
            out.setdefault(result.outcome, result)
    return out


def test_the_trail_renders_for_every_outcome(episodes):
    assert len(episodes) >= 4, f"only saw {sorted(episodes)}"
    for outcome, result in episodes.items():
        for style in ("text", "markdown", "json"):
            rendered = render_episode(result, style=style)
            assert rendered, f"{outcome}/{style} rendered empty"
            assert result.customer_id in rendered
        assert outcome in render_episode(result)


def test_the_trail_renders_every_failure_class(mixed_dataset):
    policy = SequencerPolicy()
    seen = set()
    for event in mixed_dataset.failure_events:
        if event.failure_class in seen:
            continue
        seen.add(event.failure_class)
        result = run_episode(
            policy,
            mixed_dataset.customer(event.customer_id),
            event,
            mixed_dataset.balance_of(event.customer_id),
            mixed_dataset.payment_history,
            seed=42,
        )
        rendered = render_episode(result)
        assert event.failure_class in rendered
        assert event.return_code in rendered
    assert len(seen) >= 3


def test_rejects_an_unknown_style(episodes):
    with pytest.raises(ValueError, match="style must be"):
        render_episode(next(iter(episodes.values())), style="html")


def test_trail_numbers_come_from_reasoning_not_recomputation(episodes):
    """The renderer must not have its own opinion about the arithmetic.

    Every rupee figure printed for a candidate has to be the one the policy
    recorded at decision time. A trail that re-derives its numbers can quietly
    disagree with the run it claims to describe.
    """
    checked = 0
    for result in episodes.values():
        rendered = render_episode(result)
        for entry in result.decision_log:
            candidates = entry.reasoning.get("candidates") or []
            if not candidates:
                continue
            best = candidates[0]
            for key in ("expected_recovery", "expected_bounce", "expected_churn", "value"):
                value = abs(best[key])
                shown = "Rs 0" if value < 0.5 else f"Rs {value:,.0f}"
                assert shown in rendered, (
                    f"{key}={best[key]} was not printed as recorded "
                    f"(looked for {shown!r})"
                )
                checked += 1
    assert checked > 0, "no scored candidates appeared in any trail"


def test_trail_probabilities_match_the_recorded_ones(episodes):
    for result in episodes.values():
        rendered = render_episode(result)
        for entry in result.decision_log:
            candidates = entry.reasoning.get("candidates") or []
            if not candidates:
                continue
            best = candidates[0]
            assert f"P(clear) {best['p_clear']:.2f}" in rendered
            assert f"opt-out risk {best['p_optout']:.3f}" in rendered


def test_json_style_round_trips(episodes):
    for result in episodes.values():
        payload = json.loads(render_episode(result, style="json"))
        assert payload["customer_id"] == result.customer_id
        assert payload["outcome"] == result.outcome
        assert payload["net_value"] == result.net_value
        assert len(payload["decisions"]) == len(result.decision_log)


def test_repeated_decisions_are_collapsed_not_dropped(mixed_dataset):
    """Folding identical days must summarise, never hide."""
    policy = SequencerPolicy()
    for event in mixed_dataset.failure_events[:60]:
        result = run_episode(
            policy,
            mixed_dataset.customer(event.customer_id),
            event,
            mixed_dataset.balance_of(event.customer_id),
            mixed_dataset.payment_history,
            seed=42,
        )
        rendered = render_episode(result)
        collapsed = re.findall(r"unchanged for (\d+) days", rendered)
        if collapsed:
            printed = rendered.count("DECISION:")
            assert printed < len(result.decision_log)
            assert all(int(n) > 1 for n in collapsed)
            return
    pytest.skip("no episode with repeated identical decisions in this sample")


def test_the_comparison_line_names_the_other_policy(mixed_dataset):
    event = mixed_dataset.failure_events[0]
    args = (
        mixed_dataset.customer(event.customer_id),
        event,
        mixed_dataset.balance_of(event.customer_id),
        mixed_dataset.payment_history,
    )
    agent = run_episode(SequencerPolicy(), *args, seed=42)
    legacy = run_episode(LegacyFixedSchedule(), *args, seed=42, strict_compliance=False)
    rendered = render_episode(agent, comparison=legacy)
    assert legacy.policy_name in rendered
    assert f"{legacy.notifications_sent} notifications" in rendered


# --- reproducibility ---------------------------------------------------------


def test_multiclass_run_is_reproducible():
    def run(seed: int):
        dataset = generate_dataset(n_customers=250, seed=seed)
        policy = SequencerPolicy()
        return [
            (
                e.failure_class,
                run_episode(
                    policy,
                    dataset.customer(e.customer_id),
                    e,
                    dataset.balance_of(e.customer_id),
                    dataset.payment_history,
                    seed=seed,
                    strict_compliance=False,
                ).net_value,
            )
            for e in dataset.failure_events
        ]

    assert run(7) == run(7)
    assert run(7) != run(8)
