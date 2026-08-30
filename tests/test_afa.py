"""Tests for failure class 5: AFA incomplete (RBI Sec 8(a), return code 1A).

The properties worth pinning here are not the same ones classes 1-4 needed.
This is the first class where a policy ACTION changes the truth, so alongside
the usual "is the model right" tests there are three that guard the mechanism
itself:

* a nudge is OUT OF BAND - no notification, no NPCI attempt. That is the whole
  reason it is cheaper than presenting, and it is the property most likely to
  regress the next time somebody touches the simulator.
* the ceiling can nudge too, so the agent cannot beat it by using an action the
  oracle was denied.
* the class-1 nudge finding is untouched, because the switch is class-dependent
  rather than a flipped global default.
"""

from __future__ import annotations

import ast
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

import router
from agent.otp_completion import (
    MAX_NUDGES_PER_CYCLE,
    AuthOutcome,
    OtpCompletionModel,
)
from agent.sequencer import SequencerConfig, SequencerPolicy
from classes.afa_incomplete import AfaIncompleteHandler
from evaluation.simulator import run_episode
from generator import rules
from generator.afa import (
    MAX_USEFUL_NUDGES,
    completion_probability,
    otp_completed,
)
from generator.generate import agent_view, generate_dataset
from generator.models import HIDDEN_CUSTOMER_FIELDS
from generator.oracle import first_success_date, would_succeed, would_succeed_at
from policy import LegacyFixedSchedule
from policy.oracle_policy import OraclePolicy

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_customers=500, seed=1)


def afa_events(dataset):
    return [
        e
        for e in dataset.failure_events
        if e.failure_class == rules.FAILURE_CLASS_AFA_INCOMPLETE
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


def _empty_outages(bank_id: str):
    """An outage book where nothing is ever down. Isolates the AFA gate."""
    from generator.banks import BankOutageBook

    return BankOutageBook(
        seed=0,
        window_start=datetime(2000, 1, 1),
        window_end=datetime(2100, 1, 1),
        intervals={bank_id: []},
    )


# --- generation --------------------------------------------------------------


def test_afa_failures_are_coded_1a_and_only_above_the_sec_8a_ceiling(dataset):
    """The threshold link is asserted through rules, never against a literal.

    Hardcoding 15,000 here would let the test keep passing if the regulation
    constant moved while the generator did not - which is the exact failure the
    threshold living in rules.py exists to prevent.
    """
    events = afa_events(dataset)
    assert events, "no AFA failures in this dataset to check"
    for event in events:
        customer = dataset.customer(event.customer_id)
        assert event.return_code == rules.RETURN_CODE_AFA_INCOMPLETE
        assert rules.requires_otp(customer.subscription_amount)
        assert customer.requires_otp
        # And the mandate is ALIVE - this is a gate, not a death.
        assert customer.mandate_dead_from is None or (
            event.original_debit_date < customer.mandate_dead_from
        )


def test_no_mandate_below_the_ceiling_ever_returns_1a(dataset):
    """The gate cannot fire where Sec 8(a) exempts the transaction."""
    for record in dataset.payment_history:
        if record.return_code == rules.RETURN_CODE_AFA_INCOMPLETE:
            assert rules.requires_otp(record.amount), (
                f"{record.customer_id} got 1A on Rs {record.amount}, which is "
                "below the Sec 8(a) ceiling and needs no authentication"
            )


def test_authentication_is_per_presentment_not_once_and_done():
    """Sec 8(a) exempts AFA 'per transaction', so it never carries over.

    This is the structural difference from expired_card, and the one most
    likely to be quietly optimised away by somebody caching a completion.
    """
    dataset = generate_dataset(n_customers=500, seed=3)
    gated = [c for c in dataset.customers if c.requires_otp and 0.2 < c.otp_responsiveness < 0.9]
    assert gated, "need a customer who sometimes authenticates and sometimes does not"

    mixed = 0
    for customer in gated:
        results = [
            otp_completed(customer, date(2026, 6, 1) + timedelta(days=30 * k))
            for k in range(12)
        ]
        if any(results) and not all(results):
            mixed += 1
    assert mixed, (
        "no customer had a cycle they authenticated and a cycle they did not - "
        "authentication is being carried over between transactions"
    )


def test_completing_one_cycle_does_not_authenticate_the_next():
    """The same property, stated as sharply as it can be stated."""
    dataset = generate_dataset(n_customers=400, seed=5)
    for customer in dataset.customers:
        if not customer.requires_otp:
            continue
        days = [date(2026, 5, 1) + timedelta(days=k) for k in range(120)]
        done = [d for d in days if otp_completed(customer, d)]
        missed = [d for d in days if not otp_completed(customer, d)]
        if done and missed:
            # There exists a later presentment that is NOT authenticated even
            # though an earlier one was.
            assert any(m > done[0] for m in missed)
            return
    pytest.skip("no partially-responsive OTP customer in this dataset")


# --- the truth model ---------------------------------------------------------


def test_a_nudge_raises_completion_and_the_second_raises_it_less():
    for responsiveness in (0.15, 0.4, 0.7):
        p = [completion_probability(responsiveness, n) for n in range(4)]
        assert p[1] > p[0], "a nudge must raise the completion probability"
        assert p[2] > p[1]
        first, second, third = p[1] - p[0], p[2] - p[1], p[3] - p[2]
        assert second < first, "the second nudge must be worth less than the first"
        assert third < second, "returns must keep diminishing"


def test_nudging_cannot_drive_completion_to_certainty():
    """No number of messages makes a human certain to act."""
    for responsiveness in (0.05, 0.3, 0.6, 0.9):
        assert completion_probability(responsiveness, 500) < 1.0
        assert completion_probability(responsiveness, 500) <= max(responsiveness, 0.95)


def test_nudging_barely_helps_a_chronically_unresponsive_customer():
    """Responsiveness scales the lift as well as the baseline.

    A flat lift would let one message rescue somebody who ignores every
    authentication prompt, which is the opposite of what the trait means.
    """
    stubborn = completion_probability(0.02, 1) - completion_probability(0.02, 0)
    ordinary = completion_probability(0.60, 1) - completion_probability(0.60, 0)
    assert stubborn < ordinary / 5


def test_the_completion_draw_is_keyed_on_identity_not_on_nudges(dataset):
    """Common random numbers: nudging moves the threshold, never the draw.

    That is what makes 'would this have completed anyway?' an exact
    counterfactual rather than a second sample, and it is what the simulator's
    nudge attribution depends on.
    """
    from generator.afa import completion_draw

    customer = next(c for c in dataset.customers if c.requires_otp)
    day = date(2026, 7, 15)
    assert completion_draw(customer, day) == completion_draw(customer, day)
    # Monotone: anything that completes unnudged still completes when nudged.
    for k in range(60):
        d = day + timedelta(days=k)
        if otp_completed(customer, d, 0):
            assert otp_completed(customer, d, 1)


# --- the oracle: the three-way overlap ---------------------------------------


def test_would_succeed_at_requires_authentication_and_balance_and_bank(dataset):
    """Written out explicitly rather than trusting the AND to fall out.

    Three independent gates, and every pair can disagree. A customer can be
    authenticated and broke; funded and unauthenticated; or fine on both and
    blocked by the bank.
    """
    def at(day):
        return datetime.combine(day, rules.DEFAULT_PRESENTMENT_TIME)

    # Search for a customer who actually exhibits all three combinations. Not
    # every OTP-gated subscriber does - they are the highest earners on the
    # book, so plenty are never short - and picking the first one and hoping
    # would make this test pass or fail on the seed rather than on the logic.
    chosen = None
    for candidate in dataset.customers:
        if not (candidate.requires_otp and 0.15 < candidate.otp_responsiveness < 0.85):
            continue
        series = dataset.balance_of(candidate.customer_id)
        amount = candidate.subscription_amount
        days = [stamp.date() for stamp in series.index]
        funded = [d for d in days if would_succeed(candidate, d, amount, series)]
        broke = [d for d in days if not would_succeed(candidate, d, amount, series)]
        if not (funded and broke):
            continue
        done_and_broke = next((d for d in broke if otp_completed(candidate, d)), None)
        missing_and_funded = next(
            (d for d in funded if not otp_completed(candidate, d)), None
        )
        both = next((d for d in funded if otp_completed(candidate, d)), None)
        if done_and_broke and missing_and_funded and both:
            chosen = candidate
            otp_done_and_broke = done_and_broke
            otp_missing_and_funded = missing_and_funded
            both_fine = both
            break

    assert chosen is not None, "no customer exhibits all three gate combinations"
    customer = chosen
    series = dataset.balance_of(customer.customer_id)
    amount = customer.subscription_amount
    up = _empty_outages(customer.bank_id)

    # OTP done + broke -> False
    assert not would_succeed_at(customer, at(otp_done_and_broke), amount, series, up)
    # OTP missing + funded -> False
    assert not would_succeed_at(
        customer, at(otp_missing_and_funded), amount, series, up
    )
    # both satisfied -> True
    assert would_succeed_at(customer, at(both_fine), amount, series, up)


def test_the_oracle_does_not_report_an_authentication_block_as_recoverable(dataset):
    """success_curve must AND in the AFA gate, not only the balance."""
    from generator.oracle import success_curve

    customer = next(
        c
        for c in dataset.customers
        if c.requires_otp and 0.15 < c.otp_responsiveness < 0.85
    )
    series = dataset.balance_of(customer.customer_id)
    amount = customer.subscription_amount
    start, end = date(2026, 6, 1), date(2026, 8, 31)
    curve = success_curve(customer, series, amount, start=start, end=end)
    for stamp, clears in curve.items():
        if clears:
            assert otp_completed(customer, stamp.date()), (
                f"{stamp.date()} reported as clearing while the AFA gate was shut"
            )


def test_nudging_can_only_widen_the_set_of_recoverable_dates(dataset):
    customer = next(
        c
        for c in dataset.customers
        if c.requires_otp and 0.15 < c.otp_responsiveness < 0.85
    )
    series = dataset.balance_of(customer.customer_id)
    amount = customer.subscription_amount
    unnudged = first_success_date(
        customer, series, amount, after=date(2026, 6, 1), horizon_days=60, nudges=0
    )
    nudged = first_success_date(
        customer,
        series,
        amount,
        after=date(2026, 6, 1),
        horizon_days=60,
        nudges=MAX_USEFUL_NUDGES,
    )
    if unnudged is not None:
        assert nudged is not None and nudged <= unnudged


# --- the agent model ---------------------------------------------------------


def test_otp_completion_is_agent_safe():
    forbidden = {
        "generator.oracle",
        "generator.balance",
        "generator.banks",
        "generator.afa",
    }
    tree = ast.parse(
        (SRC / "agent" / "otp_completion.py").read_text(encoding="utf-8")
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not (imported & forbidden)


def test_auth_outcome_accepts_only_agent_visible_fields():
    """Four fields, all of which a real PSP has on every presentment."""
    fields = set(AuthOutcome.__dataclass_fields__)
    assert fields == {"at", "amount", "succeeded", "return_code"}
    assert not (fields & HIDDEN_CUSTOMER_FIELDS)


def test_otp_responsiveness_never_reaches_the_agent(dataset):
    assert "otp_responsiveness" in HIDDEN_CUSTOMER_FIELDS
    for customer in agent_view(dataset).customers:
        assert "otp_responsiveness" not in customer.model_dump()


def test_cold_start_returns_the_book_prior_not_a_crash_or_a_certainty():
    model = OtpCompletionModel()
    assert model.cold_start
    p = model.completion_probability()
    assert 0.0 < p < 1.0
    assert p == pytest.approx(model.book_rate)


def test_a_run_of_1a_lowers_the_estimate_and_04_raises_it():
    """The discriminating property: both are declines on a large mandate."""
    big = 19_000.0
    refused = OtpCompletionModel()
    authenticated = OtpCompletionModel()
    for k in range(8):
        day = date(2026, 1, 1) + timedelta(days=k)
        refused.observe(AuthOutcome(at=day, amount=big, succeeded=False, return_code="1A"))
        authenticated.observe(
            AuthOutcome(at=day, amount=big, succeeded=False, return_code="04")
        )
    assert refused.completion_probability() < 0.4
    assert authenticated.completion_probability() > 0.8


def test_censored_declines_are_not_authentication_evidence():
    """A 91 never reached the gate; a 54 had no live credential."""
    big = 19_000.0
    model = OtpCompletionModel()
    for code in ("91", "54", "22"):
        for k in range(6):
            model.observe(
                AuthOutcome(
                    at=date(2026, 1, 1) + timedelta(days=k),
                    amount=big,
                    succeeded=False,
                    return_code=code,
                )
            )
    assert model.evidence_count == 0
    assert model.cold_start
    assert model.completion_probability() == pytest.approx(model.book_rate)


def test_small_mandates_are_not_authentication_evidence():
    model = OtpCompletionModel()
    for k in range(10):
        model.observe(
            AuthOutcome(
                at=date(2026, 1, 1) + timedelta(days=k),
                amount=199.0,
                succeeded=False,
                return_code="04",
            )
        )
    assert model.evidence_count == 0


def test_agent_nudge_lift_diminishes_and_never_reaches_certainty():
    model = OtpCompletionModel()
    lifts = [model.nudge_lift(k) for k in range(4)]
    assert all(a > b for a, b in zip(lifts, lifts[1:])), lifts
    assert model.completion_probability(1000) < 1.0


# --- routing -----------------------------------------------------------------


def test_router_sends_1a_to_the_real_handler_not_the_old_stub():
    handler = router.route_return_code(rules.RETURN_CODE_AFA_INCOMPLETE)
    assert isinstance(handler, AfaIncompleteHandler)
    assert handler.failure_class == rules.FAILURE_CLASS_AFA_INCOMPLETE
    assert handler.implemented is True
    assert "NOT IMPLEMENTED" not in handler.rationale


def test_afa_offers_both_nudge_and_retry():
    """Unlike a dead mandate, a later presentment genuinely can clear."""
    actions = AfaIncompleteHandler().allowed_actions(None)
    assert "NUDGE" in actions and "RETRY" in actions


def test_afa_is_not_a_never_retry_class():
    assert rules.FAILURE_CLASS_AFA_INCOMPLETE not in rules.NEVER_RETRY_CLASSES
    assert rules.can_retry(rules.FAILURE_CLASS_AFA_INCOMPLETE)


def test_success_probability_is_the_product_of_both_terms(dataset):
    """P(authenticated) x P(balance), and neither term alone."""
    from policy.base import PolicyState

    event = afa_events(dataset)[0]
    customer = dataset.customer(event.customer_id)
    history = tuple(
        r for r in dataset.payment_history if r.customer_id == customer.customer_id
    )
    from generator.models import AgentCustomer, AgentFailureEvent

    state = PolicyState(
        customer=AgentCustomer.from_customer(customer),
        payment_history=history,
        failure_event=AgentFailureEvent.from_failure(event),
        today=event.original_debit_date,
        attempts_used=1,
        notifications_sent_this_cycle=0,
        cycle_end_date=event.original_debit_date + timedelta(days=29),
    )
    handler = AfaIncompleteHandler()
    day = event.original_debit_date + timedelta(days=5)

    from agent.payday import PaydayModel

    model = PaydayModel(tier=2)
    model.fit(history)
    liquidity = model.liquidity_curve(day, day, customer.subscription_amount)[day]
    completion = handler.completion_probability(state)

    assert handler.success_probability(state, day) == pytest.approx(
        completion * liquidity
    )
    # And it is strictly below the liquidity term alone, so a shut gate cannot
    # be papered over by a healthy balance.
    assert handler.success_probability(state, day) <= liquidity


# --- the simulator: the out-of-band property ---------------------------------


class AlwaysNudge:
    """Nudges on the first day of every cycle, then waits."""

    name = "always_nudge"

    def decide(self, state):
        from policy.base import RetryDecision

        if state.nudges_sent_this_cycle < MAX_NUDGES_PER_CYCLE:
            return RetryDecision(action="NUDGE", reasoning={"rule": "test"})
        return RetryDecision(action="WAIT", reasoning={"rule": "test"})


def test_a_nudge_consumes_no_attempt_and_sends_no_notification(dataset):
    """THE property most likely to regress, and the reason nudging is cheap.

    A nudge is a merchant message, not the issuer's pre-debit notice. If it
    ever started counting as a notification it would create an RBI Sec 6(c)
    opt-out opportunity that does not exist, and the whole cost argument for
    the action would be wrong.
    """
    event = afa_events(dataset)[0]
    result = play(AlwaysNudge(), dataset, event)

    assert result.nudges_sent > 0, "the test policy did not actually nudge"
    assert result.notifications_sent == 0, (
        "a nudge incremented the notification counter - it is out of band and "
        "must never create a Sec 6(c) opt-out opportunity"
    )
    assert result.outcome != "OPTED_OUT", "a nudge created a cancel opportunity"
    # Attempts are the ORIGINAL failure plus one auto-presentment per carried
    # cycle. None of them came from a nudge.
    assert result.attempts_used == 1 + result.cycles_held
    assert len(result.retry_dates) == 0


def test_a_nudge_costs_goodwill(dataset):
    event = afa_events(dataset)[0]
    customer = dataset.customer(event.customer_id)
    result = play(AlwaysNudge(), dataset, event)
    expected = (
        rules.NUDGE_GOODWILL_COST_FRACTION
        * customer.annual_ltv
        * result.nudges_sent
    )
    assert result.goodwill_cost_paid == pytest.approx(expected)
    assert result.goodwill_cost_paid > 0


def test_the_nudge_budget_is_per_cycle_not_per_episode(dataset):
    """AFA is required per transaction, so last cycle's ask does not carry."""
    event = afa_events(dataset)[0]
    result = play(AlwaysNudge(), dataset, event)
    if result.cycles_held == 0:
        pytest.skip("episode resolved in one cycle")
    assert result.nudges_sent > MAX_NUDGES_PER_CYCLE


def test_nudges_are_attributed_only_when_they_changed_the_outcome(dataset):
    """Attribution is an exact counterfactual, so it can never exceed nudges."""
    for event in afa_events(dataset):
        result = play(SequencerPolicy(), dataset, event)
        assert result.otp_completions_from_nudges <= result.nudges_sent
        if result.nudges_sent == 0:
            assert result.otp_completions_from_nudges == 0


def test_a_1a_refusal_is_recorded_as_1a_not_as_04(dataset):
    """The return code is the agent's only channel.

    Writing 04 onto an authentication refusal would teach the payday model the
    account was empty - the same bug code 91 surfaced in class 2.
    """
    seen = 0
    for event in afa_events(dataset):
        result = play(LegacyFixedSchedule(), dataset, event)
        for entry in result.decision_log:
            if "code" in entry.event:
                assert "code 04" not in entry.event or not result.presentments_refused_for_afa
        seen += result.presentments_refused_for_afa
    assert seen > 0, "no AFA refusals recorded across the class"


# --- the sequencer -----------------------------------------------------------


def test_the_sequencer_nudges_before_presenting_on_an_afa_failure(dataset):
    """The whole point of the class: ask, then present.

    Presenting into a shut gate cannot clear whatever the balance is, so a
    policy that presents first is spending an NPCI attempt and a cancel-capable
    notice to learn nothing.
    """
    nudged_first = presented_first = 0
    for event in afa_events(dataset):
        result = play(SequencerPolicy(), dataset, event)
        actions = [e.action for e in result.decision_log if e.action in ("NUDGE", "RETRY")]
        if not actions:
            continue
        if actions[0] == "NUDGE":
            nudged_first += 1
        else:
            presented_first += 1
    assert nudged_first > presented_first, (
        f"nudged first on {nudged_first}, presented first on {presented_first}"
    )


def test_the_sequencer_never_nudges_an_insufficient_funds_failure(dataset):
    """The class-1 finding stands: nudging somebody broke does not help.

    ``use_nudge`` is still False by default, and making AFA nudge must not have
    switched it on globally.
    """
    events = [
        e
        for e in dataset.failure_events
        if e.failure_class == rules.FAILURE_CLASS_INSUFFICIENT_FUNDS
    ][:60]
    assert events
    for event in events:
        result = play(SequencerPolicy(), dataset, event)
        assert result.nudges_sent == 0


def test_the_afa_nudge_switch_leaves_class_1_untouched():
    """The regression check: class-dependent, so class-1 numbers cannot move.

    Every insufficient-funds episode must score identically with the AFA nudge
    on and off. If flipping it moves a single rupee here, the switch has leaked
    out of its class.
    """
    on = SequencerPolicy(SequencerConfig(use_afa_nudge=True), name="on")
    off = SequencerPolicy(SequencerConfig(use_afa_nudge=False), name="off")
    checked = 0
    for seed in (1, 2, 3, 4, 5):
        data = generate_dataset(n_customers=400, seed=seed)
        for event in data.failure_events:
            if event.failure_class != rules.FAILURE_CLASS_INSUFFICIENT_FUNDS:
                continue
            a = play(on, data, event, seed)
            b = play(off, data, event, seed)
            assert a.outcome == b.outcome
            assert a.recovered_amount == b.recovered_amount
            assert a.net_value == pytest.approx(b.net_value)
            assert a.attempts_used == b.attempts_used
            assert a.notifications_sent == b.notifications_sent
            checked += 1
    assert checked > 200, f"only {checked} insufficient-funds episodes compared"


def test_sequencer_recovers_more_value_than_legacy_across_seeds():
    """Pooled and measured in RUPEES, because that is what this class is about.

    A recovery-rate comparison on a handful of episodes per seed would be
    measuring noise. The value figure is the claim.
    """
    legacy_money = agent_money = at_risk = 0.0
    episodes = 0
    for seed in range(1, 21):
        data = generate_dataset(n_customers=500, seed=seed)
        events = afa_events(data)
        if not events:
            continue
        for event in events:
            at_risk += data.customer(event.customer_id).subscription_amount
            legacy_money += play(LegacyFixedSchedule(), data, event, seed).recovered_amount
            agent_money += play(SequencerPolicy(), data, event, seed).recovered_amount
            episodes += 1
    assert episodes >= 50, f"only {episodes} AFA failures to judge on"
    assert agent_money > legacy_money, (
        f"agent recovered Rs {agent_money:,.0f} of Rs {at_risk:,.0f} against "
        f"legacy's Rs {legacy_money:,.0f}"
    )


def test_the_agent_never_beats_the_ceiling_on_value():
    """The leakage guard, and the reason the ceiling was taught to nudge.

    An oracle restricted to choosing dates would be an upper bound over a
    smaller action set than the agent's, and the agent could legitimately
    exceed it - which would make the ceiling meaningless rather than reveal a
    leak.
    """
    for seed in range(1, 11):
        data = generate_dataset(n_customers=500, seed=seed)
        events = afa_events(data)
        if not events:
            continue
        oracle = OraclePolicy.for_dataset(data)
        agent_money = sum(
            play(SequencerPolicy(), data, e, seed).recovered_amount for e in events
        )
        oracle_money = sum(play(oracle, data, e, seed).recovered_amount for e in events)
        assert agent_money <= oracle_money + 1e-6, (
            f"seed {seed}: agent Rs {agent_money:,.0f} > ceiling Rs {oracle_money:,.0f}"
        )


def test_the_agent_never_recovers_what_no_date_could_have():
    """Money can only be collected on a date the truth says could clear."""
    for seed in (1, 2, 3):
        data = generate_dataset(n_customers=500, seed=seed)
        for event in afa_events(data):
            customer = data.customer(event.customer_id)
            result = play(SequencerPolicy(), data, event, seed)
            if result.outcome != "RECOVERED":
                continue
            reachable = first_success_date(
                customer,
                data.balance_of(event.customer_id),
                customer.subscription_amount,
                after=event.original_debit_date,
                horizon_days=400,
                nudges=MAX_USEFUL_NUDGES,
            )
            assert reachable is not None, (
                f"{event.customer_id} recovered on a path the truth says has no "
                "clearing date at all"
            )


def test_the_no_afa_nudge_ablation_changes_what_the_agent_does():
    """The ablation must actually ablate something."""
    data = generate_dataset(n_customers=500, seed=1)
    events = afa_events(data)
    assert events
    blind = SequencerPolicy(SequencerConfig(use_afa_nudge=False), name="no_afa_nudge")
    with_nudge = sum(play(SequencerPolicy(), data, e).nudges_sent for e in events)
    without = sum(play(blind, data, e).nudges_sent for e in events)
    assert with_nudge > 0
    assert without == 0


# --- reproducibility ---------------------------------------------------------


def test_an_afa_run_is_reproducible_seed_for_seed():
    def run(seed):
        data = generate_dataset(n_customers=400, seed=seed)
        return [
            (
                e.customer_id,
                r.outcome,
                r.recovered_amount,
                r.attempts_used,
                r.nudges_sent,
                r.goodwill_cost_paid,
                r.otp_completions_from_nudges,
                r.presentments_refused_for_afa,
            )
            for e in afa_events(data)
            for r in [play(SequencerPolicy(), data, e, seed)]
        ]

    for seed in (1, 2):
        assert run(seed) == run(seed)


def test_the_dataset_reports_the_realised_afa_share(dataset):
    """Thin by count, heavy by value - and both numbers are in meta."""
    assert "afa_share" in dataset.meta
    assert "afa_value_share" in dataset.meta
    assert dataset.meta["afa_share"] > 0
    assert dataset.meta["afa_value_share"] > dataset.meta["afa_share"], (
        "the whole argument for this class is that it carries more value than "
        "its episode count suggests"
    )
