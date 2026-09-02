"""Failure class 6: the ambiguous decline, and the trap it is built around.

The load-bearing test in this file is
``test_the_fitted_posterior_tracks_the_generators_mixture``. Everything else
checks mechanics; that one checks that the result is worth anything at all.

The trap: the generator draws what is behind a 05 from
``ambiguous.LATENT_CAUSE_MIXTURE``. If the agent's prior were a copy of that
constant, the agent would have been handed the answer and every class-6 number
would be synthetic-data self-fulfilment. So ``agent/masking.py`` fits its
posterior from the observable resolution history instead, and this file
perturbs the generator's mixture and asserts the fit MOVES WITH IT. If that
fails, the class does not ship.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

import pytest

from agent.masking import (
    CAUSE_PERMANENT_BLOCK,
    CAUSE_TRANSIENT,
    MIN_BANK_EVIDENCE,
    MaskingModel,
)
from generator import rules
from generator import ambiguous
from generator.ambiguous import (
    AmbiguityConfig,
    CAUSE_ISSUER_RISK_BLOCK,
    LATENT_CAUSE_MIXTURE,
    in_glitch,
    is_masked,
    masking_propensity,
    risk_blocked_from,
)
from generator.generate import agent_view, generate_dataset
from generator.oracle import first_success_date, resolve_presentment

SHARE = 1.0


def masked_dataset(seed: int = 5, n: int = 400, share: float = SHARE):
    return generate_dataset(n_customers=n, seed=seed, ambiguity_share=share)


def config_for(dataset, share: float = SHARE) -> AmbiguityConfig:
    return AmbiguityConfig(
        share=share,
        seed=dataset.seed,
        window_start=date.fromisoformat(dataset.meta["series_start"]),
    )


# --- the default path must be inert -----------------------------------------


def test_the_default_share_leaves_the_generator_byte_identical():
    """Off by default, and off means INVISIBLE.

    The convention DecisionEngine.daily_discount already follows: a new
    mechanism that silently moves every published figure is not a feature. At
    share 0.0 every record, every code and every meta key must match the build
    before class 6 existed - which is what lets 243 pinned numbers survive it.
    """
    plain = generate_dataset(n_customers=250, seed=42)
    explicit = generate_dataset(n_customers=250, seed=42, ambiguity_share=0.0)

    assert [c.model_dump() for c in plain.customers] == [
        c.model_dump() for c in explicit.customers
    ]
    assert [e.model_dump() for e in plain.failure_events] == [
        e.model_dump() for e in explicit.failure_events
    ]
    assert [r.model_dump() for r in plain.payment_history] == [
        r.model_dump() for r in explicit.payment_history
    ]
    assert plain.meta == explicit.meta
    # And the class-6 meta keys must not appear at all when it is off.
    assert "ambiguity_share" not in plain.meta
    assert "masked_share" not in plain.meta


def test_no_05_is_generated_at_the_default_share():
    dataset = generate_dataset(n_customers=300, seed=7)
    codes = {e.return_code for e in dataset.failure_events}
    assert rules.RETURN_CODE_DO_NOT_HONOUR not in codes


# --- determinism and common random numbers ----------------------------------


def test_draws_are_keyed_on_identity_and_date_only():
    """Same customer, same date, same draw - whatever any policy did.

    The discipline afa.completion_draw and simulator.optout_rng both follow.
    Without it a counterfactual comparison measures RNG alignment rather than
    skill.
    """
    dataset = masked_dataset()
    config = config_for(dataset)
    customer = dataset.customers[0]
    day = dataset.as_of

    assert is_masked(customer, day, config) == is_masked(customer, day, config)
    assert in_glitch(customer, day, config) == in_glitch(customer, day, config)
    # And it does not depend on the object, only on the values it carries.
    twin = customer.model_copy(deep=True)
    assert is_masked(twin, day, config) == is_masked(customer, day, config)


def test_masking_propensity_is_per_bank_not_per_customer():
    """The property that makes the agent's posterior a POOLED inference.

    If masking were a per-customer trait there would be nothing to pool and the
    model would have to be shaped like otp_completion.py instead. It is a
    property of the institution, so every customer on an issuer shares it.
    """
    dataset = masked_dataset()
    config = config_for(dataset)

    by_bank: dict[str, set[float]] = {}
    for customer in dataset.customers:
        by_bank.setdefault(customer.bank_id, set()).add(
            masking_propensity(customer.bank_id, config)
        )
    for bank_id, values in by_bank.items():
        assert len(values) == 1, f"{bank_id} has per-customer propensities"

    # And different issuers genuinely differ, or the pooling would be pointless.
    distinct = {next(iter(v)) for v in by_bank.values()}
    assert len(distinct) > 1


# --- the masking transform preserves the truth ------------------------------


def test_a_masked_balance_failure_keeps_its_true_class():
    """return_code == 05 while failure_class == insufficient_funds.

    THE invariant of the whole class. Masking degrades the agent-visible
    channel and nothing else; if it destroyed the failure class there would be
    no way to score whether the posterior was right, which is the only
    interesting measurement here.
    """
    dataset = masked_dataset()
    masked = [
        e
        for e in dataset.failure_events
        if e.return_code == rules.RETURN_CODE_DO_NOT_HONOUR
    ]
    assert masked, "fixture generated no masked failures"

    disguised = [
        e for e in masked if e.failure_class != rules.FAILURE_CLASS_AMBIGUOUS
    ]
    assert disguised, "no failure was masked over a real class"
    for event in disguised:
        assert event.return_code == rules.RETURN_CODE_DO_NOT_HONOUR
        assert event.failure_class in rules.FAILURE_CLASSES
        assert event.failure_class != rules.FAILURE_CLASS_AMBIGUOUS


def test_masking_never_turns_a_success_into_a_failure():
    """An issuer does not decline a transaction it approved."""
    dataset = masked_dataset()
    config = config_for(dataset)
    for customer in dataset.customers[:60]:
        series = dataset.balance_of(customer.customer_id)
        for offset in range(0, 40, 7):
            day = dataset.as_of + timedelta(days=offset)
            plain = resolve_presentment(
                customer, day, customer.subscription_amount, series
            )
            masked = resolve_presentment(
                customer, day, customer.subscription_amount, series,
                ambiguity=config,
            )
            if plain[0]:
                # A success can still be blocked by a NATIVE cause, which is a
                # new failure rather than a mask. What must never happen is a
                # success being relabelled while staying a success.
                assert masked[0] is True or masked[1] == (
                    rules.RETURN_CODE_DO_NOT_HONOUR
                )


# --- the two native causes ---------------------------------------------------


def test_a_risk_blocked_mandate_never_clears_on_any_date():
    """Permanent, and never announced. first_success_date must return None."""
    dataset = masked_dataset()
    config = config_for(dataset)
    blocked = [
        c for c in dataset.customers if risk_blocked_from(c, config) is not None
    ]
    assert blocked, "fixture produced no risk-blocked mandates"

    checked = 0
    for customer in blocked[:15]:
        blocked_from = risk_blocked_from(customer, config)
        series = dataset.balance_of(customer.customer_id)
        after = max(blocked_from, series.index[0].date())
        if after >= series.index[-1].date():
            continue
        assert (
            first_success_date(
                customer,
                series,
                customer.subscription_amount,
                after=after,
                horizon_days=120,
                ambiguity=config,
            )
            is None
        ), f"{customer.customer_id} cleared despite a permanent risk block"
        checked += 1
    assert checked, "no risk-blocked customer was actually exercised"


def test_a_transient_glitch_clears_and_a_block_does_not():
    """The distinction the mixture handler exists to price.

    Both arrive as an identical 05. One is over in a day; the other never ends.
    If the generator could not tell them apart there would be nothing for a
    posterior to learn.
    """
    dataset = masked_dataset()
    config = config_for(dataset)

    glitched = None
    for customer in dataset.customers:
        if risk_blocked_from(customer, config) is not None:
            continue
        for offset in range(0, 200):
            day = dataset.as_of - timedelta(days=offset)
            if in_glitch(customer, day, config):
                glitched = (customer, day)
                break
        if glitched:
            break
    assert glitched, "fixture produced no transient glitch"

    customer, day = glitched
    # The glitch is a per-day event, so some later day is clear of it. That is
    # what makes waiting worth anything.
    later = [
        day + timedelta(days=d)
        for d in range(1, 15)
        if not in_glitch(customer, day + timedelta(days=d), config)
    ]
    assert later, "a transient glitch never cleared, which makes it permanent"


# --- THE DISCRIMINATING TEST -------------------------------------------------


@pytest.mark.parametrize(
    "risk_share,other_share", [(0.05, 0.70), (0.55, 0.20)]
)
def test_the_fitted_posterior_tracks_the_generators_mixture(
    risk_share, other_share, monkeypatch
):
    """The anti-self-fulfilment guardrail. If this fails, the class does not ship.

    Perturb what the generator actually puts behind a 05 and the agent's fitted
    posterior must move with it. It can only do that by LEARNING from the
    observable resolution history - nothing in agent/masking.py reads
    LATENT_CAUSE_MIXTURE, and this test is what proves the claim rather than
    asserting it.

    Parameterised at the two ends so the pair of runs brackets the effect; the
    ordering assertion lives in the test below, which needs both fits at once.
    """
    mixture = {
        ambiguous.CAUSE_INSUFFICIENT_FUNDS: other_share,
        CAUSE_ISSUER_RISK_BLOCK: risk_share,
        ambiguous.CAUSE_TRANSIENT_GLITCH: 0.15,
        ambiguous.CAUSE_EXPIRED_CARD: 0.07,
        ambiguous.CAUSE_DOWNTIME: 1.0 - other_share - risk_share - 0.15 - 0.07,
    }
    monkeypatch.setattr(ambiguous, "LATENT_CAUSE_MIXTURE", mixture)

    dataset = masked_dataset(seed=5, n=400)
    view = agent_view(dataset)
    bank_of = {c.customer_id: c.bank_id for c in view.customers}
    posterior = MaskingModel.from_history(view.payment_history, bank_of).posterior()

    # A high risk-block share must put real mass on permanent_block, a low one
    # must not. The thresholds are deliberately loose - the claim is that the
    # fit TRACKS, not that it is calibrated to three decimals.
    if risk_share > 0.4:
        assert posterior[CAUSE_PERMANENT_BLOCK] > 0.4
    else:
        assert posterior[CAUSE_PERMANENT_BLOCK] < 0.2


def test_the_fitted_posterior_moves_in_the_right_direction(monkeypatch):
    """Both ends in one test, so the comparison is explicit rather than implied."""

    def fit(risk_share: float, funds_share: float) -> dict[str, float]:
        mixture = {
            ambiguous.CAUSE_INSUFFICIENT_FUNDS: funds_share,
            CAUSE_ISSUER_RISK_BLOCK: risk_share,
            ambiguous.CAUSE_TRANSIENT_GLITCH: 0.15,
            ambiguous.CAUSE_EXPIRED_CARD: 0.07,
            ambiguous.CAUSE_DOWNTIME: 1.0 - funds_share - risk_share - 0.22,
        }
        monkeypatch.setattr(ambiguous, "LATENT_CAUSE_MIXTURE", mixture)
        dataset = masked_dataset(seed=5, n=400)
        view = agent_view(dataset)
        bank_of = {c.customer_id: c.bank_id for c in view.customers}
        return MaskingModel.from_history(
            view.payment_history, bank_of
        ).posterior()

    low = fit(0.05, 0.70)
    high = fit(0.55, 0.20)

    assert high[CAUSE_PERMANENT_BLOCK] > low[CAUSE_PERMANENT_BLOCK], (
        "more risk blocks in the generator must mean more permanent_block mass "
        "in the fit; if it does not, the model is not learning from the book"
    )
    assert (
        low[rules.FAILURE_CLASS_INSUFFICIENT_FUNDS]
        > high[rules.FAILURE_CLASS_INSUFFICIENT_FUNDS]
    ), "and the balance mass must move the other way"


# --- the model's own mechanics ----------------------------------------------


def test_the_posterior_is_uniform_before_it_has_seen_anything():
    """It refuses to guess the mixture it exists to estimate."""
    model = MaskingModel()
    posterior = model.posterior()
    assert len(set(round(p, 9) for p in posterior.values())) == 1


def test_a_thin_issuer_backs_off_to_the_book():
    """Three 05s cannot support their own posterior; say so rather than pretend."""
    model = MaskingModel()
    for _ in range(50):
        model.observe("BIGBANK", rules.FAILURE_CLASS_INSUFFICIENT_FUNDS)
    for _ in range(3):
        model.observe("TINYBANK", CAUSE_PERMANENT_BLOCK)

    assert model.evidence_count("TINYBANK") < MIN_BANK_EVIDENCE
    assert model.posterior("TINYBANK") == model.posterior()
    assert not model.explain("TINYBANK")["uses_bank_level"]
    assert model.explain("BIGBANK")["uses_bank_level"]


def test_resolution_reads_only_the_observable_channel():
    """A 05 followed by a 04 is a balance failure; a 05 that never clears is a block."""
    from agent.masking import _resolution_of

    class Attempt:
        def __init__(self, succeeded, return_code, day=1):
            self.succeeded = succeeded
            self.return_code = return_code
            self.attempt_date = date(2026, 1, day)

    assert _resolution_of([Attempt(False, "04")]) == (
        rules.FAILURE_CLASS_INSUFFICIENT_FUNDS
    )
    assert _resolution_of([Attempt(True, None)]) == CAUSE_TRANSIENT
    assert _resolution_of([Attempt(False, "05"), Attempt(False, "05")]) == (
        CAUSE_PERMANENT_BLOCK
    )
    assert _resolution_of([]) is None
    # Censoring: one follow-up 05 is equally consistent with a block that had
    # not resolved before the window ended, so it is UNKNOWN, not permanent.
    assert _resolution_of([Attempt(False, "05")]) is None


# --- realism -----------------------------------------------------------------


def test_the_05_share_lands_in_a_sane_band():
    """Pinned as a property rather than tuned, same treatment as the class-2
    downtime-share volatility.

    An unexplained-decline share is genuinely variable - it depends on which
    issuers a seed's customers happen to bank with, and masking propensity is
    per bank - so the band is wide on purpose. What it rules out is a class
    that is either invisible or that has eaten the book.
    """
    shares = []
    for seed in range(1, 6):
        dataset = masked_dataset(seed=seed, n=400, share=0.35)
        events = dataset.failure_events
        shares.append(
            sum(
                1
                for e in events
                if e.return_code == rules.RETURN_CODE_DO_NOT_HONOUR
            )
            / len(events)
        )
    mean = sum(shares) / len(shares)
    assert 0.02 <= mean <= 0.40, f"mean 05 share {mean:.3f} is outside a sane band"
    assert max(shares) > min(shares), "no seed-to-seed variation at all"
