"""Dataset-level tests: reproducibility, failure realism, agent/truth split.

Kept separate from tests/test_generator.py because these build whole datasets
and are therefore the slow ones.
"""

from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pytest

from generator.generate import (
    LEGACY_RETRY_OFFSETS,
    agent_view,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from generator.models import HIDDEN_CUSTOMER_FIELDS, HIDDEN_FAILURE_FIELDS
from generator.oracle import would_succeed
from generator.rules import MAX_ATTEMPTS_PER_CYCLE


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_customers=300, seed=42)


# --- 1. reproducibility ------------------------------------------------------


def test_same_seed_produces_an_identical_dataset(dataset):
    other = generate_dataset(n_customers=300, seed=42)

    assert [c.model_dump() for c in other.customers] == [
        c.model_dump() for c in dataset.customers
    ]
    assert [e.model_dump() for e in other.failure_events] == [
        e.model_dump() for e in dataset.failure_events
    ]
    assert [r.model_dump() for r in other.payment_history] == [
        r.model_dump() for r in dataset.payment_history
    ]
    assert other.meta == dataset.meta
    for customer_id, series in dataset.balance_series.items():
        np.testing.assert_array_equal(
            other.balance_series[customer_id].to_numpy(), series.to_numpy()
        )


def test_a_different_seed_produces_a_different_dataset(dataset):
    other = generate_dataset(n_customers=300, seed=43)
    assert [c.model_dump() for c in other.customers] != [
        c.model_dump() for c in dataset.customers
    ]


# --- 5. failure realism ------------------------------------------------------

# ONE definition of the band, read by both failure-rate tests below.
#
# They used to carry their own literals and drifted apart when the fifth class
# widened the spread: the per-seed test still demanded <= 0.20 while the
# multi-seed test allowed 0.24, so the two disagreed about what counts as a
# realistic world. It was not flaky only because the per-seed test is pinned to
# three hand-picked seeds - it would have failed the moment a fourth was added.
# Sharing the constants is the fix, because it makes the disagreement
# impossible rather than merely absent today.
#
# FAILURE_RATE_MEAN_BAND is the real quality gate and is asserted on the MEAN
# across seeds. FAILURE_RATE_SEED_BAND is wider on purpose: it is a sanity
# check that no individual seed produces an absurd world, not a claim about the
# process. CONTEXT is explicit that single runs are not evidence, so the
# per-seed assertion is deliberately the weaker of the two.
FAILURE_RATE_MEAN_BAND = (0.10, 0.20)
FAILURE_RATE_SEED_BAND = (0.05, 0.24)


@pytest.mark.parametrize("seed", [42, 7, 2024])
def test_no_single_seed_produces_an_absurd_failure_rate(seed):
    """A sanity check on one run, and deliberately nothing more.

    The rate is never imposed, so this is a genuine property of the process -
    but a proportion measured over ~85 failures swings hard, and any bound
    tight enough to be interesting here would be measuring the seed rather than
    the generator. The claim that matters is made on the mean, in the test
    below.
    """
    ds = generate_dataset(n_customers=400, seed=seed)
    rate = ds.natural_failure_rate
    low, high = FAILURE_RATE_SEED_BAND
    assert low <= rate <= high, f"seed {seed} gave a failure rate of {rate:.3f}"


def test_failure_rate_mean_and_spread_across_seeds(multi_seed_meta):
    """Single runs are not evidence - report mean and spread.

    The MEAN is the gate; the max is a spread allowance, and it has been
    widened once per generated class rather than tuned. Each new failure mode
    adds its own share on top of the insufficient-funds base, which is the
    point of adding it: bank outages took the four-class book to ~14.6% mean,
    and the AFA gate takes the five-class book to ~17.1%. Both sit inside the
    5-20% band the brief asks for; only the worst individual seed clips it, on
    roughly 1 seed in 20, which is a small-sample property of a proportion over
    ~70 failures rather than an unrealistic world.

    Both bounds come from the shared constants above, so this test and the
    per-seed sanity check cannot drift apart again.

    What must NOT happen in response to this test is tuning
    ``OTP_RESPONSIVENESS_BETA`` down to buy headroom. That constant is a claim
    about how often people complete an authentication challenge, and it is not
    for sale to a threshold.
    """
    rates = np.array([m["natural_failure_rate"] for m in multi_seed_meta])
    seed_low, seed_high = FAILURE_RATE_SEED_BAND
    mean_low, mean_high = FAILURE_RATE_MEAN_BAND
    assert seed_low <= rates.min() and rates.max() <= seed_high
    assert mean_low <= rates.mean() <= mean_high, (
        f"mean failure rate {rates.mean():.3f}"
    )


def test_downtime_share_is_volatile_because_outages_are_correlated(multi_seed_meta):
    """A property worth pinning rather than smoothing over.

    Insufficient funds is an independent per-customer draw, so its share is
    stable across seeds. An outage is one event that fails every customer on
    that issuer at once, so whether a large bank happened to be down on a busy
    debit day swings the whole seed. Mean lands near the 10% target; individual
    seeds range from about 1% to over 20%, and that spread is the correlation
    showing up rather than a tuning failure.
    """
    shares = np.array([m["downtime_share"] for m in multi_seed_meta])
    assert 0.03 <= shares.mean() <= 0.18, f"mean downtime share {shares.mean():.3f}"
    assert shares.max() > shares.min(), "no seed-to-seed variation at all"


def test_failure_events_are_exactly_the_debits_that_really_bounced(dataset):
    failed_ids = {e.customer_id for e in dataset.failure_events}

    from generator import rules

    for event in dataset.failure_events:
        customer = dataset.customer(event.customer_id)
        series = dataset.balance_of(event.customer_id)
        assert event.failure_class in rules.FAILURE_CLASSES

        if event.failure_class == "insufficient_funds":
            assert event.return_code == "04"
            assert event.balance_at_failure < customer.subscription_amount
            assert not would_succeed(
                customer, event.original_debit_date, customer.subscription_amount, series
            )
        elif event.failure_class == rules.FAILURE_CLASS_AFA_INCOMPLETE:
            # Refused at the authentication gate, which sits IN FRONT OF the
            # account. So the balance is not implicated and must not be
            # asserted on: the debit never reached it. What is implicated is
            # the Sec 8(a) ceiling - this can only happen above it.
            from generator.rules import requires_otp

            assert event.return_code == rules.RETURN_CODE_AFA_INCOMPLETE
            assert requires_otp(customer.subscription_amount)
            assert customer.mandate_dead_from is None or (
                event.original_debit_date < customer.mandate_dead_from
            )
        elif event.failure_class == rules.FAILURE_CLASS_DOWNTIME:
            # The bank was unavailable at the presentment hour. Nothing about
            # the mandate or the balance is implicated - and specifically, no
            # dead-mandate flag is involved.
            from datetime import datetime

            from generator.generate import outage_book_for

            at = datetime.combine(
                event.original_debit_date, rules.DEFAULT_PRESENTMENT_TIME
            )
            assert outage_book_for(dataset).is_down(customer.bank_id, at)
            assert event.return_code == rules.RETURN_CODE_DOWNTIME
        else:
            # A dead mandate fails whatever the balance says, and the return
            # code is the only thing that tells the agent why.
            assert customer.mandate_dead_from is not None
            assert event.original_debit_date >= customer.mandate_dead_from
            assert event.return_code != "04"
            assert rules.classify(event.return_code) == event.failure_class

    # And nobody was left out: every customer whose last attempt failed is here.
    last_attempt = {}
    for record in dataset.payment_history:
        prior = last_attempt.get(record.customer_id)
        if prior is None or record.attempt_date > prior.attempt_date:
            last_attempt[record.customer_id] = record
    really_failed = {
        cid for cid, rec in last_attempt.items() if not rec.succeeded
    }
    assert really_failed == failed_ids


def _attempts_per_cycle(dataset) -> dict[tuple[str, object], int]:
    """Count attempts per *billing cycle*, not per calendar month.

    A retry on debit day 28 + 7 lands in the following month, so cycles are
    keyed by the original debit date they belong to.
    """
    by_customer: dict[str, list] = {}
    for record in dataset.payment_history:
        by_customer.setdefault(record.customer_id, []).append(record)

    counts: dict[tuple[str, object], int] = {}
    for customer_id, records in by_customer.items():
        debit_day = dataset.customer(customer_id).debit_day_of_month
        records.sort(key=lambda r: r.attempt_date)
        cycle = None
        for record in records:
            if record.attempt_date.day == debit_day:
                cycle = record.attempt_date  # only originals fall on the debit day
            assert cycle is not None, "a retry with no originating debit"
            key = (customer_id, cycle)
            counts[key] = counts.get(key, 0) + 1
    return counts


def test_payment_history_respects_the_npci_attempt_ceiling(dataset):
    counts = _attempts_per_cycle(dataset)
    assert max(counts.values()) <= MAX_ATTEMPTS_PER_CYCLE


def test_history_is_truncated_to_tenure(dataset):
    counts: dict[str, set] = {}
    for record in dataset.payment_history:
        counts.setdefault(record.customer_id, set()).add(
            (record.attempt_date.year, record.attempt_date.month)
        )
    for customer in dataset.customers:
        cycles = len(counts.get(customer.customer_id, ()))
        assert cycles <= min(customer.tenure_months, 12)


def test_history_attempts_agree_with_the_balance_truth(dataset):
    """Each attempt is resolved by whatever actually blocked it.

    Four mechanisms can stop a presentment, and the return code says which:

    * a dead mandate (54/22) - fails on every date whatever the balance says;
    * a bank outage (91) - the debit never reached the issuer, so the balance
      is irrelevant and unknowable;
    * AFA incomplete (1A) - the authentication gate sits in front of the
      account, so the debit was refused before the balance was read;
    * otherwise the balance (04, or success).

    Only the LAST is a statement about ``would_succeed``. Asserting that the
    balance explains a downtime or authentication failure would be asserting
    something false - and the corresponding bug (counting censored declines as
    balance evidence) is one this project has already had once, in class 2.
    """
    from datetime import datetime

    from generator import rules
    from generator.afa import otp_completed
    from generator.generate import outage_book_for

    outages = outage_book_for(dataset)
    checked = downtime_seen = afa_seen = 0

    for record in dataset.payment_history[:2000]:
        customer = dataset.customer(record.customer_id)
        if customer.mandate_dead_from is not None and (
            record.attempt_date >= customer.mandate_dead_from
        ):
            assert not record.succeeded
            assert record.return_code not in ("04", None)
            continue

        at = datetime.combine(record.attempt_date, rules.DEFAULT_PRESENTMENT_TIME)
        if rules.classify(record.return_code) == rules.FAILURE_CLASS_DOWNTIME:
            assert not record.succeeded
            assert outages.is_down(customer.bank_id, at), (
                f"{record.customer_id} got code 91 on {record.attempt_date} but "
                f"{customer.bank_id} was up at the presentment hour"
            )
            downtime_seen += 1
            continue

        if rules.classify(record.return_code) == rules.FAILURE_CLASS_AFA_INCOMPLETE:
            # The fourth mechanism, and the fourth thing that is NOT the
            # balance. The authentication gate sits in front of the account, so
            # a 1A says nothing about the money - asserting would_succeed here
            # would be asserting something false, exactly as it would for a 91.
            assert not record.succeeded
            assert rules.requires_otp(record.amount), (
                f"{record.customer_id} got code 1A on Rs {record.amount}, below "
                "the Sec 8(a) ceiling where no authentication is required"
            )
            assert not otp_completed(customer, record.attempt_date)
            afa_seen += 1
            continue

        # Everything else really is the balance - and the bank was up for it,
        # and any authentication it needed was completed.
        assert outages.is_up(customer.bank_id, at)
        series = dataset.balance_of(record.customer_id)
        assert record.succeeded == would_succeed(
            customer, record.attempt_date, record.amount, series
        )
        checked += 1

    assert checked > 100
    assert downtime_seen > 0, "no downtime failures in this sample to check"
    assert afa_seen > 0, "no AFA refusals in this sample to check"


def test_retries_follow_the_legacy_schedule_and_stop_on_success(dataset):
    by_customer: dict[str, list] = {}
    for record in dataset.payment_history:
        by_customer.setdefault(record.customer_id, []).append(record)

    for customer_id, records in by_customer.items():
        customer = dataset.customer(customer_id)
        records.sort(key=lambda r: r.attempt_date)
        originals = [r for r in records if r.attempt_date.day == customer.debit_day_of_month]
        for original in originals:
            retries = [
                r
                for r in records
                if original.attempt_date
                < r.attempt_date
                <= original.attempt_date + timedelta(days=max(LEGACY_RETRY_OFFSETS))
                and r is not original
            ]
            if original.succeeded:
                assert retries == [], "a successful debit should not be retried"
            offsets = [(r.attempt_date - original.attempt_date).days for r in retries]
            assert offsets == list(LEGACY_RETRY_OFFSETS[: len(offsets)])
            assert all(not r.succeeded for r in retries[:-1]), "stopped late"


# --- 6. agent view leakage ---------------------------------------------------


def test_agent_view_contains_no_hidden_fields(dataset):
    view = agent_view(dataset)

    for customer in view.customers:
        fields = set(customer.model_dump())
        assert fields.isdisjoint(HIDDEN_CUSTOMER_FIELDS), fields & HIDDEN_CUSTOMER_FIELDS

    for event in view.failure_events:
        fields = set(event.model_dump())
        assert fields.isdisjoint(HIDDEN_FAILURE_FIELDS)

    assert not hasattr(view, "balance_series")

    # Nothing hidden survives serialisation either, under any nesting.
    blob = json.dumps(
        {
            "customers": [c.model_dump() for c in view.customers],
            "failure_events": [e.model_dump() for e in view.failure_events],
            "payment_history": [r.model_dump() for r in view.payment_history],
        },
        default=str,
    )
    for hidden in HIDDEN_CUSTOMER_FIELDS | HIDDEN_FAILURE_FIELDS:
        assert hidden not in blob, f"{hidden} leaked into the agent view"

    for frame in (
        view.customers_frame(),
        view.failure_events_frame(),
        view.payment_history_frame(),
    ):
        assert set(frame.columns).isdisjoint(
            HIDDEN_CUSTOMER_FIELDS | HIDDEN_FAILURE_FIELDS
        )


def test_agent_customers_cannot_be_built_with_hidden_fields():
    from generator.models import AgentCustomer

    with pytest.raises(Exception):
        AgentCustomer(  # type: ignore[call-arg]  # the point: this must not typecheck OR run
            customer_id="CUST_0001",
            subscription_amount=499.0,
            rail="nach",
            debit_day_of_month=5,
            tenure_months=9,
            monthly_ltv=5988.0,
            salary_day=1,  # forbidden
        )


def test_agent_view_keeps_everything_the_agent_is_allowed_to_see(dataset):
    view = agent_view(dataset)
    assert len(view.customers) == len(dataset.customers)
    assert len(view.failure_events) == len(dataset.failure_events)
    assert len(view.payment_history) == len(dataset.payment_history)
    assert view.as_of == dataset.as_of
    assert view.customers[0].customer_id == dataset.customers[0].customer_id


# --- persistence -------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_save_and_load_roundtrip(dataset, tmp_path, fmt):
    save_dataset(dataset, tmp_path / fmt, fmt=fmt)
    restored = load_dataset(tmp_path / fmt)

    assert [c.model_dump() for c in restored.customers] == [
        c.model_dump() for c in dataset.customers
    ]
    assert [e.model_dump() for e in restored.failure_events] == [
        e.model_dump() for e in dataset.failure_events
    ]
    assert [r.model_dump() for r in restored.payment_history] == [
        r.model_dump() for r in dataset.payment_history
    ]
    assert restored.as_of == dataset.as_of
    assert restored.seed == dataset.seed
    assert restored.natural_failure_rate == dataset.natural_failure_rate
    for customer_id, series in dataset.balance_series.items():
        restored_series = restored.balance_series[customer_id]
        np.testing.assert_allclose(restored_series.to_numpy(), series.to_numpy())
        assert restored_series.index.equals(series.index)


# --- realism: the dataset must not make the problem easier than reality ------


@pytest.fixture(scope="module")
def multi_seed_meta():
    """Never judge these on a single run - CONTEXT evaluation rule."""
    return [generate_dataset(n_customers=500, seed=s).meta for s in (1, 5, 7, 42, 99, 2024)]


def test_legacy_baseline_recovers_a_realistic_share(multi_seed_meta):
    """Published fixed-retry recovery is 35-45%.

    A synthetic world where the dumb day-2/4/7 schedule scores far above that
    has made the problem too easy, and every agent compared against it will
    look better than it deserves. This is the guard on that.
    """
    rates = np.array([m["legacy_recovery_rate"] for m in multi_seed_meta])
    assert 0.35 <= rates.mean() <= 0.45, f"mean legacy recovery {rates.mean():.3f}"
    assert rates.max() <= 0.55, f"a seed reached {rates.max():.3f}"


def test_some_failures_are_genuinely_unrecoverable(multi_seed_meta):
    """If every customer can be recovered, CANCEL is never correct and the
    stopping rule is never exercised. Income-shocked customers fix that."""
    shares = np.array([m["unrecoverable_share"] for m in multi_seed_meta])
    assert shares.min() > 0.0, "no failure is unrecoverable; stopping rule untestable"
    assert 0.05 <= shares.mean() <= 0.40, f"mean unrecoverable share {shares.mean():.3f}"


def test_failures_are_not_clustered_inside_the_legacy_schedule(multi_seed_meta):
    """The failure-to-payday gap must be wide enough that a fixed schedule
    cannot reach most of the money by luck."""
    gaps = np.array([m["median_days_to_payday"] for m in multi_seed_meta])
    beyond = np.array([m["share_failing_beyond_legacy_reach"] for m in multi_seed_meta])
    assert gaps.mean() >= 8.0, f"median gap only {gaps.mean():.1f} days"
    assert beyond.mean() >= 0.40, f"only {beyond.mean():.3f} land beyond day 7"


def test_income_shock_share_matches_the_population_setting(multi_seed_meta):
    from generator.population import INCOME_SHOCK_SHARE

    shares = np.array([m["n_income_shocked"] / m["n_customers"] for m in multi_seed_meta])
    assert shares.mean() == pytest.approx(INCOME_SHOCK_SHARE, abs=0.015)


def test_income_shocked_customers_leave_a_visible_trail(dataset):
    """The agent never sees income_shock_start. Its only evidence is a run of
    failures across consecutive cycles, so that trail must actually exist."""
    shocked = [c for c in dataset.customers if c.income_shock_start is not None]
    assert shocked, "no shocked customers in this dataset"

    by_customer: dict[str, list] = {}
    for record in dataset.payment_history:
        by_customer.setdefault(record.customer_id, []).append(record)

    trails = 0
    for customer in shocked:
        records = sorted(
            by_customer.get(customer.customer_id, []), key=lambda r: r.attempt_date
        )
        after = [r for r in records if r.attempt_date >= customer.income_shock_start]
        if len(after) >= 2 and not any(r.succeeded for r in after):
            trails += 1
    assert trails / len(shocked) >= 0.5, "shock leaves no learnable signature"


def test_income_shock_start_is_hidden_from_the_agent(dataset):
    from generator.generate import agent_view
    from generator.models import HIDDEN_CUSTOMER_FIELDS

    assert "income_shock_start" in HIDDEN_CUSTOMER_FIELDS
    view = agent_view(dataset)
    for customer in view.customers:
        assert "income_shock_start" not in customer.model_dump()
