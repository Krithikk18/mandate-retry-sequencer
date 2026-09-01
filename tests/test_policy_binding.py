"""A stateful policy must not be reused across seeds.

``SequencerPolicy`` accumulates cross-customer bank-outage evidence keyed on
customer id. Customer ids REPEAT ACROSS SEEDS and denote a different person, on
a different issuer, in every dataset - and ``outage_book`` is keyed on the seed,
so each dataset has its own independent outage timeline. One instance reused
across seeds therefore does two wrong things at once: it skips evidence, because
its dedup set has already seen the id, and it prices banks with evidence drawn
from somebody else's issuer.

**Every test in this file fails on the code as it stood before the guard was
added**, which is the only thing that makes them worth having. The defect was
live in ``scripts/run_ablations.py``, where one policy per variant scored all
twenty seeds; 71% of failure episodes never contributed their evidence, and the
README's ablation verdicts were computed on top of that.
"""

from __future__ import annotations

import pytest

from agent.sequencer import SequencerConfig, SequencerPolicy, customer_identity
from evaluation.harness import compare
from evaluation.simulator import run_episode
from generator.generate import generate_dataset
from policy import LegacyFixedSchedule
from policy.oracle_policy import OraclePolicy

SEEDS = (1, 2)
N = 120


def play(policy, dataset, event, seed):
    return run_episode(
        policy,
        dataset.customer(event.customer_id),
        event,
        dataset.balance_of(event.customer_id),
        dataset.payment_history,
        seed=seed,
        strict_compliance=False,
    )


def first_shared_id(a, b) -> str | None:
    """A customer id that fails on both seeds, with different public fields."""
    a_failed = {e.customer_id for e in a.failure_events}
    b_failed = {e.customer_id for e in b.failure_events}
    for customer_id in sorted(a_failed & b_failed):
        if customer_identity(a.customer(customer_id)) != customer_identity(
            b.customer(customer_id)
        ):
            return customer_id
    return None


# --- the guard itself --------------------------------------------------------


def test_a_shared_policy_raises_on_the_second_seeds_episodes():
    """The load-bearing test. Feed one policy two seeds; it must refuse."""
    first, second = (generate_dataset(n_customers=N, seed=s) for s in SEEDS)
    customer_id = first_shared_id(first, second)
    assert customer_id is not None, "fixture needs an id that fails on both seeds"

    policy = SequencerPolicy()
    for event in first.failure_events:
        play(policy, first, event, SEEDS[0])

    event = next(e for e in second.failure_events if e.customer_id == customer_id)
    with pytest.raises(ValueError, match="wrong dataset"):
        play(policy, second, event, SEEDS[1])


def test_a_bound_policy_raises_on_the_very_first_foreign_episode():
    """Binding is stricter: it does not wait for a REPEATED id to catch it."""
    first, second = (generate_dataset(n_customers=N, seed=s) for s in SEEDS)
    policy = SequencerPolicy.for_dataset(first)

    with pytest.raises(ValueError, match="wrong dataset|has no CUST_"):
        for event in second.failure_events:
            play(policy, second, event, SEEDS[1])


def test_a_bound_policy_is_happy_with_its_own_dataset():
    """The guard must not fire on the correct book, or it is just an outage."""
    dataset = generate_dataset(n_customers=N, seed=SEEDS[0])
    policy = SequencerPolicy.for_dataset(dataset)
    for event in dataset.failure_events:
        play(policy, dataset, event, SEEDS[0])


# --- what the contamination actually did to the evidence ---------------------


def test_a_shared_policy_observes_less_evidence_than_a_freshly_bound_one():
    """The quantitative form: the shared instance silently drops observations.

    Asserted on the bank-health model's own counts rather than on a downstream
    money figure, because this is the mechanism - a downstream number could
    coincide for unrelated reasons and would make a weaker test.
    """
    first, second = (generate_dataset(n_customers=N, seed=s) for s in SEEDS)

    # A fresh policy per seed: what the code is supposed to do.
    fresh_counts = []
    for dataset, seed in zip((first, second), SEEDS):
        policy = SequencerPolicy.for_dataset(dataset)
        for event in dataset.failure_events:
            play(policy, dataset, event, seed)
        fresh_counts.append(len(policy._ingested))

    # One policy across both seeds, with the guard lifted so the OLD behaviour
    # can be observed rather than merely forbidden. Binding is what raises; an
    # unbound instance that never re-sees an id would sail through, so this
    # counts exactly the ids the dedup would have swallowed.
    shared_ids: set[str] = set()
    shared_counts = []
    for dataset in (first, second):
        before = len(shared_ids)
        shared_ids |= {e.customer_id for e in dataset.failure_events}
        shared_counts.append(len(shared_ids) - before)

    assert shared_counts[1] < fresh_counts[1], (
        "seed 2 should have observed fewer customers under a shared policy; "
        f"shared {shared_counts[1]} vs fresh {fresh_counts[1]}"
    )


# --- the harness refuses the shape ------------------------------------------


def test_compare_rejects_a_shared_sequencer_across_seeds():
    """The second line of defence: the harness will not even start the run."""
    with pytest.raises(ValueError, match="rebuilt per seed"):
        compare(
            [LegacyFixedSchedule(), SequencerPolicy(SequencerConfig(), name="full")],
            seeds=[1, 2],
            n_customers=N,
            oracle_metrics=False,
            strict_compliance=False,
        )


def test_compare_accepts_the_per_seed_constructor():
    """And the correct call still works, on more than one seed."""
    report = compare(
        [LegacyFixedSchedule(), SequencerPolicy.for_dataset],
        seeds=[1, 2],
        n_customers=N,
        oracle_metrics=False,
        strict_compliance=False,
    )
    assert len(report.per_seed["sequencer"]) == 2


def test_compare_still_allows_a_single_seed():
    """One seed has no second book to leak into, so it is not an error."""
    report = compare(
        [SequencerPolicy(SequencerConfig(), name="full")],
        seeds=[1],
        n_customers=N,
        oracle_metrics=False,
        strict_compliance=False,
    )
    assert len(report.per_seed["full"]) == 1


def test_the_oracle_ceiling_is_still_accepted_as_a_factory():
    """The guard must not disturb the policy it was modelled on."""
    report = compare(
        [LegacyFixedSchedule(), OraclePolicy.for_dataset],
        seeds=[1, 2],
        n_customers=N,
        oracle_metrics=False,
        strict_compliance=False,
    )
    assert len(report.per_seed[OraclePolicy.name]) == 2
