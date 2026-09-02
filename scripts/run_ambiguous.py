"""Failure class 6: the ambiguous decline.

    python scripts/run_ambiguous.py
    python scripts/run_ambiguous.py --seeds 20 --n 500 --share 0.35

**This class is off by default in the generator**, so unlike every other
script here it takes a ``--share``: how much of the book's decline traffic the
issuers hide behind an unexplained ``05``. At share 0 there is nothing to
report and every other number in the repository is untouched, which is the
whole point of the flag.

WHAT MAKES THIS CLASS DIFFERENT, and it is worth stating before any number:
classes 1, 2 and 5 hide a PARAMETER behind a code that says which parameter to
estimate. Class 6 hides the QUESTION. So the agent's response is not a sixth
success model but a fitted POSTERIOR over the other five, and the success
probability is their mixture.

Reported by VALUE as well as by count, following class 5's convention: an
unexplained decline can sit on a Rs 199 mandate or a Rs 19,000 one, and a
count-only table weights them the same.

Four things are reported:

1. the realised mix - how many failures are masked, and what was really behind
   them, which only the truth side knows
2. whether the fitted posterior is any good, scored against that truth
3. money and recovery: legacy, sequencer, ceiling, and the no_masking_posterior
   ablation that treats every 05 as a plain balance failure
4. the permanent-block waste: presentments spent on mandates that were never
   coming back and never said so
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _emit import add_json_argument, emit  # noqa: E402
from agent.masking import (  # noqa: E402
    CAUSE_PERMANENT_BLOCK,
    CAUSE_TRANSIENT,
    MaskingModel,
)
from agent.sequencer import SequencerConfig, SequencerPolicy  # noqa: E402
from evaluation.simulator import run_episode  # noqa: E402
from generator import rules  # noqa: E402
from generator.ambiguous import (  # noqa: E402
    CAUSE_ISSUER_RISK_BLOCK,
    LATENT_CAUSE_MIXTURE,
)
from generator.generate import agent_view, generate_dataset  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from policy.oracle_policy import OraclePolicy  # noqa: E402

RULE = "=" * 92

ARMS = ("legacy_fixed", "sequencer", "oracle_ceiling", "no_masking_posterior")


class Tally:
    """Per-policy totals over the ambiguous failures of one run."""

    def __init__(self) -> None:
        self.episodes = 0
        self.recovered = 0
        self.money = 0.0
        self.value_at_risk = 0.0
        self.presentments = 0
        self.chosen = 0
        self.notifications = 0
        self.net_value = 0.0
        self.optouts = 0
        # Presentments spent on mandates that could never have cleared - the
        # class-6 specific waste, because nothing ever says "stop".
        self.doomed_presentments = 0

    def add(self, result, amount: float, doomed: bool) -> None:
        self.episodes += 1
        self.recovered += result.outcome == "RECOVERED"
        self.money += result.recovered_amount
        self.value_at_risk += amount
        self.presentments += result.attempts_used
        self.chosen += len(result.retry_dates)
        self.notifications += result.notifications_sent
        self.net_value += result.net_value
        self.optouts += result.outcome == "OPTED_OUT"
        if doomed:
            self.doomed_presentments += result.attempts_used

    @property
    def recovery_rate(self) -> float:
        return self.recovered / self.episodes if self.episodes else 0.0

    @property
    def value_recovered_share(self) -> float:
        return self.money / self.value_at_risk if self.value_at_risk else 0.0


def ambiguous_events(dataset):
    return [
        e
        for e in dataset.failure_events
        if e.return_code == rules.RETURN_CODE_DO_NOT_HONOUR
    ]


def play(policy, dataset, event, seed, **kwargs):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--share", type=float, default=0.35,
        help="masking share; 0 disables class 6 entirely (the generator default)",
    )
    add_json_argument(parser)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    started = time.time()

    tallies = {name: Tally() for name in ARMS}
    book_failures = 0
    masked_failures = 0
    true_behind: dict[str, int] = {}
    posterior_hits = posterior_total = 0
    fitted_posteriors: list[dict[str, float]] = []

    for seed in seeds:
        dataset = generate_dataset(
            n_customers=args.n, seed=seed, ambiguity_share=args.share
        )
        book_failures += len(dataset.failure_events)
        events = ambiguous_events(dataset)
        masked_failures += len(events)
        for event in events:
            true_behind[event.failure_class] = (
                true_behind.get(event.failure_class, 0) + 1
            )
        if not events:
            continue

        # What the agent believes, fitted from the observable channel only.
        view = agent_view(dataset)
        bank_of = {c.customer_id: c.bank_id for c in view.customers}
        fitted = MaskingModel.from_history(view.payment_history, bank_of)
        fitted_posteriors.append(fitted.posterior())

        # Score the posterior against the truth: for each masked failure whose
        # true class the agent could in principle have named, did the posterior
        # put its highest mass on the right one?
        for event in events:
            truth = event.failure_class
            if truth == rules.FAILURE_CLASS_AMBIGUOUS:
                continue  # 05-native; no component class to be right about
            posterior = fitted.posterior(
                dataset.customer(event.customer_id).bank_id
            )
            named = {
                c: p for c, p in posterior.items()
                if c not in (CAUSE_TRANSIENT, CAUSE_PERMANENT_BLOCK)
            }
            posterior_total += 1
            if named and max(named, key=named.get) == truth:
                posterior_hits += 1

        # PRIMED FROM THE WHOLE BOOK, which matters more than it looks.
        # Letting each policy accumulate evidence only from the episodes it
        # plays would fit the posterior on the customers who HAD a 05 - a
        # sample badly enriched for permanent blocks, because a blocked mandate
        # emits an unexplained decline every cycle while a transient one emits
        # one. Measured on seed 1: 0.90 mass on permanent_block against 0.64
        # fitted from the whole book. That is selection bias in the harness,
        # not a property of the model, and a deployment would not have it -
        # a PSP has its entire decline history, not just today's work queue.
        policies = {
            "legacy_fixed": LegacyFixedSchedule(),
            "sequencer": SequencerPolicy.for_dataset(dataset, masking=fitted),
            "oracle_ceiling": OraclePolicy.for_dataset(dataset),
            "no_masking_posterior": SequencerPolicy.for_dataset(
                dataset,
                SequencerConfig(use_masking_posterior=False),
                name="no_masking_posterior",
                masking=fitted,
            ),
        }
        for event in events:
            customer = dataset.customer(event.customer_id)
            amount = customer.subscription_amount
            # "Doomed" = the truth says nothing could ever have cleared it.
            doomed = event.failure_class in rules.NEVER_RETRY_CLASSES
            for name, policy in policies.items():
                tallies[name].add(play(policy, dataset, event, seed), amount, doomed)

    legacy, agent, oracle, blind = (tallies[name] for name in ARMS)
    n = legacy.episodes

    print(f"{RULE}\nFAILURE CLASS 6: AMBIGUOUS DECLINE (code 05)\n{RULE}")
    print(f"{args.seeds} seeds x {args.n} customers   masking share {args.share}")
    if n == 0:
        print("\nNo masked failures generated. Class 6 is off; nothing to report.")
        return

    # --- 1. the realised mix ------------------------------------------------
    print(f"\n{RULE}\n1. WHAT IS ACTUALLY BEHIND THE 05s\n{RULE}")
    print(f"  {masked_failures:,} of {book_failures:,} failures came back "
          f"unexplained ({masked_failures / book_failures:.1%})\n")
    print(f"  {'true class':<30}{'count':>8}{'share':>9}")
    for cause, count in sorted(true_behind.items(), key=lambda kv: -kv[1]):
        label = cause if cause != rules.FAILURE_CLASS_AMBIGUOUS else (
            "05-native (block or glitch)"
        )
        print(f"  {label:<30}{count:>8,}{count / masked_failures:>9.1%}")
    print()
    print("  Only the truth side knows this table. The agent sees 05 and")
    print("  nothing else, which is the whole reason the class needs a")
    print("  posterior rather than a model.")

    # --- 2. is the posterior any good? --------------------------------------
    print(f"\n{RULE}\n2. IS THE FITTED POSTERIOR ANY GOOD?\n{RULE}")
    mean_posterior: dict[str, float] = {}
    for posterior in fitted_posteriors:
        for cause, weight in posterior.items():
            mean_posterior[cause] = mean_posterior.get(cause, 0.0) + weight
    for cause in mean_posterior:
        mean_posterior[cause] /= len(fitted_posteriors)

    print("  Fitted from the resolution history of past 05s, across the book.")
    print("  NOTHING here reads the generator's mixture - see agent/masking.py.\n")
    print(f"  {'cause':<30}{'fitted':>10}{'generator':>12}")
    for cause, weight in sorted(mean_posterior.items(), key=lambda kv: -kv[1]):
        truth_weight = {
            CAUSE_PERMANENT_BLOCK: LATENT_CAUSE_MIXTURE.get(
                CAUSE_ISSUER_RISK_BLOCK, 0.0
            ),
            CAUSE_TRANSIENT: LATENT_CAUSE_MIXTURE.get(
                "transient_issuer_glitch", 0.0
            ),
        }.get(cause, LATENT_CAUSE_MIXTURE.get(cause, 0.0))
        print(f"  {cause:<30}{weight:>10.3f}{truth_weight:>12.3f}")
    print()
    print("  The two columns are NOT expected to match. The generator mixture")
    print("  is over causes of NATIVE 05s; the fit is over every 05 including")
    print("  masked ones, whose causes follow the book's own failure mix. What")
    print("  matters is that the fit MOVES when the generator moves, which")
    print("  tests/test_ambiguous.py asserts directly.")
    if posterior_total:
        print()
        print(f"  Top-1 accuracy naming the true class behind a MASKED 05: "
              f"{posterior_hits / posterior_total:.1%} "
              f"({posterior_hits:,}/{posterior_total:,})")

    # --- 3. money and recovery ----------------------------------------------
    print(f"\n{RULE}\n3. MONEY RECOVERED (lead with this)\n{RULE}")
    print(f"  Value at risk across {n:,} ambiguous failures: "
          f"Rs {legacy.value_at_risk:,.0f}\n")
    print(f"  {'policy':<24}{'money':>13}{'of value':>11}{'recovery':>11}"
          f"{'net value':>15}{'notices':>9}")
    for name in ARMS:
        t = tallies[name]
        print(f"  {name:<24}{t.money:>13,.0f}{t.value_recovered_share:>11.1%}"
              f"{t.recovery_rate:>11.1%}{t.net_value:>15,.0f}{t.notifications:>9,}")

    gain = agent.money - legacy.money
    print(f"\n  sequencer over legacy: Rs {gain:,.0f} "
          f"({gain / n:,.0f} per ambiguous failure)")

    # --- the ablation, reported honestly ------------------------------------
    d_money = agent.money - blind.money
    d_net = agent.net_value - blind.net_value
    print(f"\n{RULE}\n4. WHAT THE POSTERIOR IS WORTH\n{RULE}")
    print("  no_masking_posterior: the agent still knows the decline was")
    print("  unexplained, but prices every 05 as a plain balance failure -")
    print("  which is what any system without a model of masking does.\n")
    print(f"  delta: Rs {d_money:+,.0f} recovered, Rs {d_net:+,.0f} net value")
    if abs(d_money) < 1.0 and abs(d_net) < 1.0:
        print()
        print("  IDENTICAL. Report it as the negative result it is, exactly as")
        print("  no_bank_health was in class 2 - do not bury it. A posterior")
        print("  that changes no decision is a finding about the decision, not")
        print("  a failure of the inference.")

    # --- 5. the permanent-block waste ---------------------------------------
    print(f"\n{RULE}\n5. PRESENTMENTS INTO A WALL NOBODY MENTIONED\n{RULE}")
    print("  A risk-blocked mandate never clears and is never announced as")
    print("  dead. Unlike an expired card there is no 54 to route on, so a")
    print("  fixed schedule presents until its budget runs out, every cycle.\n")
    print(f"  {'policy':<24}{'presentments':>14}{'chosen':>9}{'opt-outs':>10}")
    for name in ARMS:
        t = tallies[name]
        print(f"  {name:<24}{t.presentments:>14,}{t.chosen:>9,}{t.optouts:>10,}")

    emit(
        args.json_path,
        script="run_ambiguous.py",
        seeds=args.seeds,
        n=args.n,
        payload={
            "share": args.share,
            "book_failures": book_failures,
            "masked_failures": masked_failures,
            "masked_share": masked_failures / book_failures if book_failures else 0.0,
            "true_behind": true_behind,
            "fitted_posterior_mean": mean_posterior,
            "posterior_top1_accuracy": (
                posterior_hits / posterior_total if posterior_total else None
            ),
            "posterior_scored": posterior_total,
            "value_at_risk": legacy.value_at_risk,
            "policies": {
                name: {
                    "episodes": tallies[name].episodes,
                    "recovered": tallies[name].recovered,
                    "recovery_rate": tallies[name].recovery_rate,
                    "money": tallies[name].money,
                    "value_recovered_share": tallies[name].value_recovered_share,
                    "net_value": tallies[name].net_value,
                    "presentments": tallies[name].presentments,
                    "chosen_presentments": tallies[name].chosen,
                    "notifications": tallies[name].notifications,
                    "optouts": tallies[name].optouts,
                }
                for name in ARMS
            },
            "posterior_delta_money": d_money,
            "posterior_delta_net_value": d_net,
        },
    )

    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
