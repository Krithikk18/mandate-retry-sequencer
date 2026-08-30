"""Legacy versus the sequencer across the full mixed population.

    python scripts/run_multiclass.py
    python scripts/run_multiclass.py --seeds 20 --n 500

Most of this repository is about insufficient funds, where WHEN you retry
changes the answer. This script is about the two classes where it does not.

An expired card and a revoked mandate cannot clear on any date. A fixed
schedule cannot tell - it never reads the return code - so it presents four
times and recovers nothing. The sequencer routes on the code and stops
immediately. That difference costs nothing to implement and it is measured
here.

The revoked-mandate number is reported separately because it is not an
efficiency finding. Presenting against a mandate whose consent was withdrawn
under RBI Sec 6(c) is a REGULATORY BREACH, and a count of breaches is a
different kind of statement from a count of wasted rupees.

Five of the six classes are now generated and modelled. Only
``ambiguous_decline`` remains a stub; it is not generated and no numbers are
reported for it.

MONEY IS REPORTED ALONGSIDE RATE throughout, and that became necessary with
class 5 rather than being tidiness. AFA failures occur only above the RBI
Sec 8(a) Rs 15,000 ceiling, so they are a few per cent of episodes and roughly
a third of the rupees at risk. A rate-only table weights a Rs 199 streaming
mandate and a Rs 19,000 one equally, which is the wrong weighting for a
revenue-recovery claim and would have made the most valuable class on the book
look like noise.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.sequencer import SequencerPolicy  # noqa: E402
from evaluation.economics import processing_cost_total  # noqa: E402
from evaluation.simulator import run_episode  # noqa: E402
from generator import rules  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from router import implemented_classes  # noqa: E402

RULE = "=" * 92


class Tally:
    """Per-class totals for one policy."""

    def __init__(self) -> None:
        self.episodes = 0
        self.recovered = 0
        self.money = 0.0
        self.presentments = 0
        self.notifications = 0
        self.optouts = 0
        self.breaches = 0
        self.net_value = 0.0
        self.processing = 0.0
        self.bounce = 0.0

    def add(self, result, rail: str) -> None:
        self.episodes += 1
        self.recovered += result.outcome == "RECOVERED"
        self.money += result.recovered_amount
        self.presentments += result.attempts_used
        self.notifications += result.notifications_sent
        self.optouts += result.outcome == "OPTED_OUT"
        self.breaches += result.compliance_breaches
        self.net_value += result.net_value
        self.processing += processing_cost_total(rail, result.attempts_used)
        self.bounce += result.bounce_charges_paid

    @property
    def recovery_rate(self) -> float:
        return self.recovered / self.episodes if self.episodes else 0.0

    @property
    def wasted_presentments(self) -> int:
        """Presentments made on a class where no date could ever have cleared."""
        return self.presentments

    @property
    def wasted_cost(self) -> float:
        return self.processing + self.bounce


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    started = time.time()

    policies = {"legacy_fixed": LegacyFixedSchedule(), "sequencer": SequencerPolicy()}
    tallies: dict[str, dict[str, Tally]] = {
        name: defaultdict(Tally) for name in policies
    }

    for seed in seeds:
        dataset = generate_dataset(n_customers=args.n, seed=seed)
        for event in dataset.failure_events:
            customer = dataset.customer(event.customer_id)
            episode_args = (
                customer,
                event,
                dataset.balance_of(event.customer_id),
                dataset.payment_history,
            )
            for name, policy in policies.items():
                result = run_episode(
                    policy,
                    *episode_args,
                    seed=seed,
                    # Legacy never reads the return code, so it presents
                    # against revoked mandates. Count that rather than abort:
                    # the count IS the finding.
                    strict_compliance=False,
                )
                tallies[name][event.failure_class].add(result, customer.rail)

    classes = sorted({c for t in tallies.values() for c in t})
    print(f"{RULE}\nMULTI-CLASS EVALUATION\n{RULE}")
    print(f"{args.seeds} seeds x {args.n} customers\n")

    total_failures = sum(t.episodes for t in tallies["legacy_fixed"].values())
    print("  failure mix")
    for name in classes:
        share = tallies["legacy_fixed"][name].episodes / total_failures
        flag = "" if name in implemented_classes() else "   (stub - not generated)"
        print(f"    {name:<26} {share:>6.1%}  "
              f"({tallies['legacy_fixed'][name].episodes}){flag}")

    print(f"\n{RULE}\nPER CLASS\n{RULE}")
    header = (
        f"  {'class':<26} {'policy':<14} {'recovery':>9} {'money':>12} "
        f"{'present':>9} {'notices':>9} {'optouts':>8} {'net value':>14}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in classes:
        for policy_name in policies:
            t = tallies[policy_name][name]
            if not t.episodes:
                continue
            print(f"  {name if policy_name == 'legacy_fixed' else '':<26} "
                  f"{policy_name:<14} {t.recovery_rate:>9.1%} {t.money:>12,.0f} "
                  f"{t.presentments:>9,} "
                  f"{t.notifications:>9,} {t.optouts:>8,} {t.net_value:>14,.0f}")
        print()

    print(f"{RULE}\nAGGREGATE\n{RULE}")
    print(f"  {'policy':<14} {'recovery':>9} {'money':>12} {'present':>9} "
          f"{'notices':>9} {'optouts':>8} {'breaches':>9} {'net value':>14}")
    aggregate: dict[str, Tally] = {}
    for policy_name in policies:
        totals = Tally()
        for t in tallies[policy_name].values():
            for attr in ("episodes", "recovered", "presentments", "notifications",
                         "optouts", "breaches"):
                setattr(totals, attr, getattr(totals, attr) + getattr(t, attr))
            totals.money += t.money
            totals.net_value += t.net_value
        aggregate[policy_name] = totals
        print(f"  {policy_name:<14} {totals.recovery_rate:>9.1%} "
              f"{totals.money:>12,.0f} "
              f"{totals.presentments:>9,} {totals.notifications:>9,} "
              f"{totals.optouts:>8,} {totals.breaches:>9,} {totals.net_value:>14,.0f}")

    # Does the fifth class move rupees more than it moves rate? It should: it
    # is a few per cent of episodes carrying the largest mandates on the book.
    legacy_all, agent_all = aggregate["legacy_fixed"], aggregate["sequencer"]
    d_rate = agent_all.recovery_rate - legacy_all.recovery_rate
    d_money = agent_all.money - legacy_all.money
    money_share = d_money / legacy_all.money if legacy_all.money else 0.0
    rate_share = d_rate / legacy_all.recovery_rate if legacy_all.recovery_rate else 0.0
    print()
    print(f"  sequencer over legacy: {d_rate * 100:+.1f} points of recovery "
          f"({rate_share:+.1%} relative)")
    print(f"                         Rs {d_money:+,.0f} recovered "
          f"({money_share:+.1%} relative)")
    if money_share > rate_share:
        print()
        print("  The money gain outruns the rate gain, and that is the class-5")
        print("  effect showing up in the aggregate: the episodes the sequencer")
        print("  wins that legacy loses are disproportionately the large ones.")

    # --- the cheap finding ---------------------------------------------------
    dead = rules.NEVER_RETRY_CLASSES
    legacy_dead = Tally()
    agent_dead = Tally()
    for name in dead:
        for source, target in (
            (tallies["legacy_fixed"][name], legacy_dead),
            (tallies["sequencer"][name], agent_dead),
        ):
            target.episodes += source.episodes
            target.presentments += source.presentments
            target.notifications += source.notifications
            target.processing += source.processing
            target.bounce += source.bounce
            target.breaches += source.breaches

    print(f"\n{RULE}\nPRESENTMENTS SPENT ON STRUCTURALLY DEAD MANDATES\n{RULE}")
    print("  No date could ever have cleared these. Every presentment is waste,")
    print("  and every notification is a cancel-capable message sent to somebody")
    print("  whose only possible action the notice does not even ask for.\n")
    print(f"  {'':<14} {'mandates':>10} {'presentments':>14} {'notices':>10} "
          f"{'wasted cost':>14}")
    for label, t in (("legacy_fixed", legacy_dead), ("sequencer", agent_dead)):
        print(f"  {label:<14} {t.episodes:>10,} {t.presentments:>14,} "
              f"{t.notifications:>10,} Rs {t.wasted_cost:>11,.0f}")

    saved = legacy_dead.presentments - agent_dead.presentments
    saved_cost = legacy_dead.wasted_cost - agent_dead.wasted_cost
    per_hundred = saved_cost / total_failures * 100 if total_failures else 0.0
    print(f"\n  Routing on the return code saves {saved:,} presentments and")
    print(f"  Rs {saved_cost:,.0f} across {total_failures:,} failures - "
          f"Rs {per_hundred:,.0f} per 100.")
    print("  It is roughly fifty lines of code and needs no model at all.")

    print(f"\n{RULE}\nREGULATORY: PRESENTMENTS AGAINST REVOKED MANDATES\n{RULE}")
    print("  Consent withdrawn under RBI Sec 6(c). Presenting again is not")
    print("  inefficient, it is a breach. Reported separately for that reason.\n")
    for policy_name in policies:
        t = tallies[policy_name][rules.FAILURE_CLASS_OPTED_OUT]
        print(f"    {policy_name:<14} {t.breaches:>5,} breaches "
              f"across {t.episodes:,} revoked mandates")
    print("\n  A fixed schedule cannot avoid this: it never reads the return")
    print("  code, so it cannot know consent is gone.")
    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
