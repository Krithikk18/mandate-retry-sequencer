"""Failure class 5: AFA incomplete.

    python scripts/run_afa.py
    python scripts/run_afa.py --seeds 20 --n 500

**Read this class by MONEY, not by count.** Every other class in this project
is reported recovery-rate first. This one is reported rupees first, and the
reason is the whole argument for handling it at all: AFA applies only above the
RBI Sec 8(a) Rs 15,000 ceiling, so the class rides exclusively on the largest
mandates on the book. It is thin by episode count and heavy by value, and a
recovery-rate table alone would make it look ignorable.

It is also the first INTERVENTIONAL class. In classes 1-4 the agent could only
choose a date on a curve it did not draw. Here a NUDGE causally raises the
chance the customer authenticates, so the agent can move the curve - and the
``no_afa_nudge`` ablation below is what isolates that action from mere
scheduling.

What is reported, in order:

1. the count/value split for the whole book - the framing number
2. money recovered: legacy, sequencer, oracle ceiling, and the ablation,
   in rupees and as a share of the value at risk
3. recovery rate, WITH the episode count behind it, because the class is thin
   and a rate off thirty episodes is not a result
4. nudges sent and completions attributable to them
5. presentments burned against a shut authentication gate - the legacy waste
   number, and the third distinct reason a fixed schedule fails
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _emit import add_json_argument, emit  # noqa: E402
from agent.sequencer import SequencerConfig, SequencerPolicy  # noqa: E402
from evaluation.simulator import run_episode  # noqa: E402
from generator import rules  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from policy.oracle_policy import OraclePolicy  # noqa: E402

RULE = "=" * 92


class Tally:
    """Per-policy totals over the AFA failures of one run."""

    def __init__(self) -> None:
        self.episodes = 0
        self.recovered = 0
        self.money = 0.0
        self.value_at_risk = 0.0
        self.presentments = 0
        self.chosen = 0
        self.afa_refused = 0
        self.notifications = 0
        self.nudges = 0
        self.nudge_completions = 0
        self.net_value = 0.0
        self.goodwill = 0.0
        self.cycles_held = 0
        self.per_seed_rate: list[float] = []

    def add(self, result, amount: float) -> None:
        self.episodes += 1
        self.recovered += result.outcome == "RECOVERED"
        self.money += result.recovered_amount
        self.value_at_risk += amount
        self.presentments += result.attempts_used
        self.chosen += len(result.retry_dates)
        self.afa_refused += result.presentments_refused_for_afa
        self.notifications += result.notifications_sent
        self.nudges += result.nudges_sent
        self.nudge_completions += result.otp_completions_from_nudges
        self.net_value += result.net_value
        self.goodwill += result.goodwill_cost_paid
        self.cycles_held += result.cycles_held

    @property
    def recovery_rate(self) -> float:
        return self.recovered / self.episodes if self.episodes else 0.0

    @property
    def value_recovered_share(self) -> float:
        return self.money / self.value_at_risk if self.value_at_risk else 0.0


def afa_events(dataset):
    return [
        e
        for e in dataset.failure_events
        if e.failure_class == rules.FAILURE_CLASS_AFA_INCOMPLETE
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


ARMS = ("legacy_fixed", "sequencer", "oracle_ceiling", "no_afa_nudge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    add_json_argument(parser)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    started = time.time()

    tallies = {name: Tally() for name in ARMS}
    per_seed_rates: dict[str, list[float]] = {name: [] for name in ARMS}
    seeds_with_episodes = 0

    # Book-level framing, over ALL failures rather than only this class.
    book_failures = book_value = 0
    otp_failures = otp_value = 0.0
    afa_failures = afa_value = 0.0

    for seed in seeds:
        dataset = generate_dataset(n_customers=args.n, seed=seed)

        for event in dataset.failure_events:
            customer = dataset.customer(event.customer_id)
            book_failures += 1
            book_value += customer.subscription_amount
            if customer.requires_otp:
                otp_failures += 1
                otp_value += customer.subscription_amount
            if event.failure_class == rules.FAILURE_CLASS_AFA_INCOMPLETE:
                afa_failures += 1
                afa_value += customer.subscription_amount

        events = afa_events(dataset)
        if not events:
            continue
        seeds_with_episodes += 1

        policies = {
            "legacy_fixed": LegacyFixedSchedule(),
            "sequencer": SequencerPolicy(),
            "oracle_ceiling": OraclePolicy.for_dataset(dataset),
            "no_afa_nudge": SequencerPolicy(
                SequencerConfig(use_afa_nudge=False), name="no_afa_nudge"
            ),
        }

        seed_recovered = {name: 0 for name in ARMS}
        for event in events:
            amount = dataset.customer(event.customer_id).subscription_amount
            for name, policy in policies.items():
                result = play(policy, dataset, event, seed)
                tallies[name].add(result, amount)
                seed_recovered[name] += result.outcome == "RECOVERED"
        for name in ARMS:
            per_seed_rates[name].append(seed_recovered[name] / len(events))

    legacy, agent, oracle, blind = (tallies[name] for name in ARMS)
    n = legacy.episodes

    print(f"{RULE}\nFAILURE CLASS 5: AFA INCOMPLETE (RBI Sec 8(a), return code 1A)\n{RULE}")
    print(f"{args.seeds} seeds x {args.n} customers")
    if n == 0:
        print("\nNo AFA failures generated. Nothing to report.")
        return

    print(f"\n{RULE}\n1. THE FRAMING NUMBER: THIN BY COUNT, HEAVY BY VALUE\n{RULE}")
    print("  AFA applies only above the Sec 8(a) Rs 15,000 ceiling, so this class")
    print("  rides on the largest mandates on the book and nothing else.\n")
    print(f"  {'':<34}{'by COUNT':>16}{'by VALUE':>20}")
    print(f"  {'all failures':<34}{book_failures:>16,}{book_value:>20,.0f}")
    print(
        f"  {'on OTP-gated mandates':<34}"
        f"{otp_failures / book_failures:>15.1%}{otp_value / book_value:>19.1%}"
    )
    print(
        f"  {'refused at the AFA gate (1A)':<34}"
        f"{afa_failures / book_failures:>15.1%}{afa_value / book_value:>19.1%}"
    )
    print()
    print(
        f"  That gap is the argument for handling this class. A "
        f"{afa_failures / book_failures:.1%} slice of"
    )
    print(
        f"  failures carrying {afa_value / book_value:.1%} of the rupees at risk "
        "is not a"
    )
    print("  rounding error, and a recovery-rate table would have hidden it.")

    print(f"\n{RULE}\n2. MONEY RECOVERED (lead with this)\n{RULE}")
    print(f"  Value at risk across {n:,} AFA failures: Rs {legacy.value_at_risk:,.0f}\n")
    print(
        f"  {'policy':<18}{'money':>14}{'of value':>11}{'net value':>14}"
        f"{'goodwill':>11}{'notices':>9}"
    )
    for name in ARMS:
        t = tallies[name]
        print(
            f"  {name:<18}{t.money:>14,.0f}{t.value_recovered_share:>11.1%}"
            f"{t.net_value:>14,.0f}{t.goodwill:>11,.0f}{t.notifications:>9,}"
        )
    gain = agent.money - legacy.money
    print(
        f"\n  sequencer over legacy: Rs {gain:,.0f} "
        f"({gain / n:,.0f} per AFA failure, {gain / args.seeds:,.0f} per seed)"
    )

    print(f"\n{RULE}\n3. RECOVERY RATE - AND THE EPISODES BEHIND IT\n{RULE}")
    print(f"  {n:,} AFA episodes across {seeds_with_episodes} of {args.seeds} seeds")
    print(f"  ({n / max(seeds_with_episodes, 1):.1f} per seed - see the caveat below)\n")
    print(f"  {'policy':<18}{'recovery':>10}{'recovered':>12}{'per-seed spread':>26}")
    for name in ARMS:
        t = tallies[name]
        rates = per_seed_rates[name]
        spread = (
            f"[{min(rates):.0%} .. {max(rates):.0%}]" if rates else "n/a"
        )
        print(
            f"  {name:<18}{t.recovery_rate:>10.1%}{t.recovered:>7,}/{t.episodes:<4}"
            f"{spread:>26}"
        )
    if rates:
        sd = statistics.pstdev(per_seed_rates["sequencer"])
        print(f"\n  Per-seed standard deviation of the sequencer rate: {sd:.1%}")
    print()
    print("  STATE THIS PLAINLY IN ANY WRITE-UP. The class is thin: a handful of")
    print("  episodes per seed, so the per-seed rate swings hard and the pooled")
    print("  rate carries a wide interval. The VALUE figures above are the ones")
    print("  to quote; the rate is context, not the result.")

    print(f"\n{RULE}\n4. THE INTERVENTIONAL ACTION: WHAT NUDGING IS WORTH\n{RULE}")
    print("  no_afa_nudge: the agent may only choose DATES, never ask for")
    print("  authentication. Everything else is identical.\n")
    print(
        f"  {'variant':<18}{'money':>14}{'recovery':>10}{'nudges':>9}"
        f"{'from nudges':>13}{'present':>9}{'net value':>14}"
    )
    for name in ("sequencer", "no_afa_nudge"):
        t = tallies[name]
        print(
            f"  {name:<18}{t.money:>14,.0f}{t.recovery_rate:>10.1%}{t.nudges:>9,}"
            f"{t.nudge_completions:>13,}{t.presentments:>9,}{t.net_value:>14,.0f}"
        )
    d_money = agent.money - blind.money
    d_net = agent.net_value - blind.net_value
    print(f"\n  delta: Rs {d_money:+,.0f} recovered, Rs {d_net:+,.0f} net value")
    print(
        f"  {agent.nudge_completions:,} of {agent.nudges:,} nudges opened a gate "
        "that would have stayed shut"
    )
    print("  (an exact counterfactual, not an estimate: the AFA draw is keyed on")
    print("  identity and date, never on the nudge count, so the unnudged branch")
    print("  re-tests the same uniform against a lower threshold.)")
    if abs(d_money) < 1.0 and abs(d_net) < 1.0:
        print()
        print("  IDENTICAL. Report it as the negative result it is, exactly as")
        print("  no_bank_health was in class 2 - do not bury it.")

    print(f"\n{RULE}\n5. WHY LEGACY FAILS HERE, AND IT IS A NEW REASON\n{RULE}")
    print("  Presentments are split into the ones the POLICY CHOSE and the ones")
    print("  that happened anyway. A live mandate keeps auto-billing on its")
    print("  anchor date every cycle it is carried, so a policy that HOLDS")
    print("  accumulates presentments it never decided to make. Pooling the two")
    print("  would read as the sequencer being the wasteful one, when on the")
    print("  only column it controls it makes barely half as many as legacy.\n")
    print(
        f"  {'policy':<18}{'chosen':>9}{'automatic':>11}{'total':>8}"
        f"{'refused 1A':>12}{'held':>7}"
    )
    for name in ARMS:
        t = tallies[name]
        print(
            f"  {name:<18}{t.chosen:>9,}{t.presentments - t.chosen:>11,}"
            f"{t.presentments:>8,}{t.afa_refused:>12,}{t.cycles_held:>7,}"
        )
    print()
    print(
        f"  Legacy chose {legacy.chosen:,} presentments to the sequencer's "
        f"{agent.chosen:,}, and recovered"
    )
    print(
        f"  Rs {legacy.money:,.0f} against Rs {agent.money:,.0f}. More attempts, less money:"
    )
    print("  the presentments it added were made into a gate that was shut.")
    print()
    print("  Class 1 legacy was MISTIMED - it retried before payday.")
    print("  Class 2 legacy was ACCIDENTALLY FINE - short outages had cleared.")
    print("  Class 5 legacy is STRUCTURALLY FUTILE. It presents four times")
    print("  against a mandate that refuses every one of them until a human")
    print("  completes an authentication step legacy never asks for, because it")
    print("  never reads the return code and so never learns one is outstanding.")
    print()
    print("  Each of those refusals spends an NPCI attempt and fires an RBI")
    print("  Sec 6(a) pre-debit notice - a cancel-capable message - to learn")
    print("  nothing at all. The debit never reached the account, so it did not")
    print("  even establish the customer was short.")
    print()
    print("  And because these are the largest mandates on the book, the")
    print("  futility is expensive in a way the recovery-rate column cannot show:")
    print(
        f"  Rs {legacy.value_at_risk - legacy.money:,.0f} left uncollected by legacy "
        f"against Rs {oracle.value_at_risk - oracle.money:,.0f} by the ceiling."
    )

    # Written from the very tallies the tables above were rendered from, so the
    # file cannot disagree with the report. See scripts/_emit.py.
    emit(
        args.json_path,
        script="run_afa.py",
        seeds=args.seeds,
        n=args.n,
        payload={
            "afa_failures": n,
            "seeds_with_episodes": seeds_with_episodes,
            "episodes_per_seed": n / max(seeds_with_episodes, 1),
            "book": {
                "failures": book_failures,
                "value": book_value,
                "otp_gated_count_share": otp_failures / book_failures,
                "otp_gated_value_share": otp_value / book_value,
                "afa_count_share": afa_failures / book_failures,
                "afa_value_share": afa_value / book_value,
            },
            "value_at_risk": legacy.value_at_risk,
            "policies": {
                name: {
                    "episodes": tallies[name].episodes,
                    "recovered": tallies[name].recovered,
                    "recovery_rate": tallies[name].recovery_rate,
                    "money": tallies[name].money,
                    "value_recovered_share": tallies[name].value_recovered_share,
                    "net_value": tallies[name].net_value,
                    "goodwill": tallies[name].goodwill,
                    "notifications": tallies[name].notifications,
                    "nudges": tallies[name].nudges,
                    "nudge_completions": tallies[name].nudge_completions,
                    "chosen_presentments": tallies[name].chosen,
                    "automatic_presentments": (
                        tallies[name].presentments - tallies[name].chosen
                    ),
                    "presentments": tallies[name].presentments,
                    "afa_refused": tallies[name].afa_refused,
                    "cycles_held": tallies[name].cycles_held,
                    "per_seed_rate_min": min(per_seed_rates[name]),
                    "per_seed_rate_max": max(per_seed_rates[name]),
                    "uncollected": (
                        tallies[name].value_at_risk - tallies[name].money
                    ),
                }
                for name in ARMS
            },
            "sequencer_over_legacy_money": gain,
            "sequencer_over_legacy_per_failure": gain / n,
            "per_seed_rate_sd_sequencer": statistics.pstdev(
                per_seed_rates["sequencer"]
            ),
            "nudge_delta_money": d_money,
            "nudge_delta_net_value": d_net,
            "headroom_by_value": {
                "floor": legacy.value_recovered_share,
                "ceiling": oracle.value_recovered_share,
                "agent": agent.value_recovered_share,
                "gap_points": (
                    oracle.value_recovered_share - legacy.value_recovered_share
                )
                * 100,
                "captured": (
                    (agent.value_recovered_share - legacy.value_recovered_share)
                    / (oracle.value_recovered_share - legacy.value_recovered_share)
                    if oracle.value_recovered_share != legacy.value_recovered_share
                    else None
                ),
            },
            "headroom_by_rate_captured": (
                (agent.recovery_rate - legacy.recovery_rate)
                / (oracle.recovery_rate - legacy.recovery_rate)
                if oracle.recovery_rate != legacy.recovery_rate
                else None
            ),
        },
    )

    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
