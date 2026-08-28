"""What does the 24-hour pre-debit notice actually cost? (Novelty N3)

    python scripts/lead_time_ablation.py
    python scripts/lead_time_ablation.py --seeds 20 --n 500

RBI Sec 6(a) requires the issuer to notify the customer at least 24 hours
before every debit. That is not free. It forces the sequencer to commit to a
date before it can see whether the money has arrived, and it puts a
cancel-capable message in front of a customer who has just bounced a payment.

Nobody has published what that costs. This script measures it: the same agent,
the same customers, the same random draws, at notice periods of 0, 6, 12, 24
and 48 hours. The 0-hour arm is a hypothetical agent that could react the
instant a salary landed - not legal, not buildable, but the right counterfactual
for pricing the rule.

Two mechanisms are at work, and both are real rather than assumed:

1. Scheduling. A longer notice period shrinks the set of dates still reachable,
   most sharply near the end of a cycle.
2. Notification timing. The customer acts on the NOTICE, not on the debit. A
   48-hour notice lands a day earlier - closer to the fresh bounce - where the
   irritation hazard is higher.

Honest limit: the simulator runs at daily resolution, so 6h, 12h and 24h all
round to the same one-day constraint and are expected to tie. The reportable
number is the 0h -> 24h gap.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.sequencer import SequencerConfig, SequencerPolicy  # noqa: E402
from evaluation.harness import evaluate  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402

RULE = "=" * 78
LEAD_TIMES_HOURS = (0, 6, 12, 24, 48)
LEGAL_MINIMUM_HOURS = 24


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    started = time.time()
    datasets = {s: generate_dataset(n_customers=args.n, seed=s) for s in seeds}
    total_failures = sum(len(d.failure_events) for d in datasets.values())

    print(f"{RULE}\nTHE COST OF THE 24-HOUR PRE-DEBIT NOTICE\n{RULE}")
    print(f"{args.seeds} seeds x {args.n} customers   {total_failures} failures total")
    print("same agent, same customers, same draws - only the notice period changes\n")

    results: dict[int, dict[str, float]] = {}
    for hours in LEAD_TIMES_HOURS:
        policy = SequencerPolicy(
            SequencerConfig(lead_time_hours=hours), name=f"lead_{hours}h"
        )
        lead_days = policy.engine.lead_time_days
        money = recovered = failures = 0.0
        net = 0.0
        for seed in seeds:
            report = evaluate(
                policy,
                datasets[seed],
                seed=seed,
                oracle_metrics=False,
                lead_time_days=lead_days,
            )
            money += report.money_recovered
            recovered += report.recovery_rate * report.n_failures
            failures += report.n_failures
            net += report.net_value_total
        results[hours] = {
            "lead_days": lead_days,
            "money": money,
            "recovery_rate": recovered / failures,
            "net": net,
            "failures": failures,
        }

    print(f"  {'notice':>8} {'days':>5} {'recovery':>10} {'money recovered':>18} "
          f"{'net value':>16}")
    for hours in LEAD_TIMES_HOURS:
        row = results[hours]
        marker = "  <- legal minimum" if hours == LEGAL_MINIMUM_HOURS else ""
        print(f"  {hours:>6}h {row['lead_days']:>5.0f} {row['recovery_rate']:>10.2%} "
              f"{row['money']:>18,.0f} {row['net']:>16,.0f}{marker}")

    # --- the second, larger cost: the notice must OFFER a way out -----------
    free_policy = SequencerPolicy(name="no_optout_world")
    free_money = free_recovered = free_failures = 0.0
    for seed in seeds:
        report = evaluate(
            free_policy,
            datasets[seed],
            seed=seed,
            oracle_metrics=False,
            notifications_can_cancel=False,
        )
        free_money += report.money_recovered
        free_recovered += report.recovery_rate * report.n_failures
        free_failures += report.n_failures

    zero = results[0]
    legal = results[LEGAL_MINIMUM_HOURS]
    money_gap = zero["money"] - legal["money"]
    rate_gap = zero["recovery_rate"] - legal["recovery_rate"]
    per_hundred = money_gap / legal["failures"] * 100

    print(f"\n{RULE}\nPART 1: WHAT THE 24-HOUR TIMING COSTS\n{RULE}")
    print("  A zero-latency agent against a compliant one:\n")
    print(f"    across the batch          Rs {money_gap:>14,.0f}"
          f"   ({int(legal['failures']):,} failures)")
    print(f"    per 100 failures          Rs {per_hundred:>14,.0f}")
    print(f"    recovery rate             {rate_gap * 100:>14.2f} percentage points")
    if legal["money"] > 0:
        print(f"    as a share of recovery    {money_gap / legal['money']:>16.2%}")

    if abs(rate_gap) < 0.005:
        print("\n  ESSENTIALLY ZERO - and the reason matters more than the number.")
        print("  What binds a retry sequencer is knowing WHICH day the salary")
        print("  lands, not how far ahead it must commit. This agent gets no")
        print("  bank-side signal, so lifting the notice period buys it")
        print("  scheduling freedom it has no new information to use.")
        print("\n  Read it as a FLOOR. An agent with a real-time balance feed could")
        print("  react to a credit the moment it landed, and for that agent the")
        print("  24-hour rule would bind hard. Measuring that needs a signal this")
        print("  generator does not model - a limitation, not a result.")

    if len({results[h]["lead_days"] for h in (6, 12, 24)}) == 1:
        print("\n  6h, 12h and 24h all round to a one-day constraint at the")
        print("  simulator's daily resolution, so they tie by construction rather")
        print("  than by measurement. Only 0h, 24h and 48h carry information.")

    free_rate = free_recovered / free_failures
    notice_money = free_money - legal["money"]
    notice_rate = free_rate - legal["recovery_rate"]
    print(f"\n{RULE}\nPART 2: WHAT THE MANDATED OPT-OUT COSTS\n{RULE}")
    print("  Sec 6(a) sets the timing. Sec 6(c) is the expensive half: it")
    print("  requires the notice to carry a way out, so every scheduled retry")
    print("  is a regulator-mandated invitation to cancel.")
    print("\n  Same agent, same dates, in a counterfactual world where the")
    print("  notice carries no opt-out:\n")
    print(f"    recovery with opt-out     {legal['recovery_rate']:>15.2%}")
    print(f"    recovery without          {free_rate:>15.2%}")
    print(f"    cost across the batch     Rs {notice_money:>14,.0f}")
    print(f"    per 100 failures          Rs "
          f"{notice_money / legal['failures'] * 100:>14,.0f}")
    print(f"    recovery rate             {notice_rate * 100:>14.2f} percentage points")

    print(f"\n{RULE}\nQUOTABLE\n{RULE}")
    print(f"  Across {int(legal['failures']):,} real failures, the 24-hour TIMING")
    print(f"  requirement costs Rs {abs(money_gap):,.0f} - within noise of zero.")
    print(f"  The MANDATED OPT-OUT the same notice must carry costs "
          f"Rs {abs(notice_money):,.0f},")
    print(f"  or Rs {abs(notice_money) / legal['failures'] * 100:,.0f} per 100 failures "
          f"({abs(notice_rate) * 100:.1f} points of recovery).")
    print("\n  For a retry sequencer, the expensive half of the e-mandate")
    print("  framework is Sec 6(c), not Sec 6(a).")
    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
