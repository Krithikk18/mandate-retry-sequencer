"""Does each piece of the sequencer earn its place?

    python scripts/run_ablations.py
    python scripts/run_ablations.py --seeds 20 --n 500

Each variant differs from `full` by exactly one thing. A component that can be
removed without the numbers moving is not a contribution, and reporting a weak
result honestly is worth more than claiming one that is not there.

    full             everything on
    no_payday        tier-2 model replaced with a flat base rate
    no_optout_cost   the churn term dropped from the money equation
    no_hold          HOLD removed from the action set
    with_nudge       NUDGE switched on (it is off by default - see below)
    tier1_payday     tier 1 (day-of-month frequency) instead of tier 2
    no_bank_health   no cross-customer pooling of bank outage evidence
    irritation_build the opposite irritation direction

Legacy and the oracle ceiling are shown alongside as the floor and the roof.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.sequencer import SequencerConfig, SequencerPolicy  # noqa: E402
from evaluation.harness import compare  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from policy.oracle_policy import OraclePolicy  # noqa: E402

RULE = "=" * 100

VARIANTS: dict[str, dict] = {
    "full": {},
    "no_payday": {"flat_liquidity": True},
    "no_optout_cost": {"include_optout_cost": False},
    "no_hold": {"use_hold": False},
    "with_nudge": {"use_nudge": True},
    "tier1_payday": {"payday_tier": 1},
    "no_bank_health": {"use_bank_health": False},
    "irritation_build": {"irritation_direction": "build"},
}

HEADLINE = (
    ("recovery_rate", "recovery", "{:.1%}"),
    ("money_recovered", "money", "{:,.0f}"),
    ("net_value_total", "net value", "{:,.0f}"),
    ("attempts_used_total", "attempts", "{:,.0f}"),
    ("optouts_induced", "optouts", "{:.1f}"),
    ("customers_retained", "retained", "{:.1f}"),
    ("wasted_attempts", "wasted", "{:.1f}"),
    ("cycles_held_total", "held", "{:.0f}"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--no-oracle-metrics", action="store_true")
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    started = time.time()

    print(f"{RULE}\nABLATIONS\n{RULE}")
    print(f"{args.seeds} seeds x {args.n} customers\n")
    datasets = {s: generate_dataset(n_customers=args.n, seed=s) for s in seeds}

    policies = [LegacyFixedSchedule(), OraclePolicy.for_dataset]
    for label, overrides in VARIANTS.items():
        policies.append(SequencerPolicy(SequencerConfig(**overrides), name=label))

    report = compare(
        policies,
        seeds=seeds,
        n_customers=args.n,
        baseline_name="legacy_fixed",
        ceiling_name=OraclePolicy.name,
        oracle_metrics=not args.no_oracle_metrics,
        strict_compliance=False,
        datasets=datasets,
    )

    order = ["legacy_fixed", *VARIANTS, OraclePolicy.name]
    header = f"  {'variant':<18}" + "".join(f"{label:>13}" for _, label, _ in HEADLINE)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in order:
        if name not in report.summary:
            continue
        row = f"  {name:<18}"
        for metric, _, fmt in HEADLINE:
            summary = report.summary[name].get(metric)
            row += f"{fmt.format(summary.mean) if summary and summary.values else 'n/a':>13}"
        print(row)

    print(f"\n{RULE}\nRECOVERY RATE, mean [min .. max] across {args.seeds} seeds\n{RULE}")
    for name in order:
        if name not in report.summary:
            continue
        s = report.summary[name]["recovery_rate"]
        print(f"  {name:<18} {s.mean:>7.1%}   [{s.low:.1%} .. {s.high:.1%}]")

    print(f"\n{RULE}\nWHAT EACH COMPONENT IS WORTH\n{RULE}")
    print("  Deltas are against `full`, paired within each seed - the same")
    print("  customers, the same draws - so seed noise cancels. `seeds worse`")
    print("  counts how many of the seeds the variant lost on: a real effect")
    print("  shows up on most seeds, not just in the mean.\n")
    full_rows = report.per_seed["full"]
    print(f"  {'variant':<18} {'d recovery':>12} {'d net value':>14} {'seeds worse':>12}   verdict")
    for name in VARIANTS:
        if name == "full":
            continue
        rows = report.per_seed[name]
        rec_deltas = [v.recovery_rate - f.recovery_rate for v, f in zip(rows, full_rows)]
        net_deltas = [v.net_value_total - f.net_value_total for v, f in zip(rows, full_rows)]
        delta_rec = sum(rec_deltas) / len(rec_deltas)
        delta_net = sum(net_deltas) / len(net_deltas)
        worse = sum(1 for d in net_deltas if d < 0)
        agreeing = max(worse, len(net_deltas) - worse) / len(net_deltas)

        # Three bands, not two. An effect that shows up on 15 of 20 seeds is
        # not the same claim as one that shows up on all 20, and collapsing
        # them into a single "earns its place" overstates the weaker one.
        if abs(delta_net) < 1.0 and abs(delta_rec) < 1e-9:
            # Not "a liability" and not "borderline" - literally no effect.
            # A component that changes nothing to the rupee on any seed is not
            # doing anything, and saying so plainly is the useful report.
            verdict = "NO EFFECT AT ALL - identical on every seed"
        elif agreeing < 0.75:
            verdict = "NOT CONSISTENT across seeds - do not claim it"
        elif agreeing < 0.85:
            direction = "hurts" if delta_net < 0 else "helps"
            verdict = f"BORDERLINE ({worse}/{len(net_deltas)}) - removing it {direction}, weakly"
        elif delta_net < 0:
            verdict = "removing it hurts: the component earns its place"
        else:
            verdict = "removing it HELPS - the component is a liability"
        print(f"  {name:<18} {delta_rec:>+11.1%} {delta_net:>+14,.0f} "
              f"{worse:>7}/{len(net_deltas):<4}   {verdict}")

    floor = report.summary["legacy_fixed"]["recovery_rate"].mean
    ceiling = report.summary[OraclePolicy.name]["recovery_rate"].mean
    agent = report.summary["full"]["recovery_rate"].mean
    print(f"\n{RULE}\nHEADLINE\n{RULE}")
    print(f"  legacy floor      {floor:>7.1%}")
    print(f"  sequencer         {agent:>7.1%}")
    print(f"  oracle ceiling    {ceiling:>7.1%}")
    if ceiling > floor:
        print(f"\n  headroom captured {(agent - floor) / (ceiling - floor):>7.1%}")
    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
