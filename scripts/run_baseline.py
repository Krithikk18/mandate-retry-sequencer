"""Run the fixed-schedule baseline against the oracle ceiling across 20 seeds.

    python scripts/run_baseline.py
    python scripts/run_baseline.py --seeds 5 --n 300 --irritation build

The two numbers that matter are the floor and the ceiling. Every agent built
later is scored as a fraction of the gap between them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generator.generate import generate_dataset  # noqa: E402
from evaluation.harness import compare  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from policy.oracle_policy import OraclePolicy  # noqa: E402

RULE = "=" * 78


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=500, help="customers per seed")
    parser.add_argument(
        "--irritation",
        choices=("decay", "build"),
        default="decay",
        help="direction of the opt-out irritation model; report both",
    )
    parser.add_argument("--no-oracle-metrics", action="store_true")
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    print(f"{RULE}\nBASELINE vs CEILING\n{RULE}")
    print(f"seeds 1..{args.seeds}   n={args.n}   irritation={args.irritation}\n")

    started = time.time()
    datasets = {}
    for seed in seeds:
        datasets[seed] = generate_dataset(n_customers=args.n, seed=seed)
        print(f"  generated seed {seed:>3}  "
              f"failures={datasets[seed].meta['n_failure_events']:>3}", end="\r")
    print(" " * 60, end="\r")

    legacy = LegacyFixedSchedule()
    # The oracle must be bound to ONE seed's truth: customer ids repeat across
    # seeds, so a merged truth table would be the wrong person's balance.
    # compare() calls this factory once per seed.
    report = compare(
        [legacy, OraclePolicy.for_dataset],
        seeds=seeds,
        n_customers=args.n,
        baseline_name=legacy.name,
        ceiling_name=OraclePolicy.name,
        irritation_direction=args.irritation,
        oracle_metrics=not args.no_oracle_metrics,
        strict_compliance=False,
        datasets=datasets,
    )

    print(report.render())

    floor = report.summary[legacy.name]["recovery_rate"]
    ceiling = report.summary[OraclePolicy.name]["recovery_rate"]
    print(RULE)
    print("THE SCALE EVERY AGENT IS SCORED ON")
    print(RULE)
    print(f"  fixed-schedule floor   {floor.mean:6.1%}   [{floor.low:.1%} .. {floor.high:.1%}]")
    print(f"  oracle ceiling         {ceiling.mean:6.1%}   [{ceiling.low:.1%} .. {ceiling.high:.1%}]")
    print(f"  headroom to capture    {ceiling.mean - floor.mean:6.1%} of failures")
    net_floor = report.summary[legacy.name]["net_value_total"]
    net_ceiling = report.summary[OraclePolicy.name]["net_value_total"]
    print(f"\n  net value  floor Rs {net_floor.mean:>14,.0f}"
          f"   ceiling Rs {net_ceiling.mean:>14,.0f}")
    print(f"  swing      Rs {net_ceiling.mean - net_floor.mean:>14,.0f} per {args.n} customers")
    print(f"\nelapsed {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
