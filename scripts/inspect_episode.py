"""Print the full decision log for one customer, day by day.

    python scripts/inspect_episode.py --customer CUST_0013
    python scripts/inspect_episode.py --customer CUST_0013 --policy oracle
    python scripts/inspect_episode.py --policy legacy --worst

This is the audit trail Razorpay's brief asks for: every decision, the numbers
behind it, what the truth was on that day, and how it ended. The hidden columns
are shown only because this is an inspection tool - the policy never saw them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generator.generate import generate_dataset  # noqa: E402
from generator.oracle import (  # noqa: E402
    days_to_next_salary,
    first_success_date,
    success_probability_true,
)
from evaluation.simulator import cycle_end_date, run_episode  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from policy.oracle_policy import OraclePolicy  # noqa: E402

RULE = "=" * 78


def build_policy(name: str, dataset):
    if name == "legacy":
        return LegacyFixedSchedule()
    if name == "oracle":
        return OraclePolicy.for_dataset(dataset)
    raise SystemExit(f"unknown policy {name!r}; choose legacy or oracle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer", type=str, default=None)
    parser.add_argument("--policy", choices=("legacy", "oracle"), default="legacy")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--worst",
        action="store_true",
        help="pick the episode with the most negative net value",
    )
    args = parser.parse_args()

    dataset = generate_dataset(n_customers=args.n, seed=args.seed)
    policy = build_policy(args.policy, dataset)

    events = {e.customer_id: e for e in dataset.failure_events}
    if args.customer:
        if args.customer not in events:
            raise SystemExit(
                f"{args.customer} has no failure event in seed {args.seed}. "
                f"Try one of: {', '.join(list(events)[:8])}"
            )
        chosen = args.customer
    elif args.worst:
        scored = [
            (
                run_episode(
                    policy,
                    dataset.customer(cid),
                    event,
                    dataset.balance_of(cid),
                    dataset.payment_history,
                    seed=args.seed,
                ).net_value,
                cid,
            )
            for cid, event in events.items()
        ]
        chosen = min(scored)[1]
    else:
        chosen = next(iter(events))

    event = events[chosen]
    customer = dataset.customer(chosen)
    series = dataset.balance_of(chosen)
    amount = customer.subscription_amount
    result = run_episode(
        policy, customer, event, series, dataset.payment_history, seed=args.seed
    )

    print(f"{RULE}\n{chosen}   policy={policy.name}   seed={args.seed}\n{RULE}")
    print(f"  pays Rs {amount:,.0f} on day {customer.debit_day_of_month} via {customer.rail}")
    print(f"  tenure {customer.tenure_months} months   annual LTV Rs {customer.annual_ltv:,.0f}")
    print(f"  failed on {event.original_debit_date}, cycle ends "
          f"{cycle_end_date(event.original_debit_date)}")
    print(f"  bounce charge per failed NACH debit: Rs {customer.bounce_charge:,.0f}")

    print("\n  HIDDEN (the policy never saw any of this):")
    print(f"    salary Rs {customer.salary_amount:,.0f} on day {customer.salary_day}"
          f"   burn {customer.burn_rate:.3f}/day   buffer {customer.buffer_ratio:.3f}")
    if customer.income_shock_start:
        print(f"    INCOME STOPPED on {customer.income_shock_start} - no date can clear")
    print(f"    balance at failure Rs {event.balance_at_failure:,.0f}")
    print(f"    next scheduled payday: +{days_to_next_salary(customer, event.original_debit_date)}d")
    best = first_success_date(customer, series, amount, after=event.original_debit_date)
    print(f"    earliest date that would have cleared: {best}")

    print(f"\n{RULE}\nDAY BY DAY\n{RULE}")
    print(f"  {'date':<12} {'action':<8} {'target':<12} {'balance':>10} "
          f"{'P(true)':>8}  note")
    for entry in result.decision_log:
        balance = float(series.loc[str(entry.day)])
        if entry.retry_date is not None and entry.action == "RETRY":
            probability = success_probability_true(
                customer, entry.retry_date, amount, n_sims=400
            )
            prob = f"{probability:8.2f}"
        else:
            prob = " " * 8
        target = str(entry.retry_date) if entry.retry_date else ""
        print(f"  {str(entry.day):<12} {entry.action:<8} {target:<12} "
              f"{balance:>10,.0f} {prob}  {entry.event}")
        for key, value in entry.reasoning.items():
            if key in ("policy",):
                continue
            print(f"      {key}: {value}")

    print(f"\n{RULE}\nOUTCOME: {result.outcome}\n{RULE}")
    print(f"  recovered          Rs {result.recovered_amount:>12,.0f}")
    print(f"  presentments       {result.attempts_used:>15}"
          f"   (cap {4})")
    print(f"  notifications      {result.notifications_sent:>15}"
          f"   (each one a chance to cancel)")
    print(f"  bounce charges     Rs {result.bounce_charges_paid:>12,.0f}   (paid by the customer)")
    print(f"  LTV retained       Rs {result.ltv_retained:>12,.0f}")
    print(f"  LTV lost           Rs {result.ltv_lost:>12,.0f}")
    print(f"  NET VALUE          Rs {result.net_value:>12,.0f}")
    print(f"  days to resolution {result.days_to_resolution:>15}")

    if best is not None and result.outcome != "RECOVERED":
        print(f"\n  The oracle says {best} would have cleared. This policy did not "
              f"present on it.")


if __name__ == "__main__":
    main()
