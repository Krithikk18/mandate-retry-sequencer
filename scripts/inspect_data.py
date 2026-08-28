"""Print a human-readable summary of a generated dataset.

    python scripts/inspect_data.py
    python scripts/inspect_data.py --n 2000 --seed 7 --save data/run_7

Everything here is text so it can be pasted into a notebook, an issue, or a
demo script without a plotting dependency.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generator.balance import daily_balance  # noqa: E402
from generator.generate import generate_dataset, save_dataset  # noqa: E402
from generator.oracle import days_to_next_salary, first_success_date  # noqa: E402
from generator.population import MONTH_END_SALARY_DAYS  # noqa: E402
from generator.rules import bounce_charge  # noqa: E402

RULE = "=" * 78
THIN = "-" * 78


def header(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def bar(value: float, peak: float, width: int = 46, char: str = "#") -> str:
    if peak <= 0:
        return ""
    return char * max(0, int(round(width * value / peak)))


# --- sections ----------------------------------------------------------------


def population_summary(dataset) -> None:
    header("POPULATION")
    customers = dataset.customers
    n = len(customers)
    amounts = np.array([c.subscription_amount for c in customers])
    salaries = np.array([c.salary_amount for c in customers])
    tenures = np.array([c.tenure_months for c in customers])

    print(f"customers                  {n}")
    print(f"as_of                      {dataset.as_of}")
    print(f"seed                       {dataset.seed}")
    print(
        f"balance window             {dataset.meta['series_start']} "
        f"+ {dataset.meta['series_days']} days "
        f"({dataset.meta['history_days']}d history, "
        f"{dataset.meta['future_horizon_days']}d oracle horizon)"
    )

    print(f"\nsubscription amount (Rs)   median {np.median(amounts):8,.0f}   "
          f"mean {amounts.mean():8,.0f}   max {amounts.max():8,.0f}")
    tiers = [
        ("Rs   99 -   299", amounts < 300),
        ("Rs  300 -   999", (amounts >= 300) & (amounts < 1000)),
        ("Rs 1000 -  4999", (amounts >= 1000) & (amounts < 5000)),
        ("Rs 5000 - 20000", amounts >= 5000),
    ]
    for label, mask in tiers:
        print(f"  {label}   {pct(mask.mean())}  ({int(mask.sum()):4d})")

    print(f"\nmonthly LTV (Rs)           median {np.median(amounts * 12):8,.0f}   "
          f"book {(amounts * 12).sum():12,.0f}")
    print(f"salary (Rs)                median {np.median(salaries):8,.0f}   "
          f"p10 {np.percentile(salaries, 10):8,.0f}   p90 {np.percentile(salaries, 90):8,.0f}")
    print(f"tenure (months)            mean   {tenures.mean():8.1f}   "
          f"median {np.median(tenures):6.0f}   max {tenures.max():4d}")

    rails = Counter(c.rail for c in customers)
    print("\nrail")
    for rail, count in rails.most_common():
        charge = bounce_charge(rail)
        note = f"bounce charge Rs {charge:.0f}" if charge else "no bounce charge"
        print(f"  {rail:<14} {pct(count / n)}  ({count:4d})   {note}")

    otp = sum(c.requires_otp for c in customers)
    print(f"\nabove the Rs 15,000 OTP threshold   {otp} customers ({pct(otp / n)})")
    print("  (these need OTP on every attempt, so a silent retry cannot work)")

    shocked = [c for c in customers if c.income_shock_start is not None]
    print(f"\nincome shocked (salary stopped)     {len(shocked)} customers "
          f"({pct(len(shocked) / n)})")
    print("  HIDDEN. For these no retry date can ever succeed, so CANCEL/HOLD is")
    print("  the only correct action. Without them a stopping rule is untestable.")


def salary_day_histogram(dataset) -> None:
    header("SALARY DAY HISTOGRAM  (the India-specific signal)")
    counts = Counter(c.salary_day for c in dataset.customers)
    n = len(dataset.customers)
    peak = max(counts.values())
    for day in sorted(counts):
        share = counts[day] / n
        flag = " <- month end" if day in MONTH_END_SALARY_DAYS else ""
        print(f"  day {day:2d}  {pct(share)} {bar(counts[day], peak)}{flag}")
    month_end = sum(c for d, c in counts.items() if d in MONTH_END_SALARY_DAYS)
    print(f"\n  paid on day 1 / 30 / 31: {pct(month_end / n)} of the book")

    header("DEBIT DAY HISTOGRAM  (uniform: merchants ignore payday)")
    debit = Counter(c.debit_day_of_month for c in dataset.customers)
    peak = max(debit.values())
    for day in sorted(debit):
        print(f"  day {day:2d}  {pct(debit[day] / n)} {bar(debit[day], peak, width=30)}")


def failure_summary(dataset) -> None:
    header("FAILURES  (not imposed - produced by the balance process)")
    n = len(dataset.customers)
    events = dataset.failure_events
    print(f"natural failure rate (current cycle)   {pct(dataset.natural_failure_rate)}"
          f"   ({len(events)} of {n})")
    print(f"historical first-attempt failure rate  "
          f"{pct(dataset.meta['historical_first_attempt_failure_rate'])}"
          f"   (over {dataset.meta['historical_original_attempts']} closed cycles)")
    print(f"payment history records                {len(dataset.payment_history)}")

    if not events:
        return

    failed = [dataset.customer(e.customer_id) for e in events]
    amounts = np.array([c.subscription_amount for c in failed])
    all_amounts = np.array([c.subscription_amount for c in dataset.customers])
    print(f"\nfailing customers pay a median of Rs {np.median(amounts):,.0f} "
          f"vs Rs {np.median(all_amounts):,.0f} for the book")

    rails = Counter(c.rail for c in failed)
    print("\nfailure rate by rail")
    for rail in ("upi_autopay", "nach"):
        total = sum(1 for c in dataset.customers if c.rail == rail)
        if total:
            print(f"  {rail:<14} {pct(rails[rail] / total)}  "
                  f"({rails[rail]:3d} of {total:4d})")

    exposed = sum(c.bounce_charge for c in failed)
    print(f"\nbank bounce charges already levied on failed customers: Rs {exposed:,.0f}")
    print("  (NACH only - every further NACH retry that fails adds Rs 500 more)")


def days_to_salary_distribution(dataset) -> None:
    header("DAYS FROM FAILURE TO THE NEXT PAYDAY  (the whole opportunity)")
    if not dataset.failure_events:
        print("no failures in this run")
        return

    gaps = [
        days_to_next_salary(dataset.customer(e.customer_id), e.original_debit_date)
        for e in dataset.failure_events
    ]
    gaps_arr = np.array(gaps)
    counts = Counter(gaps)
    peak = max(counts.values())
    for gap in sorted(counts):
        print(f"  +{gap:2d}d  {counts[gap]:3d} {bar(counts[gap], peak, width=40)}")
    print(f"\n  median wait {np.median(gaps_arr):.0f} days   "
          f"mean {gaps_arr.mean():.1f}   "
          f"within a week {pct(float(np.mean(gaps_arr <= 7)))}")
    print("  A fixed day-2/4/7 schedule spends all four attempts inside the first")
    print("  week, so every failure to the right of +7 is money left on the table.")


def oracle_headroom(dataset) -> None:
    header("ORACLE HEADROOM  (only computable because the data is synthetic)")
    if not dataset.failure_events:
        print("no failures in this run")
        return

    horizon = 30
    recoverable, best_offsets, legacy_wins = 0, [], 0
    legacy_offsets = dataset.meta["legacy_retry_offsets"]
    for event in dataset.failure_events:
        customer = dataset.customer(event.customer_id)
        series = dataset.balance_of(event.customer_id)
        best = first_success_date(
            customer,
            series,
            customer.subscription_amount,
            after=event.original_debit_date,
            horizon_days=horizon,
        )
        if best is not None:
            recoverable += 1
            best_offsets.append((best - event.original_debit_date).days)
        for offset in legacy_offsets:
            day = event.original_debit_date + timedelta(days=offset)
            if series.loc[str(day)] >= customer.subscription_amount:
                legacy_wins += 1
                break

    total = len(dataset.failure_events)
    unrecoverable = total - recoverable
    print(f"failures recoverable within {horizon} days   {pct(recoverable / total)}"
          f"   ({recoverable} of {total})")
    print(f"never recoverable (income gone)      {pct(unrecoverable / total)}"
          f"   ({unrecoverable} of {total})")
    print(f"recovered by the legacy day-{'/'.join(map(str, legacy_offsets))} schedule "
          f"  {pct(legacy_wins / total)}   ({legacy_wins} of {total})")
    print(f"\nheadroom a perfect sequencer could capture: "
          f"{pct((recoverable - legacy_wins) / total)} of failures")
    if best_offsets:
        offsets = np.array(best_offsets)
        print(f"earliest winning retry offset: median +{np.median(offsets):.0f}d   "
              f"mean +{offsets.mean():.1f}d   p90 +{np.percentile(offsets, 90):.0f}d")
    print("\nThe two numbers that decide whether this dataset is worth evaluating on:")
    print(f"  legacy recovery {pct(legacy_wins / total)} - published fixed-retry recovery")
    print("    is 35-45%. Far above that means the world is too easy.")
    print(f"  unrecoverable   {pct(unrecoverable / total)} - if this were zero, CANCEL")
    print("    would never be correct and no stopping rule could ever be wrong.")


def balance_plot(dataset, window: int = 60) -> None:
    header(f"ONE CUSTOMER, {window} DAYS OF HIDDEN BALANCE")
    if dataset.failure_events:
        event = dataset.failure_events[0]
        customer = dataset.customer(event.customer_id)
        anchor = event.original_debit_date
    else:
        customer = dataset.customers[0]
        anchor = dataset.as_of - timedelta(days=30)

    start = anchor - timedelta(days=window // 3)
    series = daily_balance(customer, start, window)
    amount = customer.subscription_amount

    print(f"{customer.customer_id}   pays Rs {amount:,.0f} on day "
          f"{customer.debit_day_of_month} via {customer.rail}")
    print(f"HIDDEN: salary Rs {customer.salary_amount:,.0f} on day {customer.salary_day}, "
          f"burn {customer.burn_rate:.3f}/day, buffer {customer.buffer_ratio:.3f}, "
          f"late-pay chance {customer.salary_delay_prob:.2f}")
    print(f"\n  The Rs {amount:,.0f} threshold sits at column '|'; a bar reaching")
    print("  it ('+') means a debit that day would clear.\n")

    peak = float(series.max())
    width = 52
    threshold_col = int(round(width * amount / peak)) if peak > 0 else 0
    values = series.to_numpy()
    for i, (stamp, value) in enumerate(series.items()):
        day = stamp.date()
        filled = int(round(width * value / peak))
        row = list(" " * width)
        for col in range(min(filled, width)):
            row[col] = "#"
        if 0 <= threshold_col < width:
            row[threshold_col] = "|" if row[threshold_col] == " " else "+"

        marks = []
        # A jump of more than half a salary in one day is the credit landing.
        if i > 0 and value - values[i - 1] > 0.5 * customer.salary_amount:
            marks.append("SALARY IN")
        if day.day == customer.salary_day:
            marks.append("(scheduled payday)")
        if day == anchor:
            marks.append("<<< DEBIT FAILED" if value < amount else "<<< DEBIT OK")
        elif day.day == customer.debit_day_of_month:
            marks.append("debit")
        print(f"  {day}  {''.join(row)} {value:10,.0f}  {' '.join(marks)}")

    best = first_success_date(customer, series, amount, after=anchor, horizon_days=30)
    print(f"\n  oracle: earliest date after {anchor} that would have cleared -> {best}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="number of customers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None, help="directory to write to")
    parser.add_argument(
        "--fmt", choices=("parquet", "csv"), default="parquet", help="save format"
    )
    args = parser.parse_args()

    dataset = generate_dataset(n_customers=args.n, seed=args.seed)

    population_summary(dataset)
    salary_day_histogram(dataset)
    failure_summary(dataset)
    days_to_salary_distribution(dataset)
    oracle_headroom(dataset)
    balance_plot(dataset)

    if args.save:
        path = save_dataset(dataset, args.save, fmt=args.fmt)
        header("SAVED")
        for file in sorted(path.iterdir()):
            print(f"  {file}  ({file.stat().st_size / 1024:,.0f} KB)")


if __name__ == "__main__":
    main()
