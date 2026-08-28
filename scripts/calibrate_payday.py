"""Is the payday model's probability actually a probability?

    python scripts/calibrate_payday.py
    python scripts/calibrate_payday.py --seeds 5 --n 400 --true-prob

The decision engine multiplies this number by real rupees. A model that says
70% must be right about 70% of the time, or the money equation is being fed
something that merely looks like a probability. Discrimination is not enough:
a model can rank dates perfectly and still be badly scaled, and a badly scaled
probability produces confidently wrong expected values.

Reported per tier:

``Brier``  mean squared error against the realised outcome. Lower is better.
           Decomposes into calibration (is 70% right 70% of the time) and
           resolution (does it separate good dates from bad).
``ECE``    expected calibration error - the average gap between predicted and
           observed frequency, weighted by how many predictions land in each
           bin. This is the number that matters for the money equation.
``MSE_p``  optional: mean squared error against the ORACLE's true probability
           rather than the realised coin flip. Measures how close the model is
           to the best any forecaster could do.

The model is fitted only on history up to the failure date, so nothing from
the scored window leaks into the fit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.payday import PaydayModel  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from generator.oracle import success_probability_true, would_succeed  # noqa: E402

RULE = "=" * 78
TIERS = (1, 2, 3)
HORIZON_DAYS = 30
N_BINS = 10


@dataclass
class Predictions:
    predicted: list[float] = field(default_factory=list)
    actual: list[int] = field(default_factory=list)
    true_prob: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.predicted)

    @property
    def brier(self) -> float:
        return sum((p - a) ** 2 for p, a in zip(self.predicted, self.actual)) / len(self)

    @property
    def base_rate(self) -> float:
        return sum(self.actual) / len(self)

    @property
    def brier_of_always_base_rate(self) -> float:
        """What you would score by ignoring the date entirely.

        The reference point that matters: a model must beat "predict the base
        rate every time" or it has learned nothing about timing.
        """
        rate = self.base_rate
        return sum((rate - a) ** 2 for a in self.actual) / len(self)

    @property
    def mse_against_true_probability(self) -> float | None:
        if not self.true_prob:
            return None
        pairs = list(zip(self.predicted, self.true_prob))
        return sum((p - t) ** 2 for p, t in pairs) / len(pairs)

    def bins(self) -> list[tuple[float, float, float, int]]:
        """(bin_low, mean_predicted, observed_frequency, count) per decile."""
        buckets: list[list[tuple[float, int]]] = [[] for _ in range(N_BINS)]
        for p, a in zip(self.predicted, self.actual):
            index = min(N_BINS - 1, int(p * N_BINS))
            buckets[index].append((p, a))
        out = []
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            mean_p = sum(p for p, _ in bucket) / len(bucket)
            observed = sum(a for _, a in bucket) / len(bucket)
            out.append((index / N_BINS, mean_p, observed, len(bucket)))
        return out

    @property
    def ece(self) -> float:
        total = len(self)
        return sum(
            count / total * abs(mean_p - observed)
            for _, mean_p, observed, count in self.bins()
        )


def collect(seeds, n_customers, with_true_prob: bool, n_sims: int):
    results = {tier: Predictions() for tier in TIERS}
    for seed in seeds:
        dataset = generate_dataset(n_customers=n_customers, seed=seed)
        by_customer: dict[str, list] = {}
        for entry in dataset.payment_history:
            by_customer.setdefault(entry.customer_id, []).append(entry)

        for event in dataset.failure_events:
            customer = dataset.customer(event.customer_id)
            series = dataset.balance_of(event.customer_id)
            amount = customer.subscription_amount
            failed_on = event.original_debit_date

            # Fit on history up to and including the failure. Nothing later.
            history = [
                r
                for r in by_customer.get(event.customer_id, [])
                if r.attempt_date <= failed_on
            ]
            if not history:
                continue

            start = failed_on + timedelta(days=1)
            end = failed_on + timedelta(days=HORIZON_DAYS)
            if end > series.index[-1].date():
                continue

            for tier in TIERS:
                model = PaydayModel(tier=tier)
                model.fit(history)
                curve = model.liquidity_curve(start, end, amount)
                for day, predicted in curve.items():
                    results[tier].predicted.append(predicted)
                    results[tier].actual.append(
                        int(would_succeed(customer, day, amount, series))
                    )
                    if with_true_prob and (day - start).days % 5 == 0:
                        results[tier].true_prob.append(
                            success_probability_true(customer, day, amount, n_sims=n_sims)
                        )
    return results


def render(tier: int, predictions: Predictions) -> None:
    print(f"\n{RULE}\nTIER {tier}   ({len(predictions):,} predictions)\n{RULE}")
    print(f"  {'bin':>10} {'predicted':>11} {'observed':>10} {'n':>7}   reliability")
    for low, mean_p, observed, count in predictions.bins():
        gap = observed - mean_p
        bar_len = int(abs(gap) * 40)
        direction = "under" if gap > 0 else "over "
        bar = ("+" if gap > 0 else "-") * bar_len
        print(f"  {low:>5.1f}-{low + 0.1:<4.1f} {mean_p:>11.3f} {observed:>10.3f} "
              f"{count:>7,}   {direction} {bar}")

    skill = 1.0 - predictions.brier / predictions.brier_of_always_base_rate
    print(f"\n  Brier              {predictions.brier:.4f}   "
          f"(base-rate-only: {predictions.brier_of_always_base_rate:.4f})")
    print(f"  skill vs base rate {skill:>+.1%}   "
          f"{'learned something about timing' if skill > 0 else 'NO BETTER THAN GUESSING'}")
    print(f"  ECE                {predictions.ece:.4f}")
    mse = predictions.mse_against_true_probability
    if mse is not None:
        print(f"  MSE vs true P      {mse:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--true-prob", action="store_true",
                        help="also score against the oracle's true probability (slow)")
    parser.add_argument("--sims", type=int, default=200)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    print(f"{RULE}\nPAYDAY MODEL CALIBRATION\n{RULE}")
    print(f"{args.seeds} seeds x {args.n} customers, {HORIZON_DAYS}-day horizon per failure")
    print("fitted on history up to the failure date only\n")

    results = collect(seeds, args.n, args.true_prob, args.sims)
    for tier in TIERS:
        render(tier, results[tier])

    print(f"\n{RULE}\nVERDICT\n{RULE}")
    print(f"  {'tier':>6} {'Brier':>9} {'ECE':>9} {'skill':>9}")
    for tier in TIERS:
        p = results[tier]
        skill = 1.0 - p.brier / p.brier_of_always_base_rate
        print(f"  {tier:>6} {p.brier:>9.4f} {p.ece:>9.4f} {skill:>+9.1%}")

    one, two = results[1], results[2]
    print()
    if two.brier < one.brier and two.ece < one.ece:
        print(f"  Tier 2 beats tier 1 on BOTH Brier ({one.brier:.4f} -> {two.brier:.4f})")
        print(f"  and ECE ({one.ece:.4f} -> {two.ece:.4f}).")
        print("  Inferring the payday and deriving liquidity from it is worth more")
        print("  than scoring days of the month independently. That is the thesis.")
    elif two.brier < one.brier:
        print(f"  Tier 2 wins on Brier ({one.brier:.4f} -> {two.brier:.4f}) but not on")
        print(f"  ECE ({one.ece:.4f} -> {two.ece:.4f}): better at ranking dates, not")
        print("  better scaled. Check the liquidity shape before trusting the")
        print("  probabilities in the money equation.")
    else:
        print(f"  Tier 2 does NOT beat tier 1 (Brier {one.brier:.4f} -> {two.brier:.4f}).")
        print("  Something is wrong with the prior or the likelihood. Do not paper")
        print("  over this: the payday posterior is the whole novelty claim.")


if __name__ == "__main__":
    main()
