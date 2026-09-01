"""How much of the win comes from the prior being right?

    python scripts/prior_robustness.py
    python scripts/prior_robustness.py --seeds 20 --n 500

The population is drawn from the same Indian payroll distribution the tier-2
model uses as its prior. That is a real advantage the agent would not have on
live data, and it makes the calibration numbers optimistic. This script
measures how much of the advantage survives when the prior is wrong.

    correct   the true Indian distribution - the optimistic case
    flat      uniform over days 1-28: no payroll knowledge at all
    shifted   the Indian shape moved five days later: right structure,
              wrong location, which is the realistic kind of wrong
    us_style  bimodal on the 1st and 15th: a semi-monthly payroll calendar
              imported wholesale from another market

If performance collapses under `flat`, the result is a statement about the
prior. If it degrades gracefully, the method is doing the work and the prior is
a head start.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.payday import (  # noqa: E402
    INDIAN_PAYDAY_PRIOR,
    PRIOR_FLOOR_MASS,
    PaydayModel,
)
from _emit import add_json_argument, emit  # noqa: E402
from agent.sequencer import SequencerConfig, SequencerPolicy  # noqa: E402
from evaluation.harness import compare  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from policy.oracle_policy import OraclePolicy  # noqa: E402

RULE = "=" * 84


def normalise(weights: dict[int, float]) -> dict[int, float]:
    """Add the same all-days floor the real prior uses, then normalise.

    Every prior keeps a floor so no variant is structurally unable to find a
    payday - otherwise `us_style` would be handicapped by more than its shape.
    """
    floor = PRIOR_FLOOR_MASS / 31.0
    prior = {day: floor for day in range(1, 32)}
    total_weight = sum(weights.values())
    for day, mass in weights.items():
        prior[day] += (mass / total_weight) * (1.0 - PRIOR_FLOOR_MASS)
    total = sum(prior.values())
    return {day: mass / total for day, mass in prior.items()}


def shifted_prior(days: int = 5) -> dict[int, float]:
    """The Indian shape, moved later in the month."""
    return normalise({((day - 1 + days) % 31) + 1: mass for day, mass in INDIAN_PAYDAY_PRIOR.items()})


PRIORS: dict[str, dict[int, float] | None] = {
    "correct": None,  # the module default
    "flat": normalise({day: 1.0 for day in range(1, 29)}),
    "shifted": shifted_prior(5),
    "us_style": normalise({1: 0.45, 2: 0.05, 15: 0.40, 16: 0.05, 30: 0.05}),
}


def _prior_factory(label: str, prior):
    """A per-seed constructor for one prior. Named, so it closes over its own."""
    config = SequencerConfig(payday_prior=prior)

    def build(dataset):
        return SequencerPolicy.for_dataset(dataset, config, name=f"prior_{label}")

    return build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    add_json_argument(parser)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    started = time.time()
    datasets = {s: generate_dataset(n_customers=args.n, seed=s) for s in seeds}

    print(f"{RULE}\nPRIOR ROBUSTNESS\n{RULE}")
    print(f"{args.seeds} seeds x {args.n} customers")
    print("the population is drawn from the `correct` prior, so it is the")
    print("optimistic case by construction\n")

    # Per seed, for the reason in run_ablations.py: one SequencerPolicy across
    # twenty seeds carries bank evidence between books whose customer ids mean
    # different people. compare() rejects the shared shape outright.
    policies = [LegacyFixedSchedule(), OraclePolicy.for_dataset]
    for label, prior in PRIORS.items():
        policies.append(_prior_factory(label, prior))

    report = compare(
        policies,
        seeds=seeds,
        n_customers=args.n,
        baseline_name="legacy_fixed",
        ceiling_name=OraclePolicy.name,
        oracle_metrics=False,
        strict_compliance=False,
        datasets=datasets,
    )

    floor = report.summary["legacy_fixed"]["recovery_rate"].mean
    ceiling = report.summary[OraclePolicy.name]["recovery_rate"].mean
    correct = report.summary["prior_correct"]["recovery_rate"].mean

    print(f"  {'prior':<12} {'recovery':>10} {'range':>18} {'money':>12} "
          f"{'headroom':>10} {'vs correct':>12}")
    print("  " + "-" * 78)
    print(f"  {'legacy floor':<12} {floor:>10.1%} "
          f"{'':>18} {report.summary['legacy_fixed']['money_recovered'].mean:>12,.0f}"
          f" {0.0:>10.1%} {'':>12}")
    for label in PRIORS:
        name = f"prior_{label}"
        s = report.summary[name]["recovery_rate"]
        money = report.summary[name]["money_recovered"].mean
        headroom = (s.mean - floor) / (ceiling - floor) if ceiling > floor else 0.0
        delta = s.mean - correct
        marker = "" if label == "correct" else f"{delta:>+11.1%}"
        print(f"  {label:<12} {s.mean:>10.1%} [{s.low:>6.1%} .. {s.high:>6.1%}] "
              f"{money:>12,.0f} {headroom:>10.1%} {marker:>12}")
    print(f"  {'oracle roof':<12} {ceiling:>10.1%} {'':>18} "
          f"{report.summary[OraclePolicy.name]['money_recovered'].mean:>12,.0f}"
          f" {1.0:>10.1%}")

    # --- why the recovery numbers barely move -------------------------------
    # Scripts may read the truth; the agent may not. Comparing the inferred
    # payday against the real salary_day explains a flat recovery table: if the
    # priors disagree about the exact day but all land within a day or two, the
    # money will not notice, because the liquidity window is a fortnight wide.
    print(f"\n{RULE}\nHOW WRONG IS THE INFERRED PAYDAY?\n{RULE}")
    print("  Truth-vs-inference, which only the evaluator can see.\n")
    accuracy: dict[str, dict[str, float]] = {}
    print(f"  {'prior':<12} {'mean |error|':>13} {'within 2 days':>15} {'exact':>8}")
    for label, prior in PRIORS.items():
        errors: list[int] = []
        for seed in seeds[: min(4, len(seeds))]:
            dataset = datasets[seed]
            by_customer: dict[str, list] = {}
            for entry in dataset.payment_history:
                by_customer.setdefault(entry.customer_id, []).append(entry)
            for event in dataset.failure_events:
                customer = dataset.customer(event.customer_id)
                history = [
                    r
                    for r in by_customer[event.customer_id]
                    if r.attempt_date <= event.original_debit_date
                ]
                if not history:
                    continue
                model = PaydayModel(tier=2, prior=prior)
                model.fit(history)
                posterior = model.payday_posterior()
                inferred = max(posterior, key=posterior.get)
                # Days 1, 30 and 31 are all "month end" - a 31 vs 1 answer is
                # a one-day error, not a thirty-day one.
                gap = abs(inferred - customer.salary_day)
                errors.append(min(gap, 31 - gap))
        if errors:
            mean_error = sum(errors) / len(errors)
            close = sum(1 for e in errors if e <= 2) / len(errors)
            exact = sum(1 for e in errors if e == 0) / len(errors)
            accuracy[label] = {
                "mean_abs_error_days": mean_error,
                "within_2_days": close,
                "exact": exact,
            }
            print(f"  {label:<12} {mean_error:>13.2f} {close:>15.1%} {exact:>8.1%}")

    worst_label = min(
        (label for label in PRIORS if label != "correct"),
        key=lambda label: report.summary[f"prior_{label}"]["recovery_rate"].mean,
    )
    worst = report.summary[f"prior_{worst_label}"]["recovery_rate"].mean
    retained = (worst - floor) / (correct - floor) if correct > floor else 0.0

    print(f"\n{RULE}\nVERDICT\n{RULE}")
    print(f"  Worst wrong prior: `{worst_label}` at {worst:.1%}, against {correct:.1%}")
    print(f"  with the correct one and a {floor:.1%} fixed-schedule floor.")
    print(f"\n  It keeps {retained:.0%} of the advantage the correct prior buys.")
    if retained > 0.7:
        print("\n  The METHOD is doing the work. Inferring a payday from attempt")
        print("  outcomes and pricing dates against it survives a badly wrong")
        print("  prior; the payroll distribution is a head start, not the result.")
        print("\n  The table above shows WHY, and it is not that the prior is")
        print("  ignored - it clearly is not. A wrong prior roughly halves how")
        print("  often the exact payday is identified. But the liquidity window")
        print("  is about a fortnight wide, so being two or three days out still")
        print("  puts the presentment on a day with money in the account. The")
        print("  decision is coarser than the inference, and that is what makes")
        print("  it robust.")
    elif retained > 0.4:
        print("\n  Mixed. A wrong prior costs real money but the method still")
        print("  clears the fixed-schedule floor comfortably. Quote the degraded")
        print("  number, not the optimistic one, when the payroll calendar is")
        print("  unknown.")
    else:
        print("\n  MOST OF THE WIN IS THE PRIOR. On a market whose payroll calendar")
        print("  is not known in advance this agent would lose most of its edge.")
        print("  Report the flat-prior number as the headline; the correct-prior")
        print("  number is an upper bound that assumes knowledge you may not have.")
    # Written from the same summaries the tables above were rendered from, so
    # the file cannot disagree with the report. See scripts/_emit.py.
    emit(
        args.json_path,
        script="prior_robustness.py",
        seeds=args.seeds,
        n=args.n,
        payload={
            "legacy_floor": floor,
            "oracle_ceiling": report.summary[OraclePolicy.name]["recovery_rate"].mean,
            "priors": {
                label: {
                    "recovery_rate": report.summary[f"prior_{label}"][
                        "recovery_rate"
                    ].mean,
                    "money": report.summary[f"prior_{label}"]["money_recovered"].mean,
                    "vs_correct_points": (
                        report.summary[f"prior_{label}"]["recovery_rate"].mean - correct
                    )
                    * 100,
                    **accuracy.get(label, {}),
                }
                for label in PRIORS
            },
            "worst_prior": worst_label,
            "worst_prior_recovery": worst,
            "advantage_retained": retained,
        },
    )

    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
