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
    with_nudge       NUDGE on for INSUFFICIENT FUNDS (off by default - see below)
    no_afa_nudge     NUDGE off for AFA (on by default - the opposite switch)
    tier1_payday     tier 1 (day-of-month frequency) instead of tier 2
    no_bank_health   no cross-customer pooling of bank outage evidence
    no_masking_posterior  class 6 priced as if every 05 were a balance failure
    irritation_build the opposite irritation direction

``with_nudge`` and ``no_afa_nudge`` are the same action measured on two
different classes, and they point opposite ways. Nudging an insufficient-funds
customer asks somebody with no money to find some: it burns a day and loses.
Nudging an AFA customer asks them to press a button, and presenting before they
do cannot clear whatever the balance is. Both defaults are set by their own
arithmetic, which is why they are two switches rather than one.

Note on scale: ``no_afa_nudge`` moves only the ~5% of failures coded 1A, so its
effect on a book-wide average is diluted by a factor of twenty before it is
read. Judge it in scripts/run_afa.py, where the class is isolated; it is listed
here so the ablation table has no missing row.

Legacy and the oracle ceiling are shown alongside as the floor and the roof.

Counting seeds: WORSE / BETTER / TIED, and the denominator matters
-----------------------------------------------------------------
Consistency is measured over the seeds where the variant ACTUALLY BIT - the
ones where it changed the outcome at all - and not over all twenty.

The distinction is invisible on a dense class and decisive on a thin one.
Insufficient funds carries ~70 episodes per seed, so a variant that touches it
at all moves the seed total and no seed is ever exactly tied. AFA carries ~5
episodes per seed, so on most seeds a variant that only acts on that class
changes nothing whatsoever. An earlier version of this script counted every
delta that was not negative as evidence AGAINST the default configuration,
which silently folded those do-nothing seeds into the case against - and the
thinner the class, the more of them there were.

A tie is not a disagreement. It is an absence of evidence, and it has to be
reported as one.

**A low bite count is itself a finding**, not a caveat to be worked around. If a
variant bites on 8 of 20 seeds, that says the component it removes only matters
on a minority of books - which for a class-specific switch is a statement about
how thin the class is. The table prints the bite count next to every verdict so
the denominator is never hidden, and a verdict computed over very few bitten
seeds is labelled as resting on thin evidence rather than being promoted to a
clean result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _emit import add_json_argument, emit  # noqa: E402
from agent.sequencer import SequencerConfig, SequencerPolicy  # noqa: E402
from evaluation.harness import compare  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from policy.oracle_policy import OraclePolicy  # noqa: E402

RULE = "=" * 100

# A seed counts as TIED when the variant moved net value by less than a rupee
# AND did not move recovery at all. Both thresholds are the ones this script
# already used for its "no effect at all" band, reused deliberately rather than
# introduced as a new tunable: a variant that changed the book by under one
# rupee and recovered exactly the same episodes did not bite on that seed.
TIE_NET_EPSILON: float = 1.0
TIE_RATE_EPSILON: float = 1e-9

# Below this many bitten seeds, a consistency ratio is arithmetic rather than
# evidence - 3 of 4 and 15 of 20 are not the same claim. Verdicts computed over
# fewer than this are labelled as thin rather than promoted.
MIN_BITTEN_FOR_A_VERDICT: int = 8

VARIANTS: dict[str, dict] = {
    "full": {},
    "no_payday": {"flat_liquidity": True},
    "no_optout_cost": {"include_optout_cost": False},
    "no_hold": {"use_hold": False},
    "with_nudge": {"use_nudge": True},
    "no_afa_nudge": {"use_afa_nudge": False},
    "tier1_payday": {"payday_tier": 1},
    "no_bank_health": {"use_bank_health": False},
    "no_masking_posterior": {"use_masking_posterior": False},
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


def _variant_factory(label: str, overrides: dict):
    """A per-seed constructor for one ablation variant.

    A named function rather than a lambda in the loop, so each variant closes
    over its OWN label and overrides. The late-binding bug that an inline
    lambda would introduce is the same shape as the one this fixes: state from
    one iteration silently reused in the next.
    """
    config = SequencerConfig(**overrides)

    def build(dataset):
        return SequencerPolicy.for_dataset(dataset, config, name=label)

    return build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--no-oracle-metrics", action="store_true")
    parser.add_argument(
        "--ambiguity-share", type=float, default=0.0,
        help=(
            "class-6 masking share. 0.0 (the default) is the book every "
            "documented ablation number was measured on; no_masking_posterior "
            "necessarily ties on every seed there, because there are no 05s "
            "for it to act on. Raise it to make that row mean something."
        ),
    )
    add_json_argument(parser)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    started = time.time()

    print(f"{RULE}\nABLATIONS\n{RULE}")
    print(f"{args.seeds} seeds x {args.n} customers\n")
    datasets = {
        s: generate_dataset(
            n_customers=args.n, seed=s, ambiguity_share=args.ambiguity_share
        )
        for s in seeds
    }

    # One policy PER SEED, not one per variant. SequencerPolicy carries a
    # bank-health model keyed on customer id, and customer ids denote a
    # different person on a different issuer in every dataset - so a single
    # instance run across twenty seeds skips most of the evidence and files the
    # rest against the wrong bank. compare() now refuses the shared shape, and
    # the policy itself raises; this passes the per-seed constructor, exactly as
    # the oracle ceiling beside it has always been passed.
    policies = [LegacyFixedSchedule(), OraclePolicy.for_dataset]
    for label, overrides in VARIANTS.items():
        policies.append(_variant_factory(label, overrides))

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
    print("  customers, the same draws - so seed noise cancels.\n")
    print("  W / B / T counts the seeds where the variant came out WORSE than")
    print("  `full`, BETTER than it, or TIED with it. Consistency is judged over")
    print("  W + B only - the seeds where the variant actually BIT - because a")
    print("  tie is an absence of evidence, not a disagreement. The bite count")
    print("  is printed with every verdict so the denominator is never hidden.\n")
    full_rows = report.per_seed["full"]
    # Kept so --json can report exactly the rows that were printed rather than
    # recomputing them; a verifier that checks a second computation checks
    # nothing. See scripts/_emit.py.
    verdicts: dict[str, dict] = {}
    print(
        f"  {'variant':<18} {'d recovery':>12} {'d net value':>14} "
        f"{'W/B/T':>12} {'bit':>5}   verdict"
    )
    for name in VARIANTS:
        if name == "full":
            continue
        rows = report.per_seed[name]
        rec_deltas = [v.recovery_rate - f.recovery_rate for v, f in zip(rows, full_rows)]
        net_deltas = [v.net_value_total - f.net_value_total for v, f in zip(rows, full_rows)]
        delta_rec = sum(rec_deltas) / len(rec_deltas)
        delta_net = sum(net_deltas) / len(net_deltas)

        # Three buckets, not two. A seed on which the variant changed nothing is
        # NOT a seed on which the variant did better, and counting it as one is
        # how a do-nothing outcome on a thin class turns into a point against
        # the default configuration.
        worse = better = tied = 0
        for net_delta, rec_delta in zip(net_deltas, rec_deltas):
            if (
                abs(net_delta) <= TIE_NET_EPSILON
                and abs(rec_delta) <= TIE_RATE_EPSILON
            ):
                tied += 1
            elif net_delta < 0:
                worse += 1
            else:
                better += 1

        bitten = worse + better
        agreeing = max(worse, better) / bitten if bitten else 0.0

        # Three bands, not two. An effect that shows up on 15 of 20 seeds is
        # not the same claim as one that shows up on all 20, and collapsing
        # them into a single "earns its place" overstates the weaker one.
        # The verdict is phrased about THE VARIANT, not about "removing" a
        # component, because not every variant removes one. `with_nudge` ADDS
        # nudging to class 1 and `irritation_build` swaps an assumption for its
        # mirror; reading their rows as "removing it hurts" inverts what they
        # actually say. The delta is always `variant - full`, so a negative
        # number always means the variant is the worse configuration - and that
        # sentence is true whichever direction the variant moves in.
        if bitten == 0:
            # Not "a liability" and not "borderline" - literally no effect.
            # A component that changes nothing to the rupee on any seed is not
            # doing anything, and saying so plainly is the useful report.
            verdict = "NO EFFECT AT ALL - identical on every seed"
        elif agreeing < 0.75:
            verdict = f"NOT CONSISTENT on the {bitten} seeds it bit - do not claim it"
        else:
            direction = "WORSE" if delta_net < 0 else "BETTER"
            tail = (
                "the default configuration earns its place"
                if delta_net < 0
                else "the default is a liability"
            )
            if bitten < MIN_BITTEN_FOR_A_VERDICT:
                # Consistent, but over so few seeds that the ratio is arithmetic
                # rather than evidence. Say which it is instead of promoting it.
                verdict = (
                    f"consistently {direction} but only on {bitten}/"
                    f"{len(net_deltas)} seeds - THIN EVIDENCE"
                )
            elif agreeing < 0.85:
                verdict = (
                    f"BORDERLINE ({max(worse, better)}/{bitten} bitten) - "
                    f"variant is {direction.lower()}, weakly"
                )
            else:
                verdict = f"variant is {direction}: {tail}"
        verdicts[name] = {
            "delta_recovery": delta_rec,
            "delta_recovery_points": delta_rec * 100,
            "delta_net_value": delta_net,
            "worse": worse,
            "better": better,
            "tied": tied,
            "bit": bitten,
            "verdict": verdict,
        }
        print(
            f"  {name:<18} {delta_rec:>+11.1%} {delta_net:>+14,.0f} "
            f"{worse:>3}W/{better:<2}B/{tied:<2}T {bitten:>5}   {verdict}"
        )

    floor = report.summary["legacy_fixed"]["recovery_rate"].mean
    ceiling = report.summary[OraclePolicy.name]["recovery_rate"].mean
    agent = report.summary["full"]["recovery_rate"].mean
    print(f"\n{RULE}\nHEADLINE\n{RULE}")
    print(f"  legacy floor      {floor:>7.1%}")
    print(f"  sequencer         {agent:>7.1%}")
    print(f"  oracle ceiling    {ceiling:>7.1%}")
    captured = (agent - floor) / (ceiling - floor) if ceiling > floor else None
    if captured is not None:
        print(f"\n  headroom captured {captured:>7.1%}")

    # Written from the very rows the tables above were rendered from, so the
    # file cannot disagree with the report. See scripts/_emit.py.
    emit(
        args.json_path,
        script="run_ablations.py",
        seeds=args.seeds,
        n=args.n,
        payload={
            "table": {
                name: {
                    metric: (
                        report.summary[name][metric].mean
                        if report.summary.get(name, {}).get(metric)
                        and report.summary[name][metric].values
                        else None
                    )
                    for metric, _, _ in HEADLINE
                }
                for name in order
                if name in report.summary
            },
            "recovery_range": {
                name: {
                    "mean": report.summary[name]["recovery_rate"].mean,
                    "low": report.summary[name]["recovery_rate"].low,
                    "high": report.summary[name]["recovery_rate"].high,
                }
                for name in order
                if name in report.summary
            },
            "verdicts": verdicts,
            "ambiguity_share": args.ambiguity_share,
            "headline": {
                "legacy_floor": floor,
                "sequencer": agent,
                "oracle_ceiling": ceiling,
                "headroom_captured": captured,
            },
        },
    )

    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
