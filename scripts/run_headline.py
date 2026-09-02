"""The submission headline: money first, rate second.

    python scripts/run_headline.py
    python scripts/run_headline.py --seeds 20 --n 500

Razorpay's Track 3 bar reads: *"Don't just identify the problem. Show measured
money recovered across a batch, with compliant escalation, stopping rules, and
an audit trail."* Four nouns, and the first one is MONEY. Everything else in
this repository leads with a recovery rate, which answers a different question -
what share of failed mandates came back - and on a book where the failures range
from Rs 199 to Rs 19,000 those two answers can point in opposite directions.

So this script exists to put the rupees on top. It computes nothing new: every
figure below already existed in ``PolicyReport`` and was being dropped at the
presentation layer. Rates are unchanged by construction, and that is checked
rather than asserted - ``scripts/verify_docs.py`` pins them.

WHAT IS WORTH READING, in order:

1. **Net value, not gross recovered.** Gross is what came back; net is what it
   was worth after the processing fees, the bounce charges, the carrying cost of
   every held cycle, the goodwill spent on nudges and the annual LTV of every
   mandate that died on the way. A policy can raise gross and lower net, and the
   fixed schedule does exactly that.

2. **Both headroom numbers.** The share of the legacy-to-ceiling gap captured,
   computed once on the rate and once on the money. Where they diverge, the gap
   says which mandates the agent is winning - see the note the table prints
   under itself.

3. **Per 100 failed mandates.** Absolute rupees on a synthetic 500-customer book
   are not a claim about anything. The per-100 form is stable across book sizes
   and shares its denominator with the Sec 6(c) opt-out cost reported by
   ``scripts/lead_time_ablation.py``, so the two can be read together.

COSTS ARE ESTIMATES. Processing fees, the carrying-cost fraction and the
goodwill charge all come from ``generator/rules.py``, where they are marked as
estimates. They are not published Razorpay figures. The regulatory constants in
that file are citable; the money constants are not, and every net-value figure
here inherits that caveat.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _emit import add_json_argument, emit  # noqa: E402
from agent.sequencer import SequencerPolicy  # noqa: E402
from evaluation.economics import per_hundred_failures  # noqa: E402
from evaluation.harness import compare  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from policy.oracle_policy import OraclePolicy  # noqa: E402

RULE = "=" * 88

LEGACY = "legacy_fixed"
AGENT = "sequencer"
CEILING = OraclePolicy.name

# Printed in the order a reader should take them: money, then what the money
# cost, then the rate, then the normalisation.
ROWS: tuple[tuple[str, str, str], ...] = (
    ("money_recovered", "money recovered", "Rs {:,.0f}"),
    ("net_value_total", "net value", "Rs {:,.0f}"),
    ("recovery_rate", "recovery rate", "{:.1%}"),
    ("attempts_used_total", "presentments", "{:,.0f}"),
    ("optouts_induced", "opt-outs induced", "{:.1f}"),
    ("customers_retained", "mandates retained", "{:.1f}"),
    ("compliance_breaches", "compliance breaches", "{:.1f}"),
)


def _mean(report, policy: str, metric: str) -> float | None:
    summary = report.summary.get(policy, {}).get(metric)
    return summary.mean if summary and summary.values else None


def _fmt(value: float | None, template: str) -> str:
    return "n/a" if value is None else template.format(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--share", type=float, default=0.0,
        help=(
            "class-6 masking share. 0.0 is the book every documented number "
            "was measured on; raise it to price a world where issuers hide "
            "their decline reasons."
        ),
    )
    add_json_argument(parser)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    started = time.time()

    datasets = {
        s: generate_dataset(
            n_customers=args.n, seed=s, ambiguity_share=args.share
        )
        for s in seeds
    }
    report = compare(
        [LegacyFixedSchedule(), SequencerPolicy.for_dataset, OraclePolicy.for_dataset],
        seeds=seeds,
        n_customers=args.n,
        baseline_name=LEGACY,
        ceiling_name=CEILING,
        oracle_metrics=False,
        strict_compliance=False,
        datasets=datasets,
    )

    # The per-100 denominator: failure EPISODES summed across every seed, which
    # is what lead_time_ablation.py divides by. Identical for all three policies
    # - they score the same book - so it is computed once, off the baseline.
    total_failures = sum(r.n_failures for r in report.per_seed[LEGACY])

    print(f"{RULE}\nMEASURED MONEY RECOVERED\n{RULE}")
    print(f"{args.seeds} seeds x {args.n} customers   "
          f"{total_failures:,} failed mandates in total"
          + (f"   class-6 masking share {args.share}" if args.share else ""))
    print("mean across seeds, [min .. max] in brackets\n")

    header = f"  {'':<22}" + "".join(f"{p:>26}" for p in (LEGACY, AGENT, CEILING))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric, label, template in ROWS:
        line = f"  {label:<22}"
        for policy in (LEGACY, AGENT, CEILING):
            summary = report.summary.get(policy, {}).get(metric)
            if summary is None or not summary.values:
                line += f"{'n/a':>26}"
            else:
                line += f"{_fmt(summary.mean, template):>26}"
        print(line)

    # --- per 100 failed mandates --------------------------------------------
    print(f"\n  {'per 100 failed mandates':<22}")
    per_hundred: dict[str, dict[str, float | None]] = {}
    for metric, label in (("money_recovered", "money recovered"),
                          ("net_value_total", "net value")):
        line = f"    {label:<20}"
        for policy in (LEGACY, AGENT, CEILING):
            total = _mean(report, policy, metric)
            # The summary mean is per seed; the per-100 denominator is the whole
            # run, so scale the mean back up by the seed count before dividing.
            batch = None if total is None else total * len(seeds)
            value = (
                None if batch is None
                else per_hundred_failures(batch, total_failures)
            )
            per_hundred.setdefault(policy, {})[metric] = value
            line += f"{_fmt(value, 'Rs {:,.0f}'):>26}"
        print(line)

    # --- the two headroom numbers -------------------------------------------
    # TWO DEFINITIONS, and they were already both in this repository before this
    # script existed - silently, which is worse than having one.
    #
    #   per-seed  : compute the ratio inside each seed, then average the ratios.
    #               This is what harness.compare() has always stored as
    #               `headroom_captured`.
    #   pooled    : average the three quantities across seeds first, then take
    #               one ratio of the means. This is what run_ablations.py prints
    #               and what README.md quotes.
    #
    # They differ because a ratio of means is not the mean of ratios: the pooled
    # form weights seeds by how much headroom they had, the per-seed form treats
    # every seed alike. Neither is wrong. Reporting only one while the other is
    # quoted elsewhere IS wrong, so both are printed and both are registered.
    rate_headroom = _mean(report, AGENT, "headroom_captured")
    money_headroom = _mean(report, AGENT, "money_headroom_captured")

    def pooled(metric: str) -> float | None:
        floor = _mean(report, LEGACY, metric)
        ceiling = _mean(report, CEILING, metric)
        agent = _mean(report, AGENT, metric)
        if None in (floor, ceiling, agent) or ceiling <= floor:
            return None
        return (agent - floor) / (ceiling - floor)

    rate_pooled = pooled("recovery_rate")
    money_pooled = pooled("money_recovered")

    print(f"\n{RULE}\nHEADROOM CAPTURED - BY RATE AND BY MONEY\n{RULE}")
    print(f"  {'':<22}{'per-seed mean':>18}{'pooled':>12}")
    print(f"  {'by recovery rate':<22}{_fmt(rate_headroom, '{:.1%}'):>18}"
          f"{_fmt(rate_pooled, '{:.1%}'):>12}")
    print(f"  {'by money recovered':<22}{_fmt(money_headroom, '{:.1%}'):>18}"
          f"{_fmt(money_pooled, '{:.1%}'):>12}")
    print()
    print("  per-seed = mean of the per-seed ratios (harness.compare).")
    print("  pooled   = ratio of the across-seed means (run_ablations.py, and")
    print("             the figure README.md quotes). Both are reported because")
    print("             both were already in use; they answer slightly")
    print("             different questions and neither is a correction of the")
    print("             other.")
    if rate_headroom is not None and money_headroom is not None:
        gap = (money_headroom - rate_headroom) * 100
        pooled_gap = (
            None if None in (rate_pooled, money_pooled)
            else (money_pooled - rate_pooled) * 100
        )
        print()
        print(f"  money minus rate:  {gap:>+6.1f} pp per-seed"
              + ("" if pooled_gap is None else f",  {pooled_gap:>+5.1f} pp pooled"))
        print()
        if abs(gap) < 1.0:
            print("  The two agree to within a point. The agent is capturing the")
            print("  same share of the money as of the episodes, so the mandates it")
            print("  wins are of about average size and the rate is a fair summary.")
        elif gap > 0:
            print("  MONEY HEADROOM IS HIGHER THAN RATE HEADROOM. The episodes the")
            print("  agent wins that legacy loses are disproportionately the LARGE")
            print("  ones, so a rate-only table understates it. That is the class-5")
            print("  effect surfacing in the aggregate: AFA applies only above the")
            print("  Sec 8(a) Rs 15,000 ceiling, so the mandates it rescues are the")
            print("  biggest on the book.")
        else:
            print("  MONEY HEADROOM IS LOWER THAN RATE HEADROOM. The agent is")
            print("  recovering the CHEAP mandates and leaving the expensive ones")
            print("  on the table, so the recovery rate flatters it. On a")
            print("  revenue-recovery brief that is the number to fix, and it is")
            print("  reported here rather than buried because a rate-only table")
            print("  cannot show it.")

    print(f"\n{RULE}\nWHAT THESE NUMBERS ARE, AND ARE NOT\n{RULE}")
    print(f"  Book: {args.seeds} seeds x {args.n} synthetic customers, "
          f"{total_failures:,} failed mandates.")
    print("  Horizon: one billing cycle to at most 12, per episode.")
    print("  The rupees are SYNTHETIC. They are the sum over a generated book,")
    print("  not a claim about any real merchant's revenue.")
    print()
    print("  Cost parameters - processing fees, the hold carrying-cost fraction,")
    print("  the nudge goodwill charge - are ESTIMATES declared in")
    print("  generator/rules.py and marked as such there. They are NOT published")
    print("  Razorpay figures. The regulatory constants in that file are citable")
    print("  (RBI Sec 6(a)/6(c)/8(a), the NPCI attempt ceiling and window); the")
    print("  money constants are not, and every net-value figure inherits that.")
    print()
    print("  Recovery RATES are unchanged by this reporting layer. It computes")
    print("  no model and moves no decision; scripts/verify_docs.py pins the")
    print("  rates it would have to break.")

    emit(
        args.json_path,
        script="run_headline.py",
        seeds=args.seeds,
        n=args.n,
        payload={
            "ambiguity_share": args.share,
            "total_failures": total_failures,
            "policies": {
                policy: {
                    metric: _mean(report, policy, metric)
                    for metric, _, _ in ROWS
                }
                | {
                    "money_recovered_per_100": per_hundred[policy]["money_recovered"],
                    "net_value_per_100": per_hundred[policy]["net_value_total"],
                    "headroom_captured": _mean(report, policy, "headroom_captured"),
                    "money_headroom_captured": _mean(
                        report, policy, "money_headroom_captured"
                    ),
                }
                for policy in (LEGACY, AGENT, CEILING)
            },
            "ranges": {
                policy: {
                    metric: {
                        "low": report.summary[policy][metric].low,
                        "high": report.summary[policy][metric].high,
                    }
                    for metric, _, _ in ROWS
                    if report.summary.get(policy, {}).get(metric)
                    and report.summary[policy][metric].values
                }
                for policy in (LEGACY, AGENT, CEILING)
            },
            # Quoted in README.md. Emitted rather than left as arithmetic on
            # two other figures, so the checker regenerates it like anything
            # else - a derived number in a document is still a claim.
            "net_value_delta_vs_legacy": (
                _mean(report, AGENT, "net_value_total")
                - _mean(report, LEGACY, "net_value_total")
            ),
            "money_delta_vs_legacy": (
                _mean(report, AGENT, "money_recovered")
                - _mean(report, LEGACY, "money_recovered")
            ),
            "headroom": {
                "by_rate": rate_headroom,
                "by_money": money_headroom,
                "by_rate_pooled": rate_pooled,
                "by_money_pooled": money_pooled,
                "difference_points": (
                    None
                    if rate_headroom is None or money_headroom is None
                    else (money_headroom - rate_headroom) * 100
                ),
                "difference_points_pooled": (
                    None
                    if rate_pooled is None or money_pooled is None
                    else (money_pooled - rate_pooled) * 100
                ),
            },
        },
    )

    print(f"\nelapsed {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
