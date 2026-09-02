"""Do the numbers in README.md and CONTEXT.md still reproduce?

    python scripts/verify_docs.py                 # check everything
    python scripts/verify_docs.py --only downtime # one producer
    python scripts/verify_docs.py --refresh       # ignore the cache
    python scripts/verify_docs.py --list          # the registry, no runs

**Run this before any release, any submission, and any commit that touches the
generator, the population, the simulator or a policy.** Every one of those can
move a number that is written down somewhere, and nothing else in the repository
notices when it does.

Why it exists
-------------
CONTEXT.md states the rule: *every claim must have a number, and every number
must come from a reproducible run*. Nothing enforced the second half. Adding the
fifth failure class changed the population underneath figures that had been
measured on the four-class book during classes 1 and 2, and two of those were
caught only because somebody happened to be editing the paragraph beside them.
That is not a process. This is: the documented value and the producing run are
named in one place, and a mismatch is a loud, specific failure rather than a
thing a reader might notice.

How it is built
---------------
Three pieces, deliberately separate:

``PRODUCERS``
    Each documented figure comes from one experiment. A producer knows how to
    run that experiment once and hand back a dict of named fields. Most shell
    out to the script the README already tells the reader to run, with
    ``--json``; those scripts write the file straight off the tallies their
    tables were printed from, so the file cannot disagree with the report. One
    producer (``dataset``) is a library call, because the 20-seed dataset table
    is a property of the generator rather than the output of any policy.

``CLAIMS``
    The registry. One row per documented number: which file it lives in, the
    text to find it by, the producer and dotted field that computes it, the
    value the document currently states, and the tolerance. The tolerance is
    the precision the document quotes at - a figure printed as ``8.9 %`` is
    checked to ±0.05, not to fifteen decimal places - so rounding never causes
    a false alarm and a real move never hides inside one.

``compare``
    Runs each needed producer once, caches the result, and reports every
    mismatch as *file, line, claim, expected, actual, delta*. It prints a diff
    a person can act on, never a boolean.

Adding a claim is one row in ``CLAIMS``. If the field it needs does not exist
yet, add it to the producing script's ``--json`` payload - taken off the object
the report already prints from, never recomputed.

Exit status is 0 when every claim reproduces and 1 when any does not, so this
is usable from CI. ``tests/test_docs.py`` runs it under ``pytest -m slow``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE_DIR = ROOT / ".verify_cache"
RULE = "=" * 100

# The documented runs are all 20 seeds x 500 customers. CONTEXT.md: "never
# report a single run; report mean and spread across 20 seeds."
SEEDS = 20
N_CUSTOMERS = 500


# --------------------------------------------------------------------- units --
# A claim's `scale` maps the producer's field into the units the document quotes
# in, so `expected` can be written exactly as it appears on the page. A share
# stored as 0.089 and written as "8.9 %" has scale=100.
FRACTION_TO_PERCENT = 100.0

# Default tolerances, by how precisely the document quotes the figure.
TOL_PCT_1DP = 0.05      # "8.9 %"      - half of the last printed digit
TOL_PCT_2DP = 0.005     # "28.42 %"
TOL_RUPEE = 0.5         # "Rs 2,293,936" - an exact integer
TOL_COUNT = 0.0         # "152 failures" - exact
TOL_1DP = 0.05          # "23.2 days"
TOL_INT = 0.5           # "539 hours"    - printed with no decimals


# ----------------------------------------------------------------- producers --
@dataclass(frozen=True)
class Producer:
    """One experiment, run once, cached, and addressed by dotted field path."""

    name: str
    description: str
    script: str | None = None
    args: tuple[str, ...] = ()
    call: Callable[[], dict[str, Any]] | None = None
    # Extra directories whose *.py files can move this producer's numbers.
    # Everything reads src/, so that is hashed unconditionally; this is for a
    # producer that reads somewhere else. The `tests` producer counts tests, so
    # its answer changes when tests/ changes and nothing else would notice.
    extra_sources: tuple[str, ...] = ()

    def command(self, out: Path) -> list[str]:
        return [
            sys.executable,
            str(ROOT / "scripts" / self.script),
            "--seeds", str(SEEDS),
            "--n", str(N_CUSTOMERS),
            *self.args,
            "--json", str(out),
        ]

    def run(self) -> dict[str, Any]:
        if self.call is not None:
            return self.call()
        out = CACHE_DIR / f"{self.name}.raw.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            self.command(out), capture_output=True, text=True, cwd=str(ROOT)
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"producer {self.name} failed (exit {proc.returncode})\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        if not out.exists():
            raise RuntimeError(
                f"producer {self.name} exited cleanly but wrote no JSON to {out}"
            )
        return json.loads(out.read_text(encoding="utf-8"))


def dataset_stats() -> dict[str, Any]:
    """The 20-seed dataset table: mean, min and max of each generated property.

    A library producer rather than a script, because these are properties of the
    GENERATOR and not the output of any policy - ``generate_dataset`` computes
    them into ``dataset.meta`` and every script that reads them reads them from
    there. ``scripts/inspect_data.py`` renders the same fields for a single
    seed; the documented table is the multi-seed aggregate, which had no
    producer at all before this one.
    """
    from generator.generate import generate_dataset

    fields = (
        "natural_failure_rate",
        "legacy_recovery_rate",
        "unrecoverable_share",
        "median_days_to_payday",
        "share_failing_beyond_legacy_reach",
        "downtime_share",
        "afa_share",
        "afa_value_share",
        "otp_required_value_share",
        "historical_first_attempt_failure_rate",
    )
    collected: dict[str, list[float]] = {name: [] for name in fields}
    n_failures: list[int] = []
    for seed in range(1, SEEDS + 1):
        meta = generate_dataset(n_customers=N_CUSTOMERS, seed=seed).meta
        for name in fields:
            collected[name].append(float(meta[name]))
        n_failures.append(int(meta["n_failure_events"]))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "script": "generator.generate.generate_dataset (library producer)",
        "seeds": SEEDS,
        "n": N_CUSTOMERS,
        "total_failures": sum(n_failures),
    }
    for name, values in collected.items():
        payload[name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    # CONTEXT.md counts how many seeds land inside the 8-12% target band for
    # the downtime share, and makes an argument out of how few do. A documented
    # count needs a producer like any other number.
    payload["downtime_share"]["seeds_in_8_12_band"] = sum(
        1 for share in collected["downtime_share"] if 0.08 <= share <= 0.12
    )
    return payload


def generator_shape() -> dict[str, Any]:
    """The generator's SHAPE, not its results: window constants and one seed.

    Every other producer here answers "what did the policies score?". This one
    answers "how big is the world they scored in?", and it exists because that
    question had no producer at all - which is exactly how README.md came to
    describe a 60-day oracle horizon against a generator that had moved to 400.
    A results checker cannot catch that: no policy number moves when the window
    grows, so every claim kept reproducing while the prose beside them rotted.

    Also carries seed 42 on its own, because the README's quick-start snippet
    prints ``generate_dataset(n_customers=500, seed=42).natural_failure_rate``
    as a literal and that literal had drifted too. A single-seed figure would be
    a bad RESULT to quote - CONTEXT.md says never report one run - but it is a
    legitimate thing to pin when the document is showing you what the command
    prints.
    """
    from generator import rules
    from generator.generate import (
        FUTURE_HORIZON_DAYS,
        HISTORY_DAYS,
        generate_dataset,
    )

    dataset = generate_dataset(n_customers=N_CUSTOMERS, seed=42)
    return {
        "schema_version": 1,
        "script": "generator.generate constants + seed 42 (library producer)",
        "seeds": 1,
        "n": N_CUSTOMERS,
        "history_days": HISTORY_DAYS,
        "future_horizon_days": FUTURE_HORIZON_DAYS,
        "series_days": int(dataset.meta["series_days"]),
        "as_of_yyyymmdd": int(str(dataset.meta["as_of"]).replace("-", "")),
        "max_episode_cycles": rules.MAX_EPISODE_CYCLES,
        "seed_42": {
            "natural_failure_rate": dataset.meta["natural_failure_rate"],
            "n_failure_events": int(dataset.meta["n_failure_events"]),
            "n_customers": len(dataset.customers),
        },
    }


def test_counts() -> dict[str, Any]:
    """How many tests the suite has, straight from pytest's own collector.

    The README states a count, and a stated count rots the moment anybody adds
    a test - it had already drifted from 267 to 275. ``--collect-only`` imports
    the test modules but runs nothing, so this is about a second and cannot
    recurse into the slow doc check that calls it.
    """
    # NOT -q: the quiet form prints per-file counts and omits the summary line
    # this parses. Verbose collection is still under a second.
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pytest collection failed:\n{proc.stdout[-2000:]}")
    match = re.search(
        r"(\d+)/(\d+) tests collected \((\d+) deselected\)", proc.stdout
    )
    if match:
        selected, total, deselected = (int(g) for g in match.groups())
    else:
        match = re.search(r"(\d+) tests collected", proc.stdout)
        if not match:
            raise RuntimeError(
                f"could not parse a test count from pytest:\n{proc.stdout[-2000:]}"
            )
        selected = total = int(match.group(1))
        deselected = 0
    return {
        "schema_version": 1,
        "script": "pytest --collect-only (library producer)",
        "seeds": 0,
        "n": 0,
        "selected": selected,
        "total": total,
        "deselected": deselected,
    }


PRODUCERS: dict[str, Producer] = {
    "multiclass": Producer(
        "multiclass",
        "legacy vs sequencer vs ceiling across the whole mixed book",
        script="run_multiclass.py",
    ),
    "downtime": Producer(
        "downtime",
        "failure class 2, downtime failures only",
        script="run_downtime.py",
    ),
    "afa": Producer(
        "afa",
        "failure class 5, AFA failures only, reported by value",
        script="run_afa.py",
    ),
    "baseline": Producer(
        "baseline",
        "the floor and the ceiling every agent is scored against",
        script="run_baseline.py",
    ),
    "ablations": Producer(
        "ablations",
        "what each component of the agent is worth",
        script="run_ablations.py",
    ),
    "headline": Producer(
        "headline",
        "the submission headline: money, net value, per-100 and both headrooms",
        script="run_headline.py",
    ),
    "ambiguous": Producer(
        "ambiguous",
        "failure class 6, at masking share 0.35 (off by default in the generator)",
        script="run_ambiguous.py",
        args=("--share", "0.35"),
    ),
    "leadtime": Producer(
        "leadtime",
        "novelty N3: what the RBI notice rule costs, timing vs opt-out",
        script="lead_time_ablation.py",
    ),
    "prior": Producer(
        "prior",
        "method or prior? the agent re-run with deliberately wrong priors",
        script="prior_robustness.py",
    ),
    "dataset": Producer(
        "dataset",
        "properties of the generated population, 20 seeds",
        call=dataset_stats,
    ),
    "shape": Producer(
        "shape",
        "the generator's window constants, and seed 42 on its own",
        call=generator_shape,
    ),
    "tests": Producer(
        "tests",
        "how many tests the suite actually has",
        call=test_counts,
        extra_sources=("tests",),
    ),
}


# --------------------------------------------------------------------- claims --
@dataclass(frozen=True)
class Claim:
    """One documented number and the run that is supposed to produce it."""

    id: str
    doc: str              # "README.md" or "CONTEXT.md"
    anchor: str           # text to locate the claim by, for the line number
    producer: str
    path: str             # dotted path into the producer's payload
    expected: float
    tol: float
    scale: float = 1.0    # producer units -> document units
    unit: str = ""
    note: str = ""


def C(*args, **kwargs) -> Claim:  # noqa: N802 - a table this long needs a short name
    return Claim(*args, **kwargs)


R = "README.md"
X = "CONTEXT.md"

CLAIMS: list[Claim] = [
    # ---------------------------------------------------------------------
    # The multiclass aggregate table, quoted twice: above the fold and again
    # in the "Multi-class results" section.
    # ---------------------------------------------------------------------
    C("mc.money.legacy", R, "| **money recovered** | Rs 2,293,936 |",
      "multiclass", "aggregate.legacy_fixed.money", 2_293_936, TOL_RUPEE, unit="Rs"),
    C("mc.money.sequencer", R, "| **money recovered** | Rs 2,293,936 |",
      "multiclass", "aggregate.sequencer.money", 3_839_648, TOL_RUPEE, unit="Rs"),
    C("mc.recovery.legacy", R, "| recovery rate | 36.2% |",
      "multiclass", "aggregate.legacy_fixed.recovery_rate", 36.2, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("mc.recovery.sequencer", R, "| recovery rate | 36.2% |",
      "multiclass", "aggregate.sequencer.recovery_rate", 59.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("mc.notices.legacy", R, "| pre-debit notifications sent | 4,091 |",
      "multiclass", "aggregate.legacy_fixed.notifications", 4_091, TOL_COUNT),
    C("mc.notices.sequencer", R, "| pre-debit notifications sent | 4,091 |",
      "multiclass", "aggregate.sequencer.notifications", 1_633, TOL_COUNT),
    C("mc.optouts.legacy", R, "| opt-outs induced | 146 |",
      "multiclass", "aggregate.legacy_fixed.optouts", 146, TOL_COUNT),
    C("mc.optouts.sequencer", R, "| opt-outs induced | 146 |",
      "multiclass", "aggregate.sequencer.optouts", 45, TOL_COUNT),
    C("mc.breaches.legacy", R, "| **compliance breaches** | **41** | **0** |",
      "multiclass", "breaches.legacy_fixed", 41, TOL_COUNT),
    C("mc.breaches.sequencer", R, "| **compliance breaches** | **41** | **0** |",
      "multiclass", "breaches.sequencer", 0, TOL_COUNT),
    C("mc.presentments.legacy", R, "| presentments | 5,654 | 7,138 |",
      "multiclass", "aggregate.legacy_fixed.presentments", 5_654, TOL_COUNT),
    C("mc.presentments.sequencer", R, "| presentments | 5,654 | 7,138 |",
      "multiclass", "aggregate.sequencer.presentments", 7_138, TOL_COUNT),
    C("mc.delta.money", R, "**+Rs 1,545,712 recovered (+67.4%)",
      "multiclass", "delta.money", 1_545_712, TOL_RUPEE, unit="Rs"),
    C("mc.delta.money_relative", R, "**+Rs 1,545,712 recovered (+67.4%)",
      "multiclass", "delta.money_relative", 67.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("mc.delta.recovery_points", R, "The sequencer is +23.2 points of recovery",
      "multiclass", "delta.recovery_points", 23.2, TOL_PCT_1DP, unit="pp"),
    C("mc.delta.recovery_relative", R, "The sequencer is +23.2 points of recovery",
      "multiclass", "delta.recovery_relative", 64.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),

    # ------------------------------------------------------ the failure mix --
    C("mix.insufficient_funds", R, "76.8% insufficient funds",
      "multiclass", "failure_mix.insufficient_funds", 76.8, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("mix.downtime", R, "mix: 76.8% insufficient funds",
      "multiclass", "failure_mix.bank_or_psp_downtime", 8.9, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%",
      note="pooled over all 20 seeds; the 8.8% elsewhere is the mean of the "
           "per-seed shares and is a different statistic"),
    C("mix.expired_card", R, "mix: 76.8% insufficient funds",
      "multiclass", "failure_mix.expired_or_reissued_card", 6.3, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("mix.afa", R, "mix: 76.8% insufficient funds",
      "multiclass", "failure_mix.afa_incomplete", 5.6, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("mix.opted_out", R, "mix: 76.8% insufficient funds",
      "multiclass", "failure_mix.customer_opted_out", 2.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("mix.total_failures", R, "across 1,709 failures - **Rs 2,266 per 100",
      "multiclass", "total_failures", 1_709, TOL_COUNT),

    # ---------------------------------------- structurally dead mandates ----
    C("dead.saved_presentments", R, "presentments and Rs 38,730** across 1,709",
      "multiclass", "dead_mandates.saved_presentments", 300, TOL_COUNT),
    C("dead.saved_cost", R, "presentments and Rs 38,730** across 1,709",
      "multiclass", "dead_mandates.saved_cost", 38_730, TOL_RUPEE, unit="Rs"),
    C("dead.per_hundred", R, "across 1,709 failures - **Rs 2,266 per 100",
      "multiclass", "dead_mandates.saved_cost_per_hundred_failures", 2_266, TOL_RUPEE,
      unit="Rs"),

    # ------------------------------------------ per-class recovery / money --
    # NOTE: there is deliberately no row here for class 1's per-class recovery
    # rate. run_multiclass.py prints it, but neither document quotes it - and a
    # registry row for a number nobody wrote down would fail as "anchor not
    # found" forever. The registry covers what the DOCUMENTS claim, not
    # everything the scripts can produce.
    C("class2.recovery.legacy", R, "| legacy fixed | 89.5% | 334 | 1 | 186 |",
      "downtime", "policies.legacy_fixed.recovery_rate", 89.5, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("class2.recovery.sequencer", R, "| **sequencer** | **94.1%** | 311 | 1 | 139 |",
      "downtime", "policies.sequencer.recovery_rate", 94.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("class2.recovery.oracle", R, "| oracle ceiling | 94.7% | 299 | 0 | 149 |",
      "downtime", "policies.oracle_ceiling.recovery_rate", 94.7, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("class2.presentments.legacy", R, "| legacy fixed | 89.5% | 334 | 1 | 186 |",
      "downtime", "policies.legacy_fixed.presentments", 334, TOL_COUNT),
    C("class2.presentments.sequencer", R, "| **sequencer** | **94.1%** | 311 | 1 | 139 |",
      "downtime", "policies.sequencer.presentments", 311, TOL_COUNT),
    C("class2.presentments.oracle", R, "| oracle ceiling | 94.7% | 299 | 0 | 149 |",
      "downtime", "policies.oracle_ceiling.presentments", 299, TOL_COUNT),
    C("class2.notices.legacy", R, "| legacy fixed | 89.5% | 334 | 1 | 186 |",
      "downtime", "policies.legacy_fixed.notifications", 186, TOL_COUNT),
    C("class2.notices.sequencer", R, "| **sequencer** | **94.1%** | 311 | 1 | 139 |",
      "downtime", "policies.sequencer.notifications", 139, TOL_COUNT),
    C("class2.notices.oracle", R, "| oracle ceiling | 94.7% | 299 | 0 | 149 |",
      "downtime", "policies.oracle_ceiling.notifications", 149, TOL_COUNT),
    C("class2.into_outage.oracle", R, "| oracle ceiling | 94.7% | 299 | 0 | 149 |",
      "downtime", "policies.oracle_ceiling.into_outage", 0, TOL_COUNT),
    C("class2.episodes", R, "152 downtime failures out of 1,709 total",
      "downtime", "downtime_failures", 152, TOL_COUNT),
    C("class2.share_pooled", R, "152 downtime failures out of 1,709 total",
      "downtime", "downtime_share_pooled", 8.9, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),

    # ----------------------------------------------- class 2 findings -------
    C("class2.spans_two", R, "Zero of 48,779 generated bank outages span two",
      "downtime", "outages_spanning_two_windows", 0, TOL_COUNT),
    C("class2.total_outages", R, "Zero of 48,779 generated bank outages span two",
      "downtime", "total_outages", 48_779, TOL_COUNT),
    C("class2.largest_cluster", R,
      "500-customer book is 10; at 5,000 customers it is 45)",
      "downtime", "largest_same_bank_same_day_cluster", 10, TOL_COUNT),
    C("class2.bank_health_delta", R,
      "`no_bank_health` produces results identical to the rupee.",
      "downtime", "bank_health_net_value_delta", 0.0, TOL_RUPEE, unit="Rs"),
    C("class2.hours_after_clear.sequencer", R,
      "The agent recovers as often as the oracle but takes **539 hours**",
      "downtime", "policies.sequencer.mean_hours_after_clear", 539, TOL_INT,
      unit="h"),
    C("class2.hours_after_clear.oracle", R,
      "outage clears against the oracle's **37**.",
      "downtime", "policies.oracle_ceiling.mean_hours_after_clear", 37, TOL_INT,
      unit="h"),
    C("class2.discount.0.recovery", R, "| 0.00 (as specified) | 94.1% | 22.5 |",
      "downtime", "discount_sweep.0\\.0.recovery_rate", 94.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("class2.discount.0.days", R, "| 0.00 (as specified) | 94.1% | 22.5 |",
      "downtime", "discount_sweep.0\\.0.mean_days_after_clear", 22.5, TOL_1DP,
      unit="days"),
    C("class2.discount.1.recovery", R, "| 0.01 | 93.4% | **10.8** |",
      "downtime", "discount_sweep.0\\.01.recovery_rate", 93.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("class2.discount.1.days", R, "| 0.01 | 93.4% | **10.8** |",
      "downtime", "discount_sweep.0\\.01.mean_days_after_clear", 10.8, TOL_1DP,
      unit="days"),
    C("class2.discount.3.recovery", R, "| 0.03 | 94.1% | 9.2 |",
      "downtime", "discount_sweep.0\\.03.recovery_rate", 94.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("class2.discount.3.days", R, "| 0.03 | 94.1% | 9.2 |",
      "downtime", "discount_sweep.0\\.03.mean_days_after_clear", 9.2, TOL_1DP,
      unit="days"),
    C("ctx.class2.hours.sequencer", X,
      "oracle but resolves 539 hours after the outage clears, against the",
      "downtime", "policies.sequencer.mean_hours_after_clear", 539, TOL_INT,
      unit="h"),
    C("ctx.class2.hours.oracle", X, "oracle's 37. DecisionEngine.daily_discount",
      "downtime", "policies.oracle_ceiling.mean_hours_after_clear", 37, TOL_INT,
      unit="h"),
    C("ctx.class2.discount.days0", X, "delay (22.5 -> 10.8 days) for 0.7 points of recovery",
      "downtime", "discount_sweep.0\\.0.mean_days_after_clear", 22.5, TOL_1DP,
      unit="days"),
    C("ctx.class2.discount.days1", X, "delay (22.5 -> 10.8 days) for 0.7 points of recovery",
      "downtime", "discount_sweep.0\\.01.mean_days_after_clear", 10.8, TOL_1DP,
      unit="days"),
    C("ctx.class2.discount.recovery", X, "93.4%, and back to 94.1% at 3%/day - flat to within a point rather",
      "downtime", "discount_sweep.0\\.01.recovery_rate", 93.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),

    # -------------------------------------------------- AFA framing numbers --
    C("afa.book.failures", R, "| across 1,709 failures worth Rs 5,806,135 |",
      "afa", "book.failures", 1_709, TOL_COUNT),
    C("afa.book.value", R, "| across 1,709 failures worth Rs 5,806,135 |",
      "afa", "book.value", 5_806_135, TOL_RUPEE, unit="Rs"),
    C("afa.otp_gated.count", R, "| failures on OTP-gated mandates | 9.2 % | 46.8 % |",
      "afa", "book.otp_gated_count_share", 9.2, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.otp_gated.value", R, "| failures on OTP-gated mandates | 9.2 % | 46.8 % |",
      "afa", "book.otp_gated_value_share", 46.8, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.gate.count", R,
      "| **failures refused at the AFA gate (1A)** | **5.6 %** | **28.4 %** |",
      "afa", "book.afa_count_share", 5.6, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("afa.gate.value", R,
      "| **failures refused at the AFA gate (1A)** | **5.6 %** | **28.4 %** |",
      "afa", "book.afa_value_share", 28.4, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.afa.count", X, "28.4% of the rupees at risk, because AFA applies only",
      "afa", "book.afa_count_share", 5.6, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.afa.value", X, "28.4% of the rupees at risk, because AFA applies only",
      "afa", "book.afa_value_share", 28.4, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),

    # --------------------------------------------------- AFA money and rate --
    C("afa.episodes", R, "Value at risk across **96 AFA failures**: Rs 1,648,936.",
      "afa", "afa_failures", 96, TOL_COUNT),
    C("afa.value_at_risk", R, "Value at risk across **96 AFA failures**: Rs 1,648,936.",
      "afa", "value_at_risk", 1_648_936, TOL_RUPEE, unit="Rs"),
    C("afa.money.legacy", R, "| legacy fixed | 816,633 | 49.5 % | -9,220,563 | 0 | 205 |",
      "afa", "policies.legacy_fixed.money", 816_633, TOL_RUPEE, unit="Rs"),
    C("afa.share.legacy", R, "| legacy fixed | 816,633 | 49.5 % | -9,220,563 | 0 | 205 |",
      "afa", "policies.legacy_fixed.value_recovered_share", 49.5, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.net.legacy", R, "| legacy fixed | 816,633 | 49.5 % | -9,220,563 | 0 | 205 |",
      "afa", "policies.legacy_fixed.net_value", -9_220_563, TOL_RUPEE, unit="Rs"),
    C("afa.money.sequencer", R,
      "| **sequencer** | **1,123,371** | **68.1 %** | **-3,130,544** | 235,961 | 136 |",
      "afa", "policies.sequencer.money", 1_123_371, TOL_RUPEE, unit="Rs"),
    C("afa.share.sequencer", R,
      "| **sequencer** | **1,123,371** | **68.1 %** | **-3,130,544** | 235,961 | 136 |",
      "afa", "policies.sequencer.value_recovered_share", 68.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.goodwill.sequencer", R,
      "| **sequencer** | **1,123,371** | **68.1 %** | **-3,130,544** | 235,961 | 136 |",
      "afa", "policies.sequencer.goodwill", 235_961, TOL_RUPEE, unit="Rs"),
    C("afa.money.blind", R, "| `no_afa_nudge` | 1,040,834 | 63.1 % | -3,552,077 | 0 | 79 |",
      "afa", "policies.no_afa_nudge.money", 1_040_834, TOL_RUPEE, unit="Rs"),
    C("afa.share.blind", R, "| `no_afa_nudge` | 1,040,834 | 63.1 % | -3,552,077 | 0 | 79 |",
      "afa", "policies.no_afa_nudge.value_recovered_share", 63.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.money.oracle", R,
      "| oracle ceiling | 1,523,117 | 92.4 % | +372,961 | 7,056 | 91 |",
      "afa", "policies.oracle_ceiling.money", 1_523_117, TOL_RUPEE, unit="Rs"),
    C("afa.share.oracle", R,
      "| oracle ceiling | 1,523,117 | 92.4 % | +372,961 | 7,056 | 91 |",
      "afa", "policies.oracle_ceiling.value_recovered_share", 92.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.gain", R, "recovers Rs 306,738 more than legacy",
      "afa", "sequencer_over_legacy_money", 306_738, TOL_RUPEE, unit="Rs"),
    C("afa.gain_per_failure", R, "Rs 3,195 per AFA failure.**",
      "afa", "sequencer_over_legacy_per_failure", 3_195, TOL_RUPEE, unit="Rs"),
    C("afa.rate.legacy", R, "| legacy fixed | 50.0 % | 48/96 | 25 % .. 100 % |",
      "afa", "policies.legacy_fixed.recovery_rate", 50.0, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.rate.sequencer", R, "| **sequencer** | **68.8 %** | 66/96 | 33 % .. 100 % |",
      "afa", "policies.sequencer.recovery_rate", 68.8, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.rate.oracle", R, "| oracle ceiling | 92.7 % | 89/96 | 67 % .. 100 % |",
      "afa", "policies.oracle_ceiling.recovery_rate", 92.7, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.rate.blind", R, "| `no_afa_nudge` | 1,040,834 | 63.5 % | 0 | 0 | -3,552,077 |",
      "afa", "policies.no_afa_nudge.recovery_rate", 63.5, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.rate_sd", R, "per-seed standard deviation of **18.0 %**.",
      "afa", "per_seed_rate_sd_sequencer", 18.0, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("afa.nudges", R, "**Only 19 of 578 nudges actually opened a gate",
      "afa", "policies.sequencer.nudges", 578, TOL_COUNT),
    C("afa.nudge_completions", R, "**Only 19 of 578 nudges actually opened a gate",
      "afa", "policies.sequencer.nudge_completions", 19, TOL_COUNT),
    C("afa.nudge_delta_money", R,
      "**Nudging is worth Rs 82,537 of recovery and Rs 421,534 of net value.**",
      "afa", "nudge_delta_money", 82_537, TOL_RUPEE, unit="Rs"),
    C("afa.nudge_delta_net", R,
      "**Nudging is worth Rs 82,537 of recovery and Rs 421,534 of net value.**",
      "afa", "nudge_delta_net_value", 421_534, TOL_RUPEE, unit="Rs"),
    C("afa.chosen.legacy", R, "| legacy fixed | 205 | 89 | 128 |",
      "afa", "policies.legacy_fixed.chosen_presentments", 205, TOL_COUNT),
    C("afa.automatic.legacy", R, "| legacy fixed | 205 | 89 | 128 |",
      "afa", "policies.legacy_fixed.automatic_presentments", 89, TOL_COUNT),
    C("afa.refused.legacy", R, "| legacy fixed | 205 | 89 | 128 |",
      "afa", "policies.legacy_fixed.afa_refused", 128, TOL_COUNT),
    C("afa.chosen.sequencer", R, "| sequencer | 136 | 393 | 332 |",
      "afa", "policies.sequencer.chosen_presentments", 136, TOL_COUNT),
    C("afa.automatic.sequencer", R, "| sequencer | 136 | 393 | 332 |",
      "afa", "policies.sequencer.automatic_presentments", 393, TOL_COUNT),
    C("afa.refused.sequencer", R, "| sequencer | 136 | 393 | 332 |",
      "afa", "policies.sequencer.afa_refused", 332, TOL_COUNT),
    C("afa.chosen.oracle", R, "| oracle ceiling | 91 | 163 | 55 |",
      "afa", "policies.oracle_ceiling.chosen_presentments", 91, TOL_COUNT),
    C("afa.uncollected.legacy", R, "**Rs 832,303** uncollected against the ceiling's",
      "afa", "policies.legacy_fixed.uncollected", 832_303, TOL_RUPEE, unit="Rs"),
    C("afa.uncollected.oracle", R, "uncollected against the ceiling's Rs 125,819.",
      "afa", "policies.oracle_ceiling.uncollected", 125_819, TOL_RUPEE, unit="Rs"),

    # ------------------------------------------------------ headroom table --
    # Quoted twice in README.md and once in CONTEXT.md. Class 5's row comes
    # from run_afa.py; classes 1 and 2 come from run_multiclass.py, whose
    # ceiling arm exists for exactly this table.
    C("headroom.c1.floor", R,
      "| 1 insufficient funds | 32.7% | 80.4% | 47.7 pp | **73.5%** |",
      "multiclass", "headroom_by_value.insufficient_funds.floor", 32.7, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("headroom.c1.ceiling", R,
      "| 1 insufficient funds | 32.7% | 80.4% | 47.7 pp | **73.5%** |",
      "multiclass", "headroom_by_value.insufficient_funds.ceiling", 80.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("headroom.c1.gap", R,
      "| 1 insufficient funds | 32.7% | 80.4% | 47.7 pp | **73.5%** |",
      "multiclass", "headroom_by_value.insufficient_funds.gap_points", 47.7, TOL_PCT_1DP,
      unit="pp"),
    C("headroom.c1.captured", R,
      "| 1 insufficient funds | 32.7% | 80.4% | 47.7 pp | **73.5%** |",
      "multiclass", "headroom_by_value.insufficient_funds.captured", 73.5, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("headroom.c2.floor", R,
      "| 2 bank downtime | 93.9% | 99.2% | 5.3 pp | 78.7% |",
      "multiclass", "headroom_by_value.bank_or_psp_downtime.floor", 93.9, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("headroom.c2.ceiling", R,
      "| 2 bank downtime | 93.9% | 99.2% | 5.3 pp | 78.7% |",
      "multiclass", "headroom_by_value.bank_or_psp_downtime.ceiling", 99.2, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("headroom.c2.captured", R,
      "| 2 bank downtime | 93.9% | 99.2% | 5.3 pp | 78.7% |",
      "multiclass", "headroom_by_value.bank_or_psp_downtime.captured", 78.7, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("headroom.c5.floor", R,
      "| 5 AFA incomplete | 49.5% | 92.4% | 42.8 pp | **43.4%** |",
      "afa", "headroom_by_value.floor", 49.5, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("headroom.c5.ceiling", R,
      "| 5 AFA incomplete | 49.5% | 92.4% | 42.8 pp | **43.4%** |",
      "afa", "headroom_by_value.ceiling", 92.4, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("headroom.c5.gap", R,
      "| 5 AFA incomplete | 49.5% | 92.4% | 42.8 pp | **43.4%** |",
      "afa", "headroom_by_value.gap_points", 42.8, TOL_PCT_1DP, unit="pp"),
    C("headroom.c5.captured", R,
      "| 5 AFA incomplete | 49.5% | 92.4% | 42.8 pp | **43.4%** |",
      "afa", "headroom_by_value.captured", 43.4, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("headroom.c1.by_rate", R, "are 72.7 %, 87.5 % and 43.9 % — the ordering is the same.)",
      "multiclass", "headroom_by_rate.insufficient_funds", 72.7, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("headroom.c2.by_rate", R, "are 72.7 %, 87.5 % and 43.9 % — the ordering is the same.)",
      "multiclass", "headroom_by_rate.bank_or_psp_downtime", 87.5, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("headroom.c5.by_rate", R, "are 72.7 %, 87.5 % and 43.9 % — the ordering is the same.)",
      "afa", "headroom_by_rate_captured", 43.9, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.headroom.c1.captured", X,
      "1 insufficient funds 32.7%    80.4%  47.7pp    73.5%",
      "multiclass", "headroom_by_value.insufficient_funds.captured", 73.5, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("ctx.headroom.c2.captured", X,
      "2 bank downtime      93.9%    99.2%   5.3pp    78.7%",
      "multiclass", "headroom_by_value.bank_or_psp_downtime.captured", 78.7,
      TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.headroom.c5.captured", X,
      "5 AFA incomplete     49.5%    92.4%  42.8pp    43.4%",
      "afa", "headroom_by_value.captured", 43.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),

    # ------------------------------------------ the scale (run_baseline.py) --
    C("baseline.floor", R, "| fixed-schedule floor (day-2/4/7) | **36.1 %** |",
      "baseline", "policies.legacy_fixed.recovery_rate.mean", 36.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("baseline.ceiling", R, "| oracle ceiling | **69.0 %** | 58.4 - 79.3 % |",
      "baseline", "policies.oracle_ceiling.recovery_rate.mean", 69.0, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("baseline.headroom", R, "| **headroom to capture** | **32.8 pp** |",
      "baseline", "headroom_points", 32.8, TOL_PCT_1DP, unit="pp"),
    C("baseline.presentments.legacy", R, "| presentments used | 283 | 286 |",
      "baseline", "policies.legacy_fixed.attempts_used_total.mean", 283, TOL_INT),
    C("baseline.presentments.oracle", R, "| presentments used | 283 | 286 |",
      "baseline", "policies.oracle_ceiling.attempts_used_total.mean", 286, TOL_INT),
    C("baseline.retained.legacy", R, "| customers retained | 30.9 | 71.8 |",
      "baseline", "policies.legacy_fixed.customers_retained.mean", 30.9, TOL_1DP),
    C("baseline.retained.oracle", R, "| customers retained | 30.9 | 71.8 |",
      "baseline", "policies.oracle_ceiling.customers_retained.mean", 71.8, TOL_1DP),
    C("baseline.optouts.legacy", R, "| opt-outs induced | 7.3 | 1.6 |",
      "baseline", "policies.legacy_fixed.optouts_induced.mean", 7.3, TOL_1DP),
    C("baseline.optouts.oracle", R, "| opt-outs induced | 7.3 | 1.6 |",
      "baseline", "policies.oracle_ceiling.optouts_induced.mean", 1.6, TOL_1DP),
    C("baseline.breaches.legacy", R, "| compliance breaches | 2.1 | 0 |",
      "baseline", "policies.legacy_fixed.compliance_breaches.mean", 2.1, TOL_1DP),

    # ------------------------------------------ ablations table and verdicts --
    C("abl.legacy.recovery", R, "| legacy fixed | 36.1% | 114,697 | -2,034,554 | 283 | 7.3 | 30.9 |",
      "ablations", "table.legacy_fixed.recovery_rate", 36.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("abl.legacy.money", R, "| legacy fixed | 36.1% | 114,697 | -2,034,554 | 283 | 7.3 | 30.9 |",
      "ablations", "table.legacy_fixed.money_recovered", 114_697, TOL_INT, unit="Rs"),
    C("abl.legacy.net", R, "| legacy fixed | 36.1% | 114,697 | -2,034,554 | 283 | 7.3 | 30.9 |",
      "ablations", "table.legacy_fixed.net_value_total", -2_034_554, TOL_INT, unit="Rs"),
    C("abl.full.recovery", R,
      "| **sequencer** | **59.3%** | **191,982** | **-653,258** | 357 | 2.2 | 65.3 |",
      "ablations", "table.full.recovery_rate", 59.3, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("abl.full.money", R,
      "| **sequencer** | **59.3%** | **191,982** | **-653,258** | 357 | 2.2 | 65.3 |",
      "ablations", "table.full.money_recovered", 191_982, TOL_INT, unit="Rs"),
    C("abl.full.net", R,
      "| **sequencer** | **59.3%** | **191,982** | **-653,258** | 357 | 2.2 | 65.3 |",
      "ablations", "table.full.net_value_total", -653_258, TOL_INT, unit="Rs"),
    C("abl.oracle.recovery", R, "| oracle ceiling | 69.0% | 234,227 | -252,233 | 286 | 1.6 | 71.8 |",
      "ablations", "table.oracle_ceiling.recovery_rate", 69.0, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("abl.headroom_captured", R, "**Headroom captured: 70.7%.**",
      "ablations", "headline.headroom_captured", 70.7, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),

    C("abl.no_payday.recovery", R, "| `no_payday` | -16.5% | -457,312 | 18/2/0 | 20 |",
      "ablations", "verdicts.no_payday.delta_recovery_points", -16.5, TOL_PCT_1DP,
      unit="pp"),
    C("abl.no_payday.net", R, "| `no_payday` | -16.5% | -457,312 | 18/2/0 | 20 |",
      "ablations", "verdicts.no_payday.delta_net_value", -457_312, TOL_INT, unit="Rs"),
    C("abl.no_payday.worse", R, "| `no_payday` | -16.5% | -457,312 | 18/2/0 | 20 |",
      "ablations", "verdicts.no_payday.worse", 18, TOL_COUNT),
    C("abl.no_payday.bit", R, "| `no_payday` | -16.5% | -457,312 | 18/2/0 | 20 |",
      "ablations", "verdicts.no_payday.bit", 20, TOL_COUNT),
    C("abl.no_hold.recovery", R, "| `no_hold` | -9.9% | -859,263 | 20/0/0 | 20 |",
      "ablations", "verdicts.no_hold.delta_recovery_points", -9.9, TOL_PCT_1DP,
      unit="pp"),
    C("abl.no_hold.net", R, "| `no_hold` | -9.9% | -859,263 | 20/0/0 | 20 |",
      "ablations", "verdicts.no_hold.delta_net_value", -859_263, TOL_INT, unit="Rs"),
    C("abl.no_hold.worse", R, "| `no_hold` | -9.9% | -859,263 | 20/0/0 | 20 |",
      "ablations", "verdicts.no_hold.worse", 20, TOL_COUNT),
    C("abl.tier1.recovery", R, "| `tier1_payday` | -7.1% | -151,935 | 16/4/0 | 20 |",
      "ablations", "verdicts.tier1_payday.delta_recovery_points", -7.1, TOL_PCT_1DP,
      unit="pp"),
    C("abl.tier1.net", R, "| `tier1_payday` | -7.1% | -151,935 | 16/4/0 | 20 |",
      "ablations", "verdicts.tier1_payday.delta_net_value", -151_935, TOL_INT, unit="Rs"),
    C("abl.tier1.worse", R, "| `tier1_payday` | -7.1% | -151,935 | 16/4/0 | 20 |",
      "ablations", "verdicts.tier1_payday.worse", 16, TOL_COUNT),
    C("abl.with_nudge.recovery", R, "| `with_nudge` | -3.1% | -345,430 | 19/1/0 | 20 |",
      "ablations", "verdicts.with_nudge.delta_recovery_points", -3.1, TOL_PCT_1DP,
      unit="pp"),
    C("abl.with_nudge.net", R, "| `with_nudge` | -3.1% | -345,430 | 19/1/0 | 20 |",
      "ablations", "verdicts.with_nudge.delta_net_value", -345_430, TOL_INT, unit="Rs"),
    C("abl.with_nudge.worse", R, "| `with_nudge` | -3.1% | -345,430 | 19/1/0 | 20 |",
      "ablations", "verdicts.with_nudge.worse", 19, TOL_COUNT),
    C("abl.irritation.recovery", R, "| `irritation_build` | -1.2% | -38,845 | 13/7/0 | 20 |",
      "ablations", "verdicts.irritation_build.delta_recovery_points", -1.2, TOL_PCT_1DP,
      unit="pp"),
    C("abl.irritation.net", R, "| `irritation_build` | -1.2% | -38,845 | 13/7/0 | 20 |",
      "ablations", "verdicts.irritation_build.delta_net_value", -38_845, TOL_INT,
      unit="Rs"),
    C("abl.irritation.worse", R, "| `irritation_build` | -1.2% | -38,845 | 13/7/0 | 20 |",
      "ablations", "verdicts.irritation_build.worse", 13, TOL_COUNT),
    C("ctx.irritation.recovery", X,
      "no longer consistent: -1.2% recovery on 13 of 20",
      "ablations", "verdicts.irritation_build.delta_recovery_points", -1.2, TOL_PCT_1DP,
      unit="pp"),
    C("ctx.irritation.worse", X,
      "no longer consistent: -1.2% recovery on 13 of 20",
      "ablations", "verdicts.irritation_build.worse", 13, TOL_COUNT),
    C("abl.no_afa_nudge.recovery", R, "| `no_afa_nudge` | -0.3% | -21,077 | 7/13/0 | 20 |",
      "ablations", "verdicts.no_afa_nudge.delta_recovery_points", -0.3, TOL_PCT_1DP,
      unit="pp"),
    C("abl.no_afa_nudge.net", R, "| `no_afa_nudge` | -0.3% | -21,077 | 7/13/0 | 20 |",
      "ablations", "verdicts.no_afa_nudge.delta_net_value", -21_077, TOL_INT, unit="Rs"),
    C("abl.no_afa_nudge.worse", R, "| `no_afa_nudge` | -0.3% | -21,077 | 7/13/0 | 20 |",
      "ablations", "verdicts.no_afa_nudge.worse", 7, TOL_COUNT),
    C("abl.no_afa_nudge.better", R, "| `no_afa_nudge` | -0.3% | -21,077 | 7/13/0 | 20 |",
      "ablations", "verdicts.no_afa_nudge.better", 13, TOL_COUNT),
    C("abl.no_afa_nudge.tied", R, "| `no_afa_nudge` | -0.3% | -21,077 | 7/13/0 | 20 |",
      "ablations", "verdicts.no_afa_nudge.tied", 0, TOL_COUNT,
      note="the README argues explicitly that this variant ties on zero seeds"),
    C("abl.no_optout.recovery", R, "| `no_optout_cost` | +1.0% | +37,394 | 9/11/0 | 20 |",
      "ablations", "verdicts.no_optout_cost.delta_recovery_points", 1.0, TOL_PCT_1DP,
      unit="pp"),
    C("abl.no_optout.net", R, "| `no_optout_cost` | +1.0% | +37,394 | 9/11/0 | 20 |",
      "ablations", "verdicts.no_optout_cost.delta_net_value", 37_394, TOL_INT, unit="Rs"),
    C("abl.no_optout.worse", R, "| `no_optout_cost` | +1.0% | +37,394 | 9/11/0 | 20 |",
      "ablations", "verdicts.no_optout_cost.worse", 9, TOL_COUNT),
    C("abl.no_optout.better", R, "| `no_optout_cost` | +1.0% | +37,394 | 9/11/0 | 20 |",
      "ablations", "verdicts.no_optout_cost.better", 11, TOL_COUNT),
    C("ctx.no_optout.split", X,
      "worse / 11 better across 20 - no consistent effect in either direction",
      "ablations", "verdicts.no_optout_cost.better", 11, TOL_COUNT),
    C("abl.no_bank_health.net", R, "| `no_bank_health` | +0.0% | **0** | 0/0/20 | **0** |",
      "ablations", "verdicts.no_bank_health.delta_net_value", 0, TOL_RUPEE, unit="Rs"),
    C("abl.no_bank_health.tied", R, "| `no_bank_health` | +0.0% | **0** | 0/0/20 | **0** |",
      "ablations", "verdicts.no_bank_health.tied", 20, TOL_COUNT),
    C("abl.no_bank_health.bit", R, "| `no_bank_health` | +0.0% | **0** | 0/0/20 | **0** |",
      "ablations", "verdicts.no_bank_health.bit", 0, TOL_COUNT),

    # --------------------------------- the money headline (run_headline.py) --
    # Part A surfaces rupees the harness already computed and then dropped at
    # the presentation layer. Every figure here is pinned so the money claims
    # get the same treatment the rate claims have had since verify_docs existed.
    C("hl.money.legacy", R, "| money recovered | Rs 114,697 | **Rs 191,982** | Rs 234,227 |",
      "headline", "policies.legacy_fixed.money_recovered", 114_697, TOL_RUPEE,
      unit="Rs"),
    C("hl.money.sequencer", R, "| money recovered | Rs 114,697 | **Rs 191,982** | Rs 234,227 |",
      "headline", "policies.sequencer.money_recovered", 191_982, TOL_RUPEE, unit="Rs"),
    C("hl.money.oracle", R, "| money recovered | Rs 114,697 | **Rs 191,982** | Rs 234,227 |",
      "headline", "policies.oracle_ceiling.money_recovered", 234_227, TOL_RUPEE,
      unit="Rs"),
    C("hl.net.legacy", R,
      "| **net value** | **-Rs 2,034,554** | **-Rs 653,258** | **-Rs 252,233** |",
      "headline", "policies.legacy_fixed.net_value_total", -2_034_554, TOL_RUPEE,
      unit="Rs"),
    C("hl.net.sequencer", R,
      "| **net value** | **-Rs 2,034,554** | **-Rs 653,258** | **-Rs 252,233** |",
      "headline", "policies.sequencer.net_value_total", -653_258, TOL_RUPEE, unit="Rs"),
    C("hl.net.oracle", R,
      "| **net value** | **-Rs 2,034,554** | **-Rs 653,258** | **-Rs 252,233** |",
      "headline", "policies.oracle_ceiling.net_value_total", -252_233, TOL_RUPEE,
      unit="Rs"),
    C("hl.rate.legacy", R, "| recovery rate | 36.1% | **59.3%** | 69.0% |",
      "headline", "policies.legacy_fixed.recovery_rate", 36.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%",
      note="the gate on Part A: this reporting layer must not move a rate"),
    C("hl.rate.sequencer", R, "| recovery rate | 36.1% | **59.3%** | 69.0% |",
      "headline", "policies.sequencer.recovery_rate", 59.3, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("hl.rate.oracle", R, "| recovery rate | 36.1% | **59.3%** | 69.0% |",
      "headline", "policies.oracle_ceiling.recovery_rate", 69.0, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("hl.presentments.legacy", R, "| presentments | 283 | 357 | 286 |",
      "headline", "policies.legacy_fixed.attempts_used_total", 283, TOL_INT),
    C("hl.presentments.sequencer", R, "| presentments | 283 | 357 | 286 |",
      "headline", "policies.sequencer.attempts_used_total", 357, TOL_INT),
    C("hl.optouts.sequencer", R, "| opt-outs induced | 7.3 | **2.2** | 1.6 |",
      "headline", "policies.sequencer.optouts_induced", 2.2, TOL_1DP),
    C("hl.retained.sequencer", R, "| mandates retained | 30.9 | **65.3** | 71.8 |",
      "headline", "policies.sequencer.customers_retained", 65.3, TOL_1DP),

    C("hl.per100.money.legacy", R,
      "| money recovered | Rs 134,227 | **Rs 224,672** | Rs 274,110 |",
      "headline", "policies.legacy_fixed.money_recovered_per_100", 134_227, TOL_RUPEE,
      unit="Rs"),
    C("hl.per100.money.sequencer", R,
      "| money recovered | Rs 134,227 | **Rs 224,672** | Rs 274,110 |",
      "headline", "policies.sequencer.money_recovered_per_100", 224_672, TOL_RUPEE,
      unit="Rs"),
    C("hl.per100.money.oracle", R,
      "| money recovered | Rs 134,227 | **Rs 224,672** | Rs 274,110 |",
      "headline", "policies.oracle_ceiling.money_recovered_per_100", 274_110, TOL_RUPEE,
      unit="Rs"),
    C("hl.per100.net.legacy", R,
      "| net value | -Rs 2,380,987 | **-Rs 764,492** | -Rs 295,182 |",
      "headline", "policies.legacy_fixed.net_value_per_100", -2_380_987, TOL_RUPEE,
      unit="Rs"),
    C("hl.per100.net.sequencer", R,
      "| net value | -Rs 2,380,987 | **-Rs 764,492** | -Rs 295,182 |",
      "headline", "policies.sequencer.net_value_per_100", -764_492, TOL_RUPEE, unit="Rs"),
    C("hl.per100.net.oracle", R,
      "| net value | -Rs 2,380,987 | **-Rs 764,492** | -Rs 295,182 |",
      "headline", "policies.oracle_ceiling.net_value_per_100", -295_182, TOL_RUPEE,
      unit="Rs"),

    C("hl.net_delta", R,
      "sequencer is **+Rs 1,381,295 per seed** of net value against the fixed",
      "headline", "net_value_delta_vs_legacy", 1_381_295, TOL_RUPEE, unit="Rs"),

    C("hl.headroom.rate.per_seed", R,
      "| mean of per-seed ratios | 70.5% | **62.3%** |",
      "headline", "headroom.by_rate", 70.5, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("hl.headroom.money.per_seed", R,
      "| mean of per-seed ratios | 70.5% | **62.3%** |",
      "headline", "headroom.by_money", 62.3, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("hl.headroom.rate.pooled", R,
      "| ratio of across-seed means | 70.7% | **64.7%** |",
      "headline", "headroom.by_rate_pooled", 70.7, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%",
      note="must equal the 70.7% run_ablations.py prints; the two definitions "
           "reconcile here"),
    C("hl.headroom.money.pooled", R,
      "| ratio of across-seed means | 70.7% | **64.7%** |",
      "headline", "headroom.by_money_pooled", 64.7, TOL_PCT_1DP, FRACTION_TO_PERCENT,
      "%"),
    C("hl.headroom.gap.pooled", R,
      "rate headroom pooled, 8.2 points",
      "headline", "headroom.difference_points_pooled", -6.0, TOL_PCT_1DP, unit="pp"),
    C("hl.headroom.gap.per_seed", R,
      "rate headroom pooled, 8.2 points",
      "headline", "headroom.difference_points", -8.2, TOL_PCT_1DP, unit="pp"),
    C("hl.total_failures", R,
      "**What these rupees are.** 20 seeds x 500 synthetic customers, 1,709 failed",
      "headline", "total_failures", 1_709, TOL_COUNT),

    # ------------------------------------ novelty N3: the cost of the notice --
    # Added after the per-seed binding fix. Both of these tables were computed
    # by scripts that shared one SequencerPolicy across twenty seeds, and both
    # were stale against the five-class book besides - neither was pinned here,
    # which is exactly how they stayed wrong. Pinning them is the fix that
    # generalises.
    C("n3.0h.recovery", R, "| 0h | 0 | 59.39% | 3,822,717 |",
      "leadtime", "notice_periods.0h.recovery_rate", 59.39, TOL_PCT_2DP,
      FRACTION_TO_PERCENT, "%"),
    C("n3.0h.money", R, "| 0h | 0 | 59.39% | 3,822,717 |",
      "leadtime", "notice_periods.0h.money", 3_822_717, TOL_RUPEE, unit="Rs"),
    C("n3.24h.recovery", R,
      "| **24h (legal minimum)** | 1 | **59.45%** | **3,839,648** |",
      "leadtime", "notice_periods.24h.recovery_rate", 59.45, TOL_PCT_2DP,
      FRACTION_TO_PERCENT, "%"),
    C("n3.24h.money", R,
      "| **24h (legal minimum)** | 1 | **59.45%** | **3,839,648** |",
      "leadtime", "notice_periods.24h.money", 3_839_648, TOL_RUPEE, unit="Rs"),
    C("n3.48h.money", R, "| 48h | 2 | 59.45% | 3,857,840 |",
      "leadtime", "notice_periods.48h.money", 3_857_840, TOL_RUPEE, unit="Rs"),
    C("n3.timing.per_hundred", R,
      "Removing the rule entirely is worth **-Rs 991 per 100 failures and -0.06 points",
      "leadtime", "timing_cost.per_hundred_failures", -991, TOL_RUPEE, unit="Rs"),
    C("n3.timing.points", R,
      "Removing the rule entirely is worth **-Rs 991 per 100 failures and -0.06 points",
      "leadtime", "timing_cost.recovery_points", -0.06, TOL_PCT_2DP, unit="pp"),
    C("n3.optout.per_hundred", R,
      "**The mandated opt-out costs Rs 6,540 per 100 failures** (1.29 points of",
      "leadtime", "optout_cost.per_hundred_failures", 6_540, TOL_RUPEE, unit="Rs"),
    C("n3.optout.points", R,
      "**The mandated opt-out costs Rs 6,540 per 100 failures** (1.29 points of",
      "leadtime", "optout_cost.recovery_points", 1.29, TOL_PCT_2DP, unit="pp"),
    C("n3.optout.money", R,
      "recovery, Rs 111,773 across 1,709 failures), measured against a counterfactual",
      "leadtime", "optout_cost.money", 111_773, TOL_RUPEE, unit="Rs"),

    # ---------------------------------------------------- prior robustness --
    C("prior.correct.recovery", R, "| correct | 59.3% | | 2.92 days | 36.1% |",
      "prior", "priors.correct.recovery_rate", 59.3, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("prior.correct.error", R, "| correct | 59.3% | | 2.92 days | 36.1% |",
      "prior", "priors.correct.mean_abs_error_days", 2.92, 0.005, unit="days"),
    C("prior.correct.exact", R, "| correct | 59.3% | | 2.92 days | 36.1% |",
      "prior", "priors.correct.exact", 36.1, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("prior.flat.recovery", R,
      "| flat (uniform 1-28) | 58.7% | -0.6 pp | 5.52 days | 15.5% |",
      "prior", "priors.flat.recovery_rate", 58.7, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("prior.flat.error", R,
      "| flat (uniform 1-28) | 58.7% | -0.6 pp | 5.52 days | 15.5% |",
      "prior", "priors.flat.mean_abs_error_days", 5.52, 0.005, unit="days"),
    C("prior.shifted.recovery", R,
      "| shifted (+5 days) | 59.4% | +0.1 pp | 4.72 days | 6.9% |",
      "prior", "priors.shifted.recovery_rate", 59.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("prior.shifted.exact", R,
      "| shifted (+5 days) | 59.4% | +0.1 pp | 4.72 days | 6.9% |",
      "prior", "priors.shifted.exact", 6.9, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("prior.us_style.recovery", R,
      "| us_style (1st/15th) | 59.1% | -0.3 pp | 4.68 days | 24.9% |",
      "prior", "priors.us_style.recovery_rate", 59.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("prior.retained", R,
      "**97% of the advantage survives the worst wrong prior** - `flat`, at 58.7%",
      "prior", "advantage_retained", 97, 0.5, FRACTION_TO_PERCENT, "%"),
    C("prior.floor", R, "against 59.3%, over a 36.1% fixed-schedule floor.",
      "prior", "legacy_floor", 36.1, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),

    # ------------------------------------------- failure class 6 (masked) --
    # Measured at --share 0.35. The generator default is 0.0, so none of these
    # numbers exist on the book every other claim here is measured on - which
    # is exactly why the flag defaults off.
    C("c6.masked_failures", R,
      "unexplained (28.6%)**:",
      "ambiguous", "masked_failures", 591, TOL_COUNT),
    C("c6.book_failures", R,
      "20 seeds x 500 customers, share 0.35 — **591 of 2,063 failures come back",
      "ambiguous", "book_failures", 2_063, TOL_COUNT),
    C("c6.masked_share", R, "unexplained (28.6%)**:",
      "ambiguous", "masked_share", 28.6, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("c6.posterior_accuracy", R,
      "still names the true class behind a *masked* 05 correctly **79.3% of the time**",
      "ambiguous", "posterior_top1_accuracy", 79.3, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("c6.posterior_scored", R,
      "(138/174).",
      "ambiguous", "posterior_scored", 174, TOL_COUNT),
    C("c6.value_at_risk", R,
      "Value at risk across 591 failures: **Rs 1,461,571**.",
      "ambiguous", "value_at_risk", 1_461_571, TOL_RUPEE, unit="Rs"),
    C("c6.money.legacy", R,
      "| legacy fixed | 791,343 | 54.1% | 71.4% | -7,426,553 | 963 |",
      "ambiguous", "policies.legacy_fixed.money", 791_343, TOL_RUPEE, unit="Rs"),
    C("c6.rate.legacy", R,
      "| legacy fixed | 791,343 | 54.1% | 71.4% | -7,426,553 | 963 |",
      "ambiguous", "policies.legacy_fixed.recovery_rate", 71.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("c6.money.sequencer", R,
      "| **sequencer** | **1,040,373** | **71.2%** | **80.9%** | **-2,256,868** | 517 |",
      "ambiguous", "policies.sequencer.money", 1_040_373, TOL_RUPEE, unit="Rs"),
    C("c6.share.sequencer", R,
      "| **sequencer** | **1,040,373** | **71.2%** | **80.9%** | **-2,256,868** | 517 |",
      "ambiguous", "policies.sequencer.value_recovered_share", 71.2, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("c6.net.sequencer", R,
      "| **sequencer** | **1,040,373** | **71.2%** | **80.9%** | **-2,256,868** | 517 |",
      "ambiguous", "policies.sequencer.net_value", -2_256_868, TOL_RUPEE, unit="Rs"),
    C("c6.money.blind", R,
      "| `no_masking_posterior` | 1,045,200 | 71.5% | 81.7% | -2,162,173 | 735 |",
      "ambiguous", "policies.no_masking_posterior.money", 1_045_200, TOL_RUPEE,
      unit="Rs"),
    C("c6.money.oracle", R,
      "| oracle ceiling | 1,215,108 | 83.1% | 84.3% | -835,045 | 276 |",
      "ambiguous", "policies.oracle_ceiling.money", 1_215_108, TOL_RUPEE, unit="Rs"),
    C("c6.posterior_delta_money", R,
      "necessarily does — the posterior costs **Rs 4,827 of recovery and Rs 94,695 of",
      "ambiguous", "posterior_delta_money", -4_827, TOL_RUPEE, unit="Rs"),
    C("c6.posterior_delta_net", R,
      "necessarily does — the posterior costs **Rs 4,827 of recovery and Rs 94,695 of",
      "ambiguous", "posterior_delta_net_value", -94_695, TOL_RUPEE, unit="Rs"),

    # ------------------------------------------- the generator's own shape --
    # These are not results, and that is the point. No policy figure moves when
    # the oracle horizon changes, so the results checker above sailed past a
    # README that described a 60-day window over a 400-day generator for two
    # failure classes. A number nothing produces is a number nothing protects.
    C("shape.history_days", R,
      "as_of - 365d ......................... as_of ................ as_of + 400d",
      "shape", "history_days", 365, TOL_COUNT, unit="days"),
    C("shape.horizon_days", R,
      "as_of - 365d ......................... as_of ................ as_of + 400d",
      "shape", "future_horizon_days", 400, TOL_COUNT, unit="days"),
    C("shape.series_days", R,
      "**765 days of daily balance in total** per customer: 365 of observed history",
      "shape", "series_days", 765, TOL_COUNT, unit="days"),
    C("shape.history_days.prose", R,
      "**765 days of daily balance in total** per customer: 365 of observed history",
      "shape", "history_days", 365, TOL_COUNT, unit="days"),
    C("shape.horizon_days.prose", R,
      "plus a 400-day oracle horizon. The horizon is long because the oracle has to be",
      "shape", "future_horizon_days", 400, TOL_COUNT, unit="days"),
    C("shape.max_cycles", R,
      "the first — `MAX_EPISODE_CYCLES` is 12, so a policy that holds all year runs off",
      "shape", "max_episode_cycles", 12, TOL_COUNT, unit="cycles"),
    C("shape.balance_series_horizon", R,
      "- **`balance_series`** — the hidden daily truth, running 400 days past `as_of`",
      "shape", "future_horizon_days", 400, TOL_COUNT, unit="days"),
    C("shape.seed42.failure_rate", R,
      "print(ds.natural_failure_rate)                     # 0.16",
      "shape", "seed_42.natural_failure_rate", 0.16, 0.005),
    C("shape.seed42.n_customers", R,
      "ds = generate_dataset(n_customers=500, seed=42)   # fully reproducible",
      "shape", "seed_42.n_customers", 500, TOL_COUNT),
    C("shape.as_of", R,
      "Anchored on `as_of` (default `2026-08-01`), the moment the retry decision is",
      "shape", "as_of_yyyymmdd", 20260801, TOL_COUNT,
      note="as an int, because the comparator is numeric"),

    # -------------------------------------------------- the suite's own size --
    C("tests.count", R, "pytest                      # 298 tests",
      "tests", "selected", 298, TOL_COUNT, unit="tests"),

    # --------------------------------------------- the 20-seed dataset table --
    C("ds.failure_rate.mean", R,
      "| natural failure rate (current cycle) | **17.1 %** | 13.2 % | 20.4 % |",
      "dataset", "natural_failure_rate.mean", 17.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("ds.failure_rate.min", R,
      "| natural failure rate (current cycle) | **17.1 %** | 13.2 % | 20.4 % |",
      "dataset", "natural_failure_rate.min", 13.2, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("ds.failure_rate.max", R,
      "| natural failure rate (current cycle) | **17.1 %** | 13.2 % | 20.4 % |",
      "dataset", "natural_failure_rate.max", 20.4, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("ds.legacy_recovery.mean", R,
      "| legacy day-2/4/7 recovery | **38.3 %** | 26.0 % | 50.0 % |",
      "dataset", "legacy_recovery_rate.mean", 38.3, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("ds.legacy_recovery.min", R,
      "| legacy day-2/4/7 recovery | **38.3 %** | 26.0 % | 50.0 % |",
      "dataset", "legacy_recovery_rate.min", 26.0, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("ds.unrecoverable.mean", R,
      "| unrecoverable within 30 days | **29.6 %** | 20.7 % | 41.6 % |",
      "dataset", "unrecoverable_share.mean", 29.6, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("ds.days_to_payday.mean", R,
      "| median days from failure to next payday | **11.1** | 9 | 13 |",
      "dataset", "median_days_to_payday.mean", 11.1, TOL_1DP, unit="days"),
    C("ds.beyond_day_7.mean", R,
      "| failures landing beyond day 7 | **69.8 %** | 60.9 % | 80.0 % |",
      "dataset", "share_failing_beyond_legacy_reach.mean", 69.8, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("ds.downtime_share.mean", R,
      "| downtime share of failures (code 91), mean of per-seed shares |",
      "dataset", "downtime_share.mean", 8.8, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%",
      note="mean of the per-seed shares; the 8.9% in the failure mix is the "
           "pooled ratio and is a different statistic"),
    C("ds.downtime_share.min", R,
      "| downtime share of failures (code 91), mean of per-seed shares |",
      "dataset", "downtime_share.min", 1.2, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ds.downtime_share.max", R,
      "| downtime share of failures (code 91), mean of per-seed shares |",
      "dataset", "downtime_share.max", 21.9, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ds.downtime_share.body", R,
      "**Realised share: 8.8% of failures carry code 91** — the **mean of the twenty",
      "dataset", "downtime_share.mean", 8.8, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ds.downtime_share.in_band", R,
      "20 land inside the 8-12% target band.",
      "dataset", "downtime_share.seeds_in_8_12_band", 4, TOL_COUNT, unit="seeds"),
    C("ctx.downtime_share.mean", X,
      "per-seed shares is 8.8% of failures carrying code 91 (the pooled ratio",
      "dataset", "downtime_share.mean", 8.8, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.downtime_share.pooled", X,
      "is 8.9%), but individual seeds run from 1.2% to 21.9% and only 4 of 20",
      "downtime", "downtime_share_pooled", 8.9, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.downtime_share.min", X,
      "is 8.9%), but individual seeds run from 1.2% to 21.9% and only 4 of 20",
      "dataset", "downtime_share.min", 1.2, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.downtime_share.max", X,
      "is 8.9%), but individual seeds run from 1.2% to 21.9% and only 4 of 20",
      "dataset", "downtime_share.max", 21.9, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.downtime_share.in_band", X,
      "is 8.9%), but individual seeds run from 1.2% to 21.9% and only 4 of 20",
      "dataset", "downtime_share.seeds_in_8_12_band", 4, TOL_COUNT, unit="seeds"),
    C("ds.afa_share.mean", R,
      "| AFA share of failures (code 1A) | **5.7 %** | 2.2 % | 12.1 % |",
      "dataset", "afa_share.mean", 5.7, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ds.afa_value_share.mean", R,
      "| AFA share of failures **by value** | **27.8 %** | 12.8 % | 48.7 % |",
      "dataset", "afa_value_share.mean", 27.8, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.failure_rate", X,
      "17.1% across 20 seeds, because AFA is a genuinely new way to fail rather",
      "dataset", "natural_failure_rate.mean", 17.1, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    C("ctx.legacy_recovery", X, "(37.5% -> 38.3%) and unrecoverable share.",
      "dataset", "legacy_recovery_rate.mean", 38.3, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
    # Pooled, not the per-seed mean - the sentence now says which, because the
    # two differ in the first decimal and the file quotes both.
    C("ctx.afa_share.pooled", X,
      "4. The AFA class is THIN by episode count - 5.6% of failures pooled across",
      "afa", "book.afa_count_share", 5.6, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("ctx.afa_share.mean", X,
      "20 seeds (5.7% as a mean of the per-seed shares),",
      "dataset", "afa_share.mean", 5.7, TOL_PCT_1DP, FRACTION_TO_PERCENT, "%"),
    C("readme.still_open.legacy_recovery", R,
      "- **Legacy recovery is 38.3 % mean but ranges 26.0–50.0 % across seeds.**",
      "dataset", "legacy_recovery_rate.mean", 38.3, TOL_PCT_1DP,
      FRACTION_TO_PERCENT, "%"),
]


# ------------------------------------------------------------------ plumbing --
def _hash_sources(digest: "hashlib._Hash", paths: Iterable[Path]) -> None:
    for path in sorted(paths):
        if "__pycache__" in path.parts or not path.exists():
            continue
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())


def fingerprint(producer: Producer | None = None) -> str:
    """Hash everything that can move THIS producer's numbers.

    Per producer rather than repository-wide, and that is not a micro-
    optimisation: a whole-repo hash meant that editing one row of the registry
    invalidated all six cached runs and cost ten minutes to re-answer a question
    nothing had changed the answer to. A tool people will not wait for is a tool
    people will not run.

    What is hashed: everything under ``src`` (any of it can move any number),
    the producing script itself, and ``_emit.py`` which shapes its payload. For
    a library producer it is the source of the producing function instead, plus
    anything named in ``extra_sources``. Documents are deliberately NOT hashed -
    a producer does not read them, so rewording a paragraph must not force a
    re-run.

    ``extra_sources`` exists because the first version of this got it wrong in
    exactly the way the whole tool is meant to prevent. The ``tests`` producer
    counts the suite, so its answer moves whenever a test is added or removed -
    but tests/ was not hashed, so the cache key did not move with it and a
    stale count would have been served as a passing claim. A checker that
    cannot invalidate its own cache is a checker that certifies whatever it
    happened to measure first.
    """
    digest = hashlib.sha256()
    digest.update(f"{SEEDS}x{N_CUSTOMERS}".encode())
    _hash_sources(digest, (ROOT / "src").rglob("*.py"))
    for folder in getattr(producer, "extra_sources", ()) if producer else ():
        _hash_sources(digest, (ROOT / folder).rglob("*.py"))
    if producer is None:
        _hash_sources(digest, (ROOT / "scripts").glob("*.py"))
    elif producer.script is not None:
        _hash_sources(
            digest, [ROOT / "scripts" / producer.script, ROOT / "scripts" / "_emit.py"]
        )
    else:
        digest.update(inspect.getsource(producer.call).encode())
    return digest.hexdigest()[:16]


def load_cached(name: str, stamp: str) -> dict[str, Any] | None:
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return blob["payload"] if blob.get("fingerprint") == stamp else None


def store_cached(name: str, stamp: str, payload: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{name}.json").write_text(
        json.dumps({"fingerprint": stamp, "payload": payload}, indent=2, default=str),
        encoding="utf-8",
    )


class Missing(Exception):
    """The producer has no such field. Loud on purpose."""


def dig(payload: dict[str, Any], path: str) -> Any:
    """Walk a dotted path. ``\\.`` escapes a dot inside one key (``0\\.01``)."""
    node: Any = payload
    parts: list[str] = []
    current = ""
    index = 0
    while index < len(path):
        char = path[index]
        if char == "\\" and index + 1 < len(path) and path[index + 1] == ".":
            current += "."
            index += 2
            continue
        if char == ".":
            parts.append(current)
            current = ""
        else:
            current += char
        index += 1
    parts.append(current)

    for part in parts:
        if not isinstance(node, dict) or part not in node:
            available = sorted(node)[:12] if isinstance(node, dict) else type(node).__name__
            raise Missing(f"no field {path!r} (stuck at {part!r}; have {available})")
        node = node[part]
    return node


def find_lines(doc_text: str, anchor: str) -> list[int]:
    return [
        number
        for number, line in enumerate(doc_text.splitlines(), start=1)
        if anchor in line
    ]


@dataclass
class Result:
    claim: Claim
    status: str              # "ok" | "MISMATCH" | "NOT FOUND" | "NO FIELD"
    lines: list[int] = field(default_factory=list)
    actual: float | None = None
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status != "ok"


def check(claims: Iterable[Claim], payloads: dict[str, dict[str, Any]],
          docs: dict[str, str]) -> list[Result]:
    results: list[Result] = []
    for claim in claims:
        lines = find_lines(docs[claim.doc], claim.anchor)
        if not lines:
            results.append(
                Result(claim, "NOT FOUND", [], None,
                       "the anchor text is not in the document any more - the "
                       "claim was reworded or removed, so this registry row is "
                       "stale")
            )
            continue
        try:
            raw = dig(payloads[claim.producer], claim.path)
        except Missing as exc:
            results.append(Result(claim, "NO FIELD", lines, None, str(exc)))
            continue
        if raw is None:
            results.append(
                Result(claim, "NO FIELD", lines, None,
                       f"{claim.path} is null in this run")
            )
            continue
        actual = float(raw) * claim.scale
        # The tolerance is half of the last printed digit, so a correctly
        # rounded figure lands EXACTLY on it - 30.95 documented as "30.9" is
        # off by precisely 0.05. Without the epsilon, binary floating point
        # decides those cases at the fifteenth decimal and the tool reports a
        # false mismatch on a number that is written down correctly.
        if abs(actual - claim.expected) <= claim.tol + 1e-9:
            results.append(Result(claim, "ok", lines, actual))
        else:
            results.append(
                Result(claim, "MISMATCH", lines, actual,
                       f"delta {actual - claim.expected:+,.4g}")
            )
    return results


def render(results: list[Result]) -> str:
    lines: list[str] = []
    failures = [r for r in results if r.failed]

    lines.append(RULE)
    lines.append("DOCUMENTED NUMBERS THAT NO LONGER REPRODUCE")
    lines.append(RULE)
    if not failures:
        lines.append(f"  none. all {len(results)} claims check out.")
        return "\n".join(lines)

    for result in failures:
        claim = result.claim
        where = ", ".join(f"{claim.doc}:{n}" for n in result.lines) or claim.doc
        lines.append("")
        lines.append(f"  {result.status}  {where}   [{claim.id}]")
        lines.append(f"      claim     {claim.anchor.strip()}")
        lines.append(f"      producer  {claim.producer} -> {claim.path}")
        if result.actual is None:
            lines.append(f"      problem   {result.detail}")
        else:
            unit = f" {claim.unit}" if claim.unit else ""
            lines.append(f"      documents {claim.expected:,.4g}{unit}")
            lines.append(
                f"      measured  {result.actual:,.4g}{unit}   "
                f"({result.detail}, tolerance +-{claim.tol:g})"
            )
        if claim.note:
            lines.append(f"      note      {claim.note}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", action="append", default=None,
                        choices=sorted(PRODUCERS),
                        help="check only claims from this producer (repeatable)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-run every producer, ignoring the cache")
    parser.add_argument("--list", action="store_true",
                        help="print the registry and exit without running anything")
    args = parser.parse_args()

    claims = [c for c in CLAIMS if not args.only or c.producer in args.only]

    if args.list:
        print(f"{RULE}\nREGISTRY: {len(claims)} claims\n{RULE}")
        for name, producer in PRODUCERS.items():
            rows = [c for c in claims if c.producer == name]
            if not rows:
                continue
            source = producer.script or "library producer"
            print(f"\n  {name}  ({source}) - {producer.description}")
            for claim in rows:
                print(f"    {claim.id:<34} {claim.doc:<11} {claim.path}")
        return 0

    needed = sorted({c.producer for c in claims})
    payloads: dict[str, dict[str, Any]] = {}

    print(f"{RULE}\nVERIFYING {len(claims)} DOCUMENTED NUMBERS "
          f"({SEEDS} seeds x {N_CUSTOMERS} customers)\n{RULE}")
    for name in needed:
        producer = PRODUCERS[name]
        stamp = fingerprint(producer)
        cached = None if args.refresh else load_cached(name, stamp)
        if cached is not None:
            print(f"  {name:<12} cached      ({stamp})")
            payloads[name] = cached
            continue
        started = time.time()
        print(f"  {name:<12} running {producer.script or 'library producer'} ...",
              end="", flush=True)
        payloads[name] = producer.run()
        store_cached(name, stamp, payloads[name])
        print(f" {time.time() - started:.0f}s")

    docs = {
        name: (ROOT / name).read_text(encoding="utf-8")
        for name in {c.doc for c in claims}
    }
    results = check(claims, payloads, docs)
    print()
    print(render(results))

    failures = [r for r in results if r.failed]
    print()
    print(RULE)
    print(f"  {len(results) - len(failures)} of {len(results)} claims reproduce; "
          f"{len(failures)} do not")
    print(RULE)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
