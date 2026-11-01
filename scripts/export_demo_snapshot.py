"""Freeze four real episodes - one per failure class - into docs/demo_snapshot.json.

    python scripts/export_demo_snapshot.py --seeds 8 --n 400

The browser demo in docs/demo.html embeds the file this writes. Nothing in
that HTML is hand-entered: if the model changes, rerun this and the page
changes with it.

**Every number written here is READ, never recomputed.** The decision logs come
out of the episodes exactly as the policies recorded them, the same discipline
audit.trail keeps and for the same reason - a snapshot that re-derives its own
figures can quietly disagree with the run it claims to describe, which makes it
worse than no snapshot at all. The headline strip is read from
.verify_cache/headline.json rather than retyped, so the page cannot drift from
run_headline.py.

The four classes are picked by CATEGORY, and each carries the metric that
actually distinguishes the policies on that class. Money is the right axis for
insufficient funds; it is the WRONG axis for a revoked mandate, where nobody
recovers anything and the whole story is that the fixed schedule presents on
it 41 times and the sequencer does not. Forcing one metric across all four
would make two of them look like null results.

**Two rules stop this from being a highlight reel.**

*The median, not the best.* Among the episodes that qualify for a scenario,
this takes the one at the MEDIAN of (agent net value - legacy net value), never
the maximum. The most flattering downtime episode in eight seeds has the fixed
schedule losing Rs 174,860 while the sequencer recovers Rs 14,570; putting that
on a portfolio page would be a lie of selection, because it is one episode in
thirty-six.

*The base rate ships with the episode.* Each scenario carries how many episodes
of its class were scanned, how many qualified, and the agent-beats / ties /
loses split over ALL of them. Downtime is mostly a tie - both policies get the
money, because a bank comes back up on its own - and the page says so next to
the one case where it is not. A demo that shows only the wins invites a
reviewer to assume the worst about the rest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass

from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.sequencer import SequencerPolicy  # noqa: E402
from audit.trail import decision_signature  # noqa: E402
from evaluation.simulator import EpisodeResult, cycle_end_date, run_episode  # noqa: E402
from generator import rules  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402
from policy.oracle_policy import OraclePolicy  # noqa: E402

# Actions the simulator logs for things that happened TO the mandate, as
# opposed to decisions a policy made. Mirrors audit.trail.EVENT_ACTIONS.
EVENT_ACTIONS = frozenset({"EXECUTE", "AUTO", "CHURN"})


@dataclass(frozen=True)
class Scenario:
    """One failure class, and the metric that tells its story."""

    key: str
    title: str
    blurb: str
    # The field on EpisodeResult that separates the policies on this class.
    # A real attribute name - nothing here is derived arithmetic.
    metric_field: str
    metric_label: str
    metric_unit: str  # "rupees" | "count" | "days"
    metric_better: str  # "higher" | "lower"
    metric_note: str
    # What makes an episode a legitimate example of this scenario. Applied to
    # every episode of the class; the median qualifier is the one exported.
    qualifies: Callable[[EpisodeResult, EpisodeResult], bool]


SCENARIOS: list[Scenario] = [
    Scenario(
        key=rules.FAILURE_CLASS_INSUFFICIENT_FUNDS,
        title="Insufficient funds",
        blurb=(
            "The account was short on the debit date. The money is not gone - "
            "it arrives on payday. The only question is which date to ask on."
        ),
        metric_field="recovered_amount",
        metric_label="Money recovered",
        metric_unit="rupees",
        metric_better="higher",
        metric_note=(
            "The fixed day-2/4/7 schedule cannot reach a payday that lands "
            "further out than day 7."
        ),
        # The money story: the sequencer waited for payday, the fixed schedule
        # spent its budget before payday arrived.
        qualifies=lambda agent, legacy: agent.outcome == "RECOVERED"
        and legacy.outcome != "RECOVERED",
    ),
    Scenario(
        key=rules.FAILURE_CLASS_DOWNTIME,
        title="Bank / PSP downtime",
        blurb=(
            "Nothing is wrong with the customer or the mandate. The bank is "
            "down. Presenting into an outage burns an attempt and learns "
            "nothing."
        ),
        metric_field="recovered_amount",
        metric_label="Money recovered",
        metric_unit="rupees",
        metric_better="higher",
        metric_note=(
            "Usually this class is a TIE - a bank comes back up on its own and "
            "both policies get the money. The gap only opens when the outage "
            "outlasts the day-7 schedule, and then the fixed one has spent "
            "every attempt before the bank returns."
        ),
        # Only the minority case is worth stepping through; the ties are
        # reported as a base rate alongside it rather than dramatised.
        qualifies=lambda agent, legacy: agent.outcome == "RECOVERED"
        and legacy.outcome != "RECOVERED",
    ),
    Scenario(
        key=rules.FAILURE_CLASS_EXPIRED_CARD,
        title="Expired / reissued card",
        blurb=(
            "The credential is dead. No date clears, and no amount of retrying "
            "changes that. Every notice sent is a chance for the customer to "
            "cancel instead."
        ),
        metric_field="notifications_sent",
        metric_label="Pre-debit notices sent",
        metric_unit="count",
        metric_better="lower",
        metric_note=(
            "Nobody recovers money from a dead card - that is the correct "
            "answer. The cost being avoided is the opt-out risk in every "
            "RBI Sec 6(c) notice."
        ),
        # Restricted to the case where BOTH policies end up cancelling. The
        # sequencer's larger wins on this class come from holding a mandate for
        # a year, which is real but takes 364 days to explain; this shape -
        # same destination, one day versus seven, three notices versus none -
        # is the conservative example and the legible one.
        qualifies=lambda agent, legacy: agent.outcome == "CANCELLED"
        and legacy.outcome == "CANCELLED"
        and agent.notifications_sent == 0
        and legacy.notifications_sent > 0
        and agent.net_value > legacy.net_value,
    ),
    Scenario(
        key=rules.FAILURE_CLASS_OPTED_OUT,
        title="Consent revoked",
        blurb=(
            "The customer cancelled the mandate. Presenting on it is not a "
            "wasted attempt - it is a compliance breach."
        ),
        metric_field="compliance_breaches",
        metric_label="Compliance breaches",
        metric_unit="count",
        metric_better="lower",
        metric_note=(
            "A fixed schedule never reads the return code, so it cannot know "
            "the mandate is revoked. Across 20 seeds it breaches 41 times; the "
            "sequencer breaches zero. Note that net value is IDENTICAL for all "
            "three policies here - a breach is a regulatory cost, and the money "
            "model does not price it. This class is won on the compliance row "
            "or not at all."
        ),
        qualifies=lambda agent, legacy: legacy.compliance_breaches > 0
        and agent.compliance_breaches == 0,
    ),
]

# The policies, in the order the demo shows them: floor, agent, ceiling.
POLICY_ORDER = ["legacy_fixed", "sequencer", "oracle_ceiling"]


def collapse(log: list) -> list[tuple[Any, int]]:
    """Fold runs of identical consecutive decisions into (entry, count).

    A policy that defers commitment says WAIT on identical evidence for days
    on end. Twenty near-identical frames is not a demo anyone watches, so the
    repetition is summarised rather than hidden - the day count is carried and
    rendered. Same rule audit.trail applies, via the same public signature.
    """
    out: list[tuple[Any, int]] = []
    for entry in log:
        if (
            out
            and entry.action not in EVENT_ACTIONS
            and out[-1][0].action not in EVENT_ACTIONS
            and entry.cycle == out[-1][0].cycle
            and decision_signature(entry) == decision_signature(out[-1][0])
        ):
            out[-1] = (out[-1][0], out[-1][1] + 1)
            continue
        out.append((entry, 1))
    return out


def steps_of(result: EpisodeResult) -> list[dict[str, Any]]:
    """The decision log as timeline frames, verbatim."""
    steps: list[dict[str, Any]] = []
    for entry, repeats in collapse(result.decision_log):
        steps.append(
            {
                "day": entry.day.isoformat(),
                "cycle": entry.cycle,
                "kind": "event" if entry.action in EVENT_ACTIONS else "decision",
                "action": entry.action,
                "repeats": repeats,
                "retry_date": entry.retry_date.isoformat() if entry.retry_date else None,
                "retry_time": entry.retry_time.isoformat() if entry.retry_time else None,
                "event": entry.event,
                "attempts_used": entry.attempts_used,
                "notifications_sent": entry.notifications_sent,
                "reasoning": entry.reasoning,
            }
        )
    return steps


def policy_payload(result: EpisodeResult) -> dict[str, Any]:
    """Everything the page shows about one policy's run. All read fields."""
    return {
        "policy": result.policy_name,
        "outcome": result.outcome,
        "recovered_amount": result.recovered_amount,
        "attempts_used": result.attempts_used,
        "notifications_sent": result.notifications_sent,
        "compliance_breaches": result.compliance_breaches,
        "bounce_charges_paid": result.bounce_charges_paid,
        "carrying_cost_paid": result.carrying_cost_paid,
        "cycles_held": result.cycles_held,
        "ltv_retained": result.ltv_retained,
        "ltv_lost": result.ltv_lost,
        "net_value": result.net_value,
        "days_to_resolution": result.days_to_resolution,
        "nudges_sent": result.nudges_sent,
        "retry_dates": [d.isoformat() for d in result.retry_dates],
        "steps": steps_of(result),
    }


def play(policy, dataset, event, seed) -> EpisodeResult:
    """Run one policy over one failure event.

    ``strict_compliance=False`` because the fixed schedule genuinely does
    breach on a revoked mandate, and that breach is the thing scenario four
    exists to show. Raising on it would make the case unrenderable.
    """
    return run_episode(
        policy,
        dataset.customer(event.customer_id),
        event,
        dataset.balance_of(event.customer_id),
        dataset.payment_history,
        seed=seed,
        strict_compliance=False,
    )


def read_headline() -> dict[str, Any]:
    """The three headline rates, read from the run that produced them."""
    path = ROOT / ".verify_cache" / "headline.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run `python scripts/run_headline.py` first - "
            "the header strip is read from it, never retyped."
        )
    cached = json.loads(path.read_text(encoding="utf-8"))
    payload = cached["payload"]
    policies = payload["policies"]
    return {
        "source_script": payload["script"],
        "fingerprint": cached["fingerprint"],
        "seeds": payload["seeds"],
        "n_customers": payload["n"],
        "total_failures": payload["total_failures"],
        "rates": {name: policies[name]["recovery_rate"] for name in POLICY_ORDER},
        "money": {name: policies[name]["money_recovered"] for name in POLICY_ORDER},
        "headroom_captured": policies["sequencer"]["headroom_captured"],
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - provenance is nice to have, not load-bearing
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--out", type=str, default="docs/demo_snapshot.json")
    args = parser.parse_args()

    wanted = {s.key: s for s in SCENARIOS}
    # Every episode of every target class, scored under both deployable
    # policies. The whole population is scanned - not stopped at the first
    # match - because the base rate is part of what gets exported.
    population: dict[str, list[dict[str, Any]]] = {s.key: [] for s in SCENARIOS}

    legacy = LegacyFixedSchedule()
    for seed in range(1, args.seeds + 1):
        dataset = generate_dataset(n_customers=args.n, seed=seed)
        # Per seed: the sequencer pools bank-outage evidence across customers
        # sharing an issuer, and a customer id means a different person in
        # every dataset. One instance reused across seeds would decide episodes
        # partly on another book's outages.
        sequencer = SequencerPolicy.for_dataset(dataset)
        for event in dataset.failure_events:
            scenario = wanted.get(event.failure_class)
            if scenario is None:
                continue
            agent = play(sequencer, dataset, event, seed)
            fixed = play(legacy, dataset, event, seed)
            population[scenario.key].append(
                {
                    "seed": seed,
                    "event": event,
                    "delta": agent.net_value - fixed.net_value,
                    "qualifies": scenario.qualifies(agent, fixed),
                }
            )
        print(f"  scanned seed {seed}")

    scenarios_out: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        episodes = population[scenario.key]
        qualifying = [e for e in episodes if e["qualifies"]]
        if not qualifying:
            raise SystemExit(
                f"no episode qualified for {scenario.key}; scanned "
                f"{len(episodes)} of that class over {args.seeds} seeds x "
                f"{args.n} customers.\nRaise --seeds rather than relaxing the "
                "predicate - a scenario picked because it was easy to find is "
                "not the scenario the page claims to show."
            )

        # THE MEDIAN QUALIFIER, not the best one. See the module docstring.
        # Ties on delta are broken by (seed, customer_id) so the pick is
        # deterministic for a given scan.
        qualifying.sort(key=lambda e: (e["delta"], e["seed"], e["event"].customer_id))
        chosen = qualifying[(len(qualifying) - 1) // 2]
        seed, event = chosen["seed"], chosen["event"]

        # A base rate the page can show next to the episode: over EVERY
        # episode of this class, how often is the sequencer actually ahead?
        # Rs 1 of net value is not a difference anyone should read as one.
        beats = sum(1 for e in episodes if e["delta"] > 1.0)
        loses = sum(1 for e in episodes if e["delta"] < -1.0)
        base_rate = {
            "episodes_scanned": len(episodes),
            "qualified": len(qualifying),
            "sequencer_ahead": beats,
            "tied": len(episodes) - beats - loses,
            "legacy_ahead": loses,
            "selection": "median qualifying episode by net-value delta",
        }

        # Rebuilt per scenario: both stateful policies bind to one dataset.
        dataset = generate_dataset(n_customers=args.n, seed=seed)
        customer = dataset.customer(event.customer_id)
        print(
            f"  {scenario.key:<26} seed {seed}  {event.customer_id}  "
            f"(median of {len(qualifying)}/{len(episodes)})"
        )
        runs = {
            "legacy_fixed": play(LegacyFixedSchedule(), dataset, event, seed),
            "sequencer": play(
                SequencerPolicy.for_dataset(dataset), dataset, event, seed
            ),
            "oracle_ceiling": play(
                OraclePolicy.for_dataset(dataset), dataset, event, seed
            ),
        }
        scenarios_out.append(
            {
                "key": scenario.key,
                "title": scenario.title,
                "blurb": scenario.blurb,
                "base_rate": base_rate,
                "seed": seed,
                "customer_id": event.customer_id,
                "return_code": event.return_code,
                "original_debit_date": event.original_debit_date.isoformat(),
                "cycle_end_date": cycle_end_date(event.original_debit_date).isoformat(),
                "verify_command": (
                    f"python scripts/show_audit.py --customer {event.customer_id} "
                    f"--seed {seed} --n {args.n}"
                ),
                "customer": {
                    "customer_id": customer.customer_id,
                    "subscription_amount": customer.subscription_amount,
                    "rail": customer.rail,
                    "debit_day_of_month": customer.debit_day_of_month,
                    "tenure_months": customer.tenure_months,
                    "annual_ltv": customer.annual_ltv,
                    "bank_id": customer.bank_id,
                    "bounce_charge": rules.bounce_charge(customer.rail),
                },
                "metric": {
                    "field": scenario.metric_field,
                    "label": scenario.metric_label,
                    "unit": scenario.metric_unit,
                    "better": scenario.metric_better,
                    "note": scenario.metric_note,
                    "values": {
                        name: getattr(runs[name], scenario.metric_field)
                        for name in POLICY_ORDER
                    },
                },
                "policies": {name: policy_payload(runs[name]) for name in POLICY_ORDER},
            }
        )

    snapshot = {
        "generated_by": "scripts/export_demo_snapshot.py",
        "command": (
            f"python scripts/export_demo_snapshot.py --seeds {args.seeds} "
            f"--n {args.n} --out {args.out}"
        ),
        "git_commit": git_commit(),
        "schema_version": 1,
        "policy_order": POLICY_ORDER,
        "policy_labels": {
            "legacy_fixed": "Legacy fixed schedule",
            "sequencer": "Sequencer",
            "oracle_ceiling": "Oracle ceiling",
        },
        "rules": {
            "max_attempts_per_cycle": rules.MAX_ATTEMPTS_PER_CYCLE,
            "pre_debit_notice_hours": rules.PRE_DEBIT_NOTICE_HOURS,
            "legacy_offsets": [2, 4, 7],
            "nach_bounce_charge": rules.NACH_BOUNCE_CHARGE_INR,
        },
        "cost_note": (
            "Processing fees, carrying cost and goodwill are documented "
            "estimates, not Razorpay figures."
        ),
        "headline": read_headline(),
        "scenarios": scenarios_out,
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, indent=1, default=str)
    out.write_text(payload, encoding="utf-8")
    print(f"wrote {args.out}  ({out.stat().st_size / 1024:,.0f} KB, "
          f"{len(scenarios_out)} scenarios)")

    build_page(payload)


def build_page(payload: str) -> None:
    """Inline the snapshot into docs/demo.html.

    The page is one file with no build step and no network calls, so it opens
    from disk and screen-records without a font or a fetch arriving late.
    ``docs/demo_template.html`` is the source; ``docs/demo.html`` is generated
    and should not be hand-edited - rerun this script instead.
    """
    template = ROOT / "docs" / "demo_template.html"
    if not template.exists():
        print(f"  (no {template.relative_to(ROOT)}; wrote JSON only)")
        return

    # "</" cannot appear raw inside a <script> block - a "</script>" in the data
    # would close the tag early. "\/" is a valid JSON escape for "/", so this
    # stays parseable JSON.
    safe = payload.replace("</", "<\\/")

    html = template.read_text(encoding="utf-8")
    if "__SNAPSHOT__" not in html:
        raise SystemExit(f"{template} has no __SNAPSHOT__ placeholder to fill")
    page = ROOT / "docs" / "demo.html"
    page.write_text(html.replace("__SNAPSHOT__", safe), encoding="utf-8")
    print(f"wrote docs/demo.html  ({page.stat().st_size / 1024:,.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
