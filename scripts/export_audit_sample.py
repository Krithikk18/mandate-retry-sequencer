"""Render a deliberately mixed set of episodes to docs/audit_samples.md.

    python scripts/export_audit_sample.py

The selection is the point. A sample chosen to show the agent winning is
marketing, not evidence, so this picks by CATEGORY and includes the cases that
do not flatter it - notably episodes where the fixed schedule beat the
sequencer outright, and where the sequencer gave up on a mandate that could
still have been recovered.

If a category has no example in the seeds scanned, the file says so rather than
quietly shipping a shorter list.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.sequencer import SequencerPolicy  # noqa: E402
from audit.trail import render_episode  # noqa: E402
from evaluation.simulator import EpisodeResult, run_episode  # noqa: E402
from generator import rules  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from generator.oracle import first_success_date  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402


@dataclass
class Case:
    """One category of episode worth showing, and why it is worth showing."""

    key: str
    title: str
    why: str
    match: Callable[[EpisodeResult, EpisodeResult, bool], bool]


CASES: list[Case] = [
    Case(
        "recovery",
        "A clean recovery",
        "The ordinary case: infer the payday, price the dates, present once on "
        "the best one.",
        lambda agent, legacy, salvageable: agent.outcome == "RECOVERED"
        and agent.notifications_sent <= 2,
    ),
    Case(
        "beats_legacy",
        "Where the sequencer earns its keep",
        "The fixed schedule spent its whole budget and lost the mandate; the "
        "sequencer waited for payday.",
        lambda agent, legacy, salvageable: agent.outcome == "RECOVERED"
        and legacy.outcome in ("CANCELLED", "OPTED_OUT"),
    ),
    Case(
        "legacy_wins",
        "Where the fixed schedule beat the sequencer",
        "This is here on purpose. A sample where the agent only ever wins is "
        "curated, and a reviewer should assume the worst of it.",
        lambda agent, legacy, salvageable: legacy.net_value > agent.net_value + 1.0,
    ),
    Case(
        "hold",
        "A HOLD: nothing this cycle is worth presenting on",
        "No date in the remaining cycle clears the money equation, so the "
        "mandate is carried rather than burned.",
        lambda agent, legacy, salvageable: agent.outcome == "HELD"
        and agent.cycles_held > 0,
    ),
    Case(
        "correct_cancel",
        "A correct CANCEL on a customer whose income stopped",
        "The oracle confirms no date could ever have cleared. Giving up is the "
        "right answer, and the stopping rule found it.",
        lambda agent, legacy, salvageable: agent.outcome == "CANCELLED"
        and not salvageable,
    ),
    Case(
        "wrong_cancel",
        "A CANCEL that was wrong",
        "The oracle says this mandate was still recoverable. The stopping rule "
        "gave up on a live customer - the failure mode worth watching.",
        lambda agent, legacy, salvageable: agent.outcome == "CANCELLED"
        and salvageable,
    ),
    Case(
        "optout",
        "An opt-out on the pre-debit notification",
        "RBI Sec 6(c) in action: the notice offered a way out and the customer "
        "took it. The whole annual LTV goes with it.",
        lambda agent, legacy, salvageable: agent.outcome == "OPTED_OUT",
    ),
    Case(
        "churn",
        "A held mandate that drifted away",
        "Holding is not free. This customer was carried for months and left of "
        "their own accord before the money ever arrived.",
        lambda agent, legacy, salvageable: agent.outcome == "CHURNED",
    ),
    Case(
        "afa_nudge",
        "An authentication gate, and the one action that opens it",
        "The only class where the agent can CHANGE the success probability "
        "rather than pick the best point on a fixed curve. Watch P(clear) rise "
        "as the nudges land, then the presentment go in behind them - and note "
        "that no notification was sent to do it, because a nudge is out of "
        "band and carries no Sec 6(c) opt-out.",
        lambda agent, legacy, salvageable: agent.failure_class
        == rules.FAILURE_CLASS_AFA_INCOMPLETE
        and agent.nudges_sent > 0
        and agent.outcome == "RECOVERED",
    ),
    Case(
        "nach",
        "A NACH mandate, where a failed retry costs the customer Rs 500",
        "Same equation, different rail. The bounce term raises the bar a date "
        "has to clear before it is worth presenting on.",
        lambda agent, legacy, salvageable: agent.customer is not None
        and agent.customer.rail == "nach"
        and agent.outcome == "RECOVERED",
    ),
    Case(
        "long_wait",
        "A long wait for a distant payday",
        "The failure lands just after payday, so the next opportunity is weeks "
        "out. A fixed schedule cannot reach it at all.",
        lambda agent, legacy, salvageable: agent.outcome == "RECOVERED"
        and agent.days_to_resolution >= 14,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--out", type=str, default="docs/audit_samples.md")
    args = parser.parse_args()

    # Rebuilt inside the seed loop below - see there.
    legacy = LegacyFixedSchedule()
    found: dict[str, tuple[EpisodeResult, EpisodeResult, int]] = {}

    for seed in range(1, args.seeds + 1):
        dataset = generate_dataset(n_customers=args.n, seed=seed)
        # Per seed: the sequencer pools bank evidence across customers sharing
        # an issuer, and a customer id means a different person in every
        # dataset. One instance across the scanned seeds would render episodes
        # decided partly on another book's outages.
        sequencer = SequencerPolicy.for_dataset(dataset)
        for event in dataset.failure_events:
            if len(found) == len(CASES):
                break
            args_ = (
                dataset.customer(event.customer_id),
                event,
                dataset.balance_of(event.customer_id),
                dataset.payment_history,
            )
            agent = run_episode(sequencer, *args_, seed=seed, strict_compliance=False)
            fixed = run_episode(legacy, *args_, seed=seed, strict_compliance=False)
            salvageable = (
                first_success_date(
                    dataset.customer(event.customer_id),
                    dataset.balance_of(event.customer_id),
                    dataset.customer(event.customer_id).subscription_amount,
                    after=event.original_debit_date,
                    horizon_days=30,
                )
                is not None
            )
            for case in CASES:
                if case.key not in found and case.match(agent, fixed, salvageable):
                    found[case.key] = (agent, fixed, seed)
                    break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Audit samples",
        "",
        "Rendered by `scripts/export_audit_sample.py`. Every number below is",
        "read from what the policy recorded at decision time - the renderer",
        "does not recompute anything.",
        "",
        "These are picked by CATEGORY, not by outcome. Two of them are cases",
        "where the agent did worse than the fixed schedule or gave up on a",
        "recoverable mandate; a sample where the agent only wins would not be",
        "worth reading.",
        "",
        "| # | case | outcome | agent net | legacy net |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, case in enumerate(CASES, start=1):
        if case.key not in found:
            lines.append(f"| {index} | {case.title} | *not found in sample* | | |")
            continue
        agent, fixed, _ = found[case.key]
        lines.append(
            f"| {index} | {case.title} | {agent.outcome} | "
            f"Rs {agent.net_value:,.0f} | Rs {fixed.net_value:,.0f} |"
        )
    lines.append("")

    for index, case in enumerate(CASES, start=1):
        lines.append(f"## {index}. {case.title}")
        lines.append("")
        lines.append(case.why)
        lines.append("")
        if case.key not in found:
            lines.append(
                f"> No episode matching this category appeared in "
                f"{args.seeds} seeds x {args.n} customers. Reported rather "
                f"than silently omitted."
            )
            lines.append("")
            continue
        agent, fixed, seed = found[case.key]
        lines.append(f"`--customer {agent.customer_id} --seed {seed}`")
        lines.append("")
        lines.append(render_episode(agent, style="markdown", comparison=fixed))
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    missing = [c.key for c in CASES if c.key not in found]
    print(f"wrote {out} with {len(found)}/{len(CASES)} cases")
    if missing:
        print(f"not found in this sample: {', '.join(missing)}")


if __name__ == "__main__":
    main()
