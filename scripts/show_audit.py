"""Print the full audit trail for one customer.

    python scripts/show_audit.py --customer CUST_0013 --seed 42
    python scripts/show_audit.py --policy legacy --style json
    python scripts/show_audit.py --pick hold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The trail uses box-drawing characters; the Windows console defaults to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audit.trail import render_episode  # noqa: E402
from evaluation.simulator import run_episode  # noqa: E402
from generator.generate import generate_dataset  # noqa: E402
from agent.sequencer import SequencerPolicy  # noqa: E402
from policy import LegacyFixedSchedule  # noqa: E402


def build_policy(name: str):
    if name == "sequencer":
        return SequencerPolicy()
    if name == "legacy":
        return LegacyFixedSchedule()
    raise SystemExit(f"unknown policy {name!r}")


def play(policy, dataset, event, seed):
    return run_episode(
        policy,
        dataset.customer(event.customer_id),
        event,
        dataset.balance_of(event.customer_id),
        dataset.payment_history,
        seed=seed,
        strict_compliance=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer", type=str, default=None)
    parser.add_argument("--policy", choices=("sequencer", "legacy"), default="sequencer")
    parser.add_argument("--style", choices=("text", "markdown", "json"), default="text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--pick", choices=("recovered", "hold", "optout", "cancel"),
                        default=None, help="pick the first episode with this outcome")
    args = parser.parse_args()

    dataset = generate_dataset(n_customers=args.n, seed=args.seed)
    policy = build_policy(args.policy)
    other = build_policy("legacy" if args.policy == "sequencer" else "sequencer")
    events = {e.customer_id: e for e in dataset.failure_events}

    if args.customer:
        if args.customer not in events:
            raise SystemExit(
                f"{args.customer} has no failure in seed {args.seed}. "
                f"Try: {', '.join(list(events)[:8])}"
            )
        chosen = events[args.customer]
    elif args.pick:
        wanted = {"recovered": "RECOVERED", "hold": "HELD",
                  "optout": "OPTED_OUT", "cancel": "CANCELLED"}[args.pick]
        chosen = next(
            (e for e in dataset.failure_events
             if play(policy, dataset, e, args.seed).outcome == wanted),
            None,
        )
        if chosen is None:
            raise SystemExit(f"no {wanted} episode in seed {args.seed}")
    else:
        chosen = dataset.failure_events[0]

    result = play(policy, dataset, chosen, args.seed)
    comparison = play(other, dataset, chosen, args.seed)
    print(render_episode(result, style=args.style, comparison=comparison))


if __name__ == "__main__":
    main()
