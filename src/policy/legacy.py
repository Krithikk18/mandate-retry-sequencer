"""The fixed-schedule baseline: retry on day +2, +4, +7, then cancel.

These offsets are the published industry convention, not a strawman. Merchant
dunning documentation and the default retry ladders shipped by subscription
billing platforms cluster on exactly this shape - a couple of days, then a
couple more, then about a week - and it is what most Indian merchants running
e-mandates actually do. The point of the comparison is not that the baseline is
stupid; it is that a schedule fixed in *retry index* cannot align with an event
that moves in *calendar time*, namely payday.

The baseline plays by identical rules to every other policy:

* at most ``rules.MAX_ATTEMPTS_PER_CYCLE`` presentments per cycle,
* every retry at least 24 hours after the decision (RBI Sec 6(a)),
* every presentment inside the execution window,
* offsets that fall outside the billing cycle are skipped, not clamped.

Crippling it would invalidate the comparison, so it gets the same compliance
treatment and the same audit trail as the agent.
"""

from __future__ import annotations

from datetime import timedelta

from generator import rules

from .base import Policy, PolicyState, RetryDecision

# Retry on the 2nd, 4th and 7th day after the original failure. Three retries,
# which with the original presentment is exactly the NPCI ceiling of four.
LEGACY_RETRY_OFFSETS: tuple[int, ...] = (2, 4, 7)


class LegacyFixedSchedule(Policy):
    """Retry at fixed offsets regardless of the customer. Cancel when spent."""

    name = "legacy_fixed"

    def __init__(self, offsets: tuple[int, ...] = LEGACY_RETRY_OFFSETS) -> None:
        if len(offsets) + 1 > rules.MAX_ATTEMPTS_PER_CYCLE:
            raise ValueError(
                f"{len(offsets)} retries plus the original exceeds the NPCI "
                f"ceiling of {rules.MAX_ATTEMPTS_PER_CYCLE} attempts per cycle"
            )
        self.offsets = offsets

    def decide(self, state: PolicyState) -> RetryDecision:
        # attempts_used counts the original failed presentment, so the number
        # of retries already made is one less.
        retries_made = max(0, state.attempts_used - 1)
        original = state.failure_event.original_debit_date
        earliest = state.earliest_compliant_retry_date

        remaining = self.offsets[retries_made:]
        skipped: list[dict[str, object]] = []

        for offset in remaining:
            candidate = original + timedelta(days=offset)
            if state.attempts_remaining <= 0:
                break
            if candidate < earliest:
                # The 24-hour notice window has already closed on this offset.
                skipped.append({"offset": offset, "reason": "lead_time_missed"})
                continue
            if candidate > state.cycle_end_date:
                skipped.append({"offset": offset, "reason": "outside_cycle"})
                continue
            return RetryDecision(
                action="RETRY",
                retry_date=candidate,
                reasoning={
                    "policy": self.name,
                    "rule": "fixed offset from original failure",
                    "offset_days": offset,
                    "retries_made": retries_made,
                    "attempts_remaining": state.attempts_remaining,
                    "skipped_offsets": skipped,
                    "payday_considered": False,
                    "optout_cost_considered": False,
                },
            )

        return RetryDecision(
            action="CANCEL",
            reasoning={
                "policy": self.name,
                "rule": "schedule exhausted",
                "retries_made": retries_made,
                "offsets": list(self.offsets),
                "skipped_offsets": skipped,
                "attempts_remaining": state.attempts_remaining,
            },
        )
