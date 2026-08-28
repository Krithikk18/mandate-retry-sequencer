"""The opt-out (cancellation) hazard attached to pre-debit notifications.

Under the RBI e-mandate framework every debit attempt must be preceded by a
notification at least 24 hours earlier, and that notification must offer the
customer a way out. So each retry is not free: scheduling one obliges a fresh,
regulator-mandated invitation to cancel the subscription.

Who sends it matters, and getting this wrong signals a misreading of the
payment flow: RBI places the notification duty on the ISSUER - the customer's
bank or UPI app - not on the merchant or its PSP. The sequencer schedules the
presentment; the issuer notifies and handles the opt-out. The sequencer cannot
suppress, soften or reword the message. It can only decide whether to trigger
one, which is precisely why the attrition has to be priced into the decision
rather than managed away.

This is the tension the sequencer has to price. Retrying more often raises the
chance of catching the customer with money, and simultaneously raises the
chance of losing their lifetime value entirely.

The hazard is shaped by three things:

annoyance
    Each extra notification inside the same billing cycle multiplies the rate
    (default 1.4x per extra notice). The second reminder in a week reads very
    differently from the first.

tenure
    Long-tenure customers are stickier. The rate is scaled by
    1 / (1 + tenure_months / TENURE_SCALE_MONTHS), so a 4-year subscriber is
    roughly five times less likely to cancel than a brand-new one.

recency
    How irritation moves with time since the bounce. THE DIRECTION IS AN
    ASSUMPTION, NOT A FINDING, and it is the single modelling choice most
    capable of inverting the project's conclusion, so it is a switch rather
    than a constant. See ``IrritationDirection`` below.

The result is capped at OPTOUT_CEILING: no single notification cancels a
quarter of a book.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .models import Customer

ANNOYANCE_MULTIPLIER: float = 1.4  # per extra notification in the same cycle
TENURE_SCALE_MONTHS: float = 12.0
RECENCY_WEIGHT: float = 0.5  # size of the time-dependent swing, either way
RECENCY_DECAY_DAYS: float = 7.0
OPTOUT_CEILING: float = 0.25

# "decay"  - frustration is freshest right after the bounce and fades. A notice
#            the next morning rides on it; one a fortnight later does not.
#            Makes WAITING cheap and HAMMERING expensive.
# "build"  - resentment accumulates while the debt drags on unresolved. A
#            notice weeks later lands on someone already tired of the merchant.
#            Makes waiting expensive and prompt resolution cheap.
#
# Both are plausible and neither is measured. Every headline result must be
# reported under both directions; a conclusion that survives only one of them
# is a conclusion about this constant, not about retry policy.
IrritationDirection = Literal["decay", "build"]
DEFAULT_IRRITATION_DIRECTION: IrritationDirection = "decay"


def recency_multiplier(
    days_since_first_failure: float,
    direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
) -> float:
    """Time-dependent component of the hazard, in [1.0, 1.0 + RECENCY_WEIGHT].

    Both directions share endpoints and scale, so flipping ``direction`` is a
    clean ablation: it changes only which end of the time axis is expensive.
    """
    fade = math.exp(-days_since_first_failure / RECENCY_DECAY_DAYS)
    if direction == "decay":
        return 1.0 + RECENCY_WEIGHT * fade
    if direction == "build":
        return 1.0 + RECENCY_WEIGHT * (1.0 - fade)
    raise ValueError(f"direction must be 'decay' or 'build', got {direction!r}")


def optout_probability(
    customer: Customer,
    notifications_this_cycle: int,
    days_since_first_failure: float,
    direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
) -> float:
    """Chance this customer cancels in response to the current notification.

    ``notifications_this_cycle`` counts the notification being sent now, so 1
    means "first notice of the cycle". Zero notifications means no cancel
    opportunity was created, hence zero probability.
    """
    if notifications_this_cycle <= 0:
        return 0.0
    if days_since_first_failure < 0:
        raise ValueError("days_since_first_failure must be >= 0")

    annoyance = ANNOYANCE_MULTIPLIER ** (notifications_this_cycle - 1)
    tenure_factor = 1.0 / (1.0 + customer.tenure_months / TENURE_SCALE_MONTHS)
    recency = recency_multiplier(days_since_first_failure, direction)

    p = customer.optout_sensitivity * annoyance * tenure_factor * recency
    return float(min(p, OPTOUT_CEILING))


def draw_optout(
    customer: Customer,
    notifications_this_cycle: int,
    days_since_first_failure: float,
    rng: np.random.Generator,
    direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
) -> bool:
    """Sample whether the customer actually cancels. Consumes one draw."""
    p = optout_probability(
        customer, notifications_this_cycle, days_since_first_failure, direction
    )
    return bool(rng.random() < p)


def survival_probability(
    customer: Customer,
    notification_offsets: list[float] | tuple[float, ...],
    direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
) -> float:
    """Chance the customer survives a whole planned retry schedule.

    ``notification_offsets`` are days after the first failure at which each
    notification lands, in order. Useful for scoring a candidate schedule
    without simulating it.
    """
    survival = 1.0
    for i, offset in enumerate(notification_offsets, start=1):
        survival *= 1.0 - optout_probability(customer, i, offset, direction)
    return survival
