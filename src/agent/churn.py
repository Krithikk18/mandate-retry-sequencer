"""The agent's own estimate of opt-out risk.

``generator.optout.optout_probability`` is the TRUE hazard, and it reads
``customer.optout_sensitivity`` - a hidden per-customer field. The agent cannot
call it. This module is the merchant-side stand-in: the same structural claims
(annoyance compounds, tenure protects, timing matters) driven only by fields a
PSP actually holds.

The one thing the agent cannot know is how twitchy a particular customer is, so
it substitutes the book-level average. That is exactly what a real risk model
does, and it is why the agent's opt-out term is an expectation over the
population rather than a per-customer truth.

Caveat worth stating in any write-up: the functional form here matches the
generator's. On real data the merchant would fit this from its own churn
history and the form would be somewhat wrong. Getting the shape right for free
flatters the agent, though the ablation in scripts/run_ablations.py measures
what the term is worth at all, which is the question that matters.
"""

from __future__ import annotations

import math
from typing import Literal

from generator.models import AgentCustomer

# Book-level mean opt-out rate per notification. A merchant reads this off its
# own cancellation data; it cannot read any individual's sensitivity.
# ESTIMATE, and the single most important number in the churn term.
ASSUMED_OPTOUT_SENSITIVITY: float = 0.03

ANNOYANCE_MULTIPLIER: float = 1.4  # per extra notification in the same cycle
TENURE_SCALE_MONTHS: float = 12.0
RECENCY_WEIGHT: float = 0.5
RECENCY_DECAY_DAYS: float = 7.0
OPTOUT_CEILING: float = 0.25

IrritationDirection = Literal["decay", "build"]
DEFAULT_IRRITATION_DIRECTION: IrritationDirection = "decay"


def recency_multiplier(
    days_since_failure: float,
    direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
) -> float:
    """Time-dependent component, in [1.0, 1.0 + RECENCY_WEIGHT].

    ``decay``: frustration is freshest right after the bounce and fades.
    ``build``: resentment accumulates while the debt drags on.
    Neither is measured. Both are run in the ablations.
    """
    fade = math.exp(-max(0.0, days_since_failure) / RECENCY_DECAY_DAYS)
    if direction == "decay":
        return 1.0 + RECENCY_WEIGHT * fade
    if direction == "build":
        return 1.0 + RECENCY_WEIGHT * (1.0 - fade)
    raise ValueError(f"direction must be 'decay' or 'build', got {direction!r}")


def estimated_optout_probability(
    customer: AgentCustomer,
    notifications_this_cycle: int,
    days_since_failure: float,
    direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
) -> float:
    """P(this notification triggers a cancellation), from public fields only.

    ``notifications_this_cycle`` counts the notice being sent now, so 1 means
    "first notice of the cycle". Zero means no cancel opportunity is created
    and the probability is zero.
    """
    if notifications_this_cycle <= 0:
        return 0.0

    annoyance = ANNOYANCE_MULTIPLIER ** (notifications_this_cycle - 1)
    tenure_factor = 1.0 / (1.0 + customer.tenure_months / TENURE_SCALE_MONTHS)
    recency = recency_multiplier(days_since_failure, direction)

    return float(
        min(ASSUMED_OPTOUT_SENSITIVITY * annoyance * tenure_factor * recency, OPTOUT_CEILING)
    )


def survival_probability(
    customer: AgentCustomer,
    notification_offsets: list[float] | tuple[float, ...],
    direction: IrritationDirection = DEFAULT_IRRITATION_DIRECTION,
) -> float:
    """Chance the mandate survives a whole planned schedule of notices."""
    survival = 1.0
    for index, offset in enumerate(notification_offsets, start=1):
        survival *= 1.0 - estimated_optout_probability(customer, index, offset, direction)
    return survival
