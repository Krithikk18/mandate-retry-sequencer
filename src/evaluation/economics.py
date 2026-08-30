"""The money equation.

    net_value = recovered_amount
              - attempts_used * processing_cost[rail]
              - bounce_charges_paid
              - cycles_held * subscription_amount * HOLD_CARRYING_COST_FRACTION
              - goodwill_cost_paid
              - ltv_lost

The asymmetry between HOLD and CANCEL is the whole point of having a HOLD
action:

    CANCEL  -> the mandate is dead. ltv_lost = annual_ltv.
    HOLD    -> nothing recovered this cycle, but the mandate survives and
               re-presents next cycle. ltv_lost = 0 - and a carrying cost is
               charged for every cycle it is carried.

For a customer whose income has genuinely stopped, no retry date can succeed.
A fixed schedule burns three presentments and then kills a mandate that was
structurally alive. HOLD keeps the relationship instead. But holding is NOT
free, and an earlier build made exactly that mistake: with ltv_lost the only
cost of ending a mandate, HOLD strictly dominated CANCEL and no stopping rule
could ever be correct, because there was never a price for being wrong.

Holding an unpaying subscriber now costs three things that compound per cycle:

1. ``HOLD_CARRYING_COST_FRACTION`` of the subscription price - the marginal
   cost of continuing to serve someone who is not paying.
2. A fresh presentment every cycle, since the mandate keeps auto-billing on its
   anchor date. On NACH each of those bounces and levies Rs 500 on the
   customer. This term dwarfs the other two on that rail.
3. ``NATURAL_ATTRITION_PER_CYCLE`` - held customers drift away on their own,
   and when they do the annual LTV is lost anyway, just later and after paying
   the carrying cost all the way there.

So HOLD buys time, and time has a price. Whether it is ever worth cancelling
instead is an empirical question about the size of that price relative to the
LTV; scripts/sensitivity_holding.py sweeps it.

``goodwill_cost_paid`` is the price of out-of-band nudges. A nudge is the one
action that CHANGES the customer's state rather than selecting a date, so it
has to cost something or the objective would buy recovery with unlimited
dunning. It is charged per nudge at ``rules.NUDGE_GOODWILL_COST_FRACTION`` of
annual LTV, and it is deliberately separate from the notification count: a
nudge creates no RBI Sec 6(c) opt-out opportunity, which is exactly why it is
the cheaper instrument and exactly why it still cannot be free.

Two accounting conventions, both deliberate:

1. ``bounce_charges_paid`` is a charge on the CUSTOMER, not on the merchant. It
   is subtracted from merchant net value anyway, because a sequencer that
   racks up Rs 500 bank charges on the people it is trying to retain is
   destroying the relationship it is being paid to protect. Internalising the
   externality is the only way the objective sees that.
2. ``attempts_used`` includes the original failed presentment. It is a constant
   offset across every policy on the same episode, so comparisons are
   unaffected, and the absolute figure then answers "what did this cycle cost
   to process" rather than "what did the retries cost".
"""

from __future__ import annotations

from generator import rules
from generator.models import AgentCustomer

# Outcomes in which the mandate is gone and its future value with it.
# CHURNED is natural attrition on a held customer: not the RBI opt-out, just
# somebody who stopped paying and eventually stopped caring.
MANDATE_KILLED_OUTCOMES: frozenset[str] = frozenset(
    {"OPTED_OUT", "CANCELLED", "CHURNED"}
)


def mandate_survives(outcome: str) -> bool:
    """True when the customer is still on the book after this episode."""
    return outcome not in MANDATE_KILLED_OUTCOMES


def processing_cost_total(rail: str, attempts_used: int) -> float:
    """Merchant-side cost of every presentment made this cycle."""
    if attempts_used < 0:
        raise ValueError("attempts_used must be >= 0")
    return rules.processing_cost(rail) * attempts_used


def carrying_cost_total(subscription_amount: float, cycles_held: int) -> float:
    """Cost of serving an unpaying subscriber across ``cycles_held`` cycles."""
    if cycles_held < 0:
        raise ValueError("cycles_held must be >= 0")
    return rules.hold_carrying_cost(subscription_amount) * cycles_held


def ltv_lost(customer: AgentCustomer, outcome: str) -> float:
    """Annual value destroyed by this episode. Zero unless the mandate died."""
    return customer.annual_ltv if not mandate_survives(outcome) else 0.0


def ltv_retained(customer: AgentCustomer, outcome: str) -> float:
    """Annual value still on the book after this episode."""
    return customer.annual_ltv if mandate_survives(outcome) else 0.0


def goodwill_cost_total(customer: AgentCustomer, nudges_sent: int) -> float:
    """Goodwill spent on ``nudges_sent`` out-of-band messages."""
    if nudges_sent < 0:
        raise ValueError("nudges_sent must be >= 0")
    return rules.NUDGE_GOODWILL_COST_FRACTION * customer.annual_ltv * nudges_sent


def net_value(
    customer: AgentCustomer,
    outcome: str,
    recovered_amount: float,
    attempts_used: int,
    bounce_charges_paid: float,
    cycles_held: int = 0,
    goodwill_cost_paid: float = 0.0,
) -> float:
    """Net rupee value of one episode to the merchant.

    ``goodwill_cost_paid`` defaults to zero so every existing call site keeps
    the number it had: a policy that never nudges is scored exactly as before.
    """
    return (
        recovered_amount
        - processing_cost_total(customer.rail, attempts_used)
        - bounce_charges_paid
        - carrying_cost_total(customer.subscription_amount, cycles_held)
        - goodwill_cost_paid
        - ltv_lost(customer, outcome)
    )


def cost_per_rupee_recovered(
    total_cost: float, money_recovered: float
) -> float | None:
    """Cost of every rupee actually collected. None when nothing was collected.

    Returning None rather than infinity keeps a policy that recovers nothing
    from silently looking cheap in an average.
    """
    if money_recovered <= 0:
        return None
    return total_cost / money_recovered
