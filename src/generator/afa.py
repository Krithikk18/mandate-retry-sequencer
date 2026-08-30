"""Additional Factor of Authentication: the hidden truth behind failure class 5.

RBI Sec 8(a) exempts recurring debits from AFA only **up to Rs 15,000 per
transaction**. Above that ceiling every debit needs its own authentication, so
a presentment on a large mandate is refused with code ``1A`` unless the
customer has completed the OTP challenge for *that* debit.

Three properties make this class different from the four before it.

**It is per-presentment, not once-and-done.** The wording is "per transaction".
A customer who authenticated last cycle must authenticate again this cycle, and
again for each retry inside a cycle. That is the structural difference from
``expired_or_reissued_card``: the mandate is not dead, it is *gated*, and the
gate closes again behind every attempt.

**The gate sits in front of the account.** An unauthenticated debit is refused
before anybody reads the balance, so it returns 1A and never 04 - whatever the
balance happened to be. ``generator.generate._resolve_attempt`` encodes that
ordering, and the oracle mirrors it.

**It is the first condition a policy can CHANGE.** A balance is a fact and a
bank outage is a fact; both can be estimated but neither can be moved. AFA
completion is a customer action, and an out-of-band nudge ("tap the link to
authorise your subscription") causally raises the chance it happens. That is
why this class needs simulator support none of the others did - and why
``otp_responsiveness`` is a trait of the PERSON rather than a property of the
attempt.

The draw is deterministic in ``Customer.balance_seed`` and the presentment
date, the same way the balance process is, so the truth is reproducible without
threading a global RNG through the oracle. Crucially the *uniform* is keyed on
identity alone and NOT on the nudge count: nudging raises the threshold the
same draw is compared against, never the draw itself. That is the common
random numbers discipline ``simulator.optout_rng`` already follows, and it is
what makes "would this have completed anyway?" an exact counterfactual rather
than an estimate.
"""

from __future__ import annotations

import hashlib
from datetime import date

from .models import Customer

# The ceiling a nudge can drive completion toward. Never 1.0: no number of
# messages makes a human certain to act, and a model that says otherwise would
# let the agent buy recovery with dunning. ESTIMATE.
MAX_NUDGED_COMPLETION: float = 0.95

# Share of the remaining gap to that ceiling that one nudge closes, BEFORE
# scaling by the customer's own responsiveness. Diminishing by construction:
# the second nudge works on what the first left behind, so it is worth
# materially less, and the sequence converges rather than accumulating.
# ESTIMATE - there is no published figure for OTP-completion lift from a
# dunning message, and this is not the contribution of the project.
NUDGE_GAP_CLOSED: float = 0.45

# How many nudges a policy is credited with when the question is "could ANY
# compliant policy have recovered this?" - the ceiling and the recoverability
# classification, never the generated data.
#
# Two, because the third is not worth sending: at NUDGE_GAP_CLOSED the third
# nudge closes under a sixth of what the first did while costing the same
# goodwill and, unlike the first, a day of scheduling. Capping the ceiling here
# rather than at "unlimited" keeps it ATTAINABLE - a ceiling computed against
# an action nobody would rationally take is not a ceiling, it is a fiction that
# makes every real policy look worse than it is.
MAX_USEFUL_NUDGES: int = 2


def completion_probability(responsiveness: float, nudges: int = 0) -> float:
    """P(this customer completes AFA for one presentment, after ``nudges``.

    ``responsiveness`` is both the unprompted completion rate AND the rate at
    which nudging works, which is one assumption rather than two: somebody who
    ignores authentication prompts ignores the reminder about them as well.
    Scaling the lift that way keeps a nudge from rescuing a chronically
    unresponsive customer, which a flat lift would wrongly let it do.

        p(0) = responsiveness
        p(n) = responsiveness + gap * (1 - (1 - g*responsiveness) ** n)

    where ``gap`` is the room left below MAX_NUDGED_COMPLETION. A customer who
    always completes has no gap, so nudging them does nothing - correctly, and
    without a special case.
    """
    if nudges < 0:
        raise ValueError("nudges must be >= 0")
    responsiveness = min(max(responsiveness, 0.0), 1.0)
    gap = max(0.0, MAX_NUDGED_COMPLETION - responsiveness)
    if gap == 0.0 or nudges == 0:
        return responsiveness
    per_nudge = NUDGE_GAP_CLOSED * responsiveness
    return responsiveness + gap * (1.0 - (1.0 - per_nudge) ** nudges)


def completion_draw(customer: Customer, day: date) -> float:
    """The uniform this customer's presentment on ``day`` is judged against.

    Keyed on identity and date ONLY. Two policies that reach the same date on
    the same customer draw the same number, so any difference in whether AFA
    completed is a difference in the probability they created by nudging - not
    RNG alignment. Same discipline as ``simulator.optout_rng``.
    """
    key = f"{customer.balance_seed}|afa|{day.isoformat()}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def afa_required(customer: Customer) -> bool:
    """True when this mandate's debits are above the Sec 8(a) ceiling.

    Reads ``rules.requires_otp`` through the model property rather than
    comparing against 15,000 here, so the threshold lives in exactly one place.
    """
    return customer.requires_otp


def otp_completed(customer: Customer, day: date, nudges: int = 0) -> bool:
    """Ground truth: had this customer authenticated a debit on ``day``?

    Vacuously True below the Sec 8(a) ceiling - there is nothing to complete,
    so the gate is open and the balance decides. That is what lets every caller
    apply this unconditionally instead of branching on the amount.
    """
    if not afa_required(customer):
        return True
    return completion_draw(customer, day) < completion_probability(
        customer.otp_responsiveness, nudges
    )
