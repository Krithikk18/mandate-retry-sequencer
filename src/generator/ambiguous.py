"""Ambiguous decline: the hidden truth behind failure class 6.

A ``05`` — "do not honour" — **is not a failure mode**. It is a MASK over one
of the others. The issuer declined and gave no reason, but there is always a
reason: the customer was broke, the credential was dead, the bank hiccuped, the
risk engine said no. The return code hides which.

**This is the first class where the FAILURE CLASS ITSELF is unobserved.**
Classes 1, 2 and 5 hide a PARAMETER behind a code that at least tells you which
parameter to estimate — a payday, an outage window, a responsiveness trait. The
code says "you are looking for a balance" and the agent goes and looks for one.
Class 6 hides the QUESTION. The agent does not know what to estimate, so before
it can price a date it has to decide what kind of failure it is looking at.

That is why the handler is not a sixth success model but a POSTERIOR over the
other five (see ``classes/ambiguous.py``). The taxonomy closes on itself.

Two things live in this module, and they are deliberately different shapes.

**The latent cause** is per-failure. Behind each 05 is one of the real reasons,
drawn from ``LATENT_CAUSE_MIXTURE``.

**The masking propensity is PER BANK**, drawn once per issuer and never per
customer. Some acquirers pass through a meaningful decline reason; others
return 05 for everything. That is a property of the institution, not of the
person, and it is the whole reason the agent's posterior is a CROSS-CUSTOMER
POOLED inference shaped like ``agent/bank_health.py`` rather than a
per-customer one shaped like ``agent/otp_completion.py``:

    otp_completion.py   whether YOU authenticate says nothing about anyone
                        else, so there is nothing to pool and the model is
                        fitted per customer
    bank_health.py      an outage is one event across every mandate on the
                        issuer, so evidence pools and a mandate that never
                        failed still inherits a sharp estimate
    masking.py          how often an ISSUER hides its reasons is learned from
                        every 05 it ever returned, across all its customers -
                        the same shape as bank_health, for the same reason

**issuer_risk_block is a new kind of dead.** It is permanent — no date ever
clears — but it has no entry in ``rules.NEVER_RETRY_CLASSES`` and it is never
announced as dead, because announcing it is exactly what the issuer declines to
do. A 54 tells the merchant to stop; a 05 over a risk block tells it nothing,
so a fixed schedule keeps presenting into a wall it will never see. That is
what makes this cause expensive rather than merely unrecoverable, and it is why
it cannot reuse ``Customer.mandate_dead_class``, whose validator admits only
the two classes that ARE announced.

Determinism follows ``afa.completion_draw`` exactly: blake2b over
``balance_seed | tag | date``, keyed on identity and date ONLY, never on any
policy action. Two policies reaching the same customer on the same date draw
the same number, so counterfactuals stay exact and common random numbers hold.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta

from . import rules
from .models import Customer

# --- the latent causes -------------------------------------------------------
# Two of these have no unmasked counterpart anywhere else in the generator:
# a risk block and a transient issuer glitch are only ever seen as a 05.
CAUSE_INSUFFICIENT_FUNDS: str = rules.FAILURE_CLASS_INSUFFICIENT_FUNDS
CAUSE_EXPIRED_CARD: str = rules.FAILURE_CLASS_EXPIRED_CARD
CAUSE_DOWNTIME: str = rules.FAILURE_CLASS_DOWNTIME
CAUSE_ISSUER_RISK_BLOCK: str = "issuer_risk_block"
CAUSE_TRANSIENT_GLITCH: str = "transient_issuer_glitch"

# What is actually behind a 05, as a distribution. ESTIMATES, every one - no
# acquirer publishes the decomposition of its own do-not-honour traffic, which
# is precisely why the class is interesting. Justified individually:
#
#   insufficient_funds 0.50  The dominant cause of every other failure class is
#                            also the dominant cause here. NACH return-code data
#                            puts balance failures at 60-70% of returns; masked
#                            traffic is shaded lower because risk blocks are
#                            over-represented in the codes issuers decline to
#                            explain.
#   issuer_risk_block  0.25  Velocity rules, suspected fraud, an internal
#                            blacklist. Issuers deliberately do NOT explain
#                            these - explaining them tells a fraudster which
#                            rule fired - so they are concentrated in 05 far
#                            beyond their share of all declines.
#   transient_glitch   0.15  Sub-outage flakiness: one switch, one route, a few
#                            hours. Too small to register as a bank outage in
#                            banks.py, too real to ignore, and it is the cause
#                            that makes a cautious retry worth anything at all.
#   expired_card       0.07  Some issuers return 05 rather than 54 on a dead
#                            credential. Low because most return the specific
#                            code, which is why class 3 exists separately.
#   downtime           0.03  Lowest because a real outage usually DOES surface
#                            as 91 - the acquirer knows the issuer is
#                            unreachable and says so.
LATENT_CAUSE_MIXTURE: dict[str, float] = {
    CAUSE_INSUFFICIENT_FUNDS: 0.50,
    CAUSE_ISSUER_RISK_BLOCK: 0.25,
    CAUSE_TRANSIENT_GLITCH: 0.15,
    CAUSE_EXPIRED_CARD: 0.07,
    CAUSE_DOWNTIME: 0.03,
}

assert abs(sum(LATENT_CAUSE_MIXTURE.values()) - 1.0) < 1e-9, (
    "the latent cause mixture must be a distribution"
)

#: Causes that only ever appear as a 05 - no unmasked counterpart exists.
NATIVE_CAUSES: frozenset[str] = frozenset(
    {CAUSE_ISSUER_RISK_BLOCK, CAUSE_TRANSIENT_GLITCH}
)

# --- per-bank masking propensity ---------------------------------------------
# The share of an issuer's failed presentments that come back as 05 instead of
# a meaningful code. Drawn once per bank from a Beta-shaped spread over this
# band, so the book contains both issuers that explain themselves and issuers
# that never do. ESTIMATES.
MASKING_PROPENSITY_MIN: float = 0.05
MASKING_PROPENSITY_MAX: float = 0.60

# How long a transient glitch lasts. Hours, not days: this is the cause that
# rewards a cautious retry, and if it lasted longer than the RBI notice period
# it would be indistinguishable from a bank outage - which class 2 already
# showed no compliant policy can react to. ESTIMATES.
GLITCH_MIN_HOURS: float = 24.0
GLITCH_MAX_HOURS: float = 72.0

# Risk blocks are placed uniformly across the observed history, so the book
# contains mandates blocked long ago and mandates blocked last month. Matches
# HISTORY_DAYS in generate.py; kept local so this module imports nothing from
# there and stays usable by the oracle.
RISK_BLOCK_WINDOW_DAYS: int = 365


@dataclass(frozen=True)
class AmbiguityConfig:
    """How much of the book's decline traffic is masked, and by whom.

    ``share`` scales every bank's masking propensity. **It defaults to 0.0 and
    the default must stay 0.0**: at zero this module is inert, every resolution
    path is byte-identical to the build before class 6 existed, and every
    number already documented still reproduces. That is the same convention
    ``DecisionEngine.daily_discount`` follows, and for the same reason - a new
    mechanism that silently moves every published figure is not a feature.
    """

    share: float = 0.0
    seed: int = 0
    #: First day of the simulated window. Risk blocks are placed relative to it
    #: so the flag is a property of the book rather than of a magic constant;
    #: the generator sets it from ``series_start``.
    window_start: date | None = None

    @property
    def enabled(self) -> bool:
        return self.share > 0.0


def _draw(customer: Customer, tag: str, day: date) -> float:
    """A uniform keyed on identity and date only. See the module docstring."""
    key = f"{customer.balance_seed}|{tag}|{day.isoformat()}".encode()
    return int.from_bytes(
        hashlib.blake2b(key, digest_size=8).digest(), "big"
    ) / float(1 << 64)


def _bank_draw(bank_id: str, seed: int, tag: str) -> float:
    """A uniform keyed on the ISSUER, drawn once and stable for the book.

    Same construction as ``banks._bank_seed``: the propensity is a property of
    the institution across the whole window, not something redrawn per attempt.
    """
    key = f"{bank_id}|{seed}|{tag}".encode()
    return int.from_bytes(
        hashlib.blake2b(key, digest_size=8).digest(), "big"
    ) / float(1 << 64)


def masking_propensity(bank_id: str, config: AmbiguityConfig) -> float:
    """P(this issuer returns 05 instead of the real code), for one failure.

    Per bank and constant across customers, which is the property the agent's
    pooled posterior depends on: an issuer that hides its reasons hides them
    for everybody, so evidence from one mandate transfers to the next.
    """
    if not config.enabled:
        return 0.0
    spread = _bank_draw(bank_id, config.seed, "mask")
    base = MASKING_PROPENSITY_MIN + spread * (
        MASKING_PROPENSITY_MAX - MASKING_PROPENSITY_MIN
    )
    return min(1.0, base * config.share)


def is_masked(customer: Customer, day: date, config: AmbiguityConfig) -> bool:
    """Would this issuer have hidden the reason for a failure on ``day``?

    Called only on presentments that ALREADY failed - masking rewrites the code
    on a decline, it never causes one. See ``oracle.resolve_presentment``.
    """
    if not config.enabled:
        return False
    return _draw(customer, "mask", day) < masking_propensity(
        customer.bank_id, config
    )


def latent_cause(customer: Customer, day: date, config: AmbiguityConfig) -> str:
    """Which real reason sits behind a NATIVE 05 on ``day``.

    Only consulted for failures this module itself creates. A masked 04 already
    has a true cause - the balance - and does not draw from the mixture; that
    is what keeps the fitted posterior in ``agent/masking.py`` an inference
    about the book rather than a readback of this constant.
    """
    draw = _draw(customer, "cause", day)
    cumulative = 0.0
    for cause, weight in LATENT_CAUSE_MIXTURE.items():
        cumulative += weight
        if draw < cumulative:
            return cause
    return CAUSE_INSUFFICIENT_FUNDS


def native_share() -> float:
    """Share of the mixture that has no unmasked counterpart.

    These are the causes that must be GENERATED rather than masked, because
    nothing else in the generator produces them.
    """
    return sum(LATENT_CAUSE_MIXTURE[c] for c in NATIVE_CAUSES)


def risk_blocked_from(customer: Customer, config: AmbiguityConfig) -> date | None:
    """The date this mandate was silently risk-blocked, if it ever was.

    Permanent and never announced. Unlike ``mandate_dead_from`` there is no
    matching entry in ``NEVER_RETRY_CLASSES`` and no code that says "stop", so
    a schedule that does not model it presents until its attempt budget runs
    out, every cycle, forever.
    """
    if not config.enabled:
        return None
    # One draw per customer, on a fixed anchor date so the flag is a property
    # of the PERSON rather than of any particular attempt - a risk block does
    # not come and go.
    anchor = date(2000, 1, 1)
    if _draw(customer, "riskblock", anchor) >= (
        LATENT_CAUSE_MIXTURE[CAUSE_ISSUER_RISK_BLOCK]
        * masking_propensity(customer.bank_id, config)
    ):
        return None
    if config.window_start is None:
        # No window bound: treat the block as having always been in force.
        # Conservative in the right direction - it never invents a date the
        # oracle and the generator could disagree about.
        return date.min
    # Placed uniformly across the observed history so some mandates are blocked
    # well before the current cycle and some only recently.
    offset = int(_draw(customer, "riskblock_when", anchor) * RISK_BLOCK_WINDOW_DAYS)
    return config.window_start + timedelta(days=offset)


def glitch_hours(customer: Customer, day: date) -> float:
    """How long a transient glitch beginning on ``day`` lasts."""
    return GLITCH_MIN_HOURS + _draw(customer, "glitch_len", day) * (
        GLITCH_MAX_HOURS - GLITCH_MIN_HOURS
    )


def in_glitch(customer: Customer, day: date, config: AmbiguityConfig) -> bool:
    """Is this customer's issuer glitching for them on ``day``?

    Deliberately per (customer, day) rather than per bank-hour: a transient
    glitch is one route or one token failing, not the issuer going down. The
    bank-wide event is ``banks.BankOutageBook`` and it is a different thing.
    """
    if not config.enabled:
        return False
    threshold = LATENT_CAUSE_MIXTURE[CAUSE_TRANSIENT_GLITCH] * masking_propensity(
        customer.bank_id, config
    )
    if _draw(customer, "glitch", day) >= threshold:
        return False
    return True


def glitch_clears_on(customer: Customer, day: date) -> date:
    """First date a glitch that started on ``day`` is over.

    Used by the oracle: a cautious retry after this date clears, one before it
    does not. That is the whole reason a single careful retry is the right
    response to an unexplained decline.
    """
    return day + timedelta(days=max(1, int(glitch_hours(customer, day) // 24) + 1))
