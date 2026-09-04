"""Regulatory and network constants.

These are real Indian regulations, not tunable model parameters. They live in
one module so the generator, the agent, the baseline and the evaluator all read
the same numbers. Read from here; never hardcode a threshold at a call site.

Sources
-------
RBI Digital Payments E-mandate Framework, 2026
    Circular RBI/DPSS/2026-27/396, 21 April 2026.

    * Sec 6(a) - a pre-debit notification must reach the customer at least 24
      hours before EVERY debit attempt.
    * Sec 6(c) - that notification must carry an AFA-validated opt-out.
    * Sec 7    - a post-debit notification is also required.
    * Sec 8(a) - no AFA (OTP) required up to Rs 15,000 per transaction.
    * Sec 8(b) - that exemption rises to Rs 1,00,000 for insurance premia,
      mutual fund subscriptions and credit card bill payments.

    Note on who notifies: RBI places the notification duty on the ISSUER (the
    customer's bank or UPI app), not on the merchant or its PSP. A retry
    sequencer schedules the presentment; the issuer sends the notice and
    handles the opt-out. The sequencer must price the resulting attrition, but
    it does not send the message.

NPCI UPI Autopay
    * At most 4 debit attempts per billing cycle (1 original + 3 retries).
    * Execution restricted to the 00:00-07:00 IST non-peak window.

NACH
    * Return code "04" = Balance Insufficient.
    * A failed NACH debit levies a bank bounce charge on the CUSTOMER
      (~Rs 500, varies by bank). Failed UPI Autopay debits carry no such
      charge. Never blend these into a single per-attempt cost.
"""

from __future__ import annotations

from datetime import time

# --- RBI e-mandate -----------------------------------------------------------
PRE_DEBIT_NOTICE_HOURS: int = 24
POST_DEBIT_NOTICE_REQUIRED: bool = True  # Sec 7

# Sec 8(a) general ceiling, and the Sec 8(b) ceiling for the named categories.
OTP_EXEMPTION_THRESHOLD_INR: float = 15_000.0
OTP_EXEMPTION_THRESHOLD_HIGH_VALUE_INR: float = 1_00_000.0
HIGH_VALUE_MANDATE_CATEGORIES: frozenset[str] = frozenset(
    {"insurance", "mutual_fund", "credit_card_bill"}
)

# --- NPCI UPI Autopay --------------------------------------------------------
MAX_ATTEMPTS_PER_CYCLE: int = 4  # 1 original + 3 retries
UPI_AUTOPAY_WINDOW_START: time = time(0, 0)
UPI_AUTOPAY_WINDOW_END: time = time(7, 0)

# The hour a presentment is actually filed, inside the window above.
#
# This lives here rather than in policy/base.py because the GENERATOR needs it:
# a bank outage is an hour-resolution process, so resolving a presentment
# against it requires knowing what time of day the debit hits. generator/ must
# never import policy/ - policy already imports generator, and the reverse
# would be circular - so the constant belongs on the generator side and
# policy/base.py re-exports it.
DEFAULT_PRESENTMENT_TIME: time = time(2, 0)

# --- NACH --------------------------------------------------------------------
RETURN_CODE_INSUFFICIENT_FUNDS: str = "04"
NACH_BOUNCE_CHARGE_INR: float = 500.0

# Return codes for the other failure classes. These are REPRESENTATIVE of the
# NACH/ISO-8583 families rather than an exhaustive mapping - real acquirers see
# dozens of codes per class and normalise them upstream. What matters here is
# that the router dispatches on the code, as a production system would, rather
# than on a label the generator helpfully attached.
RETURN_CODE_EXPIRED_CARD: str = "54"  # ISO 8583: expired card
RETURN_CODE_MANDATE_REVOKED: str = "22"  # NACH: mandate cancelled by customer
RETURN_CODE_DOWNTIME: str = "91"  # ISO 8583: issuer or switch inoperative
RETURN_CODE_AFA_INCOMPLETE: str = "1A"  # ISO 8583: strong authentication required
RETURN_CODE_DO_NOT_HONOUR: str = "05"  # ISO 8583: no reason given

RAILS: tuple[str, ...] = ("upi_autopay", "nach")

# --- per-presentment processing cost -----------------------------------------
# ESTIMATES, not published figures. Razorpay does not publish per-mandate
# presentment economics, and the real number is contract-specific. They are
# here so no call site invents its own, and so a sensitivity run can move them
# in one place. Order of magnitude only: UPI Autopay presentments are cheap,
# NACH presentments carry higher per-item handling.
PROCESSING_COST_INR: dict[str, float] = {
    "upi_autopay": 5.0,
    "nach": 15.0,
}

# --- cost of carrying an unpaying subscriber ---------------------------------
# HOLD keeps a mandate alive through a cycle it did not pay for. That is not
# free: the customer keeps consuming the service. Content delivery, support,
# infrastructure and operational overhead all continue.
#
# ESTIMATE, not a published figure - it is a fraction of the subscription price
# standing in for marginal cost to serve. Swept in
# scripts/sensitivity_holding.py so no conclusion rests on the exact value.
HOLD_CARRYING_COST_FRACTION: float = 0.15

# Held customers also leave of their own accord. This is not the RBI opt-out
# (that fires on a pre-debit notification); it is ordinary churn by somebody
# who has stopped paying and eventually stops caring. Per held cycle.
NATURAL_ATTRITION_PER_CYCLE: float = 0.05

# --- cost of an out-of-band nudge --------------------------------------------
# A nudge is a message the merchant sends directly ("your payment failed, tap
# here to authorise it"). It is NOT the RBI pre-debit notification: it consumes
# no NPCI attempt and creates no Sec 6(c) opt-out opportunity, which is exactly
# why it is cheaper than presenting. It is not free either - goodwill is
# finite, and a sequencer that dunned its way to a good number would be selling
# something the merchant did not agree to spend.
#
# ESTIMATE, as a fraction of annual LTV per nudge. It lives here rather than in
# agent/decision_engine.py for the same reason DEFAULT_PRESENTMENT_TIME does:
# the EVALUATOR has to charge it, evaluation/ must not import agent/, and a
# second constant on the evaluation side would be a second knob for one number.
# agent/decision_engine.py re-exports this name so every existing import site
# keeps working.
NUDGE_GOODWILL_COST_FRACTION: float = 0.002

# How many billing cycles an episode may run before the evaluation stops.
# Twelve matches the horizon annual_ltv is denominated in, so the two sides of
# the money equation are measured over the same period.
MAX_EPISODE_CYCLES: int = 12

# --- failure taxonomy --------------------------------------------------------
# The six classes the sequencer must handle. They share one decision engine and
# differ only in their success model and permitted actions. All six have a
# handler; five are generated by default and `ambiguous_decline` is emitted only
# when `generate_dataset(ambiguity_share=...)` is non-zero. They are named here
# so the router and the class modules have a single canonical spelling to
# import.
FAILURE_CLASS_INSUFFICIENT_FUNDS: str = "insufficient_funds"
FAILURE_CLASS_DOWNTIME: str = "bank_or_psp_downtime"
FAILURE_CLASS_EXPIRED_CARD: str = "expired_or_reissued_card"
FAILURE_CLASS_OPTED_OUT: str = "customer_opted_out"
FAILURE_CLASS_AFA_INCOMPLETE: str = "afa_incomplete"
FAILURE_CLASS_AMBIGUOUS: str = "ambiguous_decline"

FAILURE_CLASSES: frozenset[str] = frozenset(
    {
        FAILURE_CLASS_INSUFFICIENT_FUNDS,
        FAILURE_CLASS_DOWNTIME,
        FAILURE_CLASS_EXPIRED_CARD,
        FAILURE_CLASS_OPTED_OUT,
        FAILURE_CLASS_AFA_INCOMPLETE,
        FAILURE_CLASS_AMBIGUOUS,
    }
)

# Two of the six are structurally dead: the mandate cannot succeed on any date,
# so a retry is pure cost. Fixed schedules retry them anyway, which is free
# recovery for a class-aware system.
NEVER_RETRY_CLASSES: frozenset[str] = frozenset(
    {FAILURE_CLASS_EXPIRED_CARD, FAILURE_CLASS_OPTED_OUT}
)

# Presenting against these is a REGULATORY BREACH, not merely a waste of an
# attempt. Consent has been withdrawn under Sec 6(c), so there is no longer a
# mandate to present against. Retrying an expired card is futile but lawful;
# retrying a revoked mandate is not, and the simulator raises on it.
RETRY_IS_BREACH_CLASSES: frozenset[str] = frozenset({FAILURE_CLASS_OPTED_OUT})

# The router dispatches on this. An unmapped code is routed to the ambiguous
# handler rather than crashing - a real acquirer sees codes nobody documented,
# and "unknown" is itself a class with a defined cautious response.
RETURN_CODE_TO_CLASS: dict[str, str] = {
    RETURN_CODE_INSUFFICIENT_FUNDS: FAILURE_CLASS_INSUFFICIENT_FUNDS,
    RETURN_CODE_EXPIRED_CARD: FAILURE_CLASS_EXPIRED_CARD,
    RETURN_CODE_MANDATE_REVOKED: FAILURE_CLASS_OPTED_OUT,
    RETURN_CODE_DOWNTIME: FAILURE_CLASS_DOWNTIME,
    RETURN_CODE_AFA_INCOMPLETE: FAILURE_CLASS_AFA_INCOMPLETE,
    RETURN_CODE_DO_NOT_HONOUR: FAILURE_CLASS_AMBIGUOUS,
}


def classify(return_code: str | None) -> str:
    """Map a return code to a failure class. Unknown codes are ambiguous."""
    if return_code is None:
        return FAILURE_CLASS_AMBIGUOUS
    return RETURN_CODE_TO_CLASS.get(return_code, FAILURE_CLASS_AMBIGUOUS)


def requires_otp(amount: float, category: str = "general") -> bool:
    """True when a recurring debit needs AFA (OTP) on every attempt.

    ``category`` selects the Sec 8(b) ceiling for insurance premia, mutual fund
    subscriptions and credit card bills. The generated population is entirely
    ``"general"`` today; the parameter exists so the threshold is never
    hardcoded at a call site when other categories are added.
    """
    if category in HIGH_VALUE_MANDATE_CATEGORIES:
        return amount > OTP_EXEMPTION_THRESHOLD_HIGH_VALUE_INR
    return amount > OTP_EXEMPTION_THRESHOLD_INR


def processing_cost(rail: str) -> float:
    """Merchant-side cost of making one presentment on ``rail`` (estimate)."""
    try:
        return PROCESSING_COST_INR[rail]
    except KeyError:
        raise ValueError(f"unknown rail: {rail!r}") from None


def hold_carrying_cost(subscription_amount: float) -> float:
    """Cost of serving an unpaying subscriber for one held cycle."""
    return subscription_amount * HOLD_CARRYING_COST_FRACTION


def bounce_charge(rail: str) -> float:
    """Charge levied on the customer by a failed debit on ``rail``."""
    if rail == "nach":
        return NACH_BOUNCE_CHARGE_INR
    if rail == "upi_autopay":
        return 0.0
    raise ValueError(f"unknown rail: {rail!r}")


def max_retries_remaining(attempts_so_far: int) -> int:
    """Retries still permitted in the cycle after ``attempts_so_far`` attempts."""
    return max(0, MAX_ATTEMPTS_PER_CYCLE - attempts_so_far)


def in_execution_window(at: time, rail: str = "upi_autopay") -> bool:
    """True when ``at`` falls inside the rail's permitted execution window.

    UPI Autopay presentments are restricted to 00:00-07:00 IST. NACH runs on
    settlement cycles rather than a clock window, so it is unconstrained here.
    """
    if rail == "nach":
        return True
    if rail != "upi_autopay":
        raise ValueError(f"unknown rail: {rail!r}")
    return UPI_AUTOPAY_WINDOW_START <= at < UPI_AUTOPAY_WINDOW_END


def can_retry(failure_class: str) -> bool:
    """False for the classes where no date can ever succeed."""
    if failure_class not in FAILURE_CLASSES:
        raise ValueError(f"unknown failure class: {failure_class!r}")
    return failure_class not in NEVER_RETRY_CLASSES
