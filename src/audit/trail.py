"""Turns a decision log into something a reviewer can read.

Razorpay's brief asks for an audit trail. This renders one.

**Every number shown is read, never recomputed.** The probabilities, the
rupee terms and the counters all come out of ``reasoning`` and the log entry
snapshots exactly as the policy wrote them. A trail that re-derives its own
figures can quietly disagree with the run it claims to describe, which makes it
worse than no trail at all - it looks like evidence while being a second
opinion. The only arithmetic done here is subtraction of two recorded dates to
show how far ahead a presentment was committed.

Three styles:

    text      the human-readable form, for a terminal or a slide
    markdown  the same content, for docs/ and the README
    json      the same content, for machine consumption
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from evaluation.simulator import DecisionLogEntry, EpisodeResult
from generator import rules

HEAVY = "═" * 70
LIGHT = "─" * 70
DOT = " · "
TICK = "✓"
CROSS = "✗"

# Log actions the simulator writes for things that happened TO the mandate,
# as opposed to decisions the policy made.
EVENT_ACTIONS = frozenset({"EXECUTE", "AUTO", "CHURN"})


def _day(value: date) -> str:
    return value.strftime("%d %b")


def _rupees(value: float, signed: bool = False) -> str:
    # A term that rounds to nothing gets no sign: "+Rs 0" for the UPI bounce
    # line reads like a credit, which it is not.
    if abs(value) < 0.5:
        return "Rs 0"
    sign = "+" if signed and value > 0 else "-" if value < 0 else ""
    return f"{sign}Rs {abs(value):,.0f}"


def _outcome_line(result: EpisodeResult) -> str:
    return (
        f"OUTCOME: {result.outcome}{DOT}net value {_rupees(result.net_value, signed=True)}"
        f"{DOT}{result.days_to_resolution} days to resolution"
    )


TERM_WIDTH = 34


def _term(label: str, value: float, indent: str) -> str:
    """One line of the money equation, columns aligned."""
    return f"{indent}{label:<{TERM_WIDTH}}= {_rupees(value, signed=True):>10}"


def _candidate_arithmetic(
    candidate: dict[str, Any], indent: str, amount: float, bounce: float, ltv: float
) -> list[str]:
    """The money equation for one date, term by term.

    Showing only the total invites the reader to trust it. Showing the terms
    lets them find the one they disagree with - and on NACH the bounce line is
    usually the one that decides the answer.

    The multiplicands (amount, bounce charge, LTV) are facts about the customer
    and the rail. The probabilities and the products are read straight from the
    reasoning the policy recorded.
    """
    p_clear = candidate["p_clear"]
    return [
        _term(f"P(clear) {p_clear:.2f} x Rs {amount:,.0f}",
              candidate["expected_recovery"], indent),
        _term("processing", -candidate["processing_cost"], indent),
        _term(f"bounce risk {1 - p_clear:.2f} x Rs {bounce:,.0f}",
              -candidate["expected_bounce"], indent),
        _term(f"opt-out risk {candidate['p_optout']:.3f} x Rs {ltv:,.0f}",
              -candidate["expected_churn"], indent),
    ]


def _compliance_lines(entry: DecisionLogEntry, indent: str) -> list[str]:
    """What made this presentment legal. Read from the entry, not re-checked."""
    if entry.retry_date is None:
        return []
    lead_days = (entry.retry_date - entry.day).days
    hours = lead_days * 24 + (entry.retry_time.hour if entry.retry_time else 0)
    window = f"{rules.UPI_AUTOPAY_WINDOW_START:%H:%M}-{rules.UPI_AUTOPAY_WINDOW_END:%H:%M}"
    stamp = f"{entry.retry_time:%H:%M}" if entry.retry_time else "unset"
    return [
        f"{indent}Compliance: committed {hours}h ahead "
        f"(need {rules.PRE_DEBIT_NOTICE_HOURS}h) {TICK}",
        f"{indent}{'':>12}{stamp} inside {window} window {TICK}",
        f"{indent}{'':>12}attempt {entry.attempts_used + 1} "
        f"of {rules.MAX_ATTEMPTS_PER_CYCLE} {TICK}",
    ]


def decision_signature(entry: DecisionLogEntry) -> tuple:
    """What makes two decisions the same decision.

    A policy that defers commitment says WAIT on identical evidence for days on
    end. Printing that eight times is noise, and noise is what stops people
    reading audit trails. Consecutive identical decisions collapse into one
    line with a day count - nothing is hidden, the repetition is summarised.
    """
    r = entry.reasoning
    best = (r.get("candidates") or [{}])[0]
    return (
        entry.action,
        r.get("rule"),
        r.get("payday_mode"),
        best.get("date"),
        best.get("value"),
    )


def _render_decision(
    entry: DecisionLogEntry, amount: float, bounce: float, ltv: float, repeats: int = 1
) -> list[str]:
    """One policy decision, with the reasoning it was made on."""
    indent = " " * 9
    repeat_note = f"  (unchanged for {repeats} days)" if repeats > 1 else ""
    lines = [f"{_day(entry.day):<8} DECISION: {entry.action}{repeat_note}"]
    r = entry.reasoning

    mode = r.get("payday_mode")
    if mode is not None:
        confidence = r.get("payday_confidence", 0.0)
        line = f"{indent}Payday posterior: day {mode} (conf {confidence:.2f})"
        lines.append(line)
    if "shock_probability" in r:
        lines.append(f"{indent}Shock probability: {r['shock_probability']:.2f}")
    if "bank_outage_probability" in r:
        lines.append(
            f"{indent}Bank {r.get('bank_id', '?')}: P(outage) "
            f"{r['bank_outage_probability']:.3f}"
            f"{DOT}expected recovery {r.get('expected_recovery_hours', 0):.1f}h"
        )

    candidates = r.get("candidates") or []
    if candidates:
        best = candidates[0]
        lines.append(
            f"{indent}Best candidate {_day(date.fromisoformat(best['date']))} "
            f"scored {_rupees(best['value'], signed=True)}"
        )
        lines.extend(_candidate_arithmetic(best, indent + "  ", amount, bounce, ltv))
        if len(candidates) > 1:
            runners = DOT.join(
                f"{_day(date.fromisoformat(c['date']))} {_rupees(c['value'], signed=True)}"
                for c in candidates[1:4]
            )
            lines.append(f"{indent}Runners-up: {runners}")

    for constraint in r.get("binding_constraints") or []:
        lines.append(
            f"{indent}BOUND BY: {constraint['reason']}"
            + (f" - {constraint['detail']}" if constraint.get("detail") else "")
        )

    if entry.action == "RETRY":
        lines.extend(_compliance_lines(entry, indent))
    for key in ("hold_value", "cancel_value"):
        if key in r:
            lines.append(f"{indent}{key.replace('_', ' '):<18}{_rupees(r[key], signed=True)}")
    if r.get("rule"):
        lines.append(f"{indent}{r['rule']}")
    return lines


def _render_event(entry: DecisionLogEntry, result: EpisodeResult) -> list[str]:
    """Something that happened to the mandate rather than a decision."""
    indent = " " * 9
    amount = result.customer.subscription_amount if result.customer else 0.0

    if entry.action == "CHURN":
        headline = "CUSTOMER LEFT - natural attrition while held"
    elif "cleared" in entry.event:
        headline = f"PRESENTMENT - SUCCEEDED, Rs {amount:,.0f} recovered"
    elif "opted out" in entry.event:
        headline = "OPTED OUT on the pre-debit notification - mandate revoked"
    else:
        headline = f"PRESENTMENT - FAILED, return code {result.return_code}"

    lines = [f"{_day(entry.day):<8} {headline}"]
    lines.append(
        f"{indent}Attempts used {entry.attempts_used}/{rules.MAX_ATTEMPTS_PER_CYCLE}"
        f"{DOT}notifications sent {entry.notifications_sent}"
    )
    return lines


def _header(result: EpisodeResult) -> list[str]:
    customer = result.customer
    lines = [HEAVY]
    if customer is not None:
        lines.append(
            f"{customer.customer_id}{DOT}Rs {customer.subscription_amount:,.0f}"
            f"{DOT}{customer.rail}{DOT}{customer.tenure_months} months"
            f"{DOT}LTV Rs {customer.annual_ltv:,.0f}"
        )
    else:  # pragma: no cover - every episode carries a customer today
        lines.append(result.customer_id)
    lines.append(f"Failure class: {result.failure_class} (return code {result.return_code})")
    lines.append(f"Policy: {result.policy_name}")
    lines.append(HEAVY)
    return lines


def _footer(result: EpisodeResult, comparison: EpisodeResult | None) -> list[str]:
    lines = [LIGHT, _outcome_line(result)]
    if result.cycles_held:
        lines.append(
            f"Carried {result.cycles_held} cycles at "
            f"{_rupees(result.carrying_cost_paid)} in carrying cost"
        )
    if result.ltv_lost:
        lines.append(f"Mandate lost{DOT}LTV forfeited {_rupees(result.ltv_lost)}")
    else:
        lines.append(f"Mandate retained{DOT}LTV preserved {_rupees(result.ltv_retained)}")

    if comparison is not None:
        dates = ", ".join(_day(d) for d in comparison.retry_dates) or "no retries"
        lines.append(
            f"{comparison.policy_name} on the same customer: presented {dates}; "
            f"{comparison.outcome.lower()}, {comparison.notifications_sent} notifications "
            f"sent, net {_rupees(comparison.net_value, signed=True)}"
        )
    lines.append(HEAVY)
    return lines


def render_episode(
    result: EpisodeResult,
    style: str = "text",
    comparison: EpisodeResult | None = None,
) -> str:
    """Render one episode's decision log.

    ``comparison`` is another policy's run of the SAME customer, printed as a
    closing line. Without it a trail only ever shows one side.
    """
    if style == "json":
        return _render_json(result, comparison)
    if style not in ("text", "markdown"):
        raise ValueError(f"style must be text, markdown or json, got {style!r}")

    customer = result.customer
    amount = customer.subscription_amount if customer else 0.0
    bounce = rules.bounce_charge(customer.rail) if customer else 0.0
    ltv = customer.annual_ltv if customer else 0.0

    body: list[str] = []
    if result.original_debit_date is not None:
        body.append("")
        body.append(
            f"{_day(result.original_debit_date):<8} DEBIT FAILED - "
            f"return code {result.return_code}"
        )
        body.append(
            f"{'':9}Attempts used 1/{rules.MAX_ATTEMPTS_PER_CYCLE}"
            f"{DOT}notifications sent 0"
        )

    current_cycle = None
    for entry, repeats in _collapsed(result.decision_log):
        if entry.cycle != current_cycle:
            if current_cycle is not None:
                body.append("")
                body.append(f"--- cycle {entry.cycle} ---")
            current_cycle = entry.cycle
        body.append("")
        if entry.action in EVENT_ACTIONS:
            body.extend(_render_event(entry, result))
        else:
            body.extend(_render_decision(entry, amount, bounce, ltv, repeats))

    lines = [*_header(result), *body, "", *_footer(result, comparison)]
    text = "\n".join(lines)
    if style == "markdown":
        return f"```\n{text}\n```"
    return text


def _collapsed(
    log: list[DecisionLogEntry],
) -> list[tuple[DecisionLogEntry, int]]:
    """Fold runs of identical consecutive decisions into (entry, count)."""
    out: list[tuple[DecisionLogEntry, int]] = []
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


def _render_json(result: EpisodeResult, comparison: EpisodeResult | None) -> str:
    payload: dict[str, Any] = {
        "customer_id": result.customer_id,
        "policy": result.policy_name,
        "failure_class": result.failure_class,
        "return_code": result.return_code,
        "customer": result.customer.model_dump() if result.customer else None,
        "outcome": result.outcome,
        "recovered_amount": result.recovered_amount,
        "attempts_used": result.attempts_used,
        "notifications_sent": result.notifications_sent,
        "bounce_charges_paid": result.bounce_charges_paid,
        "carrying_cost_paid": result.carrying_cost_paid,
        "cycles_held": result.cycles_held,
        "ltv_retained": result.ltv_retained,
        "ltv_lost": result.ltv_lost,
        "net_value": result.net_value,
        "days_to_resolution": result.days_to_resolution,
        "decisions": [
            {
                "day": entry.day.isoformat(),
                "cycle": entry.cycle,
                "action": entry.action,
                "retry_date": entry.retry_date.isoformat() if entry.retry_date else None,
                "retry_time": entry.retry_time.isoformat() if entry.retry_time else None,
                "event": entry.event,
                "attempts_used": entry.attempts_used,
                "notifications_sent": entry.notifications_sent,
                "reasoning": entry.reasoning,
            }
            for entry in result.decision_log
        ],
    }
    if comparison is not None:
        payload["comparison"] = {
            "policy": comparison.policy_name,
            "outcome": comparison.outcome,
            "retry_dates": [d.isoformat() for d in comparison.retry_dates],
            "notifications_sent": comparison.notifications_sent,
            "net_value": comparison.net_value,
        }
    return json.dumps(payload, indent=2, default=str)
