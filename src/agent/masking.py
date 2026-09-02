"""What is actually behind a 05, learned from the book rather than assumed.

``agent/bank_health.py`` pools decline evidence across every customer on an
issuer to answer "is this bank down?". This module has the same SHAPE and
answers a different question: **"when this issuer says nothing, what did it
mean?"**

Why pooled and not per customer, explicitly, because the project now has one of
each and the contrast is the point:

    payday.py          ONE customer. Another subscriber's salary date says
                       nothing about yours; pooling would be meaningless.
    otp_completion.py  ONE customer. Whether YOU authenticate says nothing
                       about whether anyone else does.
    bank_health.py     POOLED per issuer. An outage is a single event across
                       every mandate on that bank.
    masking.py         POOLED per issuer. How often an acquirer hides its
                       decline reasons is a property of the INSTITUTION, so
                       every 05 it ever returned is evidence about the next
                       one - including for a mandate that has never failed.

THE TRAP THIS MODULE IS BUILT TO AVOID
--------------------------------------
The generator draws the cause behind a 05 from
``generator.ambiguous.LATENT_CAUSE_MIXTURE``. If this model's prior were a copy
of that constant, the agent would have been told the answer and every result
about class 6 would be synthetic-data self-fulfilment - unpublishable, and
worse, unfalsifiable.

**So nothing here reads that constant.** The posterior is fitted from what a
real PSP can see: the *resolution history* of past 05s. A 05 is followed by
more attempts, and what those attempts returned is observable:

    05 then a success        the block was TRANSIENT - it cleared on its own
    05 then 04               the account was short all along
    05 then 54               the credential was dead
    05 then 1A               it was an authentication gate
    05 then 91               the issuer really was down
    05 then 05 then 05...    a permanent block nobody ever announced - but
                             only once the run is long enough that the window
                             ending is not the explanation. See
                             MIN_UNRESOLVED_RUN.

That last row is the one that matters commercially and it is inferrable purely
from the shape of the run: an unexplained decline that never resolves, however
many times you present, is a risk block whether or not anybody says so.

``tests/test_ambiguous.py`` perturbs the generator's mixture and asserts the
fitted posterior MOVES WITH IT. If that test fails the model is reading its own
answer and the class does not ship.

BACKING OFF
-----------
Per bank where the data supports it, pooled across the book where it does not.
An issuer with three 05s in the whole history cannot support its own posterior -
the estimate would be noise wearing a bank's name - so below
``MIN_BANK_EVIDENCE`` the book-level fit is used instead, and between the two
the bank estimate is shrunk toward the book by its own sample size. Same
discipline as ``bank_health``'s cold start returning the book-wide prior.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from generator import rules

#: The causes a fitted posterior can put mass on. The two that have no
#: unmasked counterpart are named here rather than imported from the generator,
#: deliberately: this module must not read the truth side, and a name is not a
#: probability. The VALUES are always fitted.
CAUSE_TRANSIENT: str = "transient"
CAUSE_PERMANENT_BLOCK: str = "permanent_block"

CAUSES: tuple[str, ...] = (
    rules.FAILURE_CLASS_INSUFFICIENT_FUNDS,
    rules.FAILURE_CLASS_EXPIRED_CARD,
    rules.FAILURE_CLASS_DOWNTIME,
    rules.FAILURE_CLASS_AFA_INCOMPLETE,
    CAUSE_TRANSIENT,
    CAUSE_PERMANENT_BLOCK,
)

# Below this many resolved 05s on one issuer, that issuer's own posterior is
# noise and the book-level fit is used instead. ESTIMATE, and deliberately not
# tuned: it is a statement about when a proportion is worth quoting, which is
# the same judgement `run_ablations.MIN_BITTEN_FOR_A_VERDICT` makes.
MIN_BANK_EVIDENCE: int = 8

# Laplace smoothing, so a cause never gets probability zero on a thin book and
# an unseen cause stays reachable. One pseudo-observation per cause.
SMOOTHING: float = 1.0

# How many further unexplained declines it takes before a run counts as a
# PERMANENT block rather than as "we have not seen it resolve yet".
#
# This is a CENSORING correction, not a tuning knob, and the project has done
# it before: PaydayModel drops censored declines rather than counting them as
# evidence the account was empty, because a 91 never reached the account. The
# same error is available here in a nastier form - a 05 near the end of the
# observed window looks permanent for the trivial reason that there was no time
# left for it to resolve. Counting that as a block would inflate
# permanent_block by exactly the amount of window that happens to be missing,
# and the agent would then hold mandates that were merely young.
MIN_UNRESOLVED_RUN: int = 2

# What the model believes before it has seen anything at all. UNIFORM on
# purpose: any other choice would be smuggling in a guess about the mixture,
# which is precisely the thing this module must not do. The book overwhelms it
# within a few dozen observations.
def _uniform() -> dict[str, float]:
    return {cause: 1.0 / len(CAUSES) for cause in CAUSES}


@dataclass
class MaskingModel:
    """Posterior over the true cause of an unexplained decline.

    Online in the same way ``BankHealthModel`` is: ``observe`` takes one
    resolved 05 at a time, so the same object serves a batch backfill and a
    live stream.
    """

    _by_bank: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float)), init=False
    )
    _pooled: dict[str, float] = field(
        default_factory=lambda: defaultdict(float), init=False
    )

    # --- ingestion -----------------------------------------------------------

    def observe(self, bank_id: str, cause: str) -> None:
        """Record one 05 whose true cause was later revealed by what followed."""
        if cause not in CAUSES:
            raise ValueError(f"unknown cause {cause!r}")
        self._by_bank[bank_id][cause] += 1.0
        self._pooled[cause] += 1.0

    def observe_customer(self, bank_id: str, attempts: list[Any]) -> None:
        """Fold one customer's whole attempt history in, resolved 05s only.

        The online entry point, used by ``SequencerPolicy`` as it meets each
        customer. Deduplication is the caller's job - the policy already keys
        that on customer id for bank evidence and this rides in the same pass.
        """
        attempts = sorted(attempts, key=lambda r: r.attempt_date)
        for index, attempt in enumerate(attempts):
            if attempt.succeeded or attempt.return_code != (
                rules.RETURN_CODE_DO_NOT_HONOUR
            ):
                continue
            cause = _resolution_of(attempts[index + 1 :])
            if cause is not None:
                self.observe(bank_id, cause)

    @classmethod
    def from_history(
        cls, records: Iterable[Any], bank_of: dict[str, str]
    ) -> "MaskingModel":
        """Fit from agent-visible payment history.

        ``records`` are ``AgentPaymentHistoryRecord``; ``bank_of`` maps customer
        to issuer, which the agent reads off the public ``bank_id`` field. Only
        the observable channel is touched: attempt dates, success flags and
        return codes. Nothing here can see a failure class.
        """
        model = cls()
        by_customer: dict[str, list[Any]] = defaultdict(list)
        for record in records:
            by_customer[record.customer_id].append(record)

        for customer_id, attempts in by_customer.items():
            bank_id = bank_of.get(customer_id)
            if bank_id is None:
                continue
            attempts = sorted(attempts, key=lambda r: r.attempt_date)
            for index, attempt in enumerate(attempts):
                if attempt.succeeded or attempt.return_code != (
                    rules.RETURN_CODE_DO_NOT_HONOUR
                ):
                    continue
                cause = _resolution_of(attempts[index + 1 :])
                if cause is not None:
                    model.observe(bank_id, cause)
        return model

    # --- inference -----------------------------------------------------------

    def posterior(self, bank_id: str | None = None) -> dict[str, float]:
        """P(true cause | an unexplained decline), for this issuer.

        Bank-level where the issuer has enough resolved 05s to support one,
        shrunk toward the book-level fit by its own sample size, and equal to
        the book-level fit below ``MIN_BANK_EVIDENCE``. With no evidence at all
        it is uniform - see ``_uniform``: this model refuses to guess the
        mixture it exists to estimate.
        """
        pooled = _normalise(self._pooled)
        if bank_id is None:
            return pooled
        counts = self._by_bank.get(bank_id)
        total = sum(counts.values()) if counts else 0.0
        if total < MIN_BANK_EVIDENCE:
            return pooled
        bank = _normalise(counts)
        # Shrink toward the book: weight the bank's own fit by how much
        # evidence it actually has, so a bank with nine observations is barely
        # trusted and one with hundreds is trusted almost fully.
        weight = total / (total + MIN_BANK_EVIDENCE)
        return {
            cause: weight * bank[cause] + (1.0 - weight) * pooled[cause]
            for cause in CAUSES
        }

    def evidence_count(self, bank_id: str | None = None) -> int:
        """How many resolved 05s this posterior rests on."""
        if bank_id is None:
            return int(sum(self._pooled.values()))
        return int(sum(self._by_bank.get(bank_id, {}).values()))

    def explain(self, bank_id: str | None = None) -> dict:
        """What the model believes and on how much evidence. For the audit trail."""
        posterior = self.posterior(bank_id)
        return {
            "bank_id": bank_id,
            "evidence": self.evidence_count(bank_id),
            "pooled_evidence": self.evidence_count(None),
            "uses_bank_level": (
                bank_id is not None
                and self.evidence_count(bank_id) >= MIN_BANK_EVIDENCE
            ),
            "posterior": {c: round(p, 4) for c, p in sorted(posterior.items())},
        }


def _normalise(counts: dict[str, float] | None) -> dict[str, float]:
    """Laplace-smoothed distribution over every cause, never a zero."""
    counts = counts or {}
    total = sum(counts.get(c, 0.0) for c in CAUSES) + SMOOTHING * len(CAUSES)
    if total <= 0:
        return _uniform()
    return {c: (counts.get(c, 0.0) + SMOOTHING) / total for c in CAUSES}


def _resolution_of(later: list[Any]) -> str | None:
    """What the attempts AFTER a 05 revealed about it. None when still unknown.

    The whole observable channel for this class. A 05 on its own says nothing;
    a 05 plus what happened next says a great deal, and none of it requires
    seeing a failure class.
    """
    if not later:
        return None
    unexplained = 0
    for attempt in later:
        if attempt.succeeded:
            # It cleared without anybody doing anything, so whatever blocked it
            # was temporary. This is the signature of a transient glitch.
            return CAUSE_TRANSIENT
        code = attempt.return_code
        if code == rules.RETURN_CODE_DO_NOT_HONOUR:
            unexplained += 1
            continue  # still unexplained; keep looking down the run
        mapped = rules.RETURN_CODE_TO_CLASS.get(code)
        if mapped in (
            rules.FAILURE_CLASS_INSUFFICIENT_FUNDS,
            rules.FAILURE_CLASS_EXPIRED_CARD,
            rules.FAILURE_CLASS_DOWNTIME,
            rules.FAILURE_CLASS_AFA_INCOMPLETE,
        ):
            return mapped
        return None
    # Every later attempt was also an unexplained decline and none ever
    # cleared. That is what a permanent block looks like from the outside, and
    # it is the only way to see one: nobody will ever send a code that says so.
    #
    # But only once the run is long enough to mean something. A single
    # follow-up 05 is equally consistent with a block that had not resolved YET
    # when the window ended, and treating it as permanent is the censoring bias
    # MIN_UNRESOLVED_RUN exists to refuse.
    if unexplained >= MIN_UNRESOLVED_RUN:
        return CAUSE_PERMANENT_BLOCK
    return None
