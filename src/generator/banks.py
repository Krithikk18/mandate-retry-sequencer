"""Bank and PSP outages: the hidden truth behind failure class 2.

This is the mirror image of the balance process. An insufficient-funds failure
resolves over DAYS and the fix is patience - wait for payday. A downtime
failure resolves over HOURS and patience loses: by the time a fixed schedule
fires its day+2 retry, a three-hour outage cleared two days ago, and the money
was collectable the whole time.

Two properties make this class worth modelling separately:

* It is **hour-resolution**. The balance process is daily; an outage that
  starts at 01:00 and ends at 04:00 either does or does not cover the 02:00
  presentment hour, and nothing coarser can express that.
* It is **shared**. Every customer on the same issuer fails together. That is
  the first correlated signal in the project, and it is what
  ``agent/bank_health.py`` learns from: five downtime-coded failures on one
  bank in one hour tell you something about every OTHER customer on that bank.

The outage book is a pure function of ``(seed, window)``. It is deliberately
NOT stored on ``Dataset``: any caller rebuilds the identical book from the
dataset's seed and window, so it survives ``save_dataset``/``load_dataset``
for free without teaching the persistence layer about a new table.
"""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np

# --- the book ----------------------------------------------------------------
# Realistic SHAPE, not real numbers. Per-issuer UPI Autopay mandate share is not
# published by NPCI; what is public is the broad concentration - a handful of
# large issuers carry most mandates, with a long tail of smaller banks and small
# finance banks. These weights reproduce that shape. Do not quote them as facts
# about any named bank.
BANK_SHARES: dict[str, float] = {
    "SBI": 0.22,
    "HDFC": 0.16,
    "ICICI": 0.12,
    "AXIS": 0.09,
    "KOTAK": 0.06,
    "PNB": 0.06,
    # long tail: ~29% shared across fifteen smaller institutions
    "BOB": 0.040,
    "CANARA": 0.034,
    "UNION": 0.030,
    "INDUSIND": 0.027,
    "IDFC": 0.025,
    "YES": 0.023,
    "IDBI": 0.020,
    "FEDERAL": 0.018,
    "RBL": 0.016,
    "BANDHAN": 0.012,
    "AU_SFB": 0.011,
    "EQUITAS": 0.010,
    "UJJIVAN": 0.009,
    "JANA_SFB": 0.008,
    "SURYODAY_SFB": 0.007,
}
assert abs(sum(BANK_SHARES.values()) - 1.0) < 1e-9, "bank shares must be a distribution"

# --- the outage process ------------------------------------------------------
# Outages arrive as a renewal process and their rate scales INVERSELY with book
# share. One knob buys the realistic pattern that large issuers have better
# uptime: they carry more mandates, so they invest more in the rails carrying
# them. A hardcoded two-tier "big banks vs small banks" would produce the same
# ranking without the mechanism.
#
#     rate = BASE_OUTAGE_RATE_PER_HOUR * (REFERENCE_SHARE / share) ** ELASTICITY
#
# Tuned so that roughly 10% of generated failures carry return code 91; see the
# realised figure reported by generate_dataset in `meta["downtime_share"]`.
BASE_OUTAGE_RATE_PER_HOUR: float = 0.0042
REFERENCE_SHARE: float = 0.05
SHARE_ELASTICITY: float = 0.55
MIN_EFFECTIVE_SHARE: float = 0.006  # floor, so the smallest bank is not absurd

# Duration is lognormal on hours. mu = ln(3) puts the MEDIAN at 3 hours, which
# is where real payment-rail incidents cluster; sigma = 0.75 stretches the tail
# so roughly the worst 1% run past 17-18 hours - an overnight core-banking
# problem that spans several presentment windows. Clipped at both ends: nothing
# below 30 minutes counts as an outage worth modelling, nothing above 18 hours.
OUTAGE_DURATION_LOG_MU: float = math.log(3.0)
OUTAGE_DURATION_LOG_SIGMA: float = 0.75
OUTAGE_MIN_HOURS: float = 0.5
OUTAGE_MAX_HOURS: float = 18.0


def bank_ids() -> tuple[str, ...]:
    return tuple(BANK_SHARES)


def outage_rate(bank_id: str) -> float:
    """Outages per hour for this bank. Smaller book, worse uptime."""
    share = max(BANK_SHARES.get(bank_id, MIN_EFFECTIVE_SHARE), MIN_EFFECTIVE_SHARE)
    return BASE_OUTAGE_RATE_PER_HOUR * (REFERENCE_SHARE / share) ** SHARE_ELASTICITY


def _bank_seed(bank_id: str, seed: int) -> int:
    """A stable per-bank stream. Same pattern as Customer.balance_seed."""
    digest = hashlib.blake2b(f"{bank_id}|{seed}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _intervals_for(
    bank_id: str, seed: int, window_start: datetime, window_end: datetime
) -> list[tuple[datetime, datetime]]:
    """Walk a renewal process across the window: gap, outage, gap, outage."""
    rng = np.random.default_rng(_bank_seed(bank_id, seed))
    rate = outage_rate(bank_id)
    intervals: list[tuple[datetime, datetime]] = []

    cursor = window_start
    while cursor < window_end:
        gap_hours = float(rng.exponential(1.0 / rate))
        start = cursor + timedelta(hours=gap_hours)
        if start >= window_end:
            break
        duration = float(
            np.clip(
                rng.lognormal(OUTAGE_DURATION_LOG_MU, OUTAGE_DURATION_LOG_SIGMA),
                OUTAGE_MIN_HOURS,
                OUTAGE_MAX_HOURS,
            )
        )
        end = start + timedelta(hours=duration)
        intervals.append((start, end))
        cursor = end
    return intervals


@dataclass(frozen=True)
class BankOutageBook:
    """Every bank's outage intervals across one window. Hidden truth.

    Lookups are binary searches over sorted, non-overlapping intervals, because
    the oracle asks "is this bank up?" once per candidate hour and there can be
    a great many candidate hours.
    """

    seed: int
    window_start: datetime
    window_end: datetime
    intervals: dict[str, list[tuple[datetime, datetime]]]

    def _starts(self, bank_id: str) -> list[datetime]:
        return [start for start, _ in self.intervals.get(bank_id, ())]

    def is_up(self, bank_id: str, at: datetime) -> bool:
        """True when the bank can process a presentment at this instant."""
        spans = self.intervals.get(bank_id)
        if not spans:
            return True
        index = bisect.bisect_right(self._starts(bank_id), at) - 1
        if index < 0:
            return True
        start, end = spans[index]
        return not (start <= at < end)

    def is_down(self, bank_id: str, at: datetime) -> bool:
        return not self.is_up(bank_id, at)

    def next_up_at_or_after(self, bank_id: str, at: datetime) -> datetime:
        """First instant at or after ``at`` when the bank is available.

        Lets the oracle jump over an outage instead of probing hour by hour.
        """
        spans = self.intervals.get(bank_id)
        if not spans:
            return at
        index = bisect.bisect_right(self._starts(bank_id), at) - 1
        if index < 0:
            return at
        start, end = spans[index]
        return end if start <= at < end else at

    def outage_covering(
        self, bank_id: str, at: datetime
    ) -> tuple[datetime, datetime] | None:
        """The outage interval containing ``at``, if any. For measurement."""
        spans = self.intervals.get(bank_id)
        if not spans:
            return None
        index = bisect.bisect_right(self._starts(bank_id), at) - 1
        if index < 0:
            return None
        start, end = spans[index]
        return (start, end) if start <= at < end else None

    def down_fraction(self, bank_id: str) -> float:
        """Share of the window this bank spent unavailable. Diagnostics only."""
        total = (self.window_end - self.window_start).total_seconds()
        if total <= 0:
            return 0.0
        down = sum(
            (end - start).total_seconds() for start, end in self.intervals.get(bank_id, ())
        )
        return down / total


@lru_cache(maxsize=64)
def outage_book(
    seed: int, window_start: datetime, window_end: datetime
) -> BankOutageBook:
    """The outage book for one dataset. Pure in (seed, window).

    Cached because every presentment resolution and every oracle query wants
    the same book, and there are only ~20 banks to build.
    """
    if window_end <= window_start:
        raise ValueError("window_end must be after window_start")
    return BankOutageBook(
        seed=seed,
        window_start=window_start,
        window_end=window_end,
        intervals={
            bank_id: _intervals_for(bank_id, seed, window_start, window_end)
            for bank_id in BANK_SHARES
        },
    )
