"""Pydantic record models plus the agent/truth field split.

Two families of models live here:

* the *truth* models (``Customer``, ``FailureEvent``) which carry the hidden
  generative parameters the oracle needs, and
* the *agent* models (``AgentCustomer``, ``AgentFailureEvent``) which are
  structurally incapable of carrying them.

Keeping the split at the type level (rather than filtering dicts) means a
leak is a construction error, not a silent data problem.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import rules

# Fields the retry agent must never observe. Enforced by AgentCustomer's shape
# and asserted in tests/test_generator.py::test_agent_view_has_no_leakage.
HIDDEN_CUSTOMER_FIELDS: frozenset[str] = frozenset(
    {
        "salary_day",
        "salary_amount",
        "salary_delay_prob",
        "salary_delay_days_max",
        "income_shock_start",
        "mandate_dead_from",
        "mandate_dead_class",
        "burn_rate",
        "buffer_ratio",
        "optout_sensitivity",
        "balance_seed",
    }
)

# A real PSP sees return code "04" ("insufficient funds") but never learns how
# short the account actually was, so the balance is hidden too.
HIDDEN_FAILURE_FIELDS: frozenset[str] = frozenset({"balance_at_failure"})

Rail = Literal["upi_autopay", "nach"]


class Customer(BaseModel):
    """A subscriber, including the hidden process that drives their balance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- public: visible to the agent ---------------------------------------
    customer_id: str = Field(pattern=r"^CUST_\d{4,}$")
    subscription_amount: float = Field(gt=0)
    rail: Rail
    debit_day_of_month: int = Field(ge=1, le=28)
    tenure_months: int = Field(ge=1, le=48)
    monthly_ltv: float = Field(gt=0)

    # --- hidden: income profile (oracle only) -------------------------------
    salary_day: int = Field(ge=1, le=31)
    salary_amount: float = Field(gt=0)
    salary_delay_prob: float = Field(ge=0.0, le=1.0)
    salary_delay_days_max: int = Field(ge=0, le=15)

    # Date from which income ceases entirely (job loss and similar). None for
    # the overwhelming majority. For an affected customer NO date clears the
    # mandate, so CANCEL / HOLD is the only correct action and a retry is pure
    # cost - which is what makes a stopping rule testable at all.
    income_shock_start: date | None = None

    # A mandate can die for reasons that have nothing to do with the balance:
    # the card expires, or the customer revokes consent at a pre-debit notice.
    # From this date every presentment fails with that class's return code, on
    # any date, forever. The agent learns this from the RETURN CODE it gets
    # back, never from these fields.
    mandate_dead_from: date | None = None
    mandate_dead_class: str | None = None

    # --- hidden: spending profile (oracle only) -----------------------------
    burn_rate: float = Field(gt=0, lt=1)
    buffer_ratio: float = Field(ge=0.0, lt=1.0)

    # --- hidden: behaviour (oracle only) ------------------------------------
    optout_sensitivity: float = Field(ge=0.0, le=1.0)

    # --- hidden: reproducibility -------------------------------------------
    # Per-customer seed for the balance process. Carried on the record so that
    # daily_balance(customer, ...) is deterministic without threading a global
    # RNG through the oracle. Hidden, because it indexes the truth.
    balance_seed: int = Field(ge=0)

    @field_validator("mandate_dead_class")
    @classmethod
    def _known_dead_class(cls, v: str | None) -> str | None:
        if v is not None and v not in rules.NEVER_RETRY_CLASSES:
            raise ValueError(
                f"mandate_dead_class must be one of {sorted(rules.NEVER_RETRY_CLASSES)}"
            )
        return v

    @field_validator("salary_day")
    @classmethod
    def _salary_day_in_indian_payroll_band(cls, v: int) -> int:
        # Indian payroll clusters at month end and in the first week; the
        # Payment of Wages Act caps it at the 7th (10th for large employers).
        if not (1 <= v <= 10 or 25 <= v <= 31):
            raise ValueError("salary_day must be in 1-10 or 25-31")
        return v

    @model_validator(mode="before")
    @classmethod
    def _default_monthly_ltv(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("monthly_ltv") is None:
            amount = data.get("subscription_amount")
            if amount is not None:
                data = {**data, "monthly_ltv": float(amount) * 12.0}
        return data

    @model_validator(mode="after")
    def _ltv_matches_amount(self) -> "Customer":
        if not math.isclose(self.monthly_ltv, self.subscription_amount * 12.0, rel_tol=1e-6):
            raise ValueError("monthly_ltv must equal subscription_amount * 12")
        return self

    @property
    def requires_otp(self) -> bool:
        """True when this mandate's debits need OTP (above Rs 15,000)."""
        return rules.requires_otp(self.subscription_amount)

    @property
    def bounce_charge(self) -> float:
        """What a failed debit costs this customer (NACH bounces, UPI does not)."""
        return rules.bounce_charge(self.rail)

    @property
    def annual_ltv(self) -> float:
        """Twelve months of subscription value.

        Alias for ``monthly_ltv``, which is named for the *billing period* and
        not for the horizon: it is already amount x 12. Use this name in the
        objective function so the units cannot be misread.
        """
        return self.monthly_ltv

    def public_fields(self) -> dict[str, Any]:
        """The subset of fields the agent is allowed to see."""
        return {
            k: v for k, v in self.model_dump().items() if k not in HIDDEN_CUSTOMER_FIELDS
        }


class AgentCustomer(BaseModel):
    """Customer as the retry agent sees it. Has no slot for hidden fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_id: str
    subscription_amount: float
    rail: Rail
    debit_day_of_month: int
    tenure_months: int
    monthly_ltv: float

    @property
    def requires_otp(self) -> bool:
        return rules.requires_otp(self.subscription_amount)

    @property
    def bounce_charge(self) -> float:
        return rules.bounce_charge(self.rail)

    @property
    def annual_ltv(self) -> float:
        """See ``Customer.annual_ltv``: monthly_ltv is already amount x 12."""
        return self.monthly_ltv

    @classmethod
    def from_customer(cls, customer: Customer) -> "AgentCustomer":
        return cls(**customer.public_fields())


class FailureEvent(BaseModel):
    """A subscription debit that actually failed on its original date.

    ``cycle_start_date`` is the first day of the billing cycle the debit belongs
    to; for a monthly mandate the cycle runs debit-date to debit-date, so it
    equals ``original_debit_date`` and is kept as a separate field because
    retries are scoped to the cycle, not to the attempt.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_id: str
    cycle_start_date: date
    original_debit_date: date
    return_code: str = rules.RETURN_CODE_INSUFFICIENT_FUNDS
    failure_class: str = rules.FAILURE_CLASS_INSUFFICIENT_FUNDS
    balance_at_failure: float  # hidden: real PSPs never see this

    @field_validator("failure_class")
    @classmethod
    def _known_failure_class(cls, v: str) -> str:
        # Only insufficient_funds is generated today, but the taxonomy is
        # fixed: a typo in a future class module should fail loudly here.
        if v not in rules.FAILURE_CLASSES:
            raise ValueError(f"unknown failure class: {v!r}")
        return v

    def public_fields(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.model_dump().items() if k not in HIDDEN_FAILURE_FIELDS
        }


class AgentFailureEvent(BaseModel):
    """Failure as the agent sees it: the return code, not the shortfall."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_id: str
    cycle_start_date: date
    original_debit_date: date
    return_code: str
    failure_class: str

    @classmethod
    def from_failure(cls, event: FailureEvent) -> "AgentFailureEvent":
        return cls(**event.public_fields())


class PaymentHistoryRecord(BaseModel):
    """One debit attempt on one date. Visible to the agent in full.

    See ``AgentPaymentHistoryRecord`` below: this record has no truth-family
    counterpart because nothing in it is hidden.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_id: str
    attempt_date: date
    amount: float
    succeeded: bool
    return_code: str | None = None

    @model_validator(mode="after")
    def _return_code_iff_failed(self) -> "PaymentHistoryRecord":
        if self.succeeded and self.return_code is not None:
            raise ValueError("successful attempts carry no return code")
        if not self.succeeded and self.return_code is None:
            raise ValueError("failed attempts must carry a return code")
        return self


# Payment history carries no hidden fields - a real PSP sees every attempt it
# made and the return code it got back - so the agent-facing name is an alias
# rather than a separate stripped type. It exists so policy code can import
# only Agent* names and never touch the truth family by accident.
AgentPaymentHistoryRecord = PaymentHistoryRecord
