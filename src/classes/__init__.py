"""One module per failure class, all behind the same two-function interface.

Five of the six are implemented. insufficient_funds, downtime and
afa_incomplete carry real models; expired_card and opted_out are complete but
trivial by nature - no date can succeed, so there is nothing to model. Only
ambiguous_decline is still declared rather than built; see classes/stubs.py.

The three modelled classes are deliberately different SHAPES, not three
variations on one:

    insufficient_funds  when is this customer liquid?   per-customer inference
    downtime            is this bank up right now?      cross-customer pooling
    afa_incomplete      will they authenticate, and     per-customer inference
                        can we make that more likely?   over an action we control
"""

from .base import FailureClassHandler, NeverRetryHandler

__all__ = ["FailureClassHandler", "NeverRetryHandler"]
