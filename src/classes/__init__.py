"""One module per failure class, all behind the same two-function interface.

All six are implemented. insufficient_funds, downtime, afa_incomplete and
ambiguous_decline carry real models; expired_card and opted_out are complete
but trivial by nature - no date can succeed, so there is nothing to model.

The four modelled classes are deliberately different SHAPES, not four
variations on one:

    insufficient_funds  when is this customer liquid?   per-customer inference
    downtime            is this bank up right now?      cross-customer pooling
    afa_incomplete      will they authenticate, and     per-customer inference
                        can we make that more likely?   over an action we control
    ambiguous_decline   which of the others IS this?    a fitted POSTERIOR over
                                                        the other five, pooled
                                                        per issuer

The last one closes the taxonomy on itself: it does not add a sixth success
model, it mixes the five that already exist. See classes/ambiguous.py.
"""

from .base import FailureClassHandler, NeverRetryHandler

__all__ = ["FailureClassHandler", "NeverRetryHandler"]
