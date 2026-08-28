"""One module per failure class, all behind the same two-function interface.

Only insufficient_funds has a real model. expired_card and opted_out are
complete but trivial by nature - no date can succeed, so there is nothing to
model. The rest are declared stubs; see classes/stubs.py.
"""

from .base import FailureClassHandler, NeverRetryHandler

__all__ = ["FailureClassHandler", "NeverRetryHandler"]
