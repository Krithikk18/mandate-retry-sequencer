"""Synthetic data generator for Indian subscription payment failures.

The generator doubles as a ground-truth oracle: because every customer's
income and spending process is invented here, any counterfactual question
("would a debit on date D have succeeded?") has an exact answer.
"""

__version__ = "0.1.0"
