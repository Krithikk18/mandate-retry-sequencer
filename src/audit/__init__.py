"""Rendering decision logs for human review.

Reads what the policy recorded. Never recomputes it.
"""

from .trail import render_episode

__all__ = ["render_episode"]
