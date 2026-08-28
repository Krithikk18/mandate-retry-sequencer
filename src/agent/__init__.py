"""The retry agent.

Agent-safe by construction: nothing here may import ``generator.oracle``,
``generator.balance`` or the truth-family models. It sees what a PSP sees.
"""
