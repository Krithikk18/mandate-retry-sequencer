"""Every number in README.md and CONTEXT.md still reproduces.

    pytest -m slow tests/test_docs.py

**Deselected from the default run**, because it re-runs the whole experiment
suite at 20 seeds x 500 customers and takes about ten minutes. `pyproject.toml`
carries `-m "not slow"` in `addopts`, so `pytest` on its own stays fast and this
file runs only when asked for.

That is a deliberate trade and worth being explicit about: this is a
**release gate**, not an inner-loop test. CONTEXT.md's evaluation rules say
every number must come from a reproducible run, and the failure it exists to
catch is slow-moving - a change to the generator, the population, the simulator
or a policy silently invalidating a figure written down weeks earlier. That
happened when the fifth failure class landed. Run it before any submission, any
release, and any commit that touches those four things.

The registry, the producers and the comparator all live in
``scripts/verify_docs.py``; this file only wires them to pytest so a red build
is possible. Running that script directly gives the same report with the full
diff, which is what you want when something fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def verification():
    """Run every producer once and check every claim. Cached across the module."""
    import verify_docs

    stamp = verify_docs.fingerprint()
    payloads = {}
    for name in sorted({claim.producer for claim in verify_docs.CLAIMS}):
        producer = verify_docs.PRODUCERS[name]
        cached = verify_docs.load_cached(name, stamp)
        if cached is None:
            cached = producer.run()
            verify_docs.store_cached(name, stamp, cached)
        payloads[name] = cached

    docs = {
        doc: (ROOT / doc).read_text(encoding="utf-8")
        for doc in {claim.doc for claim in verify_docs.CLAIMS}
    }
    return verify_docs, verify_docs.check(verify_docs.CLAIMS, payloads, docs)


def test_every_registry_anchor_is_still_in_the_document(verification):
    """A claim whose anchor text has gone is a registry row nobody maintained.

    Separated from the value check because it is a different failure: the number
    did not move, the sentence around it did, and the fix is to re-anchor the
    row rather than to re-measure anything.
    """
    _, results = verification
    stale = [r for r in results if r.status == "NOT FOUND"]
    assert not stale, "\n".join(
        f"{r.claim.doc}: anchor gone for [{r.claim.id}] - {r.claim.anchor!r}"
        for r in stale
    )


def test_every_registered_field_exists_in_its_producer(verification):
    """A claim pointing at a field no producer emits verifies nothing at all."""
    _, results = verification
    missing = [r for r in results if r.status == "NO FIELD"]
    assert not missing, "\n".join(
        f"[{r.claim.id}] {r.claim.producer} -> {r.claim.path}: {r.detail}"
        for r in missing
    )


def test_documented_numbers_reproduce(verification):
    """The gate. Any documented figure that moved fails here, with the diff."""
    verify_docs, results = verification
    mismatches = [r for r in results if r.status == "MISMATCH"]
    assert not mismatches, "\n" + verify_docs.render(results)
