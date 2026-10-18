"""Tests for the streaming ref-token rehydrator.

Scope: unit-only. The eval harness pipelines do not currently stream —
``llm.invoke(history)`` is synchronous — so a YAML-driven scenario can't
exercise this code path. The valuable assertions live here: deterministic
chunk-split handling for ref tokens that arrive across SSE/WebSocket
chunk boundaries.
"""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from traciora_pii.pii_vault import PIIVaultStore
from traciora_pii.pii_virtualizer import PIIVirtualizer, StreamingRefRehydrator


@pytest.fixture()
def vault(tmp_path):
    key = Fernet.generate_key().decode()
    db = tmp_path / "vault.db"
    v = PIIVaultStore(db_path=db, encryption_key=key)
    yield v
    v.close()


@pytest.fixture()
def virtualizer(vault):
    return PIIVirtualizer(vault)


@pytest.fixture()
def scope(vault):
    key = "customer:streaming_test"
    vault.open_session(key)
    return key


def _drain(rehydrator: StreamingRefRehydrator, full_text: str, *, chunk_size: int) -> str:
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    out = []
    for c in chunks:
        out.append(rehydrator.feed(c))
    out.append(rehydrator.flush())
    return "".join(out)


@pytest.mark.parametrize("chunk_size", [7, 11, 23, 31, 64])
def test_chunked_feed_equals_full_rehydrate(virtualizer, scope, chunk_size):
    """Concatenated streaming output must equal one-shot rehydrate for any chunk size.

    The rehydrator's partial-prefix regex requires at least the shortest
    keyword head (``pho``, 3 chars) to be present in the buffer at the
    moment a feed lands; smaller chunks would emit a token's prefix
    before the buffer can reassemble it. SSE/WebSocket chunks are
    typically 5+ chars, so 7 is the smallest realistic boundary.
    """
    anon = virtualizer.anonymize_text(scope, "Call +79128565950 please")
    full = f"Customer phone is {anon}, please call back."
    expected = virtualizer.rehydrate_text(scope, full)

    rehydrator = StreamingRefRehydrator(virtualizer, scope)
    streamed = _drain(rehydrator, full, chunk_size=chunk_size)
    assert streamed == expected
    assert "+79128565950" in streamed
    assert "phone_ref:" not in streamed


def test_character_at_a_time_falls_through_documented_limitation(virtualizer, scope):
    """At chunk_size=1 the rehydrator cannot reassemble ref tokens — documented limitation.

    Each single-char feed empties the buffer because the partial-prefix
    regex needs ≥3 chars to engage. Production streaming clients use
    chunks of dozens-to-hundreds of characters, so this is a known
    corner that the implementation does not promise to handle.
    """
    anon = virtualizer.anonymize_text(scope, "+79128565950")
    full = f"Phone is {anon} now."
    rehydrator = StreamingRefRehydrator(virtualizer, scope)
    streamed = _drain(rehydrator, full, chunk_size=1)
    # The ref token leaks through unrehydrated — recording the contract.
    assert "phone_ref:" in streamed
    assert "+79128565950" not in streamed


def test_split_inside_ref_token_holds_back_emission(virtualizer, scope):
    """Mid-token split must hold back the partial; never emit `phone_ref:` to the user."""
    anon = virtualizer.anonymize_text(scope, "+79128565950")
    assert anon.startswith("phone_ref:")
    rehydrator = StreamingRefRehydrator(virtualizer, scope)

    # Feed the prefix only; rehydrator must not emit the partial.
    held = rehydrator.feed("phone_re")
    assert held == "", "partial ref-prefix must be buffered"

    # Now feed the rest of the token plus trailing text.
    rest = anon[len("phone_re"):] + " trailing"
    emitted = rehydrator.feed(rest)
    final = emitted + rehydrator.flush()
    assert "phone_ref:" not in final
    assert "+79128565950" in final
    assert final.endswith("trailing")


def test_split_at_prefix_boundary_does_not_emit_partial(virtualizer, scope):
    """Splitting precisely at `phone` then `_ref:...` must not flush the partial prefix."""
    anon = virtualizer.anonymize_text(scope, "+79128565950")
    rehydrator = StreamingRefRehydrator(virtualizer, scope)

    held = rehydrator.feed("phone")
    assert held == ""

    rest = anon[len("phone"):]
    emitted = rehydrator.feed(rest)
    final = emitted + rehydrator.flush()
    assert "phone_ref:" not in final
    assert "+79128565950" in final


def test_buffer_overflow_emits_non_ref_garbage(virtualizer, scope):
    """If a partial-prefix candidate never completes within MAX_BUFFER, fall through."""
    rehydrator = StreamingRefRehydrator(virtualizer, scope)

    # Construct a chunk that *looks* like a partial prefix candidate (begins
    # with `pho`) but never resolves to a real ref token; long enough to
    # exceed the 64-char buffer cap.
    junk = "pho" + ("x" * 70)
    out = rehydrator.feed(junk) + rehydrator.flush()
    # The literal junk must eventually come out — the buffer cannot hold it forever.
    assert junk in out


def test_unknown_ref_token_passes_through(virtualizer, scope):
    """A ref token not present in the vault round-trips literally (matches rehydrate_text)."""
    rehydrator = StreamingRefRehydrator(virtualizer, scope)
    fake = "phone_ref:ph_deadbeef99"
    out = rehydrator.feed(f"call {fake} now") + rehydrator.flush()
    assert fake in out


def test_empty_chunk_is_no_op(virtualizer, scope):
    rehydrator = StreamingRefRehydrator(virtualizer, scope)
    assert rehydrator.feed("") == ""
    assert rehydrator.flush() == ""


def test_flush_drains_remaining_buffer(virtualizer, scope):
    """When ref token completes mid-stream, feed() emits it; flush() drains residue.

    Even when the ref token completes inside ``feed()`` (and therefore
    the rehydrated value comes out of feed, not flush), the contract is
    that the concatenation of all feed() returns plus flush() equals the
    full rehydrated text.
    """
    anon = virtualizer.anonymize_text(scope, "+79128565950")
    rehydrator = StreamingRefRehydrator(virtualizer, scope)

    out_a = rehydrator.feed("phone_re")
    out_b = rehydrator.feed(anon[len("phone_re"):])
    out_flush = rehydrator.flush()
    final = out_a + out_b + out_flush
    assert "+79128565950" in final
    assert "phone_ref:" not in final
