"""Tests for the chunker.

Run:
    python -m src.test_chunker
"""

from __future__ import annotations

from pathlib import Path

from .chunker import Chunk, _parse_frontmatter, _parse_turns, chunk_corpus, chunk_transcript


REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO_ROOT / "transcripts"


def test_parse_frontmatter_extracts_keys():
    text = "---\nparticipant: Sarah K.\nrole: therapist\n---\nbody here"
    meta, body = _parse_frontmatter(text)
    assert meta["participant"] == "Sarah K."
    assert meta["role"] == "therapist"
    assert "body here" in body


def test_parse_frontmatter_no_frontmatter():
    meta, body = _parse_frontmatter("just body, no frontmatter")
    assert meta == {}
    assert body == "just body, no frontmatter"


def test_parse_turns_simple():
    body = "ALEX: Hello there.\n\nSARAH: Hi back.\n\nALEX: How are you?"
    turns = _parse_turns(body)
    assert len(turns) == 3
    assert turns[0] == ("ALEX", "Hello there.")
    assert turns[1] == ("SARAH", "Hi back.")
    assert turns[2] == ("ALEX", "How are you?")


def test_parse_turns_multiline():
    body = "SARAH: This is a long answer\nthat wraps across lines.\n\nALEX: I see."
    turns = _parse_turns(body)
    assert len(turns) == 2
    assert "wraps across lines" in turns[0][1]


def test_chunk_real_transcript_has_context():
    """Real-corpus smoke test: chunking 01_sarah produces turns with the
    expected speakers and non-empty context windows (except at the edges)."""
    path = TRANSCRIPTS / "01_sarah.md"
    chunks = chunk_transcript(path)
    assert len(chunks) > 10, "expected many turns for Sarah's interview"
    assert chunks[0].speaker == "ALEX", "interview should open with the moderator"
    assert "SARAH" in {c.speaker for c in chunks}, "Sarah should appear as a speaker"
    # Middle chunks should have both context_before and context_after
    middle = chunks[len(chunks) // 2]
    assert middle.context_before, "middle chunk should have prior turn"
    assert middle.context_after, "middle chunk should have next turn"


def test_chunk_corpus_covers_all_transcripts():
    """Loading the whole corpus should give one big flat list of chunks
    spanning every transcript file."""
    chunks = chunk_corpus(TRANSCRIPTS)
    interview_ids = {c.interview_id for c in chunks}
    assert len(interview_ids) == 10, f"expected 10 interviews, got {len(interview_ids)}"
    assert len(chunks) > 100, "expected at least 100 chunks across the corpus"


def test_chunk_to_embedding_text_includes_context():
    """The text we embed should include surrounding turns, not just the turn itself."""
    chunk = Chunk(
        interview_id="test",
        turn_index=5,
        speaker="SARAH",
        text="the buffer was zero",
        context_before="ALEX: What went wrong?",
        context_after="ALEX: Ouch.",
    )
    embedded = chunk.to_embedding_text()
    assert "What went wrong" in embedded
    assert "buffer was zero" in embedded
    assert "Ouch" in embedded


def test_chunk_id_is_stable():
    """chunk_id should be deterministic and zero-padded for sorting."""
    chunk = Chunk(interview_id="01_sarah", turn_index=7, speaker="SARAH", text="x")
    assert chunk.chunk_id == "01_sarah#turn_007"


if __name__ == "__main__":
    import inspect
    import sys

    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures.append((name, e))
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - len(failures)} passed, {len(failures)} failed")
    sys.exit(1 if failures else 0)
