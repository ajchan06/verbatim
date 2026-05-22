"""Transcript chunking.

The fundamental retrieval unit in Verbatim is the **speaker turn** — one
chunk of text spoken by one person before another person speaks. This is
the natural unit of interview content; chunking by character count (the
default in most RAG tutorials) would split a single insight across
multiple chunks and dilute its searchability.

For each speaker turn we also keep the *previous and next turns* as
context. This matters because interviewer questions ("Why did you
cancel?") often contain words the participant won't repeat in their
answer ("Because the buffer time was zero..."). Without the surrounding
context, a query about "cancellation reasons" wouldn't retrieve the
answer turn, only the question turn.

Chunk shape (one per speaker turn):
    {
        "interview_id": "01_sarah",
        "turn_index": 7,           # ordinal position of this turn
        "speaker": "SARAH",
        "text": "...",             # what the speaker said in this turn
        "context_before": "...",   # the previous turn (often the question)
        "context_after": "...",    # the following turn
        "metadata": {...}          # interview frontmatter (participant, role, etc)
    }

When we embed a chunk we embed `f"{context_before}\n{speaker}: {text}\n{context_after}"`
so that retrieval matches on the surrounding conversational context as
well as the turn itself.

When we *display* a chunk we usually show just `text` (clean) and surface
the surrounding turns separately if the user wants more context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# Matches lines like "SARAH: I made an account..." or "ALEX: Thanks for joining."
# Speaker is uppercase letters/spaces, immediately followed by a colon and a space.
_TURN_RE = re.compile(r"^([A-Z][A-Z ]+):\s+(.*)$")


@dataclass
class Chunk:
    """One speaker turn from one interview, ready to embed and retrieve."""

    interview_id: str
    turn_index: int
    speaker: str
    text: str
    context_before: str = ""
    context_after: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.interview_id}#turn_{self.turn_index:03d}"

    def to_embedding_text(self) -> str:
        """The text we feed to the embedding model.

        Includes surrounding context so that questions and answers
        retrieve together — the interviewer's wording often contains
        keywords the participant doesn't repeat.
        """
        parts = []
        if self.context_before:
            parts.append(self.context_before)
        parts.append(f"{self.speaker}: {self.text}")
        if self.context_after:
            parts.append(self.context_after)
        return "\n".join(parts)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Pull YAML-ish frontmatter from the top of a markdown file.

    We don't pull in a full YAML parser; the frontmatter format is
    simple key: value lines and we just need a few fields. Returns
    (frontmatter_dict, remaining_body).
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    header_block, body = parts[1], parts[2]
    meta = {}
    for line in header_block.strip().splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        meta[key.strip()] = val.strip()
    return meta, body


def _parse_turns(body: str) -> list[tuple[str, str]]:
    """Walk the transcript body and group lines into (speaker, text) turns.

    Lines that don't start with a speaker label are appended to the
    current turn (some real transcripts wrap long turns across multiple
    lines). Blank lines are turn separators.
    """
    turns: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_speaker is not None and current_lines:
            text = " ".join(line.strip() for line in current_lines if line.strip())
            if text:
                turns.append((current_speaker, text))

    for line in body.splitlines():
        stripped = line.strip()
        m = _TURN_RE.match(stripped) if stripped else None
        if m:
            flush()
            current_speaker = m.group(1).strip()
            current_lines = [m.group(2)]
        elif stripped:
            if current_speaker is not None:
                current_lines.append(stripped)
        # blank line: keep accumulating; turns are separated by speaker change
    flush()
    return turns


def chunk_transcript(path: Path) -> list[Chunk]:
    """Parse one transcript file into chunks (one per speaker turn).

    Includes context_before / context_after so retrieval has room to match
    on conversational context, not just the literal turn.
    """
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    interview_id = path.stem  # e.g. "01_sarah"
    turns = _parse_turns(body)

    chunks: list[Chunk] = []
    for i, (speaker, turn_text) in enumerate(turns):
        before = ""
        after = ""
        if i > 0:
            prev_speaker, prev_text = turns[i - 1]
            before = f"{prev_speaker}: {prev_text}"
        if i < len(turns) - 1:
            next_speaker, next_text = turns[i + 1]
            after = f"{next_speaker}: {next_text}"
        chunks.append(
            Chunk(
                interview_id=interview_id,
                turn_index=i,
                speaker=speaker,
                text=turn_text,
                context_before=before,
                context_after=after,
                metadata=meta,
            )
        )
    return chunks


def chunk_corpus(transcript_dir: Path) -> list[Chunk]:
    """Chunk every .md transcript in a directory into a flat list of chunks."""
    chunks: list[Chunk] = []
    for path in sorted(transcript_dir.glob("*.md")):
        chunks.extend(chunk_transcript(path))
    return chunks
