"""Naive RAG pipeline (the baseline).

This is the simplest possible "real" pipeline:
  1. Embed the question with the same encoder we used at ingest
  2. Retrieve the top-K most-similar chunks across all interviews
  3. Stuff those chunks into a prompt
  4. Ask Claude to answer using only that context
  5. Return the answer plus the interviews we touched

We build this BEFORE the agent pipeline for one reason: so we can measure
what the agent actually adds. If the agent doesn't beat naive RAG on the
evals, we don't need the agent.

Known weaknesses we expect to see in the eval scores:
  - Multi-interview questions ("which interviewees mentioned X?") are
    bounded by K — if K=10 and the right answer is in 5 interviews but our
    top 10 chunks happen to all come from 3 of them, we miss two.
  - Synthesis questions ("top three reasons") get a single shot; the
    model can't go back and search for more.
  - Specific-fact questions ("Chloe's team size") only work if the right
    turn is in the top K.

We'll quantify these gaps once we run the eval, then design the agent's
tools to specifically address them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from .types import Citation, Pipeline, PipelineAnswer
from .vector_store import Retrieval, VectorStore


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


# Claude Sonnet — best balance of quality and cost for the agent + naive RAG.
# If the eval shows naive RAG is good enough on this corpus, you could swap
# to claude-haiku-4-5 to cut cost ~5x.
CLAUDE_MODEL = "claude-sonnet-4-6"


SYSTEM_PROMPT = """\
You are Verbatim, an assistant that answers questions about a corpus of
customer research interviews.

Rules you MUST follow:
1. Only use facts present in the provided interview excerpts. Do not
   invent details or speculate beyond what's there.
2. When the user asks "which interviewees said X" or "who mentioned Y",
   list every distinct interview_id whose excerpts support the claim.
3. When you reference a specific claim, cite the interview_id in
   brackets, e.g. [05_diana].
4. Prefer the participant's actual words. Short verbatim quotes are
   better than paraphrase. Wrap quotes in double quotes.
5. If the excerpts don't contain enough information to answer, say so
   honestly. Do not pretend to know.

At the end of every answer, on its own final line, output:
    INTERVIEWS_USED: <comma-separated list of interview_ids you actually used>
This line is parsed by tooling; keep it exact.
"""


def _format_retrievals(retrievals: list[Retrieval]) -> str:
    """Render retrieved chunks into a prompt-friendly text block."""
    parts: list[str] = []
    for r in retrievals:
        c = r.chunk
        participant = c.metadata.get("participant", "")
        role = c.metadata.get("role", "")
        header = f"[{c.interview_id}] {participant} — {role}".strip(" —")
        parts.append(
            f"--- {header} (turn {c.turn_index}, similarity={r.score:.2f}) ---\n"
            + (f"(prior turn) {c.context_before}\n" if c.context_before else "")
            + f"{c.speaker}: {c.text}\n"
            + (f"(next turn) {c.context_after}\n" if c.context_after else "")
        )
    return "\n".join(parts)


_INTERVIEWS_USED_RE = re.compile(r"INTERVIEWS_USED:\s*(.+?)\s*$", re.MULTILINE)


def _parse_interviews_used(answer_text: str) -> list[str]:
    """Pull the trailing INTERVIEWS_USED line out of the model's answer."""
    m = _INTERVIEWS_USED_RE.search(answer_text)
    if not m:
        return []
    raw = m.group(1).strip()
    if raw.lower() in {"none", "n/a", "-"}:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _strip_interviews_used_line(answer_text: str) -> str:
    return _INTERVIEWS_USED_RE.sub("", answer_text).strip()


class NaiveRagPipeline(Pipeline):
    """Retrieve-once, generate-once. The baseline."""

    name = "naive_rag"

    def __init__(self, *, top_k: int = 10, collection_name: str = "interviews"):
        self.top_k = top_k
        self.store = VectorStore(collection_name=collection_name)
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env

        if self.store.count() == 0:
            raise RuntimeError(
                "Vector store is empty. Run `python -m src.ingest` first."
            )

    def answer(self, question: str) -> PipelineAnswer:
        # Step 1: retrieve
        retrievals = self.store.search(question, k=self.top_k)

        # Step 2: build prompt
        context_block = _format_retrievals(retrievals)
        user_message = (
            f"Question: {question}\n\n"
            f"Relevant interview excerpts:\n\n{context_block}"
        )

        # Step 3: generate
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        # Step 4: parse the model's declared interview list
        declared = _parse_interviews_used(raw_text)
        clean_answer = _strip_interviews_used_line(raw_text)

        # Build citations from the retrievals that came from declared interviews.
        citations = [
            Citation(
                interview_id=r.chunk.interview_id,
                speaker=r.chunk.speaker,
                excerpt=r.chunk.text,
                score=r.score,
            )
            for r in retrievals
            if r.chunk.interview_id in declared
        ]

        return PipelineAnswer(
            answer=clean_answer,
            citations=citations,
            interviews_used=declared,
            metadata={
                "top_k": self.top_k,
                "model": CLAUDE_MODEL,
                "retrieved_interviews": sorted(
                    {r.chunk.interview_id for r in retrievals}
                ),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
