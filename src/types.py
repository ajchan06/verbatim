"""Core types for Verbatim.

The Pipeline protocol is the contract every implementation must satisfy. By
defining it before any implementation exists, we can:

1. Build the eval harness first and run it against a stub
2. Swap implementations (naive RAG, agent v1, agent v2) without touching evals
3. A/B compare implementations by running the same evals against each

This separation of "what the system does" from "how it does it" is the move
that lets us measure iteration honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Citation:
    """A pointer back to a specific moment in a specific interview.

    Citations are how we tie answers to sources. Every claim in an answer
    should be backed by at least one citation; otherwise we can't trust it.
    """

    interview_id: str  # e.g. "05_diana"
    speaker: str | None = None  # e.g. "DIANA" — None for whole-interview citations
    excerpt: str = ""  # the verbatim text being cited
    score: float | None = None  # retrieval similarity score, if available


@dataclass
class PipelineAnswer:
    """The result of a pipeline answering a single question.

    `answer` is the natural-language response shown to the user.
    `citations` are the underlying retrievals that should support the answer.
    `interviews_used` is the unique set of interview_ids the pipeline consulted —
        this is what the retrieval eval scores against.
    `metadata` is whatever else the pipeline wants to surface (latency, token
        counts, tool calls made, etc).
    """

    answer: str
    citations: list[Citation] = field(default_factory=list)
    interviews_used: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Pipeline(Protocol):
    """A Pipeline answers natural-language questions about a corpus of interviews.

    Implementations:
      - StubPipeline: returns canned data for testing the harness itself
      - NaiveRagPipeline: retrieve-once-then-stuff-into-prompt baseline
      - AgentPipeline: tool-using agent that decides what to fetch

    Each implementation is a separate file. The eval harness doesn't care
    which one it's running.
    """

    name: str  # human-readable identifier shown in eval output

    def answer(self, question: str) -> PipelineAnswer:
        """Answer a question about the corpus. Should not raise on bad input —
        return a PipelineAnswer with an explanatory message instead."""
        ...
