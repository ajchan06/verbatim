"""Agent pipeline: Claude with tool use, multi-step retrieval.

Where naive RAG is a single retrieve-and-stuff, the agent is a loop:
  - Claude sees the question
  - Decides which tool to call
  - Gets the result
  - Decides whether to call another tool or answer
  - (repeat up to MAX_STEPS)

Tools we expose:
  1. search_interviews(query, k)   — semantic search over the corpus
  2. get_full_transcript(id)       — full text of one interview
  3. find_quotes(phrase, max=10)   — keyword / phrase search for verbatim quotes
  4. list_interviews()             — corpus index (id, participant, role, recruited_as)

Why these four:
  search_interviews   — the workhorse, same as naive RAG
  get_full_transcript — for "what did X say about Y" — load Sarah's whole
                        transcript instead of guessing which chunks to pull
  find_quotes         — semantic search misses exact-phrase needs ("who said
                        'forty-eight bucks'?"); keyword search complements it
  list_interviews     — gives the agent a corpus map so it can pick targets
                        instead of blind-searching ("who are the team admins?")

The agent is the thing that should *beat* naive RAG on the eval. Where
naive RAG is bounded by K (it can only see what its single retrieval
returned), the agent can do follow-up retrievals when its first one
was incomplete.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from .chunker import chunk_transcript
from .types import Citation, Pipeline, PipelineAnswer
from .vector_store import VectorStore


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_STEPS = 8  # safety bound on the agent loop


SYSTEM_PROMPT = """\
You are Verbatim, an assistant that answers questions about a corpus of
customer research interviews.

You have tools to search the corpus, pull full transcripts, find exact
quotes, and list the interview index. Use them iteratively — your first
search rarely surfaces everything.

Workflow you should generally follow:
1. If the question asks "who", "which interviewees", "how many", or
   otherwise requires enumerating across the corpus, do a broad
   search_interviews first, then follow up if you suspect you're
   missing relevant interviews. Use list_interviews to see the full set.
2. If the question is about a specific named person, prefer
   get_full_transcript over guessing which chunks to search for.
3. For exact-phrase or quote questions, use find_quotes.
4. Stop searching once you have enough evidence; don't burn calls.

Answer rules:
- Only use facts present in tool results. Don't invent details.
- When citing claims, reference the interview_id in brackets, e.g.
  [05_diana].
- Prefer verbatim quotes (in double quotes) over paraphrase.
- If the evidence is insufficient, say so honestly.

At the end of every answer, on its own final line, output:
    INTERVIEWS_USED: <comma-separated interview_ids you actually used>
This line is parsed by tooling; keep it exact.
"""


# ---------- Tool schemas (what Claude sees) ----------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_interviews",
        "description": (
            "Semantic search over interview turns. Returns the most relevant "
            "speaker turns across all interviews. Best for thematic questions "
            "('what do users say about pricing?') and finding evidence across "
            "the corpus. Returns chunks with interview_id, speaker, text, and "
            "surrounding context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default 10, max 25).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_full_transcript",
        "description": (
            "Load the full text of a single interview by interview_id. Use this "
            "when the question is specifically about one named person or one "
            "interview and you'd rather see the whole conversation than guess "
            "which chunks to search for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "interview_id": {
                    "type": "string",
                    "description": "The interview_id, e.g. '05_diana'.",
                },
            },
            "required": ["interview_id"],
        },
    },
    {
        "name": "find_quotes",
        "description": (
            "Literal substring search across all turns. Use this when you want "
            "exact phrase matches (e.g. specific numbers, product names, or "
            "quotes) that semantic search may miss. Case-insensitive."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phrase": {
                    "type": "string",
                    "description": "Substring to find. Case-insensitive.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matches to return (default 10).",
                },
            },
            "required": ["phrase"],
        },
    },
    {
        "name": "list_interviews",
        "description": (
            "Return a directory of all interviews in the corpus: interview_id, "
            "participant, role, and how they were recruited (e.g. 'churned trial "
            "user', 'active paying customer'). Use this to understand the corpus "
            "shape before searching."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ---------- Parsing helpers (shared with naive_rag) ----------

_INTERVIEWS_USED_RE = re.compile(r"INTERVIEWS_USED:\s*(.+?)\s*$", re.MULTILINE)


def _parse_interviews_used(answer_text: str) -> list[str]:
    m = _INTERVIEWS_USED_RE.search(answer_text)
    if not m:
        return []
    raw = m.group(1).strip()
    if raw.lower() in {"none", "n/a", "-"}:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _strip_interviews_used_line(answer_text: str) -> str:
    return _INTERVIEWS_USED_RE.sub("", answer_text).strip()


# ---------- Pipeline ----------


class AgentPipeline(Pipeline):
    """Tool-using agent. Runs multi-step retrieval."""

    name = "agent_v1"

    def __init__(
        self,
        *,
        collection_name: str = "interviews",
        transcript_dir: Path = REPO_ROOT / "transcripts",
    ):
        self.store = VectorStore(collection_name=collection_name)
        self.transcript_dir = transcript_dir
        self.client = Anthropic()

        if self.store.count() == 0:
            raise RuntimeError(
                "Vector store is empty. Run `python -m src.ingest` first."
            )

    # ---------- Tool implementations ----------

    def _tool_search_interviews(self, query: str, k: int = 10) -> dict[str, Any]:
        k = max(1, min(25, k))
        retrievals = self.store.search(query, k=k)
        return {
            "results": [
                {
                    "interview_id": r.chunk.interview_id,
                    "speaker": r.chunk.speaker,
                    "text": r.chunk.text,
                    "context_before": r.chunk.context_before,
                    "context_after": r.chunk.context_after,
                    "score": round(r.score, 3),
                    "participant": r.chunk.metadata.get("participant", ""),
                }
                for r in retrievals
            ]
        }

    def _tool_get_full_transcript(self, interview_id: str) -> dict[str, Any]:
        path = self.transcript_dir / f"{interview_id}.md"
        if not path.exists():
            return {"error": f"interview not found: {interview_id}"}
        chunks = chunk_transcript(path)
        if not chunks:
            return {"error": f"no parseable turns in {interview_id}"}
        meta = chunks[0].metadata
        return {
            "interview_id": interview_id,
            "participant": meta.get("participant", ""),
            "role": meta.get("role", ""),
            "recruited_as": meta.get("recruited_as", ""),
            "turns": [
                {"speaker": c.speaker, "text": c.text}
                for c in chunks
            ],
        }

    def _tool_find_quotes(self, phrase: str, max_results: int = 10) -> dict[str, Any]:
        needle = phrase.lower()
        matches: list[dict[str, Any]] = []
        for path in sorted(self.transcript_dir.glob("*.md")):
            chunks = chunk_transcript(path)
            for c in chunks:
                if needle in c.text.lower():
                    matches.append(
                        {
                            "interview_id": c.interview_id,
                            "speaker": c.speaker,
                            "text": c.text,
                            "turn_index": c.turn_index,
                        }
                    )
                    if len(matches) >= max_results:
                        return {"matches": matches}
        return {"matches": matches}

    def _tool_list_interviews(self) -> dict[str, Any]:
        entries = []
        for path in sorted(self.transcript_dir.glob("*.md")):
            chunks = chunk_transcript(path)
            if not chunks:
                continue
            meta = chunks[0].metadata
            entries.append(
                {
                    "interview_id": path.stem,
                    "participant": meta.get("participant", ""),
                    "role": meta.get("role", ""),
                    "recruited_as": meta.get("recruited_as", ""),
                }
            )
        return {"interviews": entries}

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "search_interviews":
            return self._tool_search_interviews(**args)
        if name == "get_full_transcript":
            return self._tool_get_full_transcript(**args)
        if name == "find_quotes":
            return self._tool_find_quotes(**args)
        if name == "list_interviews":
            return self._tool_list_interviews()
        return {"error": f"unknown tool: {name}"}

    # ---------- The loop ----------

    def answer(self, question: str) -> PipelineAnswer:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": question}
        ]
        tool_calls_made: list[dict[str, Any]] = []
        interviews_touched: set[str] = set()
        input_tokens_total = 0
        output_tokens_total = 0

        for step in range(MAX_STEPS):
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
            input_tokens_total += response.usage.input_tokens
            output_tokens_total += response.usage.output_tokens

            # Append assistant turn to history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # The model finished without calling more tools — extract final text
                final_text = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                declared = _parse_interviews_used(final_text)
                clean = _strip_interviews_used_line(final_text)
                citations = self._build_citations(tool_calls_made, declared)
                return PipelineAnswer(
                    answer=clean,
                    citations=citations,
                    interviews_used=declared,
                    metadata={
                        "steps": step + 1,
                        "model": CLAUDE_MODEL,
                        "tool_calls": [t["name"] for t in tool_calls_made],
                        "interviews_touched": sorted(interviews_touched),
                        "input_tokens": input_tokens_total,
                        "output_tokens": output_tokens_total,
                    },
                )

            if response.stop_reason == "tool_use":
                # Execute each tool the model requested and append results
                tool_result_blocks: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    args = block.input or {}
                    result = self._dispatch_tool(block.name, args)
                    tool_calls_made.append(
                        {"name": block.name, "input": args, "result_summary": _summarize_result(result)}
                    )
                    for iid in _interviews_in_result(result):
                        interviews_touched.add(iid)
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    )
                messages.append({"role": "user", "content": tool_result_blocks})
                continue

            # Unexpected stop reason — bail with whatever we have
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            return PipelineAnswer(
                answer=final_text or f"(stopped: {response.stop_reason})",
                interviews_used=sorted(interviews_touched),
                metadata={"steps": step + 1, "stop_reason": response.stop_reason},
            )

        # Hit MAX_STEPS — return what we have
        return PipelineAnswer(
            answer="(agent did not converge within step limit)",
            interviews_used=sorted(interviews_touched),
            metadata={
                "steps": MAX_STEPS,
                "model": CLAUDE_MODEL,
                "tool_calls": [t["name"] for t in tool_calls_made],
                "interviews_touched": sorted(interviews_touched),
            },
        )

    def _build_citations(
        self, tool_calls: list[dict[str, Any]], declared: list[str]
    ) -> list[Citation]:
        """Best-effort citation reconstruction from search results that
        touched declared interviews."""
        citations: list[Citation] = []
        for call in tool_calls:
            if call["name"] != "search_interviews":
                continue
            for r in call["result_summary"].get("results", []):
                if r["interview_id"] in declared:
                    citations.append(
                        Citation(
                            interview_id=r["interview_id"],
                            speaker=r.get("speaker"),
                            excerpt=r.get("text", ""),
                            score=r.get("score"),
                        )
                    )
        return citations


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a compact-but-useful summary of a tool result for metadata."""
    if "results" in result:
        return {
            "results": [
                {
                    "interview_id": r.get("interview_id"),
                    "speaker": r.get("speaker"),
                    "text": r.get("text", "")[:200],
                    "score": r.get("score"),
                }
                for r in result["results"]
            ]
        }
    if "turns" in result:
        return {"interview_id": result.get("interview_id"), "turn_count": len(result["turns"])}
    return result


def _interviews_in_result(result: dict[str, Any]) -> set[str]:
    """Pull all interview_ids referenced anywhere in a tool result."""
    found: set[str] = set()
    if "results" in result:
        for r in result["results"]:
            if "interview_id" in r:
                found.add(r["interview_id"])
    if "matches" in result:
        for m in result["matches"]:
            if "interview_id" in m:
                found.add(m["interview_id"])
    if "interview_id" in result:
        found.add(result["interview_id"])
    if "interviews" in result:
        for e in result["interviews"]:
            if "interview_id" in e:
                found.add(e["interview_id"])
    return found
