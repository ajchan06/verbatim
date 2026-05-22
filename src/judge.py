"""LLM-as-judge for faithfulness.

The substring check in eval.py is a floor; this is the more rigorous test:
given an answer and the cited interview excerpts, does the answer's
content actually appear in those excerpts, or did the model hallucinate?

We use Haiku as the judge — it's much cheaper than Sonnet and faithfulness
checking is a fairly easy task (does X appear in Y, yes or no). For a
production system you'd want to also test the judge itself against
hand-labeled examples, but for a weekend demo Haiku-as-judge is plenty.

This module is optional — the eval harness runs without it. Use it when
you want a deeper read on a specific run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from .chunker import chunk_transcript


REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

JUDGE_MODEL = "claude-haiku-4-5-20251001"

JUDGE_SYSTEM = """\
You are a faithfulness judge. You will be given an answer and a list of
interview excerpts. Decide whether each substantive claim in the answer
is supported by the excerpts.

Return JSON only, no other text:

{
  "supported": true | false,
  "unsupported_claims": ["..."],
  "reasoning": "one sentence"
}

A claim is supported if a reasonable reader would agree the excerpts
contain enough to back it up. Don't require word-for-word identity —
paraphrase is fine. Mark unsupported only when the answer states
something the excerpts do not contain or contradict.
"""


def judge_answer(
    question: str,
    answer: str,
    interview_ids: list[str],
    transcript_dir: Path = REPO_ROOT / "transcripts",
) -> dict:
    """Score an answer for faithfulness against the interviews it claims to use."""
    if not interview_ids:
        return {
            "supported": False,
            "unsupported_claims": ["no interviews cited"],
            "reasoning": "answer cited no interviews",
        }

    excerpts: list[str] = []
    for iid in interview_ids:
        path = transcript_dir / f"{iid}.md"
        if not path.exists():
            excerpts.append(f"[{iid}] (NOT FOUND)")
            continue
        chunks = chunk_transcript(path)
        body = "\n".join(f"{c.speaker}: {c.text}" for c in chunks)
        excerpts.append(f"[{iid}]\n{body}")
    excerpts_block = "\n\n---\n\n".join(excerpts)

    user_message = (
        f"Question: {question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Interview excerpts:\n\n{excerpts_block}"
    )

    client = Anthropic()
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text").strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "supported": False,
            "unsupported_claims": [],
            "reasoning": f"judge returned non-JSON: {raw[:200]}",
        }
