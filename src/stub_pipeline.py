"""Stub pipeline for testing the eval harness.

This pipeline doesn't do retrieval or generation. It returns hand-coded answers
based on simple keyword matching against the question. Its purpose is to verify
the eval harness works end-to-end before we build the real RAG/agent pipelines.

If we run evals against this stub and the harness produces sensible scores
(some questions pass, some fail, in expected ways), we know the harness itself
is correct. Then any failures we see later are pipeline failures, not eval bugs.
"""

from __future__ import annotations

from .types import Citation, Pipeline, PipelineAnswer


class StubPipeline(Pipeline):
    """A deliberately-imperfect pipeline. Some keyword rules will get things
    right; others will miss. That asymmetry is exactly what we want — it lets
    us verify the eval harness can distinguish good answers from bad ones."""

    name = "stub"

    # Hand-coded rules: keywords in question -> (interview_ids, answer text)
    # Deliberately imperfect — some rules are over-broad, some too narrow,
    # so we get a mix of passing/failing eval cases to verify scoring works.
    _RULES: list[tuple[list[str], list[str], str]] = [
        # (keywords, interview_ids, answer)
        (
            ["onboarding", "setup", "steps"],
            ["01_sarah", "02_marcus", "04_tom"],
            "Sarah, Marcus, and Tom all said the onboarding had too many setup steps before they could share a link.",
        ),
        (
            ["mobile", "app"],
            ["03_priya", "05_diana"],  # deliberately missing Yuki — should fail recall
            "Priya and Diana both said the mobile app is worse than the web app.",
        ),
        (
            ["round-robin", "round robin", "routing"],
            ["05_diana", "06_jamal", "08_rafael"],
            "Round-robin routing is the most loved feature among Team-plan customers (Diana, Jamal, Rafael).",
        ),
        (
            ["pricing", "price", "cost", "expensive"],
            ["02_marcus", "09_chloe", "10_yuki", "01_sarah"],  # extra Sarah — should hurt precision
            "Marcus, Chloe, and Yuki all mentioned pricing as a concern.",
        ),
        (
            ["salesforce"],
            ["06_jamal", "08_rafael", "09_chloe"],
            "Jamal, Rafael, and Chloe all said the Salesforce integration is unreliable.",
        ),
        (
            ["notion"],
            ["05_diana", "07_lena"],
            "Diana and Lena both asked for a Notion integration.",
        ),
        (
            ["sarah", "buffer", "trust"],
            ["01_sarah"],
            "Sarah lost trust in the product when a client booked back-to-back with no buffer — the default buffer time was zero.",
        ),
        (
            ["yuki", "savvycal", "switch", "competitor"],
            ["10_yuki"],
            "Yuki switched to SavvyCal specifically for its email-embed feature, which lets you paste availability as clickable times directly into an email.",
        ),
        (
            ["chloe", "team size", "how big", "how many"],
            ["09_chloe"],
            "Chloe's team is 35 people in customer support.",
        ),
        (
            ["loved", "favorite", "best", "most-loved"],
            ["05_diana", "06_jamal", "08_rafael"],
            "The most-loved feature among Team-plan customers is round-robin routing.",
        ),
        (
            ["churn", "trial", "quit", "cancel"],
            ["01_sarah", "02_marcus", "03_priya", "04_tom"],
            "Top reasons trial users churn: too many onboarding setup steps, pricing too high for small teams, and a poor mobile app experience.",
        ),
        (
            ["booking page", "design", "positive", "compliment"],
            ["02_marcus", "05_diana", "07_lena"],  # missing Yuki — recall miss
            "Marcus, Diana, and Lena all spoke positively about the booking page design.",
        ),
    ]

    def answer(self, question: str) -> PipelineAnswer:
        q = question.lower()

        # Find the best-matching rule (most keywords matched)
        best_score = 0
        best_rule = None
        for keywords, interviews, ans in self._RULES:
            score = sum(1 for kw in keywords if kw in q)
            if score > best_score:
                best_score = score
                best_rule = (interviews, ans)

        if best_rule is None:
            return PipelineAnswer(
                answer="I don't have an answer for that question.",
                interviews_used=[],
                citations=[],
                metadata={"matched_rule": None},
            )

        interviews, ans = best_rule
        return PipelineAnswer(
            answer=ans,
            interviews_used=list(interviews),
            citations=[
                Citation(interview_id=iid, excerpt="(stub — no real excerpt)")
                for iid in interviews
            ],
            metadata={"matched_rule_keywords_hit": best_score},
        )
