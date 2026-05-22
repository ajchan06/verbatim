"""Eval harness for Verbatim.

Loads the ground-truth Q&A set, runs each question through a Pipeline
implementation, and scores the results on:

  - Retrieval precision  — of the interviews the pipeline used, how many were
                           in the expected set?
  - Retrieval recall     — of the expected interviews, how many did the
                           pipeline use?
  - F1                   — harmonic mean of the two
  - Faithfulness         — for synthesis/single-fact questions, does the
                           answer actually contain the expected substring(s)?
                           (Cheap substring check first; LLM-as-judge is
                           layered on later in src/judge.py)

Why build this before the real pipeline:
  Every change to chunking, embedding, retrieval, or agent design becomes
  measurable. Without this, "this version feels better" is the best we can do.
  With this, we have numbers.

Run:
    python -m src.eval --pipeline stub
    python -m src.eval --pipeline naive_rag
    python -m src.eval --pipeline agent_v1
    python -m src.eval --compare stub agent_v1
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .types import Pipeline, PipelineAnswer


# ---------- Loading ground truth ----------


REPO_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_PATH = REPO_ROOT / "evals" / "ground_truth.json"


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[dict]:
    """Load the eval questions from disk."""
    with path.open() as f:
        data = json.load(f)
    return data["questions"]


# ---------- Scoring ----------


@dataclass
class QuestionResult:
    """The score for a single question. Lives in eval output files so we can
    diff runs and find regressions."""

    question_id: str
    question: str
    pipeline_name: str
    expected_interviews: list[str]
    actual_interviews: list[str]
    precision: float
    recall: float
    f1: float
    faithfulness_pass: bool | None  # None = not applicable for this question type
    latency_seconds: float
    answer: str
    notes: str = ""


def precision_recall_f1(expected: list[str], actual: list[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 on two sets of interview IDs.

    Edge cases:
      - If expected is empty: precision/recall are undefined; we return 1.0 if
        actual is also empty (correctly returned nothing) else 0.0.
      - If actual is empty but expected isn't: precision is undefined (0/0);
        we return 0.0 for precision since the pipeline returned nothing useful.
    """
    expected_set = set(expected)
    actual_set = set(actual)

    if not expected_set and not actual_set:
        return 1.0, 1.0, 1.0
    if not expected_set:
        return 0.0, 1.0, 0.0  # pipeline returned interviews when none expected
    if not actual_set:
        return 0.0, 0.0, 0.0  # pipeline returned nothing

    tp = len(expected_set & actual_set)
    precision = tp / len(actual_set)
    recall = tp / len(expected_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def faithfulness_substring(answer: str, expected_contains: list[str]) -> bool:
    """Cheap faithfulness check: does the answer literally contain at least one
    of the expected substrings? Case-insensitive.

    This is the floor. A real production system would also use LLM-as-judge to
    verify the answer is supported by the retrieved interviews — see src/judge.py.

    The reason we do this cheap check first: it's free, it's deterministic, and
    it catches the obvious failures. LLM-as-judge is for the harder cases.
    """
    ans_lower = answer.lower()
    return any(sub.lower() in ans_lower for sub in expected_contains)


# ---------- Running a pipeline ----------


def score_question(pipeline: Pipeline, question: dict) -> QuestionResult:
    """Run a single question through the pipeline and score it."""
    qid = question["id"]
    qtext = question["question"]

    # Expected interview set — different question types use different fields.
    expected = (
        question.get("expected_interviews")
        or question.get("expected_interviews_subset")
        or []
    )

    # Run the pipeline, timed.
    t0 = time.perf_counter()
    try:
        result: PipelineAnswer = pipeline.answer(qtext)
        error = None
    except Exception as e:
        # Don't let one bad question kill the whole eval run.
        result = PipelineAnswer(answer=f"(error: {e})", interviews_used=[])
        error = str(e)
    latency = time.perf_counter() - t0

    precision, recall, f1 = precision_recall_f1(expected, result.interviews_used)

    # Faithfulness only applies to question types that specify expected
    # substrings (single_interview_fact, synthesis).
    expected_contains = question.get("expected_answer_contains")
    faithfulness = (
        faithfulness_substring(result.answer, expected_contains)
        if expected_contains
        else None
    )

    notes = ""
    if error:
        notes = f"pipeline raised: {error}"
    elif question.get("type") == "synthesis" and "expected_interviews_subset" in question:
        notes = "scored against expected subset (not exact match)"

    return QuestionResult(
        question_id=qid,
        question=qtext,
        pipeline_name=pipeline.name,
        expected_interviews=list(expected),
        actual_interviews=list(result.interviews_used),
        precision=precision,
        recall=recall,
        f1=f1,
        faithfulness_pass=faithfulness,
        latency_seconds=latency,
        answer=result.answer,
        notes=notes,
    )


def run_eval(pipeline: Pipeline, ground_truth_path: Path = GROUND_TRUTH_PATH) -> list[QuestionResult]:
    """Run the full eval set against a pipeline."""
    questions = load_ground_truth(ground_truth_path)
    return [score_question(pipeline, q) for q in questions]


# ---------- Pretty-printing ----------


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def print_results_table(results: list[QuestionResult]) -> None:
    """Print a human-readable table of eval results."""
    pipeline_name = results[0].pipeline_name if results else "?"
    print(f"\n=== Eval results: pipeline={pipeline_name} ===\n")

    # Per-question rows
    header = f"{'Q':<4} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Faith':>6} {'Lat(s)':>7}  Notes"
    print(header)
    print("-" * len(header))
    for r in results:
        faith = "—" if r.faithfulness_pass is None else ("✓" if r.faithfulness_pass else "✗")
        print(
            f"{r.question_id:<4} "
            f"{_fmt_pct(r.precision):>6} "
            f"{_fmt_pct(r.recall):>6} "
            f"{_fmt_pct(r.f1):>6} "
            f"{faith:>6} "
            f"{r.latency_seconds:>6.2f}s  "
            f"{r.notes}"
        )

    # Aggregates
    print("-" * len(header))
    mean_p = statistics.mean(r.precision for r in results)
    mean_r = statistics.mean(r.recall for r in results)
    mean_f1 = statistics.mean(r.f1 for r in results)
    applicable = [r for r in results if r.faithfulness_pass is not None]
    faith_rate = (
        sum(1 for r in applicable if r.faithfulness_pass) / len(applicable)
        if applicable
        else None
    )
    mean_lat = statistics.mean(r.latency_seconds for r in results)
    faith_str = "n/a" if faith_rate is None else f"{_fmt_pct(faith_rate).strip()} ({len(applicable)} q)"
    print(
        f"{'MEAN':<4} "
        f"{_fmt_pct(mean_p):>6} "
        f"{_fmt_pct(mean_r):>6} "
        f"{_fmt_pct(mean_f1):>6} "
        f"{faith_str:>6}  "
        f"{mean_lat:>6.2f}s"
    )
    print()


def save_results(results: list[QuestionResult], pipeline_name: str) -> Path:
    """Persist results to disk so we can diff runs later."""
    out_dir = REPO_ROOT / "evals" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{pipeline_name}_{timestamp}.json"
    with out_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    return out_path


# ---------- Comparison mode (A/B) ----------


def print_comparison(a_results: list[QuestionResult], b_results: list[QuestionResult]) -> None:
    """Side-by-side comparison of two pipelines."""
    a_name = a_results[0].pipeline_name
    b_name = b_results[0].pipeline_name
    print(f"\n=== Comparison: {a_name} vs {b_name} ===\n")

    header = f"{'Q':<4} {'F1 ' + a_name:>14} {'F1 ' + b_name:>14} {'Δ':>8}"
    print(header)
    print("-" * len(header))

    by_id_a = {r.question_id: r for r in a_results}
    by_id_b = {r.question_id: r for r in b_results}
    for qid in sorted(by_id_a.keys()):
        ra, rb = by_id_a[qid], by_id_b[qid]
        delta = rb.f1 - ra.f1
        sign = "+" if delta > 0 else ("-" if delta < 0 else " ")
        print(
            f"{qid:<4} "
            f"{_fmt_pct(ra.f1):>14} "
            f"{_fmt_pct(rb.f1):>14} "
            f"{sign}{abs(delta) * 100:6.1f}pp"
        )
    print("-" * len(header))
    mean_a = statistics.mean(r.f1 for r in a_results)
    mean_b = statistics.mean(r.f1 for r in b_results)
    mean_delta = mean_b - mean_a
    sign = "+" if mean_delta > 0 else ("-" if mean_delta < 0 else " ")
    print(f"{'MEAN':<4} {_fmt_pct(mean_a):>14} {_fmt_pct(mean_b):>14} "
          f"{sign}{abs(mean_delta) * 100:6.1f}pp\n")


# ---------- Pipeline loading ----------


def load_pipeline(name: str) -> Pipeline:
    """Load a pipeline by short name. Add new pipelines here as they're built."""
    if name == "stub":
        from .stub_pipeline import StubPipeline
        return StubPipeline()
    if name == "naive_rag":
        from .naive_rag import NaiveRagPipeline
        return NaiveRagPipeline()
    if name == "agent_v1":
        from .agent import AgentPipeline
        return AgentPipeline()
    raise SystemExit(f"unknown pipeline: {name}")


# ---------- CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evals against a pipeline.")
    parser.add_argument("--pipeline", default="stub", help="pipeline name (stub|naive_rag|agent_v1)")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="compare two pipelines side-by-side")
    parser.add_argument("--save", action="store_true", help="save results to evals/runs/")
    parser.add_argument(
        "--ground-truth",
        default=str(GROUND_TRUTH_PATH),
        help="path to ground-truth JSON (use evals/job_ground_truth.json for the job-interview corpus)",
    )
    args = parser.parse_args()
    gt_path = Path(args.ground_truth)

    if args.compare:
        a, b = args.compare
        ra = run_eval(load_pipeline(a), gt_path)
        rb = run_eval(load_pipeline(b), gt_path)
        print_results_table(ra)
        print_results_table(rb)
        print_comparison(ra, rb)
        if args.save:
            print(f"saved: {save_results(ra, a)}")
            print(f"saved: {save_results(rb, b)}")
    else:
        results = run_eval(load_pipeline(args.pipeline), gt_path)
        print_results_table(results)
        if args.save:
            print(f"saved: {save_results(results, args.pipeline)}")


if __name__ == "__main__":
    main()
