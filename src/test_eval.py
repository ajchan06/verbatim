"""Tests for the eval harness math.

These are pure-function tests — no LLM calls, no I/O, no fixtures. They run in
milliseconds and protect against silent regressions in precision/recall/F1.
The expected values are calculated by hand in the comments so a reviewer can
verify the test cases themselves, not just trust the code.

Run:
    python -m pytest src/test_eval.py -v
    # or, without pytest:
    python src/test_eval.py
"""

from __future__ import annotations

from .eval import faithfulness_substring, precision_recall_f1


# ---------- precision_recall_f1 ----------


def test_perfect_match():
    # expected == actual → all 1.0
    p, r, f = precision_recall_f1(["a", "b", "c"], ["a", "b", "c"])
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_extra_in_actual_hurts_precision():
    # expected = {a, b}; actual = {a, b, c}
    # TP=2, precision = 2/3, recall = 2/2 = 1.0, F1 = 2 * (2/3 * 1) / (5/3) = 0.8
    p, r, f = precision_recall_f1(["a", "b"], ["a", "b", "c"])
    assert p == 2 / 3
    assert r == 1.0
    assert abs(f - 0.8) < 1e-9


def test_missing_in_actual_hurts_recall():
    # expected = {a, b, c}; actual = {a, b}
    # precision = 1.0, recall = 2/3, F1 = 0.8
    p, r, f = precision_recall_f1(["a", "b", "c"], ["a", "b"])
    assert p == 1.0
    assert r == 2 / 3
    assert abs(f - 0.8) < 1e-9


def test_zero_overlap():
    p, r, f = precision_recall_f1(["a", "b"], ["c", "d"])
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_empty_actual():
    # pipeline returned nothing but something was expected → 0/0/0
    p, r, f = precision_recall_f1(["a", "b"], [])
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_empty_expected_and_actual():
    # both empty: pipeline correctly returned nothing → perfect
    p, r, f = precision_recall_f1([], [])
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_empty_expected_nonempty_actual():
    # nothing expected but pipeline returned things → precision should be 0
    p, r, f = precision_recall_f1([], ["a", "b"])
    assert p == 0.0
    assert f == 0.0


def test_duplicates_in_actual_dont_double_count():
    # the function uses set semantics; duplicates collapse
    p, r, f = precision_recall_f1(["a", "b"], ["a", "a", "b"])
    assert (p, r, f) == (1.0, 1.0, 1.0)


# ---------- faithfulness_substring ----------


def test_faithfulness_pass_simple():
    assert faithfulness_substring("Sarah's buffer was zero", ["buffer", "default"]) is True


def test_faithfulness_pass_any_substring_counts():
    # only one of the expected substrings needs to appear
    assert faithfulness_substring("the answer mentions buffer", ["buffer", "default", "zero"]) is True


def test_faithfulness_fail():
    assert faithfulness_substring("the answer is about pricing", ["buffer", "default"]) is False


def test_faithfulness_case_insensitive():
    assert faithfulness_substring("THE BUFFER WAS SET TO ZERO", ["buffer"]) is True
    assert faithfulness_substring("buffer", ["BUFFER"]) is True


# ---------- bare-bones runner so this works without pytest ----------


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
