"""MBPP coding problems as a source: the question is the description plus
one example call, the answer is the reference solution, and the score is
the fraction of hidden test calls the answer passes when run in a sandbox.
Each MBPP test is `assert f(args) == value`; we keep the call, drop the
value, and recompute the expected output by running the reference through
the same grader, so grading never needs the original assert.
"""

from __future__ import annotations

import ast
import builtins
import json
import random
from pathlib import Path

from cheatbox.sandbox import Sandbox
from cheatbox.sources import Problem

SPLITS = ["train", "validation", "prompt", "test"]
HARNESS = Path(__file__).resolve().parent.parent / "harness.py"
BUILTINS = set(dir(builtins))


class Mbpp:
    """`Mbpp()`: the google-research MBPP dataset, all of it."""

    def problems(self, n: int, seed: int) -> list[Problem]:
        """Up to `n` gradable problems, in a seeded order."""
        from datasets import load_dataset

        mbpp = load_dataset("google-research-datasets/mbpp", "full")
        rows = [r for s in SPLITS for r in mbpp[s]]
        random.Random(seed).shuffle(rows)
        problems = []
        for row in rows:
            if len(problems) == n:
                break
            if problem := convert(row):
                problems.append(problem)
        return problems

    def score(self, spec: dict, answer: str) -> float:
        """The fraction of test calls whose result matches the expected repr."""
        results = run_tests(answer, spec["calls"])
        passed = [
            same(got, want) for got, want in zip(results, spec["expected"])
        ]
        return sum(passed) / len(passed)


def run_tests(
    code: str, calls: list[str], timeout: int = 10
) -> list[str | None]:
    """Evaluate each call against `code` in a fresh sandbox; the reprs, None
    where a call raised.
    """
    files = {
        "solution.py": code,
        "calls.json": json.dumps(list(calls)),
        "harness.py": HARNESS.read_text(),
    }
    sb = Sandbox(files)
    try:
        sb.run("python harness.py", timeout=timeout)
        raw = sb.read("results.json")
    finally:
        sb.close()
    try:
        results = json.loads(raw) if raw else []
    except ValueError:
        results = []
    results = [r if isinstance(r, str) else None for r in results]
    return (results + [None] * len(calls))[: len(calls)]


def same(got: str | None, want: str) -> bool:
    """Compare two `repr` strings by Python equality when both parse as
    literals, else textually.
    """
    if got is None:
        return False
    try:
        return ast.literal_eval(got) == ast.literal_eval(want)
    except Exception:
        return got == want


def convert(row: dict) -> Problem | None:
    """One MBPP row as a Problem, or None if its tests are not reproducible
    by the grader.
    """
    if row["test_setup_code"].strip():
        return None
    parts = [split_test(t) for t in row["test_list"]]
    if not parts or any(p is None for p in parts):
        return None
    calls, values = map(list, zip(*parts))
    entry = entry_point(calls[0])
    if entry is None or any(entry_point(c) != entry for c in calls):
        return None
    if any(names(c) - BUILTINS - {entry} for c in calls):
        return None
    results = run_tests(row["code"], calls)
    if not all(same(got, want) for got, want in zip(results, values)):
        return None
    question = f"{row['text']}\nIt will be called like `{calls[0]}`."
    return Problem(question, row["code"], {"calls": calls, "expected": results})


def split_test(test: str) -> tuple[str, str] | None:
    """'assert f(x) == y' -> ('f(x)', 'y')."""
    try:
        body = ast.parse(test).body
    except SyntaxError:
        return None
    if len(body) != 1 or not isinstance(body[0], ast.Assert):
        return None
    cmp = body[0].test
    if not (
        isinstance(cmp, ast.Compare)
        and len(cmp.ops) == 1
        and isinstance(cmp.ops[0], ast.Eq)
    ):
        return None
    return ast.unparse(cmp.left), ast.unparse(cmp.comparators[0])


def entry_point(call: str) -> str | None:
    """Name of the function a test call invokes, or None if the call is not
    a plain `name(...)`.
    """
    node = ast.parse(call, mode="eval").body
    return (
        node.func.id
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        else None
    )


def names(expr: str) -> set[str]:
    """All identifiers used in an expression."""
    return {
        n.id
        for n in ast.walk(ast.parse(expr, mode="eval"))
        if isinstance(n, ast.Name)
    }
