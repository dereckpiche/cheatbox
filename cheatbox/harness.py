"""Runs inside the grading sandbox: import solution.py, evaluate each call from
calls.json, write results.json.
"""

import json
import traceback


def main() -> None:
    """Execute the solution, then record `repr` of every call, or None where it
    raised.
    """
    calls = json.load(open("calls.json"))
    results = [None] * len(calls)
    namespace = {"__name__": "__main__"}
    try:
        exec(
            compile(open("solution.py").read(), "solution.py", "exec"),
            namespace,
        )
    except BaseException:
        traceback.print_exc()
    else:
        for i, call in enumerate(calls):
            try:
                results[i] = repr(eval(call, namespace))
            except BaseException:
                traceback.print_exc()
    json.dump(results, open("results.json", "w"))


main()
