"""Run the golden-question fixture against the agent and report pass/fail.

Usage:
    python -m evals.run_evals               # uses HF_TOKEN / ANTHROPIC_API_KEY from env
    python -m evals.run_evals --limit 2     # only run the first 2 questions

This is a thin harness on top of `build_agent`. It does not benchmark cost,
latency or model-vs-model accuracy. It is a smoke check so I notice when
something obvious breaks.
"""

import argparse
import os
import sys
import time

import yaml

# allow `python evals/run_evals.py` from project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from agent import build_agent
from tools import DataframeStore, bind_store, _active_store_ctx


def run(limit: int | None) -> int:
    fixture_path = os.path.join(os.path.dirname(__file__), "golden.yaml")
    with open(fixture_path) as f:
        cases = yaml.safe_load(f)
    if limit:
        cases = cases[:limit]

    try:
        agent, model_name = build_agent(
            hf_token=os.getenv("HF_TOKEN"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2

    print(f"Model: {model_name}")
    print(f"Cases: {len(cases)}\n")

    passed = 0
    for case in cases:
        store = DataframeStore()
        token = bind_store(store)
        start = time.time()
        try:
            answer = str(agent.run(case["prompt"]))
            ok = all(needle.lower() in answer.lower() for needle in case["must_contain"])
        except Exception as e:
            answer = f"<error: {e}>"
            ok = False
        finally:
            _active_store_ctx.reset(token)

        secs = time.time() - start
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']} ({secs:.1f}s)")
        if not ok:
            print(f"  expected: {case['must_contain']}")
            print(f"  answer:   {answer[:300]}{'...' if len(answer) > 300 else ''}")
        passed += int(ok)

    print(f"\n{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    sys.exit(run(args.limit))
