"""Run the golden-question fixture against the agent and report pass/fail.

Usage:
    python -m evals.run_evals               # uses HF_TOKEN / ANTHROPIC_API_KEY from env
    python -m evals.run_evals --limit 2     # only run the first 2 questions
    python -m evals.run_evals --provider anthropic --model anthropic/claude-sonnet-5-5

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
from tools import DataframeStore


def run(limit: int | None, provider: str | None = None, model_id: str | None = None) -> int:
    fixture_path = os.path.join(os.path.dirname(__file__), "golden.yaml")
    with open(fixture_path) as f:
        cases = yaml.safe_load(f)
    if limit:
        cases = cases[:limit]

    passed = 0
    for i, case in enumerate(cases):
        # fresh store and agent per case, so no dataset or memory carries over; the
        # store is bound into the tools because the agent runs code on a worker thread
        store = DataframeStore()
        try:
            agent, model_name = build_agent(
                provider=provider, model_id=model_id, store=store,
            ) if provider else build_agent(
                hf_token=os.getenv("HF_TOKEN"), anthropic_key=os.getenv("ANTHROPIC_API_KEY"), store=store,
            )
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 2
        if i == 0:
            print(f"Model: {model_name}")
            print(f"Cases: {len(cases)}\n")
        start = time.time()
        try:
            answer = str(agent.run(case["prompt"]))
            ok = all(needle.lower() in answer.lower() for needle in case["must_contain"])
        except Exception as e:
            answer = f"<error: {e}>"
            ok = False

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
    parser.add_argument("--provider", default=None, help="anthropic | openai | gemini | hf | ollama")
    parser.add_argument("--model", default=None, help="model id; blank uses the provider default")
    args = parser.parse_args()
    sys.exit(run(args.limit, args.provider, args.model))
