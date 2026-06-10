"""smolagents CodeAgent wired up with the data tools.

Built while working through the Hugging Face Agents Course (Units 1-3).
"""

import os
import urllib.request

from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, OpenAIServerModel

from tools import ALL_TOOLS

DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "anthropic/claude-sonnet-4-6")
DEFAULT_HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are a data analysis assistant. The user asks questions in plain English; you call the
provided tools to fetch data and analyse it.

Pick a source if no dataset is loaded yet: load_hf_dataset for the built-in shortcuts, fetch_kaggle_dataset
for Kaggle, or load_dataset for a local CSV. Use describe_dataset only when you need column details you
don't already have. Use filter_data, aggregate_data, correlation_analysis, create_visualization and
generate_report as appropriate.

filter_data is persistent: after you filter, every later tool operates on the filtered subset. Call
reset_filters before an analysis that needs the full dataset again.

When create_visualization returns a marker like [CHART:/path/to/file.png], copy that marker verbatim into
your final answer so the chart is shown to the user. Do not paraphrase or drop it.

Quote concrete numbers in your final answer. Do not invent values that the tools did not return.
"""


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _build_model(hf_token, anthropic_key):
    if anthropic_key:
        # pass the key into the client rather than mutating process-wide env, so
        # one user's key can't leak into another session sharing the process
        return (
            LiteLLMModel(model_id=DEFAULT_ANTHROPIC_MODEL, api_key=anthropic_key),
            DEFAULT_ANTHROPIC_MODEL,
        )

    token = hf_token or os.getenv("HF_TOKEN")
    if token:
        return InferenceClientModel(model_id=DEFAULT_HF_MODEL, token=token), DEFAULT_HF_MODEL

    if _ollama_running():
        return OpenAIServerModel(
            model_id=DEFAULT_OLLAMA_MODEL,
            api_base=f"{OLLAMA_URL}/v1",
            api_key="ollama",
        ), f"{DEFAULT_OLLAMA_MODEL} (local Ollama)"

    raise RuntimeError(
        "No model available. Provide an HF token, an Anthropic key, or run Ollama locally."
    )


def build_agent(hf_token=None, anthropic_key=None):
    model, model_name = _build_model(hf_token, anthropic_key)
    agent = CodeAgent(
        tools=ALL_TOOLS,
        model=model,
        instructions=SYSTEM_PROMPT,
        max_steps=10,
        verbosity_level=1,
    )
    return agent, model_name
