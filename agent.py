import os
from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, OpenAIServerModel

SYSTEM_PROMPT = """You are a professional Data Analysis Agent with access to real-time data sources.

You can fetch live data from two sources:
1. Hugging Face Datasets Hub — for real-world structured datasets (Spotify, Titanic, Netflix, sales data, job postings, Airbnb, and more)
2. Kaggle — for thousands of community datasets across any domain

When a user asks a question, follow this approach:

STEP 1 — Fetch data if needed:
- For HF built-in datasets → use load_hf_dataset (use list_available_datasets to show options)
- For Kaggle datasets → use search_kaggle_datasets to find, then fetch_kaggle_dataset to load
- For a local CSV → use load_dataset

STEP 2 — Understand the data:
- Use describe_dataset to profile the data before diving into specifics

STEP 3 — Answer the question using appropriate tools:
- filter_data → to subset rows by condition
- aggregate_data → to group and summarise
- correlation_analysis → to find relationships between numeric columns
- create_visualization → to produce charts (bar, line, scatter, histogram, box, heatmap)

STEP 4 — Communicate findings clearly:
- Always explain results in plain, direct language
- Quantify insights — use numbers, percentages, rankings
- When asked for a report → use generate_report with specific findings and recommendations

Be professional, concise, and data-driven. Never guess — always use the tools to get real data.
"""

TOOLS = None  # Lazy-loaded to avoid circular imports


def _get_tools():
    from tools import (
        load_hf_dataset, list_available_datasets,
        fetch_kaggle_dataset, search_kaggle_datasets,
        load_dataset, describe_dataset, filter_data,
        aggregate_data, correlation_analysis,
        create_visualization, generate_report,
    )
    return [
        load_hf_dataset, list_available_datasets,
        fetch_kaggle_dataset, search_kaggle_datasets,
        load_dataset, describe_dataset, filter_data,
        aggregate_data, correlation_analysis,
        create_visualization, generate_report,
    ]


def _build_model(hf_token: str | None, anthropic_key: str | None):
    """
    Model priority:
    1. Anthropic key provided → Claude Sonnet (best quality)
    2. HF token provided → Qwen2.5-72B on HF Inference API (free, strong)
    3. HF_TOKEN env var set → same as above (for HF Spaces deployment)
    """
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        return LiteLLMModel(model_id="anthropic/claude-sonnet-4-6"), "Claude Sonnet 4.6 (Anthropic)"

    token = hf_token or os.getenv("HF_TOKEN")
    if token:
        return InferenceClientModel(
            model_id="Qwen/Qwen2.5-72B-Instruct",
            token=token,
        ), "Qwen2.5-72B (HF Inference)"

    # Fallback: local Ollama if running
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return OpenAIServerModel(
            model_id="qwen3-coder:30b",
            api_base="http://localhost:11434/v1",
            api_key="ollama",
        ), "qwen3-coder:30b (local Ollama)"
    except Exception:
        pass

    raise ValueError(
        "No model available. Enter a Hugging Face token (free) or an Anthropic API key."
    )


def build_agent(hf_token: str | None = None, anthropic_key: str | None = None):
    model, model_name = _build_model(hf_token, anthropic_key)
    agent = CodeAgent(
        tools=_get_tools(),
        model=model,
        instructions=SYSTEM_PROMPT,
        max_steps=15,
        verbosity_level=1,
    )
    return agent, model_name
