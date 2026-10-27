"""smolagents CodeAgent wired up with the data tools.

Built while working through the Hugging Face Agents Course (Units 1-3).
"""

import os
import urllib.request
from dataclasses import dataclass

from smolagents import CodeAgent, InferenceClientModel, LiteLLMModel, OpenAIServerModel

from tools import ALL_TOOLS, DataframeStore, bind_tools

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

SYSTEM_PROMPT = """You are a data analysis assistant. The user asks questions in plain English; you call the
provided tools to fetch data and analyse it.

Pick a source if no dataset is loaded yet: load_hf_dataset for the built-in shortcuts, fetch_kaggle_dataset
for Kaggle, load_dataset for a local CSV, load_cloud_file for s3:// gs:// az:// hf:// or https:// files, and
query_motherduck / query_snowflake / query_bigquery for warehouse tables. list_data_sources shows which
cloud sources are configured. Use describe_dataset only when you need column details you don't already have.

run_sql runs DuckDB SQL against the loaded dataset (table `data`, unfiltered copy `data_full`); prefer it
for multi-step grouping, joins between columns, window functions and top-N questions. Use filter_data,
aggregate_data, correlation_analysis, create_visualization and generate_report as appropriate.

filter_data is persistent: after you filter, every later tool operates on the filtered subset. Call
reset_filters before an analysis that needs the full dataset again.

When create_visualization returns a marker like [CHART:/path/to/file.png], copy that marker verbatim into
your final answer so the chart is shown to the user. Do not paraphrase or drop it.

Quote concrete numbers in your final answer. Do not invent values that the tools did not return.
"""


@dataclass(frozen=True)
class Provider:
    label: str
    default_model: str
    key_env: str | None  # env var holding the API key, None for local models


# default model ids are overridable per provider through env vars, and the UI lets
# you type any model id the provider serves
PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider("Anthropic Claude", os.getenv("ANTHROPIC_MODEL", "anthropic/claude-sonnet-5-5"), "ANTHROPIC_API_KEY"),
    "openai": Provider("OpenAI", os.getenv("OPENAI_MODEL", "openai/gpt-5-mini"), "OPENAI_API_KEY"),
    "gemini": Provider("Google Gemini", os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash"), "GEMINI_API_KEY"),
    "hf": Provider("Hugging Face Inference", os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct"), "HF_TOKEN"),
    # Ollama serves local models and, once signed in, cloud tags like gpt-oss:120b-cloud
    "ollama": Provider("Ollama (local or cloud)", os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"), None),
}


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def make_model(provider: str, model_id: str | None = None, api_key: str | None = None):
    """Build the smolagents model for `provider`. Returns (model, display name)."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from {sorted(PROVIDERS)}.")
    spec = PROVIDERS[provider]
    model_id = (model_id or "").strip() or spec.default_model
    key = api_key or (os.getenv(spec.key_env) if spec.key_env else None)

    if provider == "ollama":
        if not _ollama_running():
            raise RuntimeError(f"Ollama is not reachable at {OLLAMA_URL}.")
        return OpenAIServerModel(model_id=model_id, api_base=f"{OLLAMA_URL}/v1", api_key="ollama"), \
            f"{model_id} (Ollama)"
    if not key:
        raise RuntimeError(f"{spec.label} needs an API key (set {spec.key_env}).")
    if provider == "hf":
        return InferenceClientModel(model_id=model_id, token=key), model_id
    # pass the key into the client rather than mutating process-wide env, so
    # one user's key can't leak into another session sharing the process
    return LiteLLMModel(model_id=model_id, api_key=key), model_id


def _auto_provider(hf_token, anthropic_key) -> tuple[str, str | None]:
    """v1 behaviour: Claude if a key is given, then HF, then local Ollama."""
    if anthropic_key:
        return "anthropic", anthropic_key
    if hf_token or os.getenv("HF_TOKEN"):
        return "hf", hf_token or os.getenv("HF_TOKEN")
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", None
    if _ollama_running():
        return "ollama", None
    raise RuntimeError("No model available. Provide an HF token, an Anthropic key, or run Ollama locally.")


def build_agent(
    hf_token=None,
    anthropic_key=None,
    *,
    provider: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    extra_tools: list | None = None,
    store: DataframeStore | None = None,
    max_steps: int = 10,
    planning_interval: int | None = None,
    stream_outputs: bool = False,
):
    """Build the CodeAgent. With no `provider`, falls back to the v1 auto-pick so the
    eval runner and old call sites keep working. `extra_tools` takes MCP tools.

    Pass the session's `store` so the tools read and write that session's data;
    without it they use the process-wide default store."""
    if provider is None:
        provider, api_key = _auto_provider(hf_token, anthropic_key)
    model, model_name = make_model(provider, model_id, api_key)
    agent = CodeAgent(
        tools=[*(bind_tools(store, ALL_TOOLS) if store is not None else ALL_TOOLS), *(extra_tools or [])],
        model=model,
        instructions=SYSTEM_PROMPT,
        max_steps=max_steps,
        planning_interval=planning_interval,
        stream_outputs=stream_outputs,
        verbosity_level=1,
    )
    return agent, model_name
