# Changelog

## 2.1.0 - 2026-09-30

### Added

- Streamlit UI replacing Gradio: Chat with a live step-by-step agent trace
  (reasoning, code, tool output, time and tokens per step), Data (loaders, column
  profile, distributions, row selection, CSV/Parquet export), SQL (DuckDB editor
  with history), Charts (quick-chart builder and gallery), Run history (tokens,
  timing, success rate, JSON export) and Connections. Optional `st.login` sign-in.
- `run_sql`: DuckDB SQL over the active dataset (`data`) and the unfiltered one
  (`data_full`). One read-only statement per call; file access, extension installs
  and `ATTACH` are disabled on the connection.
- Cloud data: `load_cloud_file` for `s3://`, `gs://`, `az://`, `r2://`, `hf://` and
  `https://` (DuckDB httpfs); read-only `query_motherduck`, `query_snowflake`
  (key-pair or PAT auth) and `query_bigquery` (dry run first, 1 GiB scan cap);
  `list_data_sources`.
- MCP client: servers from `mcp_servers.json` (stdio or streamable HTTP, `${VAR}`
  expansion) add their tools to the agent. Example config for MotherDuck,
  BigQuery, the AWS MCP Server, Athena, Redshift, Snowflake-managed and Hugging Face.
- MCP server (`mcp_server.py`, FastMCP) exposing the tools to Claude Code, Claude
  Desktop and other MCP clients; charts come back as images.
- Model providers: OpenAI and Gemini alongside Claude, HF Inference and Ollama;
  token streaming and optional planning steps.
- Plotly JSON twin for every chart, rendered interactively in the UI.
- Tests: 56 offline (SQL sandbox, cloud guards, MCP client and server, agent
  loop and Streamlit app driven by a scripted stand-in model) plus opt-in live
  tests that load every dataset shortcut.

### Fixed

- **Per-session data never reached the agent's tools.** smolagents runs the
  generated code on a worker thread to enforce its execution timeout (1.24 did
  too), and a new thread doesn't inherit the `ContextVar` bound around
  `agent.run()`. So in 1.0 the tools fell back to one process-wide store: a CSV
  uploaded in the UI was invisible to the agent, and browser tabs shared
  datasets. Tools are now bound to the session's store inside each call
  (`bind_tools`), with a regression test.
- `titanic` shortcut: `datasets` 4+ dropped script-based datasets; now
  `phihung/titanic`.
- `sales` shortcut pointed at a dataset that no longer exists on the Hub; now
  `An-j96/SuperstoreData`.
- The chat box keeps its value when the agent launches (stable widget key).

### Changed

- smolagents 1.24 -> 1.26, pandas 3, datasets 5, kaggle 2; Python 3.11+.
- `mcp` pinned `<2`: smolagents' MCP bridge (`mcpadapt`) does not import under
  the 2.x SDK yet.
- Bar charts with a y column label the axis "mean <column>", which is what they plot.

### Removed

- Gradio app (`app.py`). It is still available at tag `v1.0.0`.

## 1.0.0 - 2026-07-07

First stable release.

### Added

- Conversational data analysis over Hugging Face Datasets, Kaggle, and local CSVs
- Twelve `smolagents` tools: loaders, describe, filter (persistent view), aggregate,
  correlation, six chart types, markdown report
- Model routing: Qwen via HF Inference API, Claude via LiteLLM, local Ollama fallback
- Per-session state via `ContextVar`-bound `DataframeStore`, so concurrent Gradio
  tabs don't share dataframes or API keys
- Pytest suite (31 tests, network tools mocked) and a golden-question eval harness
- GitHub Actions CI
- Installable package (`pip install -e .`), MIT licence

### Fixed

- Gradio 6 compatibility: `Chatbot(type="messages")` was removed upstream and
  crashed the UI on startup; the argument is gone and a UI-construction test
  now guards it
- Malformed CSV uploads return an error message instead of raising
