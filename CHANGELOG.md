# Changelog

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
