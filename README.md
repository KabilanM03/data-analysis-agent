# Data Analysis Agent

A conversational AI agent that fetches real-world data from live sources and lets you analyse it using plain English — no SQL or Python required.

Built with [smolagents](https://huggingface.co/docs/smolagents) (Hugging Face), [Gradio](https://gradio.app/), and [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct).

---

## What it does

You ask a question. The agent fetches real data, runs the analysis, and gives you a clear answer — all in one flow.

**Example questions:**
- *"Fetch Apple stock data for the last year and show me the trend"*
- *"Compare AAPL, MSFT, GOOGL and NVDA over 6 months"*
- *"Show me Tesla's key financials"*
- *"Load the Spotify dataset and find the top genres by popularity"*
- *"Load the data jobs dataset and show average salary by job title"*
- *"Create a scatter chart of energy vs danceability"*
- *"Generate a report with findings and recommendations"*

---

## Demo

![Demo Screenshot](assets/demo.png)

---

## Data Sources

| Source | What it provides | Authentication |
|---|---|---|
| [Yahoo Finance](https://finance.yahoo.com/) | Live + historical stock prices, company financials, market comparisons | None required |
| [Hugging Face Datasets Hub](https://huggingface.co/datasets) | 100,000+ real-world structured datasets | Free HF token |

**Built-in datasets (load by name):**
| Name | Description |
|---|---|
| `spotify` | 114k Spotify tracks with audio features (danceability, energy, genre) |
| `titanic` | Classic Titanic passenger survival dataset |
| `netflix` | Netflix shows and movies catalogue |
| `sales` | B2B sales transaction records |
| `data jobs` | Data science job postings with salary and skills |
| `airbnb` | NYC Airbnb listings with price and reviews |

---

## Architecture

```
User Query (plain English)
        │
        ▼
  CodeAgent (smolagents)
  LLM: Qwen2.5-72B-Instruct
        │
        ├── fetch_stock_data      → Yahoo Finance: OHLCV price history
        ├── fetch_company_info    → Yahoo Finance: fundamentals, market cap, P/E
        ├── compare_stocks        → Yahoo Finance: multi-ticker return comparison
        ├── load_hf_dataset       → HF Hub: 100k+ real-world datasets
        ├── list_available_datasets → shows built-in dataset options
        │
        ├── describe_dataset      → stats, dtypes, missing values
        ├── filter_data           → pandas query filter
        ├── aggregate_data        → groupby + aggregation
        ├── correlation_analysis  → correlation matrix
        │
        ├── create_visualization  → bar, line, scatter, histogram, box, heatmap
        └── generate_report       → structured markdown report
```

The agent uses a `CodeAgent` — it writes and executes Python code to call tools, rather than following a fixed pipeline. This means it can handle unexpected questions without breaking.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | smolagents (Hugging Face) |
| LLM | Qwen/Qwen2.5-72B-Instruct via HF Inference API |
| Market Data | yfinance (Yahoo Finance) |
| Dataset Source | Hugging Face Datasets Hub |
| Data Analysis | pandas |
| Visualisation | matplotlib, seaborn |
| UI | Gradio |
| Language | Python 3.10+ |

---

## Project Structure

```
data-analysis-agent/
├── app.py                   # Gradio UI — chat-first interface
├── agent.py                 # Agent setup, tools, system prompt
├── tools/
│   ├── __init__.py
│   ├── fetch_tools.py       # Yahoo Finance + HF Datasets fetching (4 tools)
│   ├── data_tools.py        # describe, filter, aggregate, correlate (5 tools)
│   ├── viz_tools.py         # chart generation — 6 chart types (1 tool)
│   └── report_tools.py      # structured markdown report (1 tool)
├── data/
│   └── sample_sales.csv     # fallback CSV for offline demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- A free [Hugging Face account](https://huggingface.co/join) + token with Inference API access

### 1. Clone the repository

```bash
git clone https://github.com/kabilanmani/data-analysis-agent.git
cd data-analysis-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Open `http://localhost:7860`. Enter your HF token, then start asking questions.

### 4. Try it

```
"Compare AAPL, TSLA and NVDA over the last year"
"Load the Spotify dataset and show the top 10 genres by danceability"
"What are Tesla's key financials?"
```

---

## Course Context

Built as part of the [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course/):

- **Unit 1** — Agent fundamentals: Tools, Think → Act → Observe loop
- **Unit 2** — smolagents: CodeAgent, tool creation, multi-tool orchestration
- **Unit 3** — Real-world use case: multi-source data analysis agent

---

## Author

**Kabilan Mani**
MSc Data Science & AI — Queen Mary University of London (2024, First Class)

[LinkedIn](https://linkedin.com/in/kabilanmani) · [GitHub](https://github.com/kabilanmani) · [Hugging Face](https://huggingface.co/Kabilanmani)

---

## License

MIT
