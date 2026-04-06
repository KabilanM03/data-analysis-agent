# Data Analysis Agent

A conversational AI agent that fetches real-world data from live sources and lets you analyse it using plain English — no SQL or Python required.

Built with [smolagents](https://huggingface.co/docs/smolagents) (Hugging Face), [Gradio](https://gradio.app/), and your choice of model: **Qwen2.5-72B** (free, via HF Inference API) or **Claude Sonnet** (Anthropic).

---

## Demo

![Demo Screenshot](assets/demo.png)

**Example chart output — Top 10 Spotify Genres by Popularity (50k tracks):**

![Spotify Bar Chart](plots/plot_a99bc736.png)

---

## What it does

Ask a question in plain English. The agent fetches real data, runs the analysis, and gives you a clear answer — charts included.

**Example questions:**
- *"Load the Spotify dataset and show top 10 genres by popularity"*
- *"Load the data jobs dataset and show average salary by job title"*
- *"Search Kaggle for football datasets"*
- *"Show correlation between all numeric columns"*
- *"Create a bar chart of total sales by region"*
- *"Generate a full analysis report with findings and recommendations"*

---

## Data Sources

| Source | What it provides | Authentication |
|---|---|---|
| [Hugging Face Datasets Hub](https://huggingface.co/datasets) | 100,000+ real-world structured datasets | Free HF token |
| [Kaggle](https://www.kaggle.com/datasets) | Thousands of community datasets across any domain | Kaggle API key (free) |
| Local CSV | Your own data | Upload in the UI |

**Built-in HF datasets (load by name):**

| Name | Description |
|---|---|
| `spotify` | 114k Spotify tracks with audio features (danceability, energy, genre) |
| `titanic` | Classic Titanic passenger survival dataset |
| `netflix` | Netflix shows and movies catalogue |
| `sales` | B2B sales transaction records |
| `data jobs` | Data science job postings with salary and skills |
| `airbnb` | NYC Airbnb listings with price and reviews |

---

## Model Support

| Model | How to use | Cost |
|---|---|---|
| **Qwen2.5-72B-Instruct** (default) | Enter a free HF token | Free |
| **Claude Sonnet 4.6** | Enter an Anthropic API key | ~$0.05/query |
| **Local Ollama** | Run Ollama locally — auto-detected | Free |

---

## Architecture

```
User Query (plain English)
        │
        ▼
  CodeAgent (smolagents)
  LLM: Qwen2.5-72B / Claude Sonnet / Ollama
        │
        ├── load_hf_dataset       → HF Hub: 100k+ real-world datasets
        ├── list_available_datasets → shows built-in dataset options
        ├── fetch_kaggle_dataset  → Kaggle: download + load any dataset
        ├── search_kaggle_datasets → search Kaggle by keyword
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
| LLM (default) | Qwen2.5-72B-Instruct via HF Inference API |
| LLM (optional) | Claude Sonnet 4.6 via LiteLLM + Anthropic API |
| LLM (local) | Any Ollama model (auto-detected) |
| Dataset Sources | Hugging Face Datasets Hub + Kaggle |
| Data Analysis | pandas |
| Visualisation | matplotlib, seaborn |
| UI | Gradio |
| Language | Python 3.11+ |

---

## Project Structure

```
data-analysis-agent/
├── app.py                   # Gradio UI — chat-first interface
├── agent.py                 # Agent setup, tools, system prompt, model selection
├── tools/
│   ├── __init__.py
│   ├── fetch_tools.py       # HF Datasets + Kaggle fetching (4 tools)
│   ├── data_tools.py        # describe, filter, aggregate, correlate (5 tools)
│   ├── viz_tools.py         # chart generation — 6 chart types (1 tool)
│   └── report_tools.py      # structured markdown report (1 tool)
├── plots/                   # generated chart images
├── data/
│   └── sample_sales.csv     # sample CSV for offline demo
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11+
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

Open `http://localhost:7860`. Enter your HF token (free), optionally your Kaggle credentials, then start asking questions.

### 4. Optional: Kaggle datasets

1. Go to [kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Click **API → Create New Token** — downloads `kaggle.json`
3. Enter username + key in the UI, or place `kaggle.json` in `~/.kaggle/`

### 5. Optional: Claude Sonnet

Enter your Anthropic API key in **Step 1b** to upgrade the model to Claude Sonnet 4.6.

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
