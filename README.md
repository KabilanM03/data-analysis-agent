# Data Analysis Agent

A conversational AI agent that lets you explore, analyse, and visualise any CSV dataset using plain English — no SQL or Python required.

Built with [smolagents](https://huggingface.co/docs/smolagents) (Hugging Face), [Gradio](https://gradio.app/), and [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) as the reasoning model.

---

## What it does

You upload a CSV file. You ask questions in plain English. The agent figures out which tools to use, runs the analysis, and gives you a clear answer.

**Example questions:**
- *"Describe this dataset"*
- *"What is the average sales by region?"*
- *"Create a bar chart of total sales by product"*
- *"Show me the correlation between quantity and total sales"*
- *"Filter orders where TotalSales > 1000 and summarise"*
- *"Generate a full report with findings and recommendations"*

---

## Demo

![Demo Screenshot](assets/demo.png)

---

## Architecture

```
User Query
    │
    ▼
CodeAgent (smolagents)
    │
    ├── load_dataset       → loads CSV into memory
    ├── describe_dataset   → stats, dtypes, missing values
    ├── filter_data        → pandas query filter
    ├── aggregate_data     → groupby + aggregation
    ├── correlation_analysis → correlation matrix
    ├── create_visualization → bar, line, scatter, histogram, box, heatmap
    └── generate_report    → structured markdown report
    │
    ▼
Gradio UI (chat interface + dataset preview)
```

The agent uses a `CodeAgent` — it writes and executes Python code to call tools, rather than relying on JSON blobs. This makes it more flexible and transparent.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | smolagents (Hugging Face) |
| LLM | Qwen/Qwen2.5-72B-Instruct via HF Inference API |
| Data Analysis | pandas |
| Visualisation | matplotlib, seaborn |
| UI | Gradio |
| Language | Python 3.10+ |

---

## Project Structure

```
data-analysis-agent/
├── app.py                  # Gradio UI and event wiring
├── agent.py                # Agent initialisation and system prompt
├── tools/
│   ├── __init__.py
│   ├── data_tools.py       # load, describe, filter, aggregate, correlate
│   ├── viz_tools.py        # chart generation (6 chart types)
│   └── report_tools.py     # structured markdown report generator
├── data/
│   └── sample_sales.csv    # sample dataset for demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/kabilanmani/data-analysis-agent.git
cd data-analysis-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your Hugging Face token

Create a free account at [huggingface.co](https://huggingface.co) and generate a token with Inference API access.

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

### 4. Run the app

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

### 5. Try it with the sample dataset

Upload `data/sample_sales.csv` and try:
- *"Describe this dataset"*
- *"What is the total sales by region?"*
- *"Create a bar chart of TotalSales by Product"*
- *"Generate a report with key findings"*

---

## Course Context

This project was built as part of the [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course/), specifically applying concepts from:

- **Unit 1** — Agent fundamentals: Tools, Think → Act → Observe loop
- **Unit 2** — smolagents framework: CodeAgent, tool creation, multi-tool orchestration
- **Unit 3** — Real-world use case: data analysis as an applied agent problem

---

## Author

**Kabilan Mani**
MSc Data Science & AI — Queen Mary University of London (2024, First Class)

[LinkedIn](https://linkedin.com/in/kabilanmani) · [GitHub](https://github.com/kabilanmani) · [Hugging Face](https://huggingface.co/Kabilanmani)

---

## License

MIT
