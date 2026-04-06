import os
from smolagents import CodeAgent, HfApiModel
from tools import (
    # Online data fetching
    fetch_stock_data,
    fetch_company_info,
    compare_stocks,
    load_hf_dataset,
    list_available_datasets,
    # Analysis
    load_dataset,
    describe_dataset,
    filter_data,
    aggregate_data,
    correlation_analysis,
    # Visualisation & reporting
    create_visualization,
    generate_report,
)

SYSTEM_PROMPT = """You are a professional Data Analysis Agent with access to real-time data sources.

You can fetch live data from two sources:
1. Yahoo Finance — for stock prices, company financials, and market comparisons
2. Hugging Face Datasets Hub — for real-world structured datasets (Spotify, Titanic, Netflix, sales data, job postings, Airbnb, and more)

When a user asks a question, follow this approach:

STEP 1 — Fetch data if needed:
- For stock/market questions → use fetch_stock_data or compare_stocks
- For company background → use fetch_company_info
- For structured datasets → use load_hf_dataset (use list_available_datasets to show options)
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


def build_agent(hf_token: str | None = None) -> CodeAgent:
    token = hf_token or os.getenv("HF_TOKEN")
    model = HfApiModel(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        token=token,
    )
    agent = CodeAgent(
        tools=[
            # Data sources
            fetch_stock_data,
            fetch_company_info,
            compare_stocks,
            load_hf_dataset,
            list_available_datasets,
            # Local CSV fallback
            load_dataset,
            # Analysis
            describe_dataset,
            filter_data,
            aggregate_data,
            correlation_analysis,
            # Output
            create_visualization,
            generate_report,
        ],
        model=model,
        system_prompt=SYSTEM_PROMPT,
        max_steps=12,
        verbose=True,
    )
    return agent
