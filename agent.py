import os
from smolagents import CodeAgent, HfApiModel
from tools import (
    load_dataset,
    describe_dataset,
    filter_data,
    aggregate_data,
    correlation_analysis,
    create_visualization,
    generate_report,
)

SYSTEM_PROMPT = """You are a professional Data Analysis Agent. Your job is to help users explore, analyse, and understand their datasets.

When a user asks a question or gives an instruction, follow this approach:
1. If no dataset is loaded, use `load_dataset` first.
2. Use `describe_dataset` to understand the data before diving into specifics.
3. Use the appropriate tools to answer the question — filter, aggregate, correlate, or visualise.
4. Always explain your findings in plain, clear language after running a tool.
5. When the user asks for a summary or report, use `generate_report` with clear findings and recommendations.

Be concise, professional, and data-driven. Quantify insights wherever possible.
"""


def build_agent(hf_token: str | None = None) -> CodeAgent:
    token = hf_token or os.getenv("HF_TOKEN")
    model = HfApiModel(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        token=token,
    )
    agent = CodeAgent(
        tools=[
            load_dataset,
            describe_dataset,
            filter_data,
            aggregate_data,
            correlation_analysis,
            create_visualization,
            generate_report,
        ],
        model=model,
        system_prompt=SYSTEM_PROMPT,
        max_steps=10,
        verbose=True,
    )
    return agent
