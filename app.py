"""Gradio entrypoint for the data analysis agent."""

import os
import re
import json
import logging
from dataclasses import dataclass, field

import gradio as gr
import pandas as pd

from agent import build_agent
from tools import DataframeStore, bind_store, _active_store_ctx, pop_charts, PLOTS_DIR

logger = logging.getLogger(__name__)

KAGGLE_JSON = os.path.expanduser("~/.kaggle/kaggle.json")
CHART_RE = re.compile(r"\[CHART:([^\]]+)\]")


@dataclass
class Session:
    agent: object | None = None
    model_name: str = ""
    store: DataframeStore = field(default_factory=DataframeStore)
    has_history: bool = False  # has the agent run at least once this session?


def make_session() -> Session:
    return Session()


def load_kaggle_creds() -> tuple[str, str]:
    try:
        with open(KAGGLE_JSON) as f:
            data = json.load(f)
        return data.get("username", ""), data.get("key", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return "", ""


def init_agent(hf_token, anthropic_key, kaggle_user, kaggle_key, session):
    if session is None:
        session = make_session()
    user = kaggle_user.strip()
    key = kaggle_key.strip()
    if not (user and key):
        user, key = load_kaggle_creds()
    if user and key:
        # used by the Kaggle client for this process; we deliberately do NOT write
        # a visitor's credentials to the server's ~/.kaggle/kaggle.json
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key

    try:
        agent, model_name = build_agent(
            hf_token=hf_token.strip() or None,
            anthropic_key=anthropic_key.strip() or None,
        )
    except RuntimeError as e:
        return session, str(e)
    except Exception as e:
        return session, f"Failed to initialise: {e}"

    session.agent = agent
    session.model_name = model_name
    sources = ["HF Datasets"]
    if user and key:
        sources.append("Kaggle")
    return session, f"Ready. Model: {model_name}. Sources: {', '.join(sources)}."


def upload_csv(file, session):
    if session is None:
        session = make_session()
    if file is None:
        return "", session
    df = pd.read_csv(file.name)
    session.store.set(df, name=os.path.basename(file.name))
    return f"Loaded **{os.path.basename(file.name)}** — {df.shape[0]:,} rows x {df.shape[1]} cols.", session


def chat(message, history, session):
    if not message.strip():
        return history, ""
    history = history + [{"role": "user", "content": message}]
    if session is None or session.agent is None:
        history.append({
            "role": "assistant",
            "content": "Agent not initialised. Configure on the left and click Launch agent.",
        })
        return history, ""

    token = bind_store(session.store)
    error = None
    try:
        # keep conversational memory after the first turn so follow-ups ("now plot
        # that", "what about the EU?") work; reset on the very first run
        response = session.agent.run(message, reset=not session.has_history)
        session.has_history = True
    except Exception as e:
        logger.exception("agent.run failed: %s", e)
        error = e
    finally:
        # collect any charts the run produced while the store is still bound
        produced_charts = pop_charts()
        _active_store_ctx.reset(token)

    if error is not None:
        history.append({
            "role": "assistant",
            "content": "Something went wrong while running the analysis. Try rephrasing "
                       "your question; the full error is in the server logs.",
        })
        return history, ""

    text = str(response)
    cleaned = CHART_RE.sub("", text).strip()
    if cleaned:
        history.append({"role": "assistant", "content": cleaned})

    # merge charts named in the answer with those the tool registered, de-duped, so a
    # chart still reaches the UI even if the model dropped the [CHART:] marker
    seen = set()
    for cp in [p.strip() for p in CHART_RE.findall(text)] + produced_charts:
        if cp and cp not in seen and os.path.exists(cp):
            seen.add(cp)
            history.append({"role": "assistant", "content": {"path": cp}})
    return history, ""


def clear_chat(session):
    if session is not None:
        session.has_history = False
    return [], ""


EXAMPLES = [
    "Load the Spotify dataset and show top 10 genres by popularity",
    "Load the data jobs dataset and show average salary by job title",
    "Search Kaggle for football datasets",
    "Show correlation between all numeric columns",
    "Generate a full analysis report with findings and recommendations",
]


def build_ui():
    k_user, k_key = load_kaggle_creds()
    with gr.Blocks(title="Data Analysis Agent") as demo:
        # session factory means each browser tab gets its own instance
        session = gr.State(value=None)
        demo.load(make_session, inputs=None, outputs=session)

        gr.Markdown(
            "# Data Analysis Agent\n"
            "Built on Hugging Face `smolagents`, alongside the "
            "[HF AI Agents Course](https://huggingface.co/learn/agents-course/). "
            "Ask questions about real-world data in plain English."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 1. Model")
                hf_token = gr.Textbox(label="Hugging Face token", type="password",
                                      placeholder="hf_...", info="huggingface.co/settings/tokens")
                anthropic_key = gr.Textbox(label="Anthropic API key (optional)", type="password",
                                           placeholder="sk-ant-...")

                gr.Markdown("### 2. Kaggle (optional)")
                kaggle_user = gr.Textbox(label="Kaggle username", value=k_user)
                kaggle_key = gr.Textbox(label="Kaggle API key", value=k_key, type="password")

                launch_btn = gr.Button("Launch agent", variant="primary")
                status = gr.Markdown("")

                gr.Markdown("### 3. Upload CSV (optional)")
                file_input = gr.File(label="CSV file", file_types=[".csv"])
                upload_status = gr.Markdown("")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=560, label="Chat")
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask a question about the data...",
                                     show_label=False, scale=5, lines=1, container=False)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(EXAMPLES, inputs=msg, label="Examples")
                clear_btn = gr.Button("Clear chat", size="sm")

        launch_btn.click(
            init_agent,
            inputs=[hf_token, anthropic_key, kaggle_user, kaggle_key, session],
            outputs=[session, status],
        )
        file_input.change(upload_csv, inputs=[file_input, session], outputs=[upload_status, session])
        send_btn.click(chat, inputs=[msg, chatbot, session], outputs=[chatbot, msg])
        msg.submit(chat, inputs=[msg, chatbot, session], outputs=[chatbot, msg])
        clear_btn.click(clear_chat, inputs=[session], outputs=[chatbot, msg])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        allowed_paths=[PLOTS_DIR],
    )
