import os
import json
import gradio as gr
from agent import build_agent
from tools.data_tools import set_active_df
import pandas as pd

agent = None
KAGGLE_JSON = os.path.expanduser("~/.kaggle/kaggle.json")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_kaggle_from_file() -> tuple[str, str]:
    try:
        with open(KAGGLE_JSON) as f:
            data = json.load(f)
        return data.get("username", ""), data.get("key", "")
    except Exception:
        return "", ""


def _save_kaggle_to_file(username: str, key: str):
    os.makedirs(os.path.dirname(KAGGLE_JSON), exist_ok=True)
    with open(KAGGLE_JSON, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(KAGGLE_JSON, 0o600)


# ── Agent logic ───────────────────────────────────────────────────────────────

def init_agent(hf_token: str, anthropic_key: str, kaggle_user: str, kaggle_key: str):
    global agent

    user = kaggle_user.strip()
    key  = kaggle_key.strip()
    if not (user and key):
        user, key = _load_kaggle_from_file()

    kaggle_ready = bool(user and key)
    if kaggle_ready:
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"]      = key
        if kaggle_user.strip() and kaggle_key.strip():
            _save_kaggle_to_file(user, key)

    try:
        agent_obj, model_name = build_agent(
            hf_token=hf_token.strip() or None,
            anthropic_key=anthropic_key.strip() or None,
        )
        agent = agent_obj

        sources = ["🤗 HF Datasets"]
        if kaggle_ready:
            sources.append("🏆 Kaggle")

        status_html = f"""
        <div class="status-card status-ok">
            <span class="status-dot"></span>
            <div>
                <strong>Agent ready</strong><br>
                <span class="status-detail">Model: {model_name}</span><br>
                <span class="status-detail">Sources: {" · ".join(sources)}</span>
            </div>
        </div>"""
        return status_html

    except Exception as e:
        status_html = f"""
        <div class="status-card status-err">
            <span class="status-dot"></span>
            <div><strong>Failed to initialise</strong><br>
            <span class="status-detail">{e}</span></div>
        </div>"""
        return status_html


def upload_csv(file):
    if file is None:
        return ""
    df = pd.read_csv(file.name)
    set_active_df(df, name=os.path.basename(file.name))
    return f"✅ Loaded **{os.path.basename(file.name)}** — {df.shape[0]:,} rows × {df.shape[1]} columns"


def chat(user_message: str, history: list):
    global agent
    if not user_message.strip():
        return history, ""

    if agent is None:
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ Agent not initialised yet. Complete the setup on the left and click **Launch Agent**."})
        return history, ""

    history.append({"role": "user", "content": user_message})
    try:
        response = agent.run(user_message)
        if isinstance(response, str) and "plots/" in response and response.strip().endswith(".png"):
            path = response.strip().split("Chart saved to:")[-1].strip()
            history.append({"role": "assistant", "content": f"📊 Chart saved to `{path}`."})
        else:
            history.append({"role": "assistant", "content": str(response)})
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ Error: {e}"})

    return history, ""


# ── UI ────────────────────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    ("🎵 Spotify analysis",   "Load the Spotify dataset and show top 10 genres by popularity"),
    ("💼 Jobs & salaries",    "Load the data jobs dataset and show average salary by job title"),
    ("🔎 Find Kaggle data",   "Search Kaggle for football datasets"),
    ("🔗 Correlations",       "Show correlation between all numeric columns"),
    ("📊 Bar chart",          "Create a bar chart of total sales by region"),
    ("📋 Full report",        "Generate a full analysis report with findings and recommendations"),
]

CSS = """
/* ── Google Blue override for all primary buttons ── */
:root {
    --color-accent: #4285F4 !important;
    --button-primary-background-fill: #4285F4 !important;
    --button-primary-background-fill-hover: #3367d6 !important;
    --button-primary-text-color: #ffffff !important;
    --button-primary-border-color: #4285F4 !important;
}
button.primary, .btn-primary, [data-testid="primary"] {
    background: #4285F4 !important;
    border-color: #4285F4 !important;
    color: #fff !important;
}
button.primary:hover, [data-testid="primary"]:hover {
    background: #3367d6 !important;
    border-color: #3367d6 !important;
}

/* ── Page ── */
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 50%, #1a1f2e 100%);
    border: 1px solid #2a2f3e;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 24px;
    text-align: center;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4285F4, #34a0f4, #4285F4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}
.hero p { color: #9ca3af; font-size: 1.05rem; margin: 0 0 18px; }
.badge-row { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.badge {
    background: rgba(66,133,244,0.12);
    border: 1px solid rgba(66,133,244,0.3);
    color: #93bbfd;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}

/* ── Panel title ── */
.panel-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #4285F4;
    margin: 0 0 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2f3e;
}

/* ── Steps bar ── */
.steps-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 0 14px;
}
.step-pill {
    background: rgba(66,133,244,0.08);
    border: 1px solid rgba(66,133,244,0.2);
    color: #6b7280;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
}
.step-pill.active {
    background: rgba(66,133,244,0.2);
    border-color: rgba(66,133,244,0.5);
    color: #93bbfd;
}
.step-sep { color: #374151; font-size: 0.85rem; }

/* ── Status card ── */
.status-card {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    border-radius: 10px;
    margin-top: 8px;
    font-size: 0.85rem;
}
.status-ok  { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); }
.status-err { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.3);  }
.status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}
.status-ok  .status-dot { background: #10b981; box-shadow: 0 0 6px #10b981; }
.status-err .status-dot { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
.status-card strong { color: #e2e8f0; }
.status-detail { color: #9ca3af; font-size: 0.8rem; }
"""

WELCOME_MSG = [
    {
        "role": "assistant",
        "content": (
            "👋 **Welcome to the Data Analysis Agent!**\n\n"
            "I can fetch and analyse real-world data from:\n"
            "- 🤗 **HF Datasets** — Spotify, Netflix, jobs, Titanic & more\n"
            "- 🏆 **Kaggle** — thousands of community datasets\n"
            "- 📂 **Your own CSV** — upload any file\n\n"
            "**To get started:** enter your HF token on the left and click **Launch Agent**.\n\n"
            "*Try: \"Load the Spotify dataset and show top 10 genres by popularity\"*"
        ),
    }
]


def build_ui():
    _kuser, _kkey = _load_kaggle_from_file()

    with gr.Blocks(title="Data Analysis Agent", css=CSS) as demo:

        # ── Hero ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="hero">
            <h1>🔍 Data Analysis Agent</h1>
            <p>Ask questions about real-world data in plain English — no code required.</p>
            <div class="badge-row">
                <span class="badge">🤗 HF Datasets</span>
                <span class="badge">🏆 Kaggle</span>
                <span class="badge">🤖 smolagents</span>
                <span class="badge">⚡ Qwen2.5-72B · Claude Sonnet</span>
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── Left sidebar ──────────────────────────────────────────────
            with gr.Column(scale=1, min_width=290):

                # Steps header — compact inline row
                gr.HTML("""
                <div class="steps-bar">
                    <span class="step-pill active">1 Model</span>
                    <span class="step-sep">›</span>
                    <span class="step-pill">2 Data</span>
                    <span class="step-sep">›</span>
                    <span class="step-pill">3 Launch</span>
                </div>
                """)

                with gr.Accordion("🤗 Step 1a — Hugging Face (free)", open=True):
                    hf_token_input = gr.Textbox(
                        label="HF Token",
                        placeholder="hf_xxxxxxxxxxxxxxxx",
                        type="password",
                        info="Free at huggingface.co/settings/tokens",
                    )

                with gr.Accordion("✨ Step 1b — Anthropic Claude (optional)", open=False):
                    anthropic_key_input = gr.Textbox(
                        label="Anthropic API Key",
                        placeholder="sk-ant-xxxxxxxxxxxxxxxx",
                        type="password",
                        info="console.anthropic.com · Claude Sonnet · ~$0.05/query",
                    )

                with gr.Accordion("🏆 Step 2 — Kaggle (optional)", open=bool(_kuser)):
                    kaggle_user_input = gr.Textbox(
                        label="Kaggle Username",
                        placeholder="your_username",
                        value=_kuser,
                        info="kaggle.com/settings/account → API → Create Token",
                    )
                    kaggle_key_input = gr.Textbox(
                        label="Kaggle API Key",
                        placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                        value=_kkey,
                        type="password",
                    )

                init_btn    = gr.Button("🚀 Launch Agent", variant="primary", size="lg")
                init_status = gr.HTML("")

                with gr.Accordion("📂 Upload CSV (optional)", open=False):
                    file_input    = gr.File(label="Drop CSV here", file_types=[".csv"])
                    upload_status = gr.Markdown("")

                gr.HTML('<div class="panel-title" style="margin-top:12px">💡 Try these</div>')
                example_btns = []
                for emoji_label, question in EXAMPLE_QUESTIONS:
                    btn = gr.Button(emoji_label, size="sm", variant="secondary")
                    example_btns.append((btn, question))

            # ── Chat panel ────────────────────────────────────────────────
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    value=WELCOME_MSG,
                    height=600,
                    label="",
                    show_label=False,
                    avatar_images=(
                        None,
                        "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png",
                    ),
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask anything — e.g. 'Load the Spotify dataset and show top genres'",
                        show_label=False,
                        scale=5,
                        lines=1,
                        container=False,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear chat", variant="secondary", size="sm")
                    gr.HTML('<div style="color:#4b5563;font-size:0.72rem;padding:6px 0;text-align:right;flex:1">'
                            'Built with smolagents · HF Agents Course</div>')

        # ── Events ────────────────────────────────────────────────────────
        init_btn.click(
            init_agent,
            inputs=[hf_token_input, anthropic_key_input, kaggle_user_input, kaggle_key_input],
            outputs=[init_status],
        )
        file_input.change(upload_csv, inputs=[file_input], outputs=[upload_status])
        send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
        msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
        clear_btn.click(lambda: (WELCOME_MSG, ""), outputs=[chatbot, msg_input])

        for btn, question in example_btns:
            btn.click(lambda q=question: q, outputs=[msg_input])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False)
