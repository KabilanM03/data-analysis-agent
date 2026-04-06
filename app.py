import os
import gradio as gr
from agent import build_agent
from tools.data_tools import set_active_df
import pandas as pd

agent = None


def init_agent(hf_token: str):
    global agent
    if not hf_token.strip():
        return "Please enter a valid HF token.", gr.update(interactive=False)
    try:
        agent = build_agent(hf_token=hf_token)
        return "Agent ready. Ask a question below.", gr.update(interactive=True)
    except Exception as e:
        return f"Failed to initialise: {e}", gr.update(interactive=False)


def upload_csv(file):
    if file is None:
        return "No file uploaded."
    df = pd.read_csv(file.name)
    set_active_df(df, name=os.path.basename(file.name))
    return f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {os.path.basename(file.name)}"


def chat(user_message: str, history: list):
    global agent
    if not user_message.strip():
        return history, ""
    if agent is None:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "Agent not initialised. Enter your HF token first."})
        return history, ""

    history.append({"role": "user", "content": user_message})
    try:
        response = agent.run(user_message)
        # If a chart was generated, surface the path
        if isinstance(response, str) and "plots/" in response and response.strip().endswith(".png"):
            path = response.strip().split("Chart saved to:")[-1].strip()
            history.append({"role": "assistant", "content": f"Chart generated and saved to `{path}`."})
        else:
            history.append({"role": "assistant", "content": str(response)})
    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {e}"})

    return history, ""


EXAMPLE_QUESTIONS = [
    "What datasets are available?",
    "Load the Spotify dataset and describe it",
    "What are the top 10 genres by average popularity?",
    "Create a scatter chart of energy vs danceability",
    "Fetch Apple stock data for the last year",
    "Compare AAPL, MSFT, GOOGL and NVDA over 6 months",
    "Show me Tesla's company info and key financials",
    "Load the data jobs dataset and show average salary by job title",
    "Create a bar chart of total sales by region",
    "Generate a full report with findings and recommendations",
]


def build_ui():
    with gr.Blocks(
        title="Data Analysis Agent",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; padding: 24px 0 8px; }
        .header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
        .header p { color: #666; margin: 6px 0 0; font-size: 1rem; }
        .badge { display: inline-block; background: #f0f4ff; color: #3b4cca;
                 font-size: 0.78rem; padding: 2px 10px; border-radius: 12px;
                 margin: 4px 2px; font-weight: 500; }
        """,
    ) as demo:

        gr.HTML("""
        <div class="header">
            <h1>Data Analysis Agent</h1>
            <p>Ask questions about real-world data in plain English.</p>
            <div style="margin-top:10px;">
                <span class="badge">Yahoo Finance</span>
                <span class="badge">HF Datasets Hub</span>
                <span class="badge">smolagents</span>
                <span class="badge">Qwen2.5-72B</span>
            </div>
        </div>
        """)

        with gr.Row():
            # ── Left panel ──────────────────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Setup")
                hf_token_input = gr.Textbox(
                    label="Hugging Face Token",
                    placeholder="hf_...",
                    type="password",
                    info="Free token from huggingface.co/settings/tokens",
                )
                init_btn = gr.Button("Initialise Agent", variant="primary")
                init_status = gr.Textbox(label="Status", interactive=False, lines=2)

                gr.Markdown("### Or upload your own CSV")
                file_input = gr.File(label="Upload CSV (optional)", file_types=[".csv"])
                upload_status = gr.Textbox(label="Upload status", interactive=False, lines=1)

                gr.Markdown("### Try these questions")
                example_btns = []
                for q in EXAMPLE_QUESTIONS:
                    btn = gr.Button(q, size="sm", variant="secondary")
                    example_btns.append(btn)

            # ── Right panel ─────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=560,
                    label="Agent",
                    type="messages",
                    avatar_images=(None, "https://huggingface.co/front/assets/huggingface_logo-noborder.svg"),
                    show_copy_button=True,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g. Load the Spotify dataset and show the top 5 genres by popularity",
                        show_label=False,
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear chat", size="sm", variant="secondary")

        # ── Wire up events ────────────────────────────────────────────
        init_btn.click(init_agent, inputs=[hf_token_input], outputs=[init_status, msg_input])
        file_input.change(upload_csv, inputs=[file_input], outputs=[upload_status])
        send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
        msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

        for btn in example_btns:
            btn.click(lambda q=btn.value: q, outputs=[msg_input])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False)
