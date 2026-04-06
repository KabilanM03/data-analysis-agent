import os
import gradio as gr
from agent import build_agent
from tools.data_tools import set_active_df
import pandas as pd

agent = None
chat_history = []


def init_agent(hf_token: str):
    global agent
    if not hf_token.strip():
        return "Please enter a valid HF token."
    try:
        agent = build_agent(hf_token=hf_token)
        return "Agent ready. Upload a CSV and start asking questions."
    except Exception as e:
        return f"Failed to initialise agent: {e}"


def upload_csv(file):
    if file is None:
        return "No file uploaded.", None
    df = pd.read_csv(file.name)
    set_active_df(df, name=os.path.basename(file.name))
    preview = df.head(10).to_html(index=False)
    return (
        f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns from {os.path.basename(file.name)}",
        preview,
    )


def chat(user_message: str, history: list):
    global agent, chat_history
    if agent is None:
        history.append((user_message, "Agent not initialised. Enter your HF token first."))
        return history, ""

    try:
        response = agent.run(user_message)
        # Check if a plot was generated and append image markdown
        if isinstance(response, str) and response.startswith("Chart saved to:"):
            path = response.replace("Chart saved to:", "").strip()
            history.append((user_message, f"Chart generated.\n\n![chart]({path})"))
        else:
            history.append((user_message, str(response)))
    except Exception as e:
        history.append((user_message, f"Error: {e}"))

    return history, ""


def build_ui():
    with gr.Blocks(
        title="Data Analysis Agent",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; padding: 20px 0; }
        .header h1 { font-size: 2rem; font-weight: 700; }
        .header p { color: #555; font-size: 1rem; }
        """,
    ) as demo:
        gr.HTML("""
        <div class="header">
            <h1>Data Analysis Agent</h1>
            <p>Upload a CSV dataset and ask questions in plain English. Powered by smolagents + Qwen2.5-72B.</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Setup")
                hf_token_input = gr.Textbox(
                    label="Hugging Face Token",
                    placeholder="hf_...",
                    type="password",
                )
                init_btn = gr.Button("Initialise Agent", variant="primary")
                init_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Upload Dataset")
                file_input = gr.File(label="Upload CSV", file_types=[".csv"])
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

                gr.Markdown("### Example Questions")
                gr.Markdown("""
- *Describe this dataset*
- *Show me the correlation between numeric columns*
- *Create a bar chart of sales by region*
- *What is the average salary by department?*
- *Filter rows where age > 30 and summarise*
- *Generate a full report with key findings*
                """)

            with gr.Column(scale=2):
                gr.Markdown("### Dataset Preview")
                data_preview = gr.HTML(label="Preview")

                gr.Markdown("### Chat with your Data")
                chatbot = gr.Chatbot(height=450, label="Agent")
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your data...",
                        show_label=False,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat", variant="secondary")

        # Wire up events
        init_btn.click(init_agent, inputs=[hf_token_input], outputs=[init_status])
        file_input.change(upload_csv, inputs=[file_input], outputs=[upload_status, data_preview])
        send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
        msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False)
