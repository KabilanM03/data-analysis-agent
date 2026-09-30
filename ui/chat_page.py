"""Chat page: ask questions, watch the agent work step by step."""

import io
import os

import pandas as pd
import streamlit as st

from . import session as S
from .trace import render_message, run_streaming

EXAMPLES = [
    "Load the Spotify dataset and show the top 10 genres by average popularity",
    "Load titanic and chart survival rate by passenger class",
    "Which cloud data sources are configured?",
    "Load the data jobs dataset and use SQL to find the median yearly salary by job title",
    "Load the Airbnb dataset and plot price against number of reviews, coloured by borough",
    "Generate a full analysis report for the current dataset",
]

FILE_TYPES = ["csv", "parquet", "json", "jsonl"]


def read_upload(file) -> pd.DataFrame:
    name = file.name.lower()
    data = io.BytesIO(file.getvalue())
    if name.endswith(".parquet"):
        return pd.read_parquet(data)
    if name.endswith((".json", ".jsonl")):
        return pd.read_json(data, lines=name.endswith(".jsonl"))
    return pd.read_csv(data)


def _take_example() -> None:
    # runs before the rerun, so the widget can be cleared and won't re-fire later
    st.session_state.pending_prompt = st.session_state.example_pick
    st.session_state.example_pick = None


def render() -> None:
    sess = S.get()
    st.title("Chat with your data")

    df = sess.df()
    cols = st.columns(4)
    cols[0].metric("Dataset", sess.dataset_name() or "none", border=True)
    cols[1].metric("Rows", f"{len(df):,}" if df is not None else "–", border=True)
    cols[2].metric("Charts", len(sess.charts), border=True)
    cols[3].metric("Runs", len(sess.runs), border=True)

    for i, msg in enumerate(sess.messages):
        render_message(msg, i)

    if not sess.messages:
        st.caption("Try one of these, or drop a CSV / Parquet / JSON file into the box below.")
        st.pills("Examples", EXAMPLES, label_visibility="collapsed", key="example_pick", on_change=_take_example)
    picked = st.session_state.pop("pending_prompt", None)

    submitted = st.chat_input(
        "Ask about your data…" if sess.agent else "Launch the agent in the sidebar first",
        accept_file=True, file_type=FILE_TYPES, submit_mode="disable", key="chat_box",
    )

    prompt, files = None, []
    if submitted:
        prompt, files = (submitted.text or "").strip(), list(submitted.files or [])
    elif picked:
        prompt = picked

    for f in files:
        try:
            frame = read_upload(f)
        except Exception as e:
            st.error(f"Could not read {f.name}: {e}")
            continue
        sess.load(frame, name=f.name, source="upload")
        note = f"Uploaded **{f.name}**: {len(frame):,} rows x {frame.shape[1]} cols. It is now the active dataset."
        sess.messages.append({"role": "user", "content": note})
        render_message(sess.messages[-1], len(sess.messages) - 1)

    if not prompt:
        return
    user_msg = {"role": "user", "content": prompt}
    sess.messages.append(user_msg)
    render_message(user_msg, len(sess.messages) - 1)

    if sess.agent is None:
        reply = {"role": "assistant", "content": "The agent isn't running yet. Pick a model in the sidebar and press **Launch agent**."}
        sess.messages.append(reply)
        render_message(reply, len(sess.messages) - 1)
        return

    with st.chat_message("assistant"):
        with S.bound(sess):
            result = run_streaming(sess.agent, prompt, reset=not sess.has_history,
                                   live_tokens=getattr(sess.agent, "stream_outputs", False))
    run = result.pop("run")
    run.update(model=sess.model_name, dataset=sess.dataset_name(), at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    sess.runs.append(run)
    sess.add_charts(result["charts"], source=prompt[:60])
    sess.has_history = sess.has_history or not result["failed"]
    sess.messages.append(result)
    st.rerun()  # re-draw from history so the live trace is replaced by its collapsed replay
