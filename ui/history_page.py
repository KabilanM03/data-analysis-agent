"""Run history: every agent run this session with steps, tokens and timing."""

import json

import pandas as pd
import streamlit as st

from . import session as S
from .trace import render_step


def render() -> None:
    sess = S.get()
    st.title("Run history")
    if not sess.runs:
        st.info("No agent runs yet.", icon=":material/info:")
        return

    runs = pd.DataFrame(sess.runs)
    ok = (runs["status"] == "ok").mean() * 100
    c = st.columns(5)
    c[0].metric("Runs", len(runs), border=True)
    c[1].metric("Success", f"{ok:.0f}%", border=True)
    trend = runs["duration_s"].tolist() if len(runs) >= 3 else None  # a 1-2 point sparkline is noise
    c[2].metric("Avg time", f"{runs['duration_s'].mean():.1f}s", border=True,
                chart_data=trend, chart_type="bar")
    c[3].metric("Input tokens", f"{runs['input_tokens'].sum():,}", border=True)
    c[4].metric("Output tokens", f"{runs['output_tokens'].sum():,}", border=True)

    table = runs[["at", "prompt", "model", "dataset", "status", "steps", "duration_s",
                  "input_tokens", "output_tokens", "charts"]]
    event = st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="single-row", key="runs_table",
        column_config={
            "prompt": st.column_config.TextColumn("prompt", width="large"),
            "duration_s": st.column_config.NumberColumn("time (s)", format="%.1f"),
        },
    )
    picked = event.selection.rows if event and event.selection else []
    if picked:
        run = sess.runs[picked[0]]
        st.subheader("Run detail")
        st.markdown(f"**Prompt:** {run['prompt']}")
        if run["error"]:
            st.error(run["error"])
        with st.container(border=True):
            for rec in run["trace"]:
                render_step(rec)
        st.markdown("**Answer**")
        st.markdown(run["answer"] or "_none_")

    st.download_button("Export history (JSON)", json.dumps(sess.runs, indent=2, default=str),
                       file_name="agent_runs.json", mime="application/json", icon=":material/download:")
