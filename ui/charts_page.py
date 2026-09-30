"""Charts page: build a chart by hand, and browse every chart made this session."""

import os

import streamlit as st

from tools import create_visualization, pop_charts

from . import session as S
from .trace import render_chart

CHART_TYPES = ["bar", "line", "scatter", "histogram", "box", "heatmap"]


def _builder(sess) -> None:
    df = sess.df()
    with st.expander("Quick chart", expanded=not sess.charts, icon=":material/add_chart:"):
        if df is None:
            st.caption("Load a dataset first.")
            return
        cols = [""] + [str(c) for c in df.columns]
        c1, c2, c3, c4 = st.columns(4)
        kind = c1.selectbox("Type", CHART_TYPES)
        x = c2.selectbox("X", cols[1:], disabled=kind == "heatmap")
        none = lambda c: c or "(none)"  # noqa: E731
        y = c3.selectbox("Y", cols, format_func=none, disabled=kind in {"histogram", "heatmap"})
        hue = c4.selectbox("Colour by", cols, format_func=none, disabled=kind in {"histogram", "heatmap"})
        title = st.text_input("Title (optional)")
        if st.button("Draw", type="primary", icon=":material/insert_chart:"):
            with S.bound(sess):
                out = create_visualization.forward(
                    chart_type=kind, x_column=x or str(df.columns[0]),
                    y_column="" if kind in {"histogram", "heatmap"} else y,
                    title=title, hue_column="" if kind in {"histogram", "heatmap"} else hue,
                )
                made = pop_charts()
            if made:
                sess.add_charts(made, source="quick chart")
                st.rerun()
            st.error(out)


def render() -> None:
    sess = S.get()
    st.title("Charts")
    _builder(sess)
    if not sess.charts:
        st.info("No charts yet. Ask the agent for one or use Quick chart.", icon=":material/info:")
        return
    st.caption(f"{len(sess.charts)} chart(s) this session, newest first. Interactive where available.")
    grid = st.columns(2)
    for i, chart in enumerate(reversed(sess.charts)):
        with grid[i % 2].container(border=True):
            render_chart(chart["path"], key=f"gallery-{chart['path']}")
            st.caption(f"{chart['at']} · {chart['source']}")
            if os.path.exists(chart["path"]):
                with open(chart["path"], "rb") as f:
                    st.download_button("PNG", f.read(), file_name=os.path.basename(chart["path"]),
                                       mime="image/png", icon=":material/download:",
                                       key=f"dl-{chart['path']}")
