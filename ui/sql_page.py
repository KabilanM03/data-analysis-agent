"""SQL page: query the active dataset directly with DuckDB, no LLM involved."""

import time

import duckdb
import streamlit as st

from tools.sql_tools import check_read_only, query_frames

from . import session as S

DEFAULT_SQL = "SELECT *\nFROM data\nLIMIT 100"


def _recall() -> None:
    # callback, so it runs before the text area is drawn and only when the pick changes
    if st.session_state.sql_past:
        st.session_state.sql_text = st.session_state.sql_past
    st.session_state.sql_past = ""


def render() -> None:
    sess = S.get()
    st.title("SQL")
    df = sess.df()
    if df is None:
        st.info("Load a dataset first (Data page or chat).", icon=":material/info:")
        return

    st.caption("DuckDB, read-only and sandboxed. The active dataset is `data`; the unfiltered one is `data_full`.")
    with st.expander(f"Schema: {sess.dataset_name()} ({df.shape[1]} columns)"):
        st.dataframe(
            {"column": [str(c) for c in df.columns], "type": [str(t) for t in df.dtypes]},
            hide_index=True,
        )

    if sess.sql_history:
        st.selectbox("History", [""] + sess.sql_history[::-1], key="sql_past", on_change=_recall,
                     format_func=lambda q: q.replace("\n", " ")[:90] or "–")
    st.session_state.setdefault("sql_text", DEFAULT_SQL)
    query = st.text_area("Query", height=160, key="sql_text")

    if st.button("Run query", type="primary", icon=":material/play_arrow:", shortcut="ctrl+enter"):
        err = check_read_only(query)
        if err:
            st.error(err)
            return
        frames = {"data": df}
        if sess.full_df() is not None:
            frames["data_full"] = sess.full_df()
        started = time.time()
        try:
            result = query_frames(query, frames)
        except duckdb.Error as e:
            st.error(f"Query failed: {e}")
            return
        st.session_state.sql_result = (query, result, time.time() - started)
        if query not in sess.sql_history:
            sess.sql_history.append(query)
            del sess.sql_history[:-20]

    if "sql_result" in st.session_state:
        q, result, secs = st.session_state.sql_result
        st.caption(f"{len(result):,} rows x {result.shape[1]} cols in {secs * 1000:.0f} ms")
        st.dataframe(result, height="auto" if len(result) <= 12 else 420)
        c1, c2 = st.columns(2)
        c1.download_button("Download CSV", result.to_csv(index=False), file_name="query.csv",
                           mime="text/csv", icon=":material/download:")
        if c2.button("Use as active dataset", icon=":material/move_up:",
                     help="The agent and the other pages will work on this result."):
            sess.load(result, name="sql result", source="sql")
            st.success("The query result is now the active dataset.")
