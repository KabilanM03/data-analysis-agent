"""Data page: load a dataset without the agent, then profile and explore it."""

import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from tools.cloud_tools import read_cloud_file
from tools.fetch_tools import KNOWN_DATASETS, load_hf_dataset

from . import session as S
from .chat_page import FILE_TYPES, read_upload

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "sample_sales.csv")


def _profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        top = s.mode(dropna=True)
        rows.append({
            "column": str(col),
            "type": str(s.dtype),
            "non-null": int(s.notna().sum()),
            "missing %": round(100 * s.isna().mean(), 1),
            "unique": int(s.nunique(dropna=True)),
            "min": s.min() if numeric else None,
            "mean": round(float(s.mean()), 3) if numeric and s.notna().any() else None,
            "max": s.max() if numeric else None,
            "most common": str(top.iloc[0])[:40] if not top.empty else "",
        })
    return pd.DataFrame(rows)


def _loaders(sess) -> None:
    with st.expander("Load data", expanded=sess.df() is None, icon=":material/upload:"):
        t_hf, t_cloud, t_file = st.tabs(["Hugging Face", "Cloud storage", "File"])
        with t_hf:
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            name = c1.selectbox("Shortcut or dataset id", list(KNOWN_DATASETS), accept_new_options=True)
            rows = c2.number_input("Max rows", 100, 100_000, 5000, step=1000)
            if st.button("Load from Hugging Face", icon=":material/download:"):
                with st.spinner(f"Loading {name}…"), S.bound(sess):
                    msg = load_hf_dataset.forward(dataset_name=name, max_rows=int(rows))
                (st.error if msg.startswith("Could not") else st.success)(msg.split("\n", 1)[0])
                if not msg.startswith("Could not"):
                    sess.loads.append({"name": name, "source": "huggingface", "rows": len(sess.df()),
                                       "cols": sess.df().shape[1], "at": pd.Timestamp.now().strftime("%H:%M:%S")})
        with t_cloud:
            uri = st.text_input("URI", placeholder="s3://bucket/path/file.parquet · gs://… · hf://datasets/org/name/file.parquet · https://…")
            fmt = st.segmented_control("Format", ["auto", "parquet", "csv", "json"], default="auto")
            if st.button("Load from cloud", icon=":material/cloud_download:", disabled=not uri):
                try:
                    with st.spinner("Reading…"):
                        frame = read_cloud_file(uri, fmt or "auto")
                    sess.load(frame, name=uri.rsplit("/", 1)[-1], source="cloud")
                    st.success(f"Loaded {len(frame):,} rows x {frame.shape[1]} cols.")
                except Exception as e:
                    st.error(str(e))
        with t_file:
            up = st.file_uploader("CSV, Parquet or JSON", type=FILE_TYPES)
            c1, c2 = st.columns(2)
            if up is not None and c1.button("Load file", icon=":material/upload_file:"):
                try:
                    sess.load(read_upload(up), name=up.name, source="upload")
                    st.success(f"Loaded {up.name}.")
                except Exception as e:
                    st.error(f"Could not read {up.name}: {e}")
            if c2.button("Use sample sales data", icon=":material/table_view:"):
                sess.load(pd.read_csv(SAMPLE_CSV), name="sample_sales.csv", source="sample")
                st.rerun()


def render() -> None:
    sess = S.get()
    st.title("Data")
    _loaders(sess)

    df, full = sess.df(), sess.full_df()
    if df is None:
        st.info("No dataset loaded yet.", icon=":material/info:")
        return

    st.subheader(sess.dataset_name() or "dataset")
    c = st.columns(5)
    c[0].metric("Rows", f"{len(df):,}", border=True,
                delta=f"filtered from {len(full):,}" if full is not None and len(full) != len(df) else None,
                delta_color="off")
    c[1].metric("Columns", df.shape[1], border=True)
    c[2].metric("Numeric", len(df.select_dtypes("number").columns), border=True)
    c[3].metric("Missing cells", f"{int(df.isna().sum().sum()):,}", border=True)
    mem = df.memory_usage(deep=True).sum()
    c[4].metric("Memory", f"{mem / 1024**2:.1f} MB" if mem >= 1024**2 else f"{mem / 1024:.0f} KB", border=True)

    if full is not None and len(full) != len(df) and st.button("Clear filter", icon=":material/filter_alt_off:"):
        sess.store.reset_view()
        st.rerun()

    t_rows, t_cols, t_dist = st.tabs(["Rows", "Column profile", "Distribution"])
    with t_rows:
        event = st.dataframe(df, height=420, on_select="rerun", selection_mode="multi-row",
                             key="data_rows")
        picked = event.selection.rows if event and event.selection else []
        if picked:
            st.caption(f"{len(picked)} row(s) selected")
            st.dataframe(df.iloc[picked], hide_index=False)
        c1, c2 = st.columns(2)
        c1.download_button("Download CSV", df.to_csv(index=False), file_name="data.csv",
                           mime="text/csv", icon=":material/download:")
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        c2.download_button("Download Parquet", buf.getvalue(), file_name="data.parquet",
                           icon=":material/download:")
    with t_cols:
        prof = _profile(df)
        st.dataframe(prof, hide_index=True, column_config={
            "missing %": st.column_config.ProgressColumn("missing %", min_value=0, max_value=100, format="%.1f%%"),
        })
    with t_dist:
        col = st.selectbox("Column", list(df.columns))
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
        else:
            counts = s.astype(str).value_counts().head(30).reset_index()
            counts.columns = [col, "count"]
            fig = px.bar(counts, x=col, y="count", title=f"Top values of {col}")
        st.plotly_chart(fig, width="stretch", key="dist_chart")

    if sess.loads:
        with st.expander("Load log"):
            st.dataframe(pd.DataFrame(sess.loads), hide_index=True)
