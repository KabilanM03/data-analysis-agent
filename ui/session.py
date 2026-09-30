"""Per-viewer state for the Streamlit app.

Everything lives in one `Session` object in st.session_state, so each browser
tab gets its own dataframe store, agent, MCP connections and history. The tools
still resolve their dataframe through the ContextVar in tools/_state.py; the app
binds this session's store around every tool or agent call (see `bound`).
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from tools import DataframeStore, bind_store, _active_store_ctx


@dataclass
class Session:
    store: DataframeStore = field(default_factory=DataframeStore)
    agent: object | None = None
    model_name: str = ""
    provider: str = ""
    mcp: object | None = None               # mcp_tools.MCPConnection
    has_history: bool = False               # has the agent run at least once?
    messages: list[dict] = field(default_factory=list)   # chat transcript incl. traces
    runs: list[dict] = field(default_factory=list)       # one record per agent run
    charts: list[dict] = field(default_factory=list)     # {"path", "title", "source", "at"}
    sql_history: list[str] = field(default_factory=list)
    loads: list[dict] = field(default_factory=list)      # dataset load log

    # -- dataset helpers ----------------------------------------------------
    def df(self) -> pd.DataFrame | None:
        try:
            return self.store.get()
        except ValueError:
            return None

    def full_df(self) -> pd.DataFrame | None:
        return self.store.full()

    def dataset_name(self) -> str:
        return self.store.name()

    def load(self, df: pd.DataFrame, name: str, source: str) -> None:
        self.store.set(df, name=name)
        self.loads.append({"name": name, "source": source, "rows": len(df),
                           "cols": df.shape[1], "at": time.strftime("%H:%M:%S")})

    def add_charts(self, paths: list[str], source: str) -> list[str]:
        known = {c["path"] for c in self.charts}
        new = [p for p in paths if p not in known and os.path.exists(p)]
        for p in new:
            self.charts.append({"path": p, "source": source, "at": time.strftime("%H:%M:%S")})
        return new

    def close_agent(self) -> None:
        if self.mcp is not None:
            self.mcp.close()
        self.agent, self.mcp, self.has_history = None, None, False


def get() -> Session:
    if "session" not in st.session_state:
        st.session_state.session = Session()
    return st.session_state.session


@contextmanager
def bound(session: Session):
    """Bind this session's store for the duration of a tool or agent call."""
    token = bind_store(session.store)
    try:
        yield
    finally:
        _active_store_ctx.reset(token)
