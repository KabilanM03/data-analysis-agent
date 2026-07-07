"""Per-session dataframe store, accessed by tools via a ContextVar.

Tools call `get_active_df()` / `set_active_df()` without knowing about sessions.
The Gradio handler binds a session-specific `DataframeStore` to the ContextVar
before calling `agent.run()`, so each user gets their own slice of state.

Falling back to a module-default store keeps the unit tests and CLI usage simple.

The store keeps the originally-loaded frame separate from the *active view* so a
filter can persist (later tools operate on the filtered subset) without losing the
ability to reset back to the full dataset. It also collects the chart paths a run
produced so the UI can render them even if the model drops the [CHART:] marker.
"""

import contextvars
import pandas as pd


class DataframeStore:
    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None      # full, originally-loaded frame
        self._view: pd.DataFrame | None = None    # active view (may be filtered)
        self._name: str = ""
        self._charts: list[str] = []              # charts produced in the current run

    def set(self, df: pd.DataFrame, name: str = "") -> None:
        """Load a dataset. Resets any active filter."""
        self._df = df
        self._view = df
        self._name = name

    def set_view(self, df: pd.DataFrame) -> None:
        """Replace the active view (e.g. after a filter) without touching the original."""
        self._view = df

    def reset_view(self) -> pd.DataFrame:
        """Drop any filter and restore the full dataset as the active view."""
        if self._df is None:
            raise ValueError("No dataset loaded yet.")
        self._view = self._df
        return self._df

    def get(self) -> pd.DataFrame:
        if self._view is None:
            raise ValueError(
                "No dataset loaded yet. Call load_hf_dataset, fetch_kaggle_dataset, "
                "or load_dataset first."
            )
        return self._view

    def name(self) -> str:
        return self._name

    # -- chart registry -----------------------------------------------------
    def register_chart(self, path: str) -> None:
        self._charts.append(path)

    def pop_charts(self) -> list[str]:
        """Return charts registered since the last pop, and clear the list."""
        out = self._charts[:]
        self._charts.clear()
        return out


_default_store = DataframeStore()
_active_store_ctx: contextvars.ContextVar[DataframeStore] = contextvars.ContextVar(
    "active_store", default=_default_store
)


def bind_store(store: DataframeStore):
    """Set `store` as the active store for the current context. Returns a token
    you can pass to `_active_store_ctx.reset()` to undo."""
    return _active_store_ctx.set(store)


def get_active_df() -> pd.DataFrame:
    return _active_store_ctx.get().get()


def set_active_df(df: pd.DataFrame, name: str = "dataset") -> None:
    _active_store_ctx.get().set(df, name)


def set_view(df: pd.DataFrame) -> None:
    _active_store_ctx.get().set_view(df)


def reset_view() -> pd.DataFrame:
    return _active_store_ctx.get().reset_view()


def get_active_name() -> str:
    return _active_store_ctx.get().name()


def register_chart(path: str) -> None:
    _active_store_ctx.get().register_chart(path)


def pop_charts() -> list[str]:
    return _active_store_ctx.get().pop_charts()


def check_columns(df: pd.DataFrame, *cols: str) -> str | None:
    """Return an error message if any column is missing, else None."""
    missing = [c for c in cols if c and c not in df.columns]
    if not missing:
        return None
    avail = ", ".join(df.columns[:12]) + ("..." if len(df.columns) > 12 else "")
    return f"Column(s) not in dataset: {missing}. Available: {avail}"
