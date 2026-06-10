"""Shared pytest fixtures.

The agent tools read from a ContextVar-backed dataframe store. Each test gets
a fresh store so state from one test does not leak into the next.
"""

import pandas as pd
import pytest

# the project root is put on sys.path via [tool.pytest.ini_options] pythonpath in
# pyproject.toml, so `tools` / `agent` import cleanly without a sys.path hack here
from tools._state import DataframeStore, bind_store, _active_store_ctx


@pytest.fixture(autouse=True)
def isolated_store():
    store = DataframeStore()
    token = bind_store(store)
    try:
        yield store
    finally:
        _active_store_ctx.reset(token)


@pytest.fixture
def sample_df(isolated_store):
    df = pd.DataFrame(
        {
            "region": ["UK", "UK", "EU", "EU", "US", "US"],
            "sales": [120, 200, 90, 150, 300, 250],
            "units": [10, 20, 8, 14, 25, 22],
        }
    )
    isolated_store.set(df, name="test")
    return df
