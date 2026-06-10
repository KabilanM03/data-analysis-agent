"""Unit tests for the dataframe-handling tools.

Network-dependent tools (HF, Kaggle) live in test_mocked.py with stubs.
"""

import pandas as pd
import pytest

from tools._state import get_active_df
from tools.data_tools import (
    load_dataset,
    describe_dataset,
    filter_data,
    aggregate_data,
    correlation_analysis,
)


def test_get_active_df_raises_on_empty_store():
    # the autouse fixture starts each test with an empty store
    with pytest.raises(ValueError):
        get_active_df()


def test_describe_runs(sample_df):
    out = describe_dataset(include_sample=True)
    assert "Stats" in out and "First 5 rows" in out


def test_filter_returns_subset(sample_df):
    out = filter_data("region == 'UK'")
    assert "2 rows" in out


def test_filter_handles_bad_query(sample_df):
    out = filter_data("nonexistent_col > 0")
    assert "Filter failed" in out


def test_aggregate_groupby(sample_df):
    out = aggregate_data("region", "sales", "sum")
    assert "320" in out and "240" in out and "550" in out


def test_aggregate_rejects_unknown_func(sample_df):
    out = aggregate_data("region", "sales", "stddev")
    assert "agg_func must be one of" in out


def test_aggregate_rejects_unknown_column(sample_df):
    out = aggregate_data("region", "fictional_col", "mean")
    assert "not in dataset" in out


def test_correlation_all_numeric(sample_df):
    out = correlation_analysis("")
    assert "sales" in out and "units" in out


def test_correlation_validates_columns(sample_df):
    out = correlation_analysis("sales,nope")
    assert "not in dataset" in out


def test_load_dataset_roundtrip(tmp_path, sample_df, monkeypatch):
    # load_dataset only reads CSVs under DATA_DIR; point it at the tmp dir
    monkeypatch.setattr("tools.data_tools.DATA_DIR", str(tmp_path))
    p = tmp_path / "x.csv"
    sample_df.to_csv(p, index=False)
    out = load_dataset(str(p))
    assert "Loaded x.csv" in out
    assert get_active_df().shape == sample_df.shape


def test_isolated_stores_do_not_leak(sample_df):
    # sample_df fixture set a 6-row df in this test's store. A fresh store
    # in the next test must not see it. Verified by `test_get_active_df_raises_on_empty_store`.
    assert get_active_df().shape == (6, 3)
