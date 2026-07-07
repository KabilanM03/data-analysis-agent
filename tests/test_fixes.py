"""Regression tests for the review fixes.

Each test pins a bug that previously shipped, so it can't silently come back.
The autouse `isolated_store` fixture (conftest.py) gives every test a fresh store.
"""

import os
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from tools._state import set_active_df, get_active_df, pop_charts
from tools.data_tools import (
    aggregate_data,
    filter_data,
    reset_filters,
    describe_dataset,
    load_dataset,
)
from tools.viz_tools import create_visualization
from tools.fetch_tools import fetch_kaggle_dataset


@pytest.fixture
def hue_df():
    df = pd.DataFrame(
        {
            "region": ["UK", "UK", "EU", "EU", "US", "US"],
            "channel": ["web", "store", "web", "store", "web", "store"],
            "sales": [120, 200, 90, 150, 300, 250],
        }
    )
    set_active_df(df, name="hue")
    return df


# -- bug 1: bar chart with a hue column ------------------------------------
def test_bar_with_hue_renders(hue_df):
    out = create_visualization("bar", "region", "sales", hue_column="channel")
    assert out.startswith("[CHART:")
    assert "Chart failed" not in out


def test_bar_rejects_non_numeric_y(hue_df):
    out = create_visualization("bar", "region", "channel")
    assert "not numeric" in out
    assert "[CHART:" not in out


# -- bug 2: aggregating a text column with a numeric func -------------------
def test_aggregate_text_column_gives_actionable_error(hue_df):
    out = aggregate_data("region", "channel", "mean")
    assert "text column" in out
    assert "sales" in out  # names a numeric column to use instead


# -- bug 3: unbounded tool output ------------------------------------------
def test_aggregate_output_is_capped():
    df = pd.DataFrame({"g": [f"grp{i}" for i in range(3000)], "v": range(3000)})
    set_active_df(df, name="big")
    out = aggregate_data("g", "v", "sum")
    assert "and 2950 more groups" in out
    assert len(out) < 4000  # was ~45,000 chars before the cap


def test_describe_truncates_wide_frame():
    df = pd.DataFrame({f"c{i}": [1, 2, 3] for i in range(60)})
    set_active_df(df, name="wide")
    out = describe_dataset(include_sample=False)
    assert "first 25 of 60 columns" in out
    assert "c59" in out  # but still lists every column name


# -- bug 4: filter persistence + reset -------------------------------------
def test_filter_persists_into_aggregate(hue_df):
    filter_data("region == 'UK'")
    assert get_active_df().shape[0] == 2  # view is now the filtered subset
    out = aggregate_data("region", "sales", "sum")
    assert "320" in out  # 120 + 200, UK only
    assert "550" not in out  # US total must not appear


def test_reset_filters_restores_full_dataset(hue_df):
    filter_data("region == 'UK'")
    assert get_active_df().shape[0] == 2
    out = reset_filters()
    assert "6 rows" in out
    assert get_active_df().shape[0] == 6


# -- chart registry: path reaches the store even without the marker --------
def test_create_visualization_registers_chart(hue_df):
    out = create_visualization("bar", "region", "sales")
    charts = pop_charts()
    assert len(charts) == 1
    assert os.path.exists(charts[0])
    assert charts[0] in out  # same path as the sentinel
    assert pop_charts() == []  # popping clears the registry


# -- bug 9: load_dataset path restriction ----------------------------------
def test_load_dataset_refuses_path_outside_data_dir():
    out = load_dataset("/etc/passwd")
    assert "outside the allowed data directory" in out or "Only .csv" in out


# -- bug 10: kaggle file mismatch lists available CSVs ---------------------
@patch("tools.fetch_tools._kaggle_api")
def test_fetch_kaggle_lists_csvs_on_filename_mismatch(mock_api, tmp_path):
    def fake_download(slug, path=None, unzip=True):
        for n in ("alpha.csv", "beta.csv"):
            (pd.DataFrame({"x": [1]})).to_csv(os.path.join(path, n), index=False)

    api = MagicMock()
    api.dataset_download_files.side_effect = fake_download
    mock_api.return_value = api

    out = fetch_kaggle_dataset("owner/ds", file_name="missing.csv")
    assert "No file named 'missing.csv'" in out
    assert "alpha.csv" in out and "beta.csv" in out


# -- bug 11: gradio 6 removed Chatbot(type=...); UI must build under the pin
def test_build_ui_constructs():
    gr = pytest.importorskip("gradio")
    from app import build_ui

    demo = build_ui()
    assert isinstance(demo, gr.Blocks)


# -- bug 12: a malformed CSV upload must return a message, not raise --------
def test_upload_csv_bad_file_returns_message(tmp_path):
    pytest.importorskip("gradio")
    from app import make_session, upload_csv

    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00broken\x00")

    class FakeFile:
        name = str(bad)

    msg, _session = upload_csv(FakeFile(), make_session())
    assert "Could not read" in msg
