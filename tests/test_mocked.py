"""Mocked tests for the network-bound and viz tools.

Goal is regression cover, not correctness of the third-party APIs themselves.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from tools.fetch_tools import load_hf_dataset, search_kaggle_datasets, fetch_kaggle_dataset
from tools.viz_tools import create_visualization
from tools._state import get_active_df


@patch("tools.fetch_tools.hf_load_dataset")
def test_load_hf_dataset_uses_shortcut(mock_hf, isolated_store):
    fake_ds = MagicMock()
    fake_ds.to_pandas.return_value = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
    mock_hf.return_value = fake_ds

    out = load_hf_dataset("spotify", max_rows=2)

    mock_hf.assert_called_once()
    args, _ = mock_hf.call_args
    assert args[0] == "maharshipandya/spotify-tracks-dataset"
    assert "Loaded spotify" in out
    assert get_active_df().shape == (2, 2)


@patch("tools.fetch_tools.hf_load_dataset")
def test_load_hf_dataset_returns_friendly_error(mock_hf, isolated_store):
    mock_hf.side_effect = RuntimeError("boom")
    out = load_hf_dataset("titanic", max_rows=10)
    assert "Could not load" in out and "boom" in out


@patch("tools.fetch_tools._kaggle_api")
def test_search_kaggle_datasets_when_unauthenticated(mock_api, isolated_store):
    mock_api.return_value = None
    out = search_kaggle_datasets("football")
    assert "Kaggle authentication failed" in out


@patch("tools.fetch_tools._kaggle_api")
def test_search_kaggle_datasets_returns_results(mock_api, isolated_store):
    api = MagicMock()
    api.dataset_list.return_value = [
        SimpleNamespace(ref="alice/foo", title="Foo dataset"),
        SimpleNamespace(ref="bob/bar", title="Bar dataset"),
    ]
    mock_api.return_value = api

    out = search_kaggle_datasets("anything")
    assert "alice/foo" in out and "bob/bar" in out


@patch("tools.fetch_tools._kaggle_api")
def test_fetch_kaggle_dataset_handles_no_csv(mock_api, isolated_store, tmp_path):
    api = MagicMock()
    # download_files is called with a tmp dir; if no CSV is created there, the
    # tool should report it cleanly
    api.dataset_download_files.return_value = None
    mock_api.return_value = api

    out = fetch_kaggle_dataset("alice/empty")
    assert "No CSV files inside" in out


def test_create_visualization_rejects_bad_type(sample_df):
    out = create_visualization("piechart", "region")
    assert "chart_type must be one of" in out


def test_create_visualization_rejects_bad_column(sample_df):
    out = create_visualization("bar", "fictional")
    assert "not in dataset" in out


def test_create_visualization_returns_chart_marker(sample_df):
    out = create_visualization("bar", "region", "sales")
    assert out.startswith("[CHART:")
    # extract path and verify file exists
    path = out.split("[CHART:", 1)[1].split("]", 1)[0]
    assert os.path.exists(path)
    assert path.endswith(".png")
