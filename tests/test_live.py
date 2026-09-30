"""Network checks, skipped by default. Run with: RUN_LIVE=1 pytest tests/test_live.py

These catch the things mocks can't: a shortcut dataset deleted from the Hub, or a
library upgrade that stops loading it (datasets 4 dropped script-based datasets,
which silently broke the old titanic shortcut)."""

import os

import pytest

from tools.fetch_tools import KNOWN_DATASETS, load_hf_dataset

pytestmark = pytest.mark.skipif(not os.getenv("RUN_LIVE"), reason="set RUN_LIVE=1 to hit the network")


@pytest.mark.parametrize("shortcut", list(KNOWN_DATASETS))
def test_shortcut_loads(shortcut, isolated_store):
    out = load_hf_dataset(shortcut, max_rows=50)
    assert out.startswith("Loaded"), out
    assert len(isolated_store.get()) == 50


def test_hf_file_through_duckdb(isolated_store):
    from tools.cloud_tools import load_cloud_file

    out = load_cloud_file("hf://datasets/phihung/titanic/train.csv", max_rows=100)
    assert out.startswith("Loaded"), out
