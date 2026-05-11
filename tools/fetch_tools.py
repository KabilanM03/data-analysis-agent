"""Hugging Face Datasets Hub and Kaggle fetchers."""

import os
import glob
import tempfile
import warnings

import pandas as pd
from datasets import load_dataset as hf_load_dataset
from smolagents import tool

from ._state import set_active_df

warnings.filterwarnings("ignore", category=UserWarning)


# shorthand names for the HF datasets the agent reaches for most often
KNOWN_DATASETS: dict[str, tuple[str, str]] = {
    "spotify":   ("maharshipandya/spotify-tracks-dataset", "train"),
    "titanic":   ("mstz/titanic",                          "train"),
    "netflix":   ("hugginglearners/netflix-shows",         "train"),
    "sales":     ("Thewillonline/sales_data_sample",       "train"),
    "data jobs": ("lukebarousse/data_jobs",                "train"),
    "airbnb":    ("gradio/NYC-Airbnb-Open-Data",           "train"),
}


@tool
def load_hf_dataset(dataset_name: str, split: str = "train", max_rows: int = 5000) -> str:
    """Load a dataset from Hugging Face Hub.

    Args:
        dataset_name: shorthand (spotify, titanic, etc.) or full HF id like 'org/name'.
        split: dataset split, defaults to 'train'.
        max_rows: row cap, kept low to keep things responsive.

    Returns:
        Short summary of the loaded dataset.
    """
    key = dataset_name.lower().strip()
    if key in KNOWN_DATASETS:
        hf_id, split = KNOWN_DATASETS[key]
    else:
        hf_id = dataset_name

    try:
        ds = hf_load_dataset(hf_id, split=f"{split}[:{max_rows}]")
    except Exception as e:
        names = ", ".join(KNOWN_DATASETS)
        return f"Could not load '{dataset_name}' from HF: {e}\nKnown shortcuts: {names}"

    df = ds.to_pandas()
    set_active_df(df, name=dataset_name)
    nums = df.select_dtypes(include="number").columns.tolist()
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return (
        f"Loaded {dataset_name} ({hf_id}): {df.shape[0]:,} rows x {df.shape[1]} cols.\n"
        f"Numeric: {', '.join(nums) or 'none'}\n"
        f"Text: {', '.join(cats) or 'none'}\n"
        f"Missing: {int(df.isnull().sum().sum()):,}\n\n"
        f"Preview:\n{df.head(3).to_string()}"
    )


@tool
def list_available_datasets() -> str:
    """List the built-in HF dataset shortcuts.

    Returns:
        Names and descriptions.
    """
    descs = {
        "spotify":   "114k Spotify tracks with audio features",
        "titanic":   "Titanic passenger survival",
        "netflix":   "Netflix catalogue of shows and movies",
        "sales":     "B2B sales transactions",
        "data jobs": "Data science job postings with salary",
        "airbnb":    "NYC Airbnb listings",
    }
    lines = ["Built-in HF datasets:"]
    for name, desc in descs.items():
        lines.append(f"  {name:<10} {desc}")
    lines.append("\nOr pass any HF dataset id like 'username/name'.")
    lines.append("For Kaggle, use search_kaggle_datasets / fetch_kaggle_dataset.")
    return "\n".join(lines)


def _kaggle_api():
    """Authenticated KaggleApi instance, or None on failure."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return None
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception:
        return None
    return api


@tool
def fetch_kaggle_dataset(dataset_slug: str, file_name: str = "", max_rows: int = 50000) -> str:
    """Download a Kaggle dataset and load the first CSV.

    Args:
        dataset_slug: 'owner/dataset-name'.
        file_name: specific CSV inside the dataset; empty picks the first.
        max_rows: cap on rows read into pandas.

    Returns:
        Short summary of the loaded dataset.
    """
    api = _kaggle_api()
    if api is None:
        return (
            "Kaggle authentication failed. Set KAGGLE_USERNAME and KAGGLE_KEY, "
            "or place kaggle.json in ~/.kaggle/. Get a token at "
            "https://www.kaggle.com/settings/account."
        )

    with tempfile.TemporaryDirectory() as tmp:
        try:
            api.dataset_download_files(dataset_slug, path=tmp, unzip=True)
        except Exception as e:
            return f"Download failed for {dataset_slug}: {e}"

        csvs = glob.glob(os.path.join(tmp, "**", "*.csv"), recursive=True)
        if not csvs:
            return f"No CSV files inside {dataset_slug}."

        if file_name:
            matched = [c for c in csvs if os.path.basename(c) == file_name]
            target = matched[0] if matched else csvs[0]
            if not matched:
                # TODO: surface this back to the agent so it can ask the user
                # which file they wanted; for now we silently fall back.
                pass
        else:
            target = csvs[0]

        df = pd.read_csv(target, nrows=max_rows)

    set_active_df(df, name=os.path.basename(target))
    nums = df.select_dtypes(include="number").columns.tolist()
    cats = df.select_dtypes(include="object").columns.tolist()
    return (
        f"Loaded {dataset_slug} / {os.path.basename(target)}: {df.shape[0]:,} rows x {df.shape[1]} cols.\n"
        f"Numeric: {', '.join(nums) or 'none'}\n"
        f"Text: {', '.join(cats) or 'none'}\n"
        f"Missing: {int(df.isnull().sum().sum()):,}\n\n"
        f"Preview:\n{df.head(3).to_string()}"
    )


@tool
def search_kaggle_datasets(query: str, max_results: int = 8) -> str:
    """Search Kaggle datasets by keyword.

    Args:
        query: search term, e.g. 'football'.
        max_results: how many results to return.

    Returns:
        List of matching slugs and titles.
    """
    api = _kaggle_api()
    if api is None:
        return "Kaggle authentication failed. Set KAGGLE_USERNAME and KAGGLE_KEY."
    try:
        results = api.dataset_list(search=query, page=1)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return f"No Kaggle datasets matched '{query}'."

    lines = [f"Kaggle results for '{query}':"]
    for ds in results[:max_results]:
        lines.append(f"  {ds.ref}  -  {ds.title}")
    lines.append("\nUse fetch_kaggle_dataset('<slug>') to load one.")
    return "\n".join(lines)
