"""Loaders and basic analysis tools for the active dataframe."""

import os

import pandas as pd
from smolagents import tool

from ._state import (
    get_active_df,
    set_active_df,
    set_view,
    reset_view,
    check_columns,
)

# Tool output is fed back into the LLM as an observation, so it has to stay small
# or it blows the context window (and, with paid models, the bill). These caps keep
# the largest results to a few hundred characters.
MAX_AGG_ROWS = 50
MAX_DESCRIBE_COLS = 25

# load_dataset only reads CSVs under this directory so the LLM can't be steered
# into reading arbitrary files off the host. Defaults to the project root.
DATA_DIR = os.path.abspath(
    os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), ".."))
)


@tool
def load_dataset(file_path: str) -> str:
    """Load a CSV file from disk (restricted to the project data directory).

    Args:
        file_path: path to the CSV.

    Returns:
        A short summary of the dataset.
    """
    resolved = os.path.realpath(file_path)
    if not resolved.startswith(DATA_DIR + os.sep):
        return (
            f"Refused: '{file_path}' is outside the allowed data directory. "
            "Place the CSV under the project folder, or use load_hf_dataset / "
            "fetch_kaggle_dataset instead."
        )
    if not resolved.lower().endswith(".csv"):
        return f"Only .csv files are supported; got '{os.path.basename(resolved)}'."
    try:
        df = pd.read_csv(resolved)
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"Could not read CSV: {e}"
    set_active_df(df, name=os.path.basename(resolved))
    return (
        f"Loaded {os.path.basename(resolved)}: {df.shape[0]:,} rows x {df.shape[1]} cols.\n"
        f"Columns: {', '.join(df.columns)}"
    )


@tool
def describe_dataset(include_sample: bool = True) -> str:
    """Stats, missing-value counts and an optional sample of the active dataset.

    Args:
        include_sample: include the first 5 rows.

    Returns:
        Summary text.
    """
    df = get_active_df()
    note = ""
    stat_df = df
    if df.shape[1] > MAX_DESCRIBE_COLS:
        stat_df = df.iloc[:, :MAX_DESCRIBE_COLS]
        note = (
            f"\n(Showing stats for the first {MAX_DESCRIBE_COLS} of {df.shape[1]} "
            f"columns.)\nAll columns: {', '.join(df.columns)}"
        )
    stats = stat_df.describe(include="all").to_string()
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    null_block = nulls.to_string() if not nulls.empty else "no missing values"
    out = f"Stats:{note}\n{stats}\n\nMissing:\n{null_block}"
    if include_sample:
        out += f"\n\nFirst 5 rows:\n{df.head().to_string()}"
    return out


@tool
def filter_data(condition: str) -> str:
    """Filter the active dataset and keep the result as the active view.

    After this call, aggregate_data, correlation_analysis, create_visualization and
    generate_report all operate on the filtered subset. Call reset_filters to undo.

    Args:
        condition: a pandas .query() expression, e.g. 'Age > 30' or 'Country == "UK"'.

    Returns:
        Shape and a preview of the filtered result.
    """
    df = get_active_df()
    try:
        out = df.query(condition)
    except Exception as e:
        return f"Filter failed: {e}. Use column names exactly; quote string literals."
    set_view(out)
    return (
        f"Filtered to {out.shape[0]:,} rows x {out.shape[1]} cols. This is now the "
        f"active view; later tools operate on it. Call reset_filters to restore the "
        f"full dataset.\nPreview:\n{out.head().to_string()}"
    )


@tool
def reset_filters() -> str:
    """Drop any active filter and restore the full dataset.

    Returns:
        Confirmation with the restored row count.
    """
    df = reset_view()
    return f"Filters cleared. Active dataset restored to {df.shape[0]:,} rows."


@tool
def aggregate_data(group_by: str, agg_column: str, agg_func: str) -> str:
    """Group by one column and aggregate another.

    Args:
        group_by: column to group by.
        agg_column: column to aggregate.
        agg_func: one of mean, sum, count, min, max, median.

    Returns:
        The aggregated result (top 50 groups).
    """
    df = get_active_df()
    err = check_columns(df, group_by, agg_column)
    if err:
        return err
    valid = {"mean", "sum", "count", "min", "max", "median"}
    if agg_func not in valid:
        return f"agg_func must be one of {sorted(valid)}; got '{agg_func}'."

    numeric_only_funcs = {"mean", "sum", "median"}
    if agg_func in numeric_only_funcs and not pd.api.types.is_numeric_dtype(df[agg_column]):
        nums = df.select_dtypes(include="number").columns.tolist()
        hint = ", ".join(nums) if nums else "none available"
        return (
            f"'{agg_column}' is a text column, so '{agg_func}' does not apply. "
            f"Use 'count', or pick a numeric column: {hint}."
        )

    try:
        result = df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
    except Exception as e:
        return f"Aggregation failed: {e}"
    result.columns = [group_by, f"{agg_func}_{agg_column}"]
    result = result.sort_values(result.columns[1], ascending=False)

    total = len(result)
    text = result.head(MAX_AGG_ROWS).to_string(index=False)
    if total > MAX_AGG_ROWS:
        text += f"\n... and {total - MAX_AGG_ROWS} more groups (showing top {MAX_AGG_ROWS} by {agg_func})."
    return text


@tool
def correlation_analysis(columns: str = "") -> str:
    """Correlation matrix for numeric columns.

    Args:
        columns: comma-separated columns. Empty = all numeric.

    Returns:
        Correlation matrix as text.
    """
    df = get_active_df()
    if columns.strip():
        col_list = [c.strip() for c in columns.split(",")]
        err = check_columns(df, *col_list)
        if err:
            return err
        df = df[col_list]
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return "No numeric columns to correlate."
    return numeric.corr().round(3).to_string()
