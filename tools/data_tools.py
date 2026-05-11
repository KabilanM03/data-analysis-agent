"""Loaders and basic analysis tools for the active dataframe."""

import os

import pandas as pd
from smolagents import tool

from ._state import (
    get_active_df,
    set_active_df,
    check_columns,
)


@tool
def load_dataset(file_path: str) -> str:
    """Load a CSV file from disk.

    Args:
        file_path: path to the CSV.

    Returns:
        A short summary of the dataset.
    """
    df = pd.read_csv(file_path)
    set_active_df(df, name=os.path.basename(file_path))
    return (
        f"Loaded {os.path.basename(file_path)}: {df.shape[0]:,} rows x {df.shape[1]} cols.\n"
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
    stats = df.describe(include="all").to_string()
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    null_block = nulls.to_string() if not nulls.empty else "no missing values"
    out = f"Stats:\n{stats}\n\nMissing:\n{null_block}"
    if include_sample:
        out += f"\n\nFirst 5 rows:\n{df.head().to_string()}"
    return out


@tool
def filter_data(condition: str) -> str:
    """Filter the active dataset using a pandas .query() expression.

    Args:
        condition: e.g. 'Age > 30' or 'Country == "UK"'.

    Returns:
        Shape and a preview of the filtered result.
    """
    df = get_active_df()
    try:
        out = df.query(condition)
    except Exception as e:
        return f"Filter failed: {e}. Use column names exactly; quote string literals."
    return f"Filtered: {out.shape[0]:,} rows x {out.shape[1]} cols.\nPreview:\n{out.head().to_string()}"


@tool
def aggregate_data(group_by: str, agg_column: str, agg_func: str) -> str:
    """Group by one column and aggregate another.

    Args:
        group_by: column to group by.
        agg_column: column to aggregate.
        agg_func: one of mean, sum, count, min, max, median.

    Returns:
        The aggregated result.
    """
    df = get_active_df()
    err = check_columns(df, group_by, agg_column)
    if err:
        return err
    valid = {"mean", "sum", "count", "min", "max", "median"}
    if agg_func not in valid:
        return f"agg_func must be one of {sorted(valid)}; got '{agg_func}'."
    try:
        result = df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
    except Exception as e:
        return f"Aggregation failed: {e}"
    result.columns = [group_by, f"{agg_func}_{agg_column}"]
    return result.to_string(index=False)


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
