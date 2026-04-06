import pandas as pd
import json
from smolagents import tool

# Global dataframe store
_df_store: dict[str, pd.DataFrame] = {}


def get_active_df() -> pd.DataFrame:
    if not _df_store:
        raise ValueError("No dataset loaded. Use load_dataset first.")
    key = list(_df_store.keys())[-1]
    return _df_store[key]


def set_active_df(df: pd.DataFrame, name: str = "dataset") -> None:
    _df_store[name] = df


@tool
def load_dataset(file_path: str) -> str:
    """
    Load a CSV file into memory for analysis.

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        A summary string with shape, column names, and dtypes.
    """
    df = pd.read_csv(file_path)
    set_active_df(df, name=file_path.split("/")[-1])
    summary = (
        f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns.\n"
        f"Columns: {', '.join(df.columns.tolist())}\n"
        f"Dtypes:\n{df.dtypes.to_string()}"
    )
    return summary


@tool
def describe_dataset(include_sample: bool = True) -> str:
    """
    Return descriptive statistics and an optional sample of the loaded dataset.

    Args:
        include_sample: Whether to include the first 5 rows as a sample.

    Returns:
        A string with descriptive statistics and optional sample rows.
    """
    df = get_active_df()
    stats = df.describe(include="all").to_string()
    missing = df.isnull().sum()
    missing_str = missing[missing > 0].to_string() if missing.any() else "None"
    result = f"=== Descriptive Statistics ===\n{stats}\n\n=== Missing Values ===\n{missing_str}"
    if include_sample:
        result += f"\n\n=== First 5 Rows ===\n{df.head().to_string()}"
    return result


@tool
def filter_data(condition: str) -> str:
    """
    Filter the dataset using a pandas query condition string and return the result shape and preview.

    Args:
        condition: A pandas query string, e.g. 'Age > 30' or 'Department == "Sales"'.

    Returns:
        Shape of filtered data and top 5 rows as string.
    """
    df = get_active_df()
    filtered = df.query(condition)
    return (
        f"Filtered result: {filtered.shape[0]} rows x {filtered.shape[1]} columns.\n"
        f"Preview:\n{filtered.head().to_string()}"
    )


@tool
def aggregate_data(group_by: str, agg_column: str, agg_func: str) -> str:
    """
    Group the dataset by a column and apply an aggregation function.

    Args:
        group_by: Column name to group by.
        agg_column: Column to aggregate.
        agg_func: Aggregation function — one of: mean, sum, count, min, max, median.

    Returns:
        Aggregated result as a string.
    """
    df = get_active_df()
    valid_funcs = {"mean", "sum", "count", "min", "max", "median"}
    if agg_func not in valid_funcs:
        return f"Invalid agg_func '{agg_func}'. Choose from: {', '.join(valid_funcs)}"
    result = df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
    result.columns = [group_by, f"{agg_func}_{agg_column}"]
    return result.to_string(index=False)


@tool
def correlation_analysis(columns: str = "") -> str:
    """
    Compute the correlation matrix for numeric columns in the dataset.

    Args:
        columns: Comma-separated column names to include. Leave empty for all numeric columns.

    Returns:
        Correlation matrix as a formatted string.
    """
    df = get_active_df()
    if columns.strip():
        col_list = [c.strip() for c in columns.split(",")]
        df = df[col_list]
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return "No numeric columns found for correlation analysis."
    corr = numeric_df.corr().round(3)
    return f"=== Correlation Matrix ===\n{corr.to_string()}"
