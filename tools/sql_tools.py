"""SQL over the active dataset, via an in-memory DuckDB connection.

The active view is registered as `data` and the unfiltered frame as `data_full`.
The connection is opened with external access disabled, so a query can't read
files off the host, install extensions or ATTACH databases, and only read-only
statement types are accepted.
"""

import duckdb
import pandas as pd
from smolagents import tool

from ._state import get_active_df, get_full_df, set_active_df, get_active_name

MAX_SQL_ROWS = 50  # rows echoed back to the model; the full result can be kept as the active dataset

READ_ONLY_TYPES = {duckdb.StatementType.SELECT, duckdb.StatementType.EXPLAIN}


def check_read_only(sql: str) -> str | None:
    """Return an error message unless `sql` is exactly one read-only statement."""
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as e:
        return f"SQL did not parse: {e}"
    if len(statements) != 1:
        return f"Send exactly one statement; got {len(statements)}."
    if statements[0].type not in READ_ONLY_TYPES:
        return f"Only SELECT queries are allowed; got {statements[0].type.name}."
    return None


def query_frames(sql: str, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run a read-only query against the given dataframes in a sandboxed connection."""
    con = duckdb.connect(config={"enable_external_access": False})
    try:
        for name, df in frames.items():
            con.register(name, df)
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def format_result(df: pd.DataFrame, limit: int = MAX_SQL_ROWS) -> str:
    text = df.head(limit).to_string(index=False)
    if len(df) > limit:
        text += f"\n... {len(df) - limit:,} more rows (showing {limit})."
    return f"{len(df):,} rows x {df.shape[1]} cols\n{text}"


@tool
def run_sql(query: str, save_as_active: bool = False) -> str:
    """Run a DuckDB SQL query against the loaded dataset.

    The active dataset (after any filter) is the table `data`; the unfiltered
    dataset is `data_full`. Only a single SELECT (or WITH ... SELECT) is allowed.

    Args:
        query: DuckDB SQL, e.g. 'SELECT region, SUM(sales) AS total FROM data GROUP BY 1 ORDER BY 2 DESC'.
        save_as_active: if true, the query result becomes the active dataset so later
            charts and aggregations use it.

    Returns:
        The result (first 50 rows) as text.
    """
    err = check_read_only(query)
    if err:
        return err
    frames = {"data": get_active_df()}
    full = get_full_df()
    if full is not None:
        frames["data_full"] = full
    try:
        result = query_frames(query, frames)
    except duckdb.Error as e:
        return f"Query failed: {e}"
    if save_as_active:
        set_active_df(result, name=f"sql result ({get_active_name() or 'data'})")
        return format_result(result) + "\nSaved as the active dataset."
    return format_result(result)
