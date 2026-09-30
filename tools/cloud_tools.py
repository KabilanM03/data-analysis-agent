"""Cloud data sources: object storage (S3 / GCS / Azure / R2 / HF) via DuckDB,
and read-only queries against MotherDuck, Snowflake and BigQuery.

Every connector reads its credentials from environment variables (the Streamlit
app copies them in from st.secrets), and every warehouse query is checked to be a
single SELECT before it is sent. The heavier client libraries (Snowflake,
BigQuery) are optional installs, imported only when their tool is called.
"""

import os
import re

import duckdb
import pandas as pd
from smolagents import tool

from ._state import set_active_df
from .sql_tools import check_read_only, format_result

MAX_CLOUD_ROWS = 100_000   # rows pulled into pandas from a file or warehouse query
BQ_MAX_BYTES = int(os.getenv("BQ_MAX_BYTES", str(1024**3)))  # refuse BigQuery scans over 1 GiB

CLOUD_SCHEMES = ("s3://", "gs://", "gcs://", "az://", "azure://", "abfss://", "r2://", "hf://", "https://")
READERS = {"parquet": "read_parquet", "csv": "read_csv_auto", "json": "read_json_auto"}


def _env(*names: str) -> bool:
    return all(os.getenv(n) for n in names)


def configured_sources() -> dict[str, bool]:
    """Which connectors have credentials. Used by the UI and by list_data_sources."""
    return {
        "Amazon S3": _env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY") or _env("AWS_PROFILE"),
        "Google Cloud Storage": _env("GCS_KEY_ID", "GCS_SECRET"),
        "Azure Blob": _env("AZURE_STORAGE_CONNECTION_STRING"),
        "Cloudflare R2": _env("R2_ACCOUNT_ID", "R2_KEY_ID", "R2_SECRET"),
        "MotherDuck": _env("MOTHERDUCK_TOKEN"),
        "Snowflake": _env("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER")
        and (_env("SNOWFLAKE_PRIVATE_KEY_FILE") or _env("SNOWFLAKE_PAT")),
        "BigQuery": _env("BIGQUERY_PROJECT"),
    }


def _guess_format(uri: str) -> str | None:
    path = uri.split("?", 1)[0].lower()
    if path.endswith(".parquet") or path.endswith(".pq"):
        return "parquet"
    if path.endswith((".csv", ".tsv", ".csv.gz")):
        return "csv"
    if path.endswith((".json", ".jsonl", ".ndjson")):
        return "json"
    return None


def _storage_secrets(con: duckdb.DuckDBPyConnection) -> None:
    """Register the object-storage secrets we have credentials for."""
    if _env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        con.execute(
            "CREATE SECRET (TYPE s3, KEY_ID ?, SECRET ?, REGION ?)",
            [os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"],
             os.getenv("AWS_REGION", "us-east-1")],
        )
    elif _env("AWS_PROFILE"):
        # picks up SSO / assumed-role credentials from the local AWS config
        con.execute("CREATE SECRET (TYPE s3, PROVIDER credential_chain)")
    if _env("GCS_KEY_ID", "GCS_SECRET"):
        # DuckDB reads GCS through its S3-compatible API, which needs HMAC keys
        con.execute("CREATE SECRET (TYPE gcs, KEY_ID ?, SECRET ?)",
                    [os.environ["GCS_KEY_ID"], os.environ["GCS_SECRET"]])
    if _env("AZURE_STORAGE_CONNECTION_STRING"):
        con.execute("CREATE SECRET (TYPE azure, CONNECTION_STRING ?)",
                    [os.environ["AZURE_STORAGE_CONNECTION_STRING"]])
    if _env("R2_ACCOUNT_ID", "R2_KEY_ID", "R2_SECRET"):
        con.execute("CREATE SECRET (TYPE r2, KEY_ID ?, SECRET ?, ACCOUNT_ID ?)",
                    [os.environ["R2_KEY_ID"], os.environ["R2_SECRET"], os.environ["R2_ACCOUNT_ID"]])


def read_cloud_file(uri: str, file_format: str = "auto", max_rows: int = MAX_CLOUD_ROWS) -> pd.DataFrame:
    """Read a cloud file (or glob) into pandas. Raises ValueError on bad input."""
    uri = uri.strip()
    if not uri.lower().startswith(CLOUD_SCHEMES):
        raise ValueError(
            f"Unsupported location '{uri}'. Use one of: {', '.join(CLOUD_SCHEMES)}. "
            "For a local CSV use load_dataset."
        )
    fmt = _guess_format(uri) if file_format == "auto" else file_format.lower()
    if fmt not in READERS:
        raise ValueError("Could not tell the file format; pass file_format='parquet', 'csv' or 'json'.")
    con = duckdb.connect()
    try:
        _storage_secrets(con)
        return con.execute(
            f"SELECT * FROM {READERS[fmt]}(?) LIMIT ?", [uri, int(max_rows)]
        ).fetchdf()
    finally:
        con.close()


@tool
def load_cloud_file(uri: str, file_format: str = "auto", max_rows: int = 100000) -> str:
    """Load a Parquet, CSV or JSON file from cloud storage and make it the active dataset.

    Supports s3://, gs://, az://, r2://, hf:// (Hugging Face) and https:// URLs,
    including globs like 's3://bucket/events/*.parquet'. Credentials come from the
    app's configured secrets.

    Args:
        uri: file location, e.g. 's3://my-bucket/sales/2026.parquet' or
            'hf://datasets/org/name/data/train.parquet'.
        file_format: 'auto' (from the extension), 'parquet', 'csv' or 'json'.
        max_rows: row cap for the load.

    Returns:
        Short summary of the loaded dataset.
    """
    try:
        df = read_cloud_file(uri, file_format, max_rows)
    except ValueError as e:
        return str(e)
    except duckdb.Error as e:
        return f"Could not read {uri}: {e}"
    name = uri.rstrip("/").rsplit("/", 1)[-1] or uri
    set_active_df(df, name=name)
    return (
        f"Loaded {uri}: {df.shape[0]:,} rows x {df.shape[1]} cols.\n"
        f"Columns: {', '.join(map(str, df.columns))}\n\nPreview:\n{df.head(3).to_string()}"
    )


WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|copy|put|call|execute)\b",
    re.IGNORECASE,
)


def warehouse_read_only(query: str) -> str | None:
    """Conservative check for warehouse SQL that DuckDB can't parse (BigQuery backticks,
    Snowflake-only syntax): one statement, starts with SELECT/WITH, no write keywords
    outside string literals. The real control is a read-only role on the warehouse side."""
    body = query.strip().rstrip(";")
    if ";" in body:
        return "Send exactly one statement."
    if not re.match(r"^\s*\(*\s*(select|with)\b", body, re.IGNORECASE):
        return "Only a single SELECT (or WITH ... SELECT) query is allowed."
    if WRITE_WORDS.search(re.sub(r"'[^']*'", "''", body)):
        return "Refused: the query contains a write keyword. Only read-only queries are allowed."
    return None


def _finish_query(df: pd.DataFrame, source: str, save_as_active: bool) -> str:
    if save_as_active:
        set_active_df(df, name=f"{source} query")
        return format_result(df) + "\nSaved as the active dataset."
    return format_result(df)


@tool
def query_motherduck(query: str, save_as_active: bool = True) -> str:
    """Run a read-only SQL query on MotherDuck (cloud DuckDB).

    Args:
        query: a single SELECT, e.g. 'SELECT * FROM sample_data.nyc.taxi LIMIT 1000'.
        save_as_active: make the result the active dataset (default true).

    Returns:
        The result (first 50 rows) as text.
    """
    if not _env("MOTHERDUCK_TOKEN"):
        return "MotherDuck is not configured. Set MOTHERDUCK_TOKEN."
    err = check_read_only(query)
    if err:
        return err
    try:
        con = duckdb.connect("md:", config={"motherduck_token": os.environ["MOTHERDUCK_TOKEN"]})
        try:
            df = con.execute(query).fetchdf().head(MAX_CLOUD_ROWS)
        finally:
            con.close()
    except duckdb.Error as e:
        return f"MotherDuck query failed: {e}"
    return _finish_query(df, "MotherDuck", save_as_active)


def _snowflake_connect():
    import snowflake.connector  # optional: pip install "snowflake-connector-python[pandas]"

    params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "session_parameters": {"QUERY_TAG": "data-analysis-agent"},
    }
    if os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE"):
        # key-pair auth; Snowflake is phasing out password-only sign-in
        params.update(authenticator="SNOWFLAKE_JWT",
                      private_key_file=os.environ["SNOWFLAKE_PRIVATE_KEY_FILE"],
                      private_key_file_pwd=os.getenv("SNOWFLAKE_PRIVATE_KEY_PWD"))
    else:
        # a programmatic access token goes in the password field
        params["password"] = os.environ["SNOWFLAKE_PAT"]
    return snowflake.connector.connect(**{k: v for k, v in params.items() if v is not None})


@tool
def query_snowflake(query: str, save_as_active: bool = True) -> str:
    """Run a read-only SQL query on Snowflake.

    Args:
        query: a single SELECT, e.g. 'SELECT * FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS LIMIT 1000'.
        save_as_active: make the result the active dataset (default true).

    Returns:
        The result (first 50 rows) as text.
    """
    if not configured_sources()["Snowflake"]:
        return ("Snowflake is not configured. Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER and either "
                "SNOWFLAKE_PRIVATE_KEY_FILE (key-pair) or SNOWFLAKE_PAT.")
    err = warehouse_read_only(query)
    if err:
        return err
    try:
        con = _snowflake_connect()
    except ImportError:
        return 'Snowflake client not installed: pip install "snowflake-connector-python[pandas]"'
    except Exception as e:
        return f"Snowflake connection failed: {e}"
    try:
        cur = con.cursor()
        cur.execute(query)
        df = cur.fetch_pandas_all().head(MAX_CLOUD_ROWS)
    except Exception as e:
        return f"Snowflake query failed: {e}"
    finally:
        con.close()
    return _finish_query(df, "Snowflake", save_as_active)


@tool
def query_bigquery(query: str, save_as_active: bool = True) -> str:
    """Run a read-only GoogleSQL query on BigQuery. Refuses queries that would scan more than 1 GiB.

    Args:
        query: a single SELECT, e.g.
            'SELECT name, SUM(number) n FROM `bigquery-public-data.usa_names.usa_1910_2013` GROUP BY 1 ORDER BY 2 DESC LIMIT 20'.
        save_as_active: make the result the active dataset (default true).

    Returns:
        The result (first 50 rows) and bytes scanned.
    """
    if not _env("BIGQUERY_PROJECT"):
        return "BigQuery is not configured. Set BIGQUERY_PROJECT (and GOOGLE_APPLICATION_CREDENTIALS)."
    err = warehouse_read_only(query)
    if err:
        return err
    try:
        from google.cloud import bigquery  # optional: pip install "google-cloud-bigquery[pandas]"
    except ImportError:
        return 'BigQuery client not installed: pip install "google-cloud-bigquery[pandas]"'
    try:
        client = bigquery.Client(project=os.environ["BIGQUERY_PROJECT"])
        dry = client.query(query, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
        scanned = dry.total_bytes_processed or 0
        if scanned > BQ_MAX_BYTES:
            return (f"Refused: this query would scan {scanned / 1024**3:.2f} GiB "
                    f"(limit {BQ_MAX_BYTES / 1024**3:.2f} GiB). Select fewer columns or add a filter.")
        job = client.query(query, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=BQ_MAX_BYTES))
        df = job.result(max_results=MAX_CLOUD_ROWS).to_dataframe()
    except Exception as e:
        return f"BigQuery query failed: {e}"
    return _finish_query(df, "BigQuery", save_as_active) + f"\nScanned {scanned / 1024**2:.1f} MiB."


@tool
def list_data_sources() -> str:
    """List which cloud data sources are configured for this session.

    Returns:
        Each source with configured / not configured.
    """
    lines = ["Cloud data sources:"]
    for name, ok in configured_sources().items():
        lines.append(f"  {name:<22} {'configured' if ok else 'not configured'}")
    lines.append("\nPublic https:// and hf:// files work without credentials.")
    return "\n".join(lines)
