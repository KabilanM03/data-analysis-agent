"""Tests for the v2.1 additions: SQL tool, cloud connectors' guards, MCP config
handling, the MCP server, the streaming agent loop and the Streamlit app."""

import asyncio
import os
from unittest.mock import patch

import pandas as pd
import pytest
from smolagents import CodeAgent

from tools import ALL_TOOLS, bind_tools, interactive_path, pop_charts
from tools.cloud_tools import (
    list_data_sources,
    load_cloud_file,
    query_bigquery,
    query_motherduck,
    query_snowflake,
    warehouse_read_only,
)
from tools.data_tools import filter_data
from tools.sql_tools import check_read_only, run_sql
from tools.viz_tools import create_visualization

from .scripted_model import ScriptedModel, code_turn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- run_sql ----------------------------------------------------------------
def test_run_sql_groups(sample_df):
    out = run_sql("SELECT region, SUM(sales) AS total FROM data GROUP BY 1 ORDER BY 2 DESC")
    assert out.startswith("3 rows")
    assert out.index("US") < out.index("UK") < out.index("EU")


def test_run_sql_sees_filter_and_full(sample_df):
    filter_data("region == 'UK'")
    assert "2 rows" in run_sql("SELECT * FROM data")
    assert "6 rows" in run_sql("SELECT * FROM data_full")


def test_run_sql_save_as_active(sample_df, isolated_store):
    run_sql("SELECT region, SUM(units) u FROM data GROUP BY 1", save_as_active=True)
    assert list(isolated_store.get().columns) == ["region", "u"]


@pytest.mark.parametrize("sql", [
    "DROP TABLE data",
    "INSERT INTO data VALUES ('X', 1, 1)",
    "SELECT 1; SELECT 2",
    "COPY data TO 'out.csv'",
])
def test_run_sql_rejects_writes(sample_df, sql):
    assert check_read_only(sql) is not None
    assert "rows" not in run_sql(sql)


def test_run_sql_cannot_read_host_files(sample_df):
    out = run_sql("SELECT * FROM read_csv_auto('data/sample_sales.csv')")
    assert out.startswith("Query failed")


# -- cloud connectors -------------------------------------------------------
def test_load_cloud_file_rejects_local_paths():
    for uri in ["/etc/passwd", "file:///etc/passwd", "data/sample_sales.csv", "http://169.254.169.254/x.csv"]:
        assert load_cloud_file(uri).startswith("Unsupported location")


def test_load_cloud_file_needs_format():
    assert "file format" in load_cloud_file("https://example.com/download")


@pytest.mark.parametrize("sql,ok", [
    ("SELECT * FROM `proj.ds.t`", True),
    ("with x as (select 1) select * from x", True),
    ("SELECT 'drop table x' AS note", True),        # keyword inside a string is fine
    ("DELETE FROM t", False),
    ("SELECT 1; DROP TABLE t", False),
    ("WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x", False),
])
def test_warehouse_read_only(sql, ok):
    assert (warehouse_read_only(sql) is None) == ok


def test_warehouses_report_missing_config(monkeypatch):
    for var in ["MOTHERDUCK_TOKEN", "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "BIGQUERY_PROJECT"]:
        monkeypatch.delenv(var, raising=False)
    assert "not configured" in query_motherduck("SELECT 1")
    assert "not configured" in query_snowflake("SELECT 1")
    assert "not configured" in query_bigquery("SELECT 1")
    assert "MotherDuck" in list_data_sources()


def test_snowflake_uses_key_pair_when_given(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "svc")
    monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY_FILE", "/keys/rsa.p8")
    captured = {}

    class FakeCursor:
        def execute(self, q): captured["q"] = q
        def fetch_pandas_all(self): return pd.DataFrame({"n": [1, 2]})

    class FakeConn:
        def cursor(self): return FakeCursor()
        def close(self): captured["closed"] = True

    def fake_connect(**kw):
        captured.update(kw)
        return FakeConn()

    import types, sys
    fake_mod = types.SimpleNamespace(connector=types.SimpleNamespace(connect=fake_connect))
    with patch.dict(sys.modules, {"snowflake": fake_mod, "snowflake.connector": fake_mod.connector}):
        out = query_snowflake("SELECT n FROM t", save_as_active=False)
    assert out.startswith("2 rows")
    assert captured["authenticator"] == "SNOWFLAKE_JWT"
    assert "password" not in captured
    assert captured["closed"]


# -- charts -----------------------------------------------------------------
def test_chart_writes_interactive_twin(sample_df):
    out = create_visualization("bar", "region", "sales")
    path = out.split("[CHART:", 1)[1].split("]", 1)[0]
    assert os.path.exists(interactive_path(path))


# -- MCP client config ------------------------------------------------------
def test_mcp_config_expands_env(monkeypatch):
    import mcp_tools
    monkeypatch.setenv("DEMO_TOKEN", "abc")
    params = mcp_tools.server_parameters({"url": "https://x/mcp", "headers": {"Authorization": "Bearer ${DEMO_TOKEN}"}})
    assert params["headers"]["Authorization"] == "Bearer abc"
    assert params["transport"] == "streamable-http"


def test_mcp_config_reports_missing_env(monkeypatch):
    import mcp_tools
    monkeypatch.delenv("NOT_SET_TOKEN", raising=False)
    conn = mcp_tools.connect(["s"], {"s": {"url": "https://x/mcp", "headers": {"A": "${NOT_SET_TOKEN}"}}})
    assert "NOT_SET_TOKEN" in conn.errors["s"]
    assert conn.tools == []


def test_example_mcp_config_is_valid():
    import mcp_tools
    config = mcp_tools.load_config(os.path.join(ROOT, "mcp_servers.example.json"))
    assert {"motherduck", "bigquery", "aws"} <= set(config)


# -- MCP server -------------------------------------------------------------
def test_mcp_server_lists_and_runs_tools(sample_df):
    import mcp_server

    async def go():
        names = {t.name for t in await mcp_server.mcp.list_tools()}
        sql = await mcp_server.mcp.call_tool("run_sql", {"query": "SELECT COUNT(*) AS n FROM data"})
        chart = await mcp_server.mcp.call_tool("create_chart", {"chart_type": "bar", "x_column": "region"})
        return names, sql, chart

    names, sql, chart = asyncio.run(go())
    assert {"run_sql", "load_cloud_file", "create_chart", "query_snowflake"} <= names
    assert "fetch_kaggle_dataset" not in names
    content = sql[0] if isinstance(sql, tuple) else sql
    assert "6" in content[0].text
    chart_content = chart[0] if isinstance(chart, tuple) else chart
    assert any(c.type == "image" for c in chart_content)


# -- agent loop, offline ----------------------------------------------------
def _scripted_agent(store):
    model = ScriptedModel([
        code_turn("Total sales per region with SQL.",
                  'print(run_sql(query="SELECT region, SUM(sales) AS total FROM data GROUP BY 1 ORDER BY 2 DESC"))'),
        code_turn("Chart it and answer.",
                  'c = create_visualization(chart_type="bar", x_column="region", y_column="sales")\n'
                  'final_answer("US leads with 550 in sales. " + c.split(chr(10))[0])'),
    ])
    return CodeAgent(tools=bind_tools(store, ALL_TOOLS), model=model, max_steps=4, verbosity_level=0)


def test_agent_streams_steps_and_charts(sample_df, isolated_store):
    from smolagents.memory import ActionStep, FinalAnswerStep

    events = list(_scripted_agent(isolated_store).run("Which region sells most?", stream=True))
    actions = [e for e in events if isinstance(e, ActionStep)]
    assert len(actions) == 2
    assert "550" in actions[0].observations
    final = events[-1]
    assert isinstance(final, FinalAnswerStep) and "[CHART:" in str(final.output)
    assert len(pop_charts()) == 1


# -- Streamlit app ----------------------------------------------------------
def test_streamlit_app_boots_and_runs_scripted_agent(sample_df):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT, "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception

    from ui.session import Session
    sess = Session()
    sess.store.set(sample_df, name="test")
    sess.agent, sess.model_name = _scripted_agent(sess.store), "scripted"
    at.session_state["session"] = sess
    at.chat_input[0].set_value("Which region sells most?").run()
    assert not at.exception
    assert len(sess.runs) == 1 and sess.runs[0]["status"] == "ok"
    assert sess.runs[0]["steps"] == 2
    assert "US leads" in sess.messages[-1]["content"]
    assert "[CHART:" not in sess.messages[-1]["content"]
    assert len(sess.charts) == 1


def test_agent_tools_stay_in_their_session():
    """Regression: smolagents runs generated code on a worker thread, which used to
    drop the ContextVar binding, so every session silently shared the default store."""
    from tools import DataframeStore
    from tools._state import _default_store

    mine, other = DataframeStore(), DataframeStore()
    mine.set(pd.DataFrame({"region": ["UK", "US"], "sales": [1, 550]}), name="mine")
    before = _default_store.full()
    list(_scripted_agent(mine).run("Which region sells most?", stream=True))
    assert _default_store.full() is before      # nothing leaked into the shared default
    assert other.full() is None
    assert len(mine.pop_charts()) == 1         # the chart landed in the right session
