"""Connections page: which cloud sources and MCP servers are usable, and how to
use this agent's own tools from Claude through MCP."""

import os
import re
import shutil
import sys

import pandas as pd
import streamlit as st

import mcp_tools
from tools import configured_sources

from . import session as S

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SOURCE_VARS = {
    "Amazon S3": "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (+ AWS_REGION), or AWS_PROFILE",
    "Google Cloud Storage": "GCS_KEY_ID + GCS_SECRET (HMAC keys)",
    "Azure Blob": "AZURE_STORAGE_CONNECTION_STRING",
    "Cloudflare R2": "R2_ACCOUNT_ID + R2_KEY_ID + R2_SECRET",
    "MotherDuck": "MOTHERDUCK_TOKEN",
    "Snowflake": "SNOWFLAKE_ACCOUNT + SNOWFLAKE_USER + SNOWFLAKE_PRIVATE_KEY_FILE or SNOWFLAKE_PAT",
    "BigQuery": "BIGQUERY_PROJECT + GOOGLE_APPLICATION_CREDENTIALS",
}


def _server_rows(config: dict) -> pd.DataFrame:
    rows = []
    for name, spec in config.items():
        needed = sorted(set(re.findall(r"\$\{([A-Z0-9_]+)\}", str(spec), re.IGNORECASE)))
        missing = [v for v in needed if not os.getenv(v)]
        kind = "HTTP" if "url" in spec else "stdio"
        launcher = spec.get("command", "")
        runnable = kind == "HTTP" or bool(shutil.which(launcher))
        rows.append({
            "server": name,
            "transport": kind,
            "target": spec.get("url") or " ".join([launcher, *spec.get("args", [])]),
            "needs": ", ".join(needed) or "–",
            "ready": runnable and not missing,
            "why not": "; ".join(
                ([f"missing {', '.join(missing)}"] if missing else [])
                + ([] if runnable else [f"{launcher} not installed"])
            ),
        })
    return pd.DataFrame(rows)


def render() -> None:
    sess = S.get()
    st.title("Connections")

    st.subheader("Cloud data sources")
    srcs = configured_sources()
    st.dataframe(
        pd.DataFrame([{"source": k, "configured": v, "set": SOURCE_VARS[k]} for k, v in srcs.items()]),
        hide_index=True,
        column_config={"configured": st.column_config.CheckboxColumn("configured")},
    )
    st.caption("Set these in `.streamlit/secrets.toml` (or Community Cloud's Secrets panel) or as environment "
               "variables. Public `https://` and `hf://` files need nothing.")

    st.subheader("MCP servers")
    config = mcp_tools.load_config()
    st.caption(f"From `{os.path.relpath(mcp_tools.CONFIG_PATH, ROOT)}`. Pick servers in the sidebar before launching.")
    if not config:
        st.info("No MCP servers configured. Copy `mcp_servers.example.json` to `mcp_servers.json`.")
    else:
        st.dataframe(_server_rows(config), hide_index=True,
                     column_config={"ready": st.column_config.CheckboxColumn("ready")})
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        target = c1.selectbox("Test a server", list(config))
        if c2.button("Connect and list tools", icon=":material/lan:"):
            with st.spinner(f"Connecting to {target}…"):
                conn = mcp_tools.connect([target], config)
            if conn.errors.get(target) and not conn.tools:
                st.error(conn.errors[target])
            else:
                st.success(f"{target}: {len(conn.tools)} tools")
                st.dataframe(pd.DataFrame([{"tool": t.name, "description": t.description[:160]} for t in conn.tools]),
                             hide_index=True)
            conn.close()

    if sess.mcp is not None and sess.mcp.tools_by_server:
        st.markdown("**Connected to the running agent**")
        for server, names in sess.mcp.tools_by_server.items():
            st.markdown(f"- `{server}`: {len(names)} tools")

    st.subheader("Use these tools from Claude")
    st.caption("`mcp_server.py` exposes the same tools (load, SQL, charts, warehouse queries) as an MCP server.")
    py = sys.executable
    st.code(f"claude mcp add data-agent -- {py} {os.path.join(ROOT, 'mcp_server.py')}", language="bash")
    st.code(f"{py} {os.path.join(ROOT, 'mcp_server.py')} --http --port 8765\n"
            f"claude mcp add --transport http data-agent http://127.0.0.1:8765/mcp", language="bash")
