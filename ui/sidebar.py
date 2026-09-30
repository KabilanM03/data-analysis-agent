"""Sidebar: model settings, MCP servers and the current dataset at a glance."""

import os

import streamlit as st

import mcp_tools
from agent import PROVIDERS, build_agent
from tools import ALL_TOOLS

from . import session as S


def _launch(sess, provider, model_id, api_key, servers, max_steps, planning, stream_tokens):
    sess.close_agent()
    conn = None
    if servers:
        with st.spinner(f"Connecting {len(servers)} MCP server(s)…"):
            conn = mcp_tools.connect(servers, reserved_names={t.name for t in ALL_TOOLS})
    try:
        agent, name = build_agent(
            provider=provider,
            model_id=model_id,
            api_key=api_key or None,
            extra_tools=conn.tools if conn else None,
            store=sess.store,
            max_steps=max_steps,
            planning_interval=planning or None,
            stream_outputs=stream_tokens,
        )
    except Exception as e:
        if conn:
            conn.close()
        st.error(str(e))
        return
    sess.agent, sess.model_name, sess.provider, sess.mcp = agent, name, provider, conn
    if conn and conn.errors:
        for server, err in conn.errors.items():
            st.warning(f"MCP {server}: {err}")


def render() -> None:
    sess = S.get()
    with st.sidebar:
        st.subheader("Agent")
        if sess.agent is not None:
            n_mcp = len(sess.mcp.tools) if sess.mcp else 0
            st.success(f"**{sess.model_name}**  \n{len(ALL_TOOLS)} built-in tools"
                       + (f" + {n_mcp} MCP tools" if n_mcp else ""), icon=":material/smart_toy:")
        else:
            st.info("Not launched yet. Pick a model and launch.", icon=":material/power_settings_new:")

        # outside the form so changing it reruns and updates the fields below
        provider = st.selectbox(
            "Provider", list(PROVIDERS), format_func=lambda k: PROVIDERS[k].label,
            index=list(PROVIDERS).index(sess.provider) if sess.provider in PROVIDERS else 0,
        )
        spec = PROVIDERS[provider]
        with st.form("agent_settings", border=False):
            model_id = st.text_input("Model id", placeholder=spec.default_model,
                                     help="Blank uses the default shown. Any id the provider serves works.")
            have_key = bool(spec.key_env and os.getenv(spec.key_env))
            api_key = ""
            if spec.key_env:
                api_key = st.text_input(
                    "API key", type="password",
                    placeholder="using key from secrets" if have_key else spec.key_env,
                    help="Kept in this browser session only. Leave blank to use the configured secret.",
                )
            config = mcp_tools.load_config()
            servers = st.multiselect("MCP servers", list(config), help="From mcp_servers.json. See the Connections page.")
            with st.expander("Advanced"):
                max_steps = st.slider("Max steps", 3, 25, 10)
                planning = st.number_input("Plan every N steps (0 = off)", 0, 10, 0)
                stream_tokens = st.toggle("Stream model tokens", value=True,
                                          help="Show the model's text as it is written.")
            if st.form_submit_button("Launch agent", type="primary", width="stretch",
                                     icon=":material/rocket_launch:"):
                _launch(sess, provider, model_id, api_key, servers, max_steps, planning, stream_tokens)
                st.rerun()

        st.divider()
        st.subheader("Dataset")
        df = sess.df()
        if df is None:
            st.caption("Nothing loaded. Ask the agent, drop a file in the chat, or use the Data page.")
        else:
            full = sess.full_df()
            st.markdown(f"**{sess.dataset_name() or 'dataset'}**")
            c1, c2 = st.columns(2)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Columns", df.shape[1])
            if full is not None and len(full) != len(df):
                st.caption(f"Filtered view of {len(full):,} rows")

        if sess.messages and st.button("Clear chat", icon=":material/delete_sweep:", width="stretch"):
            sess.messages.clear()
            sess.has_history = False
            st.rerun()
