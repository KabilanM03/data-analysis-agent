"""Streamlit entrypoint for the data analysis agent (v2.1).

    streamlit run streamlit_app.py

On Streamlit Community Cloud, point the app at this file. Secrets from the
Secrets panel (or .streamlit/secrets.toml locally) are copied into the process
environment, which is where the tools and agent read their credentials. If an
[auth] section is configured, viewers must sign in with st.login first.
"""

import os

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Data Analysis Agent", page_icon=":material/query_stats:", layout="wide")

# Load a local .env (HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY, etc.) so keys don't
# have to be pasted into the UI on every launch.
load_dotenv()


def _secrets_to_env() -> None:
    try:
        items = dict(st.secrets)
    except Exception:  # no secrets file: fine for local runs with .env
        return
    for key, value in items.items():
        if isinstance(value, (str, int, float)) and key.isupper():
            os.environ.setdefault(key, str(value))


def _auth_gate() -> bool:
    """True if the viewer may use the app. With no [auth] configured the app is open,
    which is right for localhost and for a Community Cloud private app (Streamlit's
    own viewer allow-list then does the gating)."""
    try:
        has_auth = "auth" in st.secrets
    except Exception:
        has_auth = False
    if not has_auth:
        return True
    if not st.user.is_logged_in:
        st.title("Data Analysis Agent")
        st.button("Sign in", on_click=st.login, type="primary", icon=":material/login:")
        return False
    allowed = [e.strip().lower() for e in os.getenv("ALLOWED_EMAILS", "").split(",") if e.strip()]
    if allowed and (st.user.email or "").lower() not in allowed:
        st.error("This account is not on the allow-list.")
        st.button("Sign out", on_click=st.logout)
        return False
    return True


_secrets_to_env()
if not _auth_gate():
    st.stop()

# imported after secrets are in the environment, since some modules read env at import time
from ui import charts_page, chat_page, connections_page, data_page, history_page, sidebar, sql_page  # noqa: E402

sidebar.render()
page = st.navigation(
    [
        st.Page(chat_page.render, title="Chat", icon=":material/chat:", url_path="chat", default=True),
        st.Page(data_page.render, title="Data", icon=":material/table_chart:", url_path="data"),
        st.Page(sql_page.render, title="SQL", icon=":material/code:", url_path="sql"),
        st.Page(charts_page.render, title="Charts", icon=":material/bar_chart:", url_path="charts"),
        st.Page(history_page.render, title="Run history", icon=":material/history:", url_path="history"),
        st.Page(connections_page.render, title="Connections", icon=":material/hub:", url_path="connections"),
    ],
    position="top",
)
page.run()
