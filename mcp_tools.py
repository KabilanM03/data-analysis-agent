"""Connect external MCP servers and hand their tools to the agent.

Servers are declared in `mcp_servers.json` (same shape as Claude Desktop's config):

    {"mcpServers": {
        "motherduck": {"command": "uvx", "args": ["mcp-server-motherduck", "--db-path", "md:"],
                       "env": {"motherduck_token": "${MOTHERDUCK_TOKEN}"}},
        "bigquery":   {"url": "https://bigquery.googleapis.com/mcp",
                       "headers": {"Authorization": "Bearer ${GCP_ACCESS_TOKEN}"}}
    }}

`${VAR}` is expanded from the environment, so secrets stay out of the file.
Entries with "command" run as stdio subprocesses; entries with "url" use
streamable HTTP. A server whose variables are missing is reported and skipped.
"""

import json
import os
import re
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
# mcp_servers.json is yours (gitignored); the example ships with the repo so the UI has something to list
CONFIG_PATH = os.getenv("MCP_CONFIG") or next(
    (p for p in (os.path.join(_HERE, "mcp_servers.json"), os.path.join(_HERE, "mcp_servers.example.json"))
     if os.path.exists(p)),
    os.path.join(_HERE, "mcp_servers.json"),
)
_VAR = re.compile(r"\$\{([A-Z0-9_]+)\}", re.IGNORECASE)


@dataclass
class MCPConnection:
    """Live connections for one session. Call close() when the session resets."""
    clients: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    tools_by_server: dict[str, list[str]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def close(self) -> None:
        for client in self.clients:
            try:
                client.disconnect()
            except Exception:
                pass
        self.clients.clear()
        self.tools.clear()


def load_config(path: str = CONFIG_PATH) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get("mcpServers", {})


def _expand(value, missing: set[str]):
    if isinstance(value, str):
        def sub(m):
            v = os.getenv(m.group(1))
            if v is None:
                missing.add(m.group(1))
                return ""
            return v
        return _VAR.sub(sub, value)
    if isinstance(value, list):
        return [_expand(v, missing) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, missing) for k, v in value.items()}
    return value


def server_parameters(spec: dict):
    """Turn one config entry into what smolagents.MCPClient accepts.
    Raises KeyError naming the missing env vars."""
    missing: set[str] = set()
    spec = _expand(spec, missing)
    if missing:
        raise KeyError(", ".join(sorted(missing)))
    if "url" in spec:
        params = {"url": spec["url"], "transport": spec.get("transport", "streamable-http")}
        if spec.get("headers"):
            params["headers"] = spec["headers"]
        return params
    from mcp import StdioServerParameters

    return StdioServerParameters(
        command=spec["command"],
        args=spec.get("args", []),
        env={**os.environ, **spec.get("env", {})},
    )


def connect(names: list[str], config: dict[str, dict] | None = None,
            reserved_names: set[str] | None = None) -> MCPConnection:
    """Connect to each named server. Failures are collected per server, not raised,
    so one bad server doesn't stop the others. Tools whose names clash with a
    built-in or an earlier server's tool are skipped (CodeAgent needs unique names)."""
    from smolagents import MCPClient

    config = load_config() if config is None else config
    taken = set(reserved_names or ())
    conn = MCPConnection()
    for name in names:
        if name not in config:
            conn.errors[name] = "not in mcp_servers.json"
            continue
        try:
            params = server_parameters(config[name])
        except KeyError as e:
            conn.errors[name] = f"missing environment variable(s): {e.args[0]}"
            continue
        try:
            client = MCPClient(params, structured_output=False)
        except Exception as e:
            conn.errors[name] = f"could not connect: {e}"
            continue
        conn.clients.append(client)
        kept = []
        for tool in client.get_tools():
            if tool.name in taken:
                conn.errors.setdefault(name, "")
                conn.errors[name] += f"skipped duplicate tool '{tool.name}'. "
                continue
            taken.add(tool.name)
            conn.tools.append(tool)
            kept.append(tool.name)
        conn.tools_by_server[name] = kept
    return conn
