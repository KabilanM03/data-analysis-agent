"""Expose the data tools as an MCP server, so Claude Desktop, Claude Code or any
MCP client can load datasets, run SQL and draw charts with them.

    python mcp_server.py                      # stdio (for Claude Desktop / Claude Code)
    python mcp_server.py --http --port 8765   # streamable HTTP at http://127.0.0.1:8765/mcp

Register with Claude Code:

    claude mcp add data-agent -- /path/to/.venv/bin/python /path/to/mcp_server.py

The server keeps one active dataset per process (the module-default store), which
is right for stdio, where each client gets its own process. Over HTTP every client
shares that one dataset, so keep HTTP mode on localhost.

The tools are the same smolagents tools the chat agent uses; only the chart tool
is wrapped, so it can return the PNG as an MCP image instead of a file path.
"""

import argparse
import inspect

from mcp.server.fastmcp import FastMCP, Image

import tools as T

# the Kaggle tools read credentials from the host and write to a temp dir; leave them
# to the chat app. Everything else is read-only or bounded.
EXPOSED = [
    T.list_available_datasets,
    T.load_hf_dataset,
    T.load_dataset,
    T.load_cloud_file,
    T.query_motherduck,
    T.query_snowflake,
    T.query_bigquery,
    T.list_data_sources,
    T.describe_dataset,
    T.filter_data,
    T.reset_filters,
    T.run_sql,
    T.aggregate_data,
    T.correlation_analysis,
    T.generate_report,
]

mcp = FastMCP(
    "data-analysis-agent",
    instructions=(
        "Load a dataset first (load_hf_dataset, load_cloud_file or a query_* tool), then analyse it "
        "with run_sql (table `data`), aggregate_data, correlation_analysis and create_chart."
    ),
)


def _as_function(tool):
    """A plain function with the tool's signature and description, which FastMCP
    can turn into an input schema."""
    sig = inspect.signature(tool.forward)
    params = [p for p in sig.parameters.values() if p.name != "self"]

    def fn(**kwargs):
        return tool.forward(**kwargs)

    fn.__signature__ = sig.replace(parameters=params)
    fn.__name__ = tool.name
    fn.__doc__ = tool.description
    return fn


for _tool in EXPOSED:
    mcp.add_tool(_as_function(_tool), name=_tool.name, description=_tool.description,
                 structured_output=False)


@mcp.tool(structured_output=False)
def create_chart(chart_type: str, x_column: str, y_column: str = "", title: str = "",
                 hue_column: str = ""):
    """Render a chart from the active dataset and return it as an image.

    Args:
        chart_type: bar | line | scatter | histogram | box | heatmap.
        x_column: column for the x-axis (or the single column for a histogram).
        y_column: column for the y-axis (skip for histogram and heatmap).
        title: chart title; auto-generated if empty.
        hue_column: optional grouping column for colour.
    """
    out = T.create_visualization.forward(chart_type=chart_type, x_column=x_column, y_column=y_column,
                                         title=title, hue_column=hue_column)
    if not out.startswith("[CHART:"):
        return out  # validation message
    path = out.split("[CHART:", 1)[1].split("]", 1)[0]
    return [f"Chart saved to {path}", Image(path=path)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
