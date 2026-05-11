"""Chart rendering tool. Saves a PNG and returns a [CHART:...] sentinel that
the Gradio chat handler picks up and turns into an inline image.
"""

import os
import uuid

import matplotlib
matplotlib.use("Agg")  # headless; Gradio renders the saved PNG
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import tool

from ._state import get_active_df, check_columns

PLOTS_DIR = os.path.abspath(
    os.environ.get("PLOTS_DIR", os.path.join(os.path.dirname(__file__), "..", "plots"))
)
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

VALID_CHARTS = {"bar", "line", "scatter", "histogram", "box", "heatmap"}


def _save(fig) -> str:
    path = os.path.join(PLOTS_DIR, f"plot_{uuid.uuid4().hex[:8]}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


@tool
def create_visualization(
    chart_type: str,
    x_column: str,
    y_column: str = "",
    title: str = "",
    hue_column: str = "",
) -> str:
    """Render a chart from the active dataset and save it as PNG.

    Args:
        chart_type: bar | line | scatter | histogram | box | heatmap.
        x_column: column for the x-axis (or single column for histogram).
        y_column: column for the y-axis (skip for histogram and heatmap).
        title: chart title; auto-generated if empty.
        hue_column: optional grouping column for colour.

    Returns:
        A '[CHART:<path>]' sentinel followed by a human-readable confirmation.
    """
    df = get_active_df()
    chart_type = chart_type.lower().strip()
    if chart_type not in VALID_CHARTS:
        return f"chart_type must be one of {sorted(VALID_CHARTS)}; got '{chart_type}'."

    err = check_columns(df, x_column, y_column, hue_column)
    if err:
        return err

    fig, ax = plt.subplots(figsize=(10, 6))
    final_title = title or f"{chart_type} of {x_column}" + (f" vs {y_column}" if y_column else "")

    try:
        if chart_type == "bar":
            if y_column:
                data = df.groupby(x_column)[y_column].mean(numeric_only=True).reset_index()
                sns.barplot(data=data, x=x_column, y=y_column, hue=hue_column or None, ax=ax)
            else:
                data = df[x_column].value_counts().reset_index()
                data.columns = [x_column, "count"]
                sns.barplot(data=data, x=x_column, y="count", ax=ax)
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "line":
            if not y_column:
                return "y_column is required for a line chart."
            sns.lineplot(data=df, x=x_column, y=y_column, hue=hue_column or None, ax=ax)
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "scatter":
            if not y_column:
                return "y_column is required for a scatter chart."
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column or None, ax=ax, alpha=0.7)

        elif chart_type == "histogram":
            sns.histplot(data=df, x=x_column, kde=True, ax=ax, color="steelblue")

        elif chart_type == "box":
            if not y_column:
                return "y_column is required for a box chart."
            sns.boxplot(data=df, x=x_column, y=y_column, hue=hue_column or None, ax=ax)
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "heatmap":
            numeric = df.select_dtypes(include="number")
            if numeric.empty:
                return "No numeric columns for a heatmap."
            sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    except Exception as e:
        plt.close(fig)
        return f"Chart failed: {e}"

    ax.set_title(final_title, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    path = _save(fig)
    return f"[CHART:{path}]\nSaved chart to {os.path.basename(path)}."
