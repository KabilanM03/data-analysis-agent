import os
import uuid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from smolagents import tool
from .data_tools import get_active_df

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def _save_fig(fig) -> str:
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
    """
    Create a chart from the loaded dataset and save it as a PNG file.

    Args:
        chart_type: One of: bar, line, scatter, histogram, box, heatmap.
        x_column: Column for the x-axis (or the single column for histogram/heatmap).
        y_column: Column for the y-axis (not needed for histogram or heatmap).
        title: Chart title. Defaults to a generated title if empty.
        hue_column: Optional column to use for colour grouping (bar, scatter, box).

    Returns:
        File path of the saved PNG chart.
    """
    df = get_active_df()
    chart_type = chart_type.lower().strip()
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_title = title or f"{chart_type.capitalize()} — {x_column}" + (f" vs {y_column}" if y_column else "")

    if chart_type == "bar":
        data = df.groupby(x_column)[y_column].mean().reset_index() if y_column else df[x_column].value_counts().reset_index()
        data.columns = [x_column, y_column or "count"]
        sns.barplot(data=data, x=x_column, y=y_column or "count", hue=hue_column or None, ax=ax)
        ax.tick_params(axis="x", rotation=45)

    elif chart_type == "line":
        if not y_column:
            return "y_column is required for a line chart."
        plot_df = df[[x_column, y_column]].dropna()
        if hue_column:
            for name, grp in plot_df.groupby(df[hue_column]):
                ax.plot(grp[x_column], grp[y_column], label=name)
            ax.legend()
        else:
            ax.plot(plot_df[x_column], plot_df[y_column])
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
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty:
            return "No numeric columns available for a heatmap."
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

    else:
        return f"Unsupported chart_type '{chart_type}'. Choose from: bar, line, scatter, histogram, box, heatmap."

    ax.set_title(plot_title, fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    path = _save_fig(fig)
    return f"Chart saved to: {path}"
