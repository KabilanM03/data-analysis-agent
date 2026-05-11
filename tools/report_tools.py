"""Markdown report generator."""

from datetime import datetime

from smolagents import tool

from ._state import get_active_df, get_active_name


@tool
def generate_report(key_findings: str, recommendations: str = "") -> str:
    """Build a short markdown report from the active dataset and supplied findings.

    Args:
        key_findings: bullet points or prose describing the analysis findings.
        recommendations: optional follow-up recommendations.

    Returns:
        Markdown report.
    """
    df = get_active_df()
    nums = df.select_dtypes(include="number").columns.tolist()
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    quality = "no missing values" if nulls.empty else "\n".join(f"- {c}: {n}" for c, n in nulls.items())

    parts = [
        f"# Analysis report ({get_active_name() or 'dataset'})",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
        "",
        "## Overview",
        f"- Rows: {df.shape[0]:,}",
        f"- Columns: {df.shape[1]} ({len(nums)} numeric, {len(cats)} text)",
        "",
        "## Data quality",
        quality,
        "",
        "## Findings",
        key_findings.strip() or "_none provided_",
    ]
    if recommendations.strip():
        parts += ["", "## Recommendations", recommendations.strip()]
    return "\n".join(parts)
