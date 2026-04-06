"""
Real-time data fetching tools.
Sources: Yahoo Finance (live market data) + Hugging Face Datasets Hub + Kaggle.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf
from datasets import load_dataset as hf_load_dataset
from smolagents import tool
from .data_tools import set_active_df


# ─── Yahoo Finance ─────────────────────────────────────────────────────────────

@tool
def fetch_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> str:
    """
    Fetch historical stock price data from Yahoo Finance and load it for analysis.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'MSFT', 'GOOGL'.
        period: Time period to fetch. Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        interval: Data interval. Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.

    Returns:
        Summary of the loaded stock data including date range, price range, and volume stats.
    """
    t = yf.Ticker(ticker.upper())
    df = t.history(period=period, interval=interval)

    if df.empty:
        return f"No data found for ticker '{ticker}'. Check the symbol and try again."

    df = df.reset_index()
    df.columns = [c.replace(" ", "_") for c in df.columns]

    # Flatten timezone-aware datetime to plain string for easy analysis
    if "Date" in df.columns:
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    elif "Datetime" in df.columns:
        df["Datetime"] = df["Datetime"].dt.strftime("%Y-%m-%d %H:%M")

    # Add useful derived columns
    df["Daily_Return_%"] = df["Close"].pct_change().mul(100).round(4)
    df["Cumulative_Return_%"] = ((df["Close"] / df["Close"].iloc[0]) - 1).mul(100).round(4)

    set_active_df(df, name=f"{ticker.upper()}_stock")

    return (
        f"Loaded {ticker.upper()} stock data: {len(df)} rows ({period} period, {interval} interval).\n"
        f"Date range: {df.iloc[0, 0]} → {df.iloc[-1, 0]}\n"
        f"Price range: ${df['Low'].min():.2f} – ${df['High'].max():.2f}\n"
        f"Latest close: ${df['Close'].iloc[-1]:.2f}\n"
        f"Columns: {', '.join(df.columns.tolist())}\n"
        f"Cumulative return over period: {df['Cumulative_Return_%'].iloc[-1]:.2f}%"
    )


@tool
def fetch_company_info(ticker: str) -> str:
    """
    Fetch company fundamentals and key metrics from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'NVDA'.

    Returns:
        Key company information: name, sector, market cap, P/E ratio, revenue, employees, and more.
    """
    t = yf.Ticker(ticker.upper())
    info = t.info

    if not info or "longName" not in info:
        return f"Could not retrieve info for '{ticker}'. Check the ticker symbol."

    def fmt(val, prefix="", suffix="", billions=False):
        if val is None:
            return "N/A"
        if billions and isinstance(val, (int, float)):
            return f"{prefix}{val / 1e9:.2f}B{suffix}"
        return f"{prefix}{val}{suffix}"

    return (
        f"=== {info.get('longName', ticker.upper())} ({ticker.upper()}) ===\n"
        f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}\n"
        f"Country: {info.get('country', 'N/A')} | Employees: {fmt(info.get('fullTimeEmployees'))}\n\n"
        f"Market Cap:     {fmt(info.get('marketCap'), '$', '', billions=True)}\n"
        f"Revenue (TTM):  {fmt(info.get('totalRevenue'), '$', '', billions=True)}\n"
        f"Net Income:     {fmt(info.get('netIncomeToCommon'), '$', '', billions=True)}\n"
        f"P/E Ratio:      {fmt(info.get('trailingPE'))}\n"
        f"EPS (TTM):      {fmt(info.get('trailingEps'), '$')}\n"
        f"Dividend Yield: {fmt(info.get('dividendYield'), suffix='%')}\n"
        f"52-Week High:   {fmt(info.get('fiftyTwoWeekHigh'), '$')}\n"
        f"52-Week Low:    {fmt(info.get('fiftyTwoWeekLow'), '$')}\n"
        f"Analyst Target: {fmt(info.get('targetMeanPrice'), '$')}\n\n"
        f"Summary: {info.get('longBusinessSummary', 'N/A')[:400]}..."
    )


@tool
def compare_stocks(tickers: str, period: str = "6mo") -> str:
    """
    Compare closing prices and returns for multiple stocks side by side.
    Loads the comparison table as the active dataset for further analysis.

    Args:
        tickers: Comma-separated ticker symbols, e.g. 'AAPL,MSFT,GOOGL,NVDA'.
        period: Time period. Options: 1mo, 3mo, 6mo, 1y, 2y, 5y.

    Returns:
        Comparison summary with returns and volatility for each ticker.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    frames = {}

    for t in ticker_list:
        try:
            hist = yf.Ticker(t).history(period=period)["Close"]
            if not hist.empty:
                frames[t] = hist
        except Exception:
            pass

    if not frames:
        return "Could not fetch data for any of the provided tickers."

    df = pd.DataFrame(frames)
    df.index = df.index.strftime("%Y-%m-%d")
    df = df.reset_index().rename(columns={"index": "Date"})
    set_active_df(df, name="stock_comparison")

    lines = [f"Loaded comparison for: {', '.join(frames.keys())} ({period})\n"]
    for t in frames:
        prices = frames[t]
        ret = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
        vol = prices.pct_change().std() * 100
        lines.append(f"  {t}: return={ret:.2f}%  daily_volatility={vol:.2f}%  latest=${prices.iloc[-1]:.2f}")

    return "\n".join(lines)


# ─── Hugging Face Datasets Hub ──────────────────────────────────────────────────

# Curated list of well-known, high-quality datasets on HF Hub
KNOWN_DATASETS = {
    "spotify":      ("maharshipandya/spotify-tracks-dataset", "train"),
    "titanic":      ("mstz/titanic",                          "train"),
    "netflix":      ("hugginglearners/netflix-shows",         "train"),
    "sales":        ("Thewillonline/sales_data_sample",       "train"),
    "data jobs":    ("lukebarousse/data_jobs",                "train"),
    "airbnb":       ("gradio/NYC-Airbnb-Open-Data",           "train"),
    "covid":        ("nid989/EpidemicQA",                     "train"),
    "cars":         ("datasets/car_models",                   "train"),
    "ecommerce":    ("turing-motors/ecommerce-product-listing","train"),
}


@tool
def load_hf_dataset(dataset_name: str, split: str = "train", max_rows: int = 5000) -> str:
    """
    Load a real dataset from the Hugging Face Datasets Hub into memory for analysis.

    Use shorthand names for popular datasets: spotify, titanic, netflix, sales, data jobs, airbnb.
    Or provide a full HF dataset ID like 'maharshipandya/spotify-tracks-dataset'.

    Args:
        dataset_name: Shorthand name (e.g. 'spotify') or full HF dataset ID.
        split: Dataset split to load, usually 'train'.
        max_rows: Maximum number of rows to load (default 5000 to keep it fast).

    Returns:
        Summary of the loaded dataset including shape, columns, and a data preview.
    """
    # Resolve shorthand names
    key = dataset_name.lower().strip()
    if key in KNOWN_DATASETS:
        hf_id, default_split = KNOWN_DATASETS[key]
        split = default_split
    else:
        hf_id = dataset_name

    try:
        ds = hf_load_dataset(hf_id, split=f"{split}[:{max_rows}]")
        df = ds.to_pandas()
    except Exception as e:
        return (
            f"Could not load '{dataset_name}' from HF Hub. Error: {e}\n"
            f"Try one of the built-in datasets: {', '.join(KNOWN_DATASETS.keys())}"
        )

    set_active_df(df, name=dataset_name)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing = df.isnull().sum().sum()

    return (
        f"Loaded '{dataset_name}' from HF Hub: {df.shape[0]:,} rows x {df.shape[1]} columns.\n"
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:8])}{'...' if len(numeric_cols) > 8 else ''}\n"
        f"Text columns ({len(cat_cols)}): {', '.join(cat_cols[:8])}{'...' if len(cat_cols) > 8 else ''}\n"
        f"Missing values: {missing:,}\n\n"
        f"Preview (first 3 rows):\n{df.head(3).to_string()}"
    )


@tool
def list_available_datasets() -> str:
    """
    List all built-in datasets available to load from the Hugging Face Hub.

    Returns:
        A formatted list of dataset names and descriptions.
    """
    descriptions = {
        "spotify":   "114k Spotify tracks with audio features (danceability, energy, tempo, genre)",
        "titanic":   "Classic Titanic passenger survival dataset",
        "netflix":   "Netflix shows and movies catalogue with ratings and genres",
        "sales":     "B2B sales transaction records by product, region, and customer",
        "data jobs": "Data science job postings with salary, skills, and location",
        "airbnb":    "NYC Airbnb listings with price, location, and reviews",
    }
    lines = ["Available built-in datasets (use with load_hf_dataset):\n"]
    for name, desc in descriptions.items():
        lines.append(f"  • '{name}' — {desc}")
    lines.append("\nOr provide any full Hugging Face dataset ID, e.g. 'username/dataset-name'.")
    lines.append("\nFor Kaggle datasets, use fetch_kaggle_dataset with a dataset slug, e.g. 'tmdb/tmdb-movie-metadata'.")
    return "\n".join(lines)


# ─── Kaggle ────────────────────────────────────────────────────────────────────

@tool
def fetch_kaggle_dataset(dataset_slug: str, file_name: str = "", max_rows: int = 50000) -> str:
    """
    Download a dataset from Kaggle and load it for analysis.
    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables (or ~/.kaggle/kaggle.json).

    Args:
        dataset_slug: Kaggle dataset slug in format 'owner/dataset-name', e.g. 'tmdb/tmdb-movie-metadata'.
        file_name: Specific CSV file name to load if the dataset has multiple files. Leave empty to auto-detect.
        max_rows: Maximum number of rows to load (default 50000).

    Returns:
        Summary of loaded dataset including shape, columns, and preview.
    """
    import os
    import tempfile
    import glob

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiClient
        api = KaggleApiClient()
    except Exception:
        try:
            import kaggle
            api = kaggle.api
            api.authenticate()
        except Exception as e:
            return (
                f"Kaggle authentication failed: {e}\n"
                "Make sure KAGGLE_USERNAME and KAGGLE_KEY are set, or ~/.kaggle/kaggle.json exists.\n"
                "Get your API key from: https://www.kaggle.com/settings/account"
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            api.dataset_download_files(dataset_slug, path=tmpdir, unzip=True)
        except Exception as e:
            return f"Failed to download '{dataset_slug}' from Kaggle: {e}"

        csv_files = glob.glob(os.path.join(tmpdir, "**", "*.csv"), recursive=True)
        if not csv_files:
            return f"No CSV files found in dataset '{dataset_slug}'."

        if file_name:
            matched = [f for f in csv_files if os.path.basename(f) == file_name]
            target = matched[0] if matched else csv_files[0]
        else:
            target = csv_files[0]

        df = pd.read_csv(target, nrows=max_rows)

    set_active_df(df, name=os.path.basename(target))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    missing = df.isnull().sum().sum()

    return (
        f"Loaded '{dataset_slug}' from Kaggle ({os.path.basename(target)}): "
        f"{df.shape[0]:,} rows x {df.shape[1]} columns.\n"
        f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:8])}{'...' if len(numeric_cols) > 8 else ''}\n"
        f"Text columns ({len(cat_cols)}): {', '.join(cat_cols[:8])}{'...' if len(cat_cols) > 8 else ''}\n"
        f"Missing values: {missing:,}\n\n"
        f"Preview (first 3 rows):\n{df.head(3).to_string()}"
    )


@tool
def search_kaggle_datasets(query: str, max_results: int = 8) -> str:
    """
    Search for datasets on Kaggle by keyword.

    Args:
        query: Search term, e.g. 'movies', 'sales', 'covid', 'football'.
        max_results: Number of results to return (default 8).

    Returns:
        List of matching Kaggle datasets with their slugs and descriptions.
    """
    try:
        import kaggle
        kaggle.api.authenticate()
        results = kaggle.api.dataset_list(search=query, max_size=None, page=1)
        if not results:
            return f"No Kaggle datasets found for '{query}'."
        lines = [f"Kaggle datasets matching '{query}':\n"]
        for ds in results[:max_results]:
            slug = f"{ds.ref}"
            size = getattr(ds, 'size', 'unknown size')
            lines.append(f"  • {slug} — {ds.title} ({size})")
        lines.append("\nUse fetch_kaggle_dataset('<slug>') to load any of these.")
        return "\n".join(lines)
    except Exception as e:
        return f"Kaggle search failed: {e}\nMake sure KAGGLE_USERNAME and KAGGLE_KEY are set."
