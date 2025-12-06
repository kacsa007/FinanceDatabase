"""Streamlit app to explore the FinanceDatabase and run quick simulations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import financedatabase as fd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import requests


st.set_page_config(
    page_title="FinanceDatabase Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass(frozen=True)
class AssetConfig:
    loader: Callable[[], Any]
    filters: Dict[str, str]
    supports_primary_listing: bool


ASSET_CONFIG: Dict[str, AssetConfig] = {
    "Equities": AssetConfig(
        loader=fd.Equities,
        filters={
            "country": "Country",
            "sector": "Sector",
            "industry_group": "Industry Group",
            "industry": "Industry",
            "currency": "Currency",
            "exchange": "Exchange",
            "market": "Market",
            "market_cap": "Market Cap Bucket",
        },
        supports_primary_listing=True,
    ),
    "ETFs": AssetConfig(
        loader=fd.ETFs,
        filters={
            "category_group": "Category Group",
            "category": "Category",
            "family": "Family",
            "currency": "Currency",
            "exchange": "Exchange",
        },
        supports_primary_listing=True,
    ),
    "Funds": AssetConfig(
        loader=fd.Funds,
        filters={
            "category_group": "Category Group",
            "category": "Category",
            "family": "Family",
            "currency": "Currency",
            "exchange": "Exchange",
        },
        supports_primary_listing=True,
    ),
    "Cryptos": AssetConfig(
        loader=fd.Cryptos,
        filters={"cryptocurrency": "Cryptocurrency", "currency": "Reference Currency"},
        supports_primary_listing=False,
    ),
    "Indices": AssetConfig(
        loader=fd.Indices,
        filters={"category": "Category", "exchange": "Exchange"},
        supports_primary_listing=False,
    ),
    "Currencies": AssetConfig(
        loader=fd.Currencies,
        filters={"category": "Category"},
        supports_primary_listing=False,
    ),
}

MAX_ROWS = 2000
DEFAULT_ROWS = 400
SESSION_RESULTS_KEY = "fd_results_cache"
SESSION_PARAMS_KEY = "fd_results_params"
LOG_PATH = Path("simulation_logs.csv")
JSON_LOG_PATH = Path("simulation_logs.jsonl")


@st.cache_resource
def load_asset_class(asset_type: str) -> Any:
    return ASSET_CONFIG[asset_type].loader()


@st.cache_data(show_spinner=False)
def get_filter_options(asset_type: str) -> Dict[str, List[str]]:
    options = load_asset_class(asset_type).show_options()
    cleaned = {}
    for key, values in options.items():
        series = pd.Series(values)
        series = series.dropna().astype(str).drop_duplicates()
        cleaned[key] = sorted(series.tolist())
    return cleaned


@st.cache_data(show_spinner=False)
def run_query(
    asset_type: str,
    filters: Dict[str, List[str]],
    only_primary: bool,
    row_cap: int,
    text_query: str,
) -> Tuple[pd.DataFrame, int]:
    loader = load_asset_class(asset_type)
    select_kwargs = {key: value for key, value in filters.items() if value}
    if only_primary and ASSET_CONFIG[asset_type].supports_primary_listing:
        select_kwargs["only_primary_listing"] = True
    data = loader.select(**select_kwargs).reset_index()

    if text_query:
        query_lower = text_query.lower()
        mask = (
            data["symbol"].astype(str).str.contains(query_lower, case=False, na=False)
            | data["name"].astype(str).str.contains(query_lower, case=False, na=False)
        )
        data = data[mask]

    total_rows = len(data)
    data = data.head(row_cap)
    return data, total_rows


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol: str, lookback_years: int) -> pd.Series:
    period = f"{lookback_years}y"
    data = yf.download(
        symbol,
        period=period,
        progress=False,
        group_by="column",
        auto_adjust=False,  # keep Adj Close column
    )

    if data.empty:
        raise ValueError("No price data returned from Yahoo Finance.")

    # yfinance often returns a 2-level MultiIndex even for single tickers.
    if isinstance(data.columns, pd.MultiIndex):
        tickers = data.columns.get_level_values(-1).unique()
        # For single ticker, drop the ticker level; otherwise, try to select the requested symbol.
        if len(tickers) == 1:
            data.columns = data.columns.get_level_values(0)
        else:
            try:
                data = data.xs(symbol, level=-1, axis=1)
            except KeyError:
                data = data.droplevel(-1, axis=1)

    # Prefer Adj Close; fall back to Close or the first available price-like column.
    if isinstance(data, pd.Series):
        prices = data
    elif "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        prices = data.iloc[:, 0] if not data.empty else pd.Series(dtype=float)

    prices = prices.dropna()
    if len(prices) < 2:
        raise ValueError("Not enough price history to simulate.")

    return prices


def simulate_paths(prices: pd.Series, days: int, paths: int) -> Tuple[np.ndarray, float, float]:
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Not enough history to simulate paths.")

    drift = log_returns.mean() - 0.5 * (log_returns.std() ** 2)
    vol = log_returns.std()
    last_price = prices.iloc[-1]
    rng = np.random.default_rng()
    shocks = rng.normal(drift, vol, size=(days, paths))
    steps = shocks.cumsum(axis=0)
    price_paths = last_price * np.exp(steps)
    return price_paths, drift, vol


def simulate_paths_bootstrap(prices: pd.Series, days: int, paths: int) -> Tuple[np.ndarray, float, float]:
    """Historical bootstrap of log returns (resample with replacement)."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Not enough history to simulate paths.")

    last_price = prices.iloc[-1]
    vol = log_returns.std()
    drift = log_returns.mean()

    rng = np.random.default_rng()
    sampled = rng.choice(log_returns.values, size=(days, paths), replace=True)
    steps = sampled.cumsum(axis=0)
    price_paths = last_price * np.exp(steps)
    return price_paths, drift, vol


def next_move_signal(prices: pd.Series) -> Dict[str, float | str]:
    returns = prices.pct_change().dropna()
    if len(returns) < 30:
        return {"label": "Unknown", "prob_up": 0.5, "vol": np.nan, "momentum": np.nan}

    vol = returns.tail(30).std()
    momentum = returns.tail(10).mean()
    vol = max(vol, 1e-6)
    score = (momentum / vol) * 3.0
    prob_up = 1 / (1 + np.exp(-score))

    if prob_up > 0.6:
        label = "Bullish tilt"
    elif prob_up < 0.4:
        label = "Bearish tilt"
    else:
        label = "Neutral"

    return {"label": label, "prob_up": prob_up, "vol": vol, "momentum": momentum}


def append_log(entry: Dict[str, Any]) -> None:
    """Append a simulation run to a local CSV log."""
    df = pd.DataFrame([entry])
    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, mode="w", header=True, index=False)


def append_json_log(entry: Dict[str, Any]) -> None:
    """Append a simulation run to a JSONL log."""
    with JSON_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(pd.Series(entry).to_json())
        f.write("\n")


def _business_days(start_date: datetime.date, periods: int) -> List[datetime.date]:
    """Return the next `periods` business days (Mon-Fri)."""
    days: List[datetime.date] = []
    cursor = start_date
    while len(days) < periods:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:  # Monday=0, Sunday=6
            days.append(cursor)
    return days


def fetch_news(symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch recent headlines via yfinance (public endpoint)."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
    except Exception:
        return []
    cleaned = []
    for item in news[:limit]:
        cleaned.append(
            {
                "title": item.get("title", "Untitled"),
                "link": item.get("link"),
                "publisher": item.get("publisher"),
                "providerPublishTime": item.get("providerPublishTime"),
            }
        )
    return cleaned


def fetch_calendar_flags(symbol: str) -> Dict[str, Any]:
    """Fetch simple upcoming calendar items (earnings/dividends) via yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
    except Exception:
        return {}
    if cal is None:
        return {}
    if hasattr(cal, "empty") and cal.empty:
        return {}
    if isinstance(cal, dict):
        return cal
    try:
        return {k: v for k, v in cal.to_dict().items()}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _sec_ticker_map() -> Dict[str, str]:
    """Map ticker -> CIK (padded string) from public SEC file."""
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "financedatabase-app/1.0 (contact: example@example.com)"}
    try:
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    mapping: Dict[str, str] = {}
    for entry in data.values():
        ticker = entry.get("ticker", "").upper()
        cik = entry.get("cik_str")
        if ticker and cik:
            mapping[ticker] = f"{int(cik):010d}"
    return mapping


@st.cache_data(show_spinner=False)
def fetch_sec_filings(symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch recent SEC filings from the public submissions endpoint."""
    mapping = _sec_ticker_map()
    cik = mapping.get(symbol.upper())
    if not cik:
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "financedatabase-app/1.0 (contact: example@example.com)"}
    try:
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    filings = data.get("filings", {}).get("recent", {})
    if not filings:
        return []

    keys = ["accessionNumber", "form", "filingDate", "periodOfReport", "primaryDocument"]
    max_len = min(
        limit,
        *[
            len(filings.get(key, []))
            for key in keys
            if isinstance(filings.get(key, []), list)
        ],
    ) or 0

    rows = []
    for idx in range(max_len):
        rows.append(
            {
                "accession": filings.get("accessionNumber", [None])[idx],
                "type": filings.get("form", [None])[idx],
                "filed": filings.get("filingDate", [None])[idx],
                "period": filings.get("periodOfReport", [None])[idx],
                "doc": filings.get("primaryDocument", [None])[idx],
            }
        )
    return rows


def render_simulation(
    symbols: List[str],
    lookback: int,
    horizon: int,
    paths: int,
    log_runs: bool,
    log_json: bool,
    use_bootstrap: bool,
) -> None:
    for symbol in symbols:
        st.subheader(f"{symbol} — Monte Carlo simulation")
        try:
            prices = fetch_history(symbol, lookback)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Could not download history for {symbol}: {exc}")
            continue

        if prices.empty:
            st.warning(f"No price data available for {symbol}.")
            continue

        try:
            if use_bootstrap:
                price_paths, drift, vol = simulate_paths_bootstrap(prices, horizon, paths)
            else:
                price_paths, drift, vol = simulate_paths(prices, horizon, paths)
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Simulation failed for {symbol}: {exc}")
            continue

        terminal_prices = price_paths[-1]
        p5, p50, p95 = np.percentile(terminal_prices, [5, 50, 95])
        days = np.arange(1, horizon + 1)

        # One-step forecast for next session (toy GBM-based).
        last_price = prices.iloc[-1]
        next_exp = last_price * np.exp(drift)
        next_std_up = last_price * np.exp(drift + vol)
        next_std_dn = last_price * np.exp(drift - vol)

        # Daily probability snapshots for the next few business sessions.
        days_to_report = [1, 2, 3, 5, 10]
        days_to_report = [d for d in days_to_report if d <= horizon]
        today = datetime.utcnow().date()
        biz_days = _business_days(today, max(days_to_report) if days_to_report else 0)
        snap_rows = []
        for d in days_to_report:
            d_idx = d - 1
            day_prices = price_paths[d_idx]
            target_date = biz_days[d_idx]
            snap_rows.append(
                {
                    "date": target_date.isoformat(),
                    "weekday": target_date.strftime("%A"),
                    "p5": float(np.percentile(day_prices, 5)),
                    "p50": float(np.percentile(day_prices, 50)),
                    "p95": float(np.percentile(day_prices, 95)),
                    "prob_above_last": float((day_prices > last_price).mean()),
                }
            )
        snap_df = pd.DataFrame(snap_rows)

        fig = go.Figure()
        display_paths = min(paths, 60)
        for idx in range(display_paths):
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=price_paths[:, idx],
                    mode="lines",
                    line=dict(width=1, color="rgba(33, 150, 243, 0.35)"),
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=days,
                y=np.median(price_paths, axis=1),
                mode="lines",
                line=dict(color="#1f77b4", width=3),
                name="Median path",
            )
        )
        fig.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis_title="Days",
            yaxis_title="Price",
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last price", f"{prices.iloc[-1]:.2f}")
        col2.metric("p5 terminal", f"{p5:.2f}")
        col3.metric("p50 terminal", f"{p50:.2f}")
        col4.metric("p95 terminal", f"{p95:.2f}")

        st.plotly_chart(fig, use_container_width=True)

        # Percentile bands over time.
        p5_path = np.percentile(price_paths, 5, axis=1)
        p50_path = np.percentile(price_paths, 50, axis=1)
        p95_path = np.percentile(price_paths, 95, axis=1)
        band_fig = go.Figure()
        band_fig.add_trace(
            go.Scatter(x=days, y=p50_path, mode="lines", name="p50", line=dict(color="#1f77b4", width=3))
        )
        band_fig.add_trace(
            go.Scatter(
                x=days,
                y=p95_path,
                mode="lines",
                name="p95",
                line=dict(color="rgba(33,150,243,0.6)", dash="dash"),
            )
        )
        band_fig.add_trace(
            go.Scatter(
                x=days,
                y=p5_path,
                mode="lines",
                name="p5",
                line=dict(color="rgba(33,150,243,0.6)", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(33,150,243,0.15)",
            )
        )
        band_fig.update_layout(
            title="Percentile bands over horizon",
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_title="Days",
            yaxis_title="Price",
        )
        st.plotly_chart(band_fig, use_container_width=True)

        # Terminal distribution histogram for quick intuition.
        st.plotly_chart(
            go.Figure(
                data=[
                    go.Histogram(
                        x=terminal_prices,
                        nbinsx=40,
                        marker=dict(color="rgba(33,150,243,0.5)"),
                        name="Terminal prices",
                    )
                ],
                layout=dict(
                    title="Terminal price distribution",
                    margin=dict(l=10, r=10, t=30, b=30),
                    xaxis_title="Price at horizon",
                    yaxis_title="Count",
                ),
            ),
            use_container_width=True,
        )

        signal = next_move_signal(prices)
        st.caption(
            f"Toy signal: {signal['label']} | "
            f"Prob up: {signal['prob_up']*100:.1f}% | "
            f"10d momentum: {signal['momentum']*100:.2f}% | "
            f"30d vol: {signal['vol']*100:.2f}%"
        )

        # Quick read on the next business day.
        if not snap_df.empty:
            next_row = snap_df.iloc[0]
            st.markdown(
                f"**Next business day view** ({next_row['weekday']} {next_row['date']}): "
                f"median ~ {next_row['p50']:.2f}, "
                f"range p5–p95: {next_row['p5']:.2f} – {next_row['p95']:.2f}, "
                f"prob > last close ({last_price:.2f}): {next_row['prob_above_last']*100:.1f}%."
            )

        with st.expander("Diagnostics and forecast"):
            st.markdown(
                "- **p5 / p50 / p95 terminal**: 5th, 50th, and 95th percentiles of the simulated price after the horizon. "
                "Interpret as a rough downside/median/upside range under the GBM assumption.\n"
                "- **Drift / vol**: Daily average return and daily volatility estimated from log returns over the lookback window.\n"
                "- **Next session range**: Expected price next session and a one-sigma band based on drift and vol.\n"
                "- **Toy signal**: Momentum vs. recent volatility; it does not incorporate the simulation sliders.\n"
                "- **Snapshots table**: For each upcoming business day, shows p5/p50/p95 and the probability of finishing above the last close. "
                "Example: p50 is the median simulated close; p5/p95 give a downside/upside band. Prob>last is the share of paths above the current close.\n"
                "- **Engines**: GBM assumes returns are normally distributed; Historical bootstrap resamples past log returns to preserve fat tails/vol clustering.\n"
                "- **VaR/ES (95%)**: Value at Risk and Expected Shortfall for terminal returns; negative numbers mean loss vs. last close."
            )
            terminal_returns = terminal_prices / last_price - 1
            var_95 = np.percentile(terminal_returns, 5)
            es_95 = (
                terminal_returns[terminal_returns <= var_95].mean()
                if (terminal_returns <= var_95).any()
                else var_95
            )
            st.write(
                {
                    "lookback_years": lookback,
                    "horizon_days": horizon,
                    "paths": paths,
                    "drift_daily": float(drift),
                    "vol_daily": float(vol),
                    "last_price": float(last_price),
                    "next_session_exp": float(next_exp),
                    "next_session_range": [float(next_std_dn), float(next_std_up)],
                    "terminal_p5_p50_p95": [float(p5), float(p50), float(p95)],
                    "series_len": len(prices),
                    "engine": "historical bootstrap" if use_bootstrap else "GBM",
                    "VaR_95_pct_return": float(var_95),
                    "ES_95_pct_return": float(es_95),
                }
            )
            if not snap_df.empty:
                st.write("Daily probability snapshots (business days, relative to last close):")
                st.dataframe(
                    snap_df.assign(prob_above_last=lambda df: (df["prob_above_last"] * 100).round(1)),
                    use_container_width=True,
                    hide_index=True,
                )

        log_entry = {
            "symbol": symbol,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "lookback_years": lookback,
            "horizon_days": horizon,
            "paths": paths,
            "drift_daily": float(drift),
            "vol_daily": float(vol),
            "last_price": float(last_price),
            "next_session_exp": float(next_exp),
            "next_session_low": float(next_std_dn),
            "next_session_high": float(next_std_up),
            "terminal_p5": float(p5),
            "terminal_p50": float(p50),
            "terminal_p95": float(p95),
            "prob_up": float(signal["prob_up"]),
            "momentum_10d": float(signal["momentum"]),
            "vol_30d": float(signal["vol"]),
            "engine": "historical_bootstrap" if use_bootstrap else "gbm",
            "VaR_95_pct_return": float(var_95),
            "ES_95_pct_return": float(es_95),
            "snapshots": snap_rows,
        }
        if log_runs:
            append_log(log_entry)
            st.info("Simulation logged to simulation_logs.csv")
        if log_json:
            append_json_log(log_entry)
            st.info("Simulation logged to simulation_logs.jsonl")


def _normalize_filters(filters: Dict[str, List[str]]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Return a hashable, sorted representation of the selected filters."""
    return tuple(sorted((key, tuple(values)) for key, values in filters.items()))


def main() -> None:
    st.title("FinanceDatabase Lab")
    st.markdown(
        "Search the FinanceDatabase locally, slice by filters, and run quick Monte Carlo "
        "experiments on selected symbols (equities, crypto, ETFs, funds, and more)."
    )

    with st.sidebar:
        st.header("Filters")
        asset_type = st.selectbox("Asset class", list(ASSET_CONFIG.keys()))
        options = get_filter_options(asset_type)

        selected_filters: Dict[str, List[str]] = {}
        for key, label in ASSET_CONFIG[asset_type].filters.items():
            selected_filters[key] = st.multiselect(label, options.get(key, []))

        only_primary = False
        if ASSET_CONFIG[asset_type].supports_primary_listing:
            only_primary = st.checkbox("Only primary listings", value=True)

        row_cap = st.slider("Max rows to display", 100, MAX_ROWS, DEFAULT_ROWS, step=100)
        text_query = st.text_input("Search symbol or name", "")
        run_button = st.button("Run query", type="primary")

    query_params = (
        asset_type,
        _normalize_filters(selected_filters),
        only_primary,
        row_cap,
        text_query.strip(),
    )

    if run_button:
        with st.spinner("Querying database..."):
            results, total_rows = run_query(
                asset_type=asset_type,
                filters=selected_filters,
                only_primary=only_primary,
                row_cap=row_cap,
                text_query=text_query.strip(),
            )
        st.session_state[SESSION_RESULTS_KEY] = (results, total_rows)
        st.session_state[SESSION_PARAMS_KEY] = query_params
    else:
        cached = st.session_state.get(SESSION_RESULTS_KEY)
        cached_params = st.session_state.get(SESSION_PARAMS_KEY)
        if cached is None:
            st.info("Choose filters and run a query from the sidebar.")
            return
        if cached_params != query_params:
            st.info("Filters changed — hit Run query to refresh the results.")
            return
        results, total_rows = cached

    st.success(f"Showing {len(results):,} of {total_rows:,} rows for {asset_type}.")
    st.dataframe(results, use_container_width=True, hide_index=True)

    if not results.empty:
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{asset_type.lower()}_results.csv",
            mime="text/csv",
        )

        selectable_symbols = results["symbol"].head(50).tolist()
        st.divider()
        st.subheader("Simulations and toy forecasts")
        st.markdown(
            "Pick a few symbols (e.g., crypto tickers, small caps) to simulate future price paths using "
            "geometric Brownian motion. Data comes from Yahoo Finance via `yfinance`."
        )

        symbols = st.multiselect(
            "Symbols to simulate",
            options=selectable_symbols,
            default=selectable_symbols[:3],
            help="Limited to the first 50 results for performance.",
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            lookback = st.slider("Lookback (years)", 1, 5, 2)
        with col_b:
            horizon = st.slider("Forecast horizon (days)", 5, 90, 21)
        with col_c:
            paths = st.slider("Simulation paths", 50, 1000, 300, step=50)
        log_runs = st.checkbox("Log simulations to CSV (simulation_logs.csv)", value=False)
        log_json = st.checkbox("Log simulations to JSONL (simulation_logs.jsonl)", value=False)
        use_bootstrap = st.checkbox("Use historical bootstrap (resample past returns)", value=False)

        if symbols:
            render_simulation(symbols, lookback, horizon, paths, log_runs, log_json, use_bootstrap)
        else:
            st.info("Select at least one symbol to simulate.")

    st.divider()
    st.subheader("News and calendar (public data)")
    if not results.empty:
        symbol_for_news = st.selectbox("Pick a symbol for headlines/calendar", results["symbol"].head(50).tolist())
        headlines = fetch_news(symbol_for_news, limit=5)
        cal = fetch_calendar_flags(symbol_for_news)
        sec_filings = fetch_sec_filings(symbol_for_news, limit=5)

        if headlines:
            st.write("Recent headlines (yfinance news):")
            for item in headlines:
                ts = item.get("providerPublishTime")
                ts_fmt = datetime.utcfromtimestamp(ts).isoformat() if ts else "n/a"
                st.write(f"- {ts_fmt} — {item['publisher']}: [{item['title']}]({item['link']})")
        else:
            st.info("No headlines available from the public yfinance feed (Yahoo may throttle or have none for this ticker).")

        if cal:
            st.write("Upcoming events (yfinance calendar):")
            st.json(cal)
        else:
            st.info("No upcoming events found via yfinance calendar.")

        if sec_filings:
            st.write("Recent SEC filings (public EDGAR):")
            st.dataframe(pd.DataFrame(sec_filings), use_container_width=True, hide_index=True)
        else:
            st.info("No SEC filings found via public EDGAR for this ticker.")

    with st.expander("Data sources & accuracy notes"):
        st.markdown(
            "- **Prices/snapshots**: yfinance (Yahoo Finance). Data can be delayed and is not guaranteed accurate; use an official feed for trading.\n"
            "- **News**: yfinance public news endpoint; some tickers have sparse/empty feeds or rate limits.\n"
            "- **Calendar**: yfinance `Ticker.calendar`; availability varies by symbol.\n"
            "- **SEC filings**: Public EDGAR JSON (no keys); if a CIK is missing or EDGAR is slow, the table can be empty.\n"
            "- **Models**: GBM and historical bootstrap are exploratory only; validate against your own data/tests before using for decisions."
        )


if __name__ == "__main__":
    main()
