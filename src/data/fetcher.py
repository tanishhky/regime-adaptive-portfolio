"""
Data acquisition module.

Downloads daily Close prices for sector ETFs, SPY, and VIX via yfinance,
and Fama-French 5-factor daily data from Ken French's website.
Computes log returns: r_t = ln(P_t / P_{t-1}).

References
----------
Fama, E.F. & French, K.R. (2015). "A five-factor model." Journal of
Financial Economics, 116(1), 1-22.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import config


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_etf_prices(
    tickers: list[str] | None = None,
    start: str = config.DATA_START_DATE,
    end: str = config.DATA_END_DATE,
    save: bool = True,
) -> pd.DataFrame:
    """Download daily Close prices for sector ETFs, benchmark, and VIX.

    Parameters
    ----------
    tickers : list[str] | None
        Ticker symbols.  Defaults to all sector ETFs + SPY + VIX.
    start, end : str
        Date range in ``'YYYY-MM-DD'`` format.
    save : bool
        If True, persist to ``data/raw/etf_prices.csv``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex × tickers, daily Close prices.
    """
    if tickers is None:
        tickers = list(config.SECTOR_ETFS.keys()) + [
            config.BENCHMARK,
            config.VIX_TICKER,
        ]

    # yfinance download — use 'Close', NEVER 'Adj Close'
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False)

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_csv(DATA_DIR / "etf_prices.csv")

    return prices


def fetch_fama_french(save: bool = True) -> pd.DataFrame:
    """Download Fama-French 5-factor daily data.

    Source: Ken French Data Library.

    Parameters
    ----------
    save : bool
        Persist to ``data/raw/ff5_daily.csv``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex with columns Mkt-RF, SMB, HML, RMW, CMA, RF
        (all in decimal, not percent).
    """
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as f:
            raw_text = f.read().decode("utf-8")

    # The CSV has a header preamble — find the line that starts with the data
    lines = raw_text.strip().split("\n")
    header_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Mkt") or "Mkt-RF" in line:
            header_idx = i
            break

    df = pd.read_csv(
        io.StringIO("\n".join(lines[header_idx:])),
        index_col=0,
    )
    # Remove any trailing text rows
    df.index = df.index.astype(str).str.strip()
    df = df[df.index.str.match(r"^\d{8}$")]
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "Date"

    # Convert from percentage to decimal
    df = df.astype(float) / 100.0

    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_DIR / "ff5_daily.csv")

    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns: r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.DataFrame
        DatetimeIndex × tickers with daily Close prices.

    Returns
    -------
    pd.DataFrame
        Log returns (first row is NaN, dropped).
    """
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.dropna(how="all")
