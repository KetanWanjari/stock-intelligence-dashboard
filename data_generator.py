"""
data_generator.py
-----------------
Generates realistic mock stock data for Indian NSE stocks.
In a live environment, replace this with yfinance / Alpha Vantage API calls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os

# ── Seed companies ────────────────────────────────────────────────────────────
COMPANIES = {
    "INFY":      {"name": "Infosys Ltd",             "sector": "IT",            "base_price": 1450.0},
    "TCS":       {"name": "Tata Consultancy Services","sector": "IT",            "base_price": 3700.0},
    "RELIANCE":  {"name": "Reliance Industries",      "sector": "Energy",        "base_price": 2900.0},
    "HDFCBANK":  {"name": "HDFC Bank Ltd",            "sector": "Banking",       "base_price": 1620.0},
    "WIPRO":     {"name": "Wipro Ltd",                "sector": "IT",            "base_price": 520.0},
    "ICICIBANK": {"name": "ICICI Bank Ltd",           "sector": "Banking",       "base_price": 1100.0},
    "HINDUNILVR":{"name": "Hindustan Unilever",       "sector": "FMCG",          "base_price": 2450.0},
    "ITC":       {"name": "ITC Ltd",                  "sector": "FMCG",          "base_price": 430.0},
    "BAJFINANCE":{"name": "Bajaj Finance Ltd",        "sector": "Finance",       "base_price": 7200.0},
    "AXISBANK":  {"name": "Axis Bank Ltd",            "sector": "Banking",       "base_price": 1050.0},
}


def generate_ohlcv(base_price: float, days: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    Simulate OHLCV data using geometric Brownian motion with realistic
    intra-day spread and volume patterns.
    """
    rng = np.random.default_rng(seed)

    mu    = 0.0003        # daily drift
    sigma = 0.015         # daily volatility

    # Generate close prices via GBM
    log_returns = rng.normal(mu - 0.5 * sigma**2, sigma, days)
    close_prices = base_price * np.exp(np.cumsum(log_returns))

    opens  = close_prices * (1 + rng.normal(0, 0.005, days))
    highs  = np.maximum(opens, close_prices) * (1 + rng.uniform(0.001, 0.015, days))
    lows   = np.minimum(opens, close_prices) * (1 - rng.uniform(0.001, 0.015, days))
    volume = rng.integers(500_000, 5_000_000, days).astype(float)

    end_date   = datetime.today().date()
    date_range = pd.bdate_range(end=end_date, periods=days)   # business days only

    df = pd.DataFrame({
        "date":   date_range,
        "open":   opens.round(2),
        "high":   highs.round(2),
        "low":    lows.round(2),
        "close":  close_prices.round(2),
        "volume": volume.astype(int),
    })
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add all required and custom metrics to a stock DataFrame."""
    df = df.copy().sort_values("date").reset_index(drop=True)

    # ── Required metrics ──────────────────────────────────────────────────────
    df["daily_return"]   = ((df["close"] - df["open"]) / df["open"] * 100).round(4)
    df["ma_7"]           = df["close"].rolling(7, min_periods=1).mean().round(2)
    df["ma_30"]          = df["close"].rolling(30, min_periods=1).mean().round(2)

    one_year_ago         = df["date"].max() - pd.Timedelta(days=365)
    yearly               = df[df["date"] >= one_year_ago]
    df["52w_high"]       = yearly["high"].max()
    df["52w_low"]        = yearly["low"].min()

    # ── Custom metric 1 — Volatility Score (20-day rolling std of returns) ───
    df["volatility_score"] = df["daily_return"].rolling(20, min_periods=1).std().round(4)

    # ── Custom metric 2 — Relative Strength Index (14-day RSI) ───────────────
    delta          = df["close"].diff()
    gain           = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss           = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs             = gain / loss.replace(0, np.nan)
    df["rsi_14"]   = (100 - (100 / (1 + rs))).round(2)

    # ── Custom metric 3 — Momentum (close vs 20-day-ago close) ───────────────
    df["momentum_20d"] = ((df["close"] / df["close"].shift(20) - 1) * 100).round(4)

    return df


def build_database(db_path: str = "stocks.db") -> None:
    """Create SQLite database and populate with mock data."""
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # ── companies table ───────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            symbol      TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            sector      TEXT NOT NULL,
            base_price  REAL NOT NULL
        )
    """)

    # ── stock_data table ──────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol           TEXT NOT NULL,
            date             TEXT NOT NULL,
            open             REAL,
            high             REAL,
            low              REAL,
            close            REAL,
            volume           INTEGER,
            daily_return     REAL,
            ma_7             REAL,
            ma_30            REAL,
            "52w_high"       REAL,
            "52w_low"        REAL,
            volatility_score REAL,
            rsi_14           REAL,
            momentum_20d     REAL,
            UNIQUE(symbol, date)
        )
    """)
    conn.commit()

    for i, (symbol, meta) in enumerate(COMPANIES.items()):
        # Insert company
        cur.execute(
            "INSERT OR REPLACE INTO companies VALUES (?, ?, ?, ?)",
            (symbol, meta["name"], meta["sector"], meta["base_price"]),
        )

        # Generate & insert stock data
        raw_df = generate_ohlcv(meta["base_price"], days=400, seed=i * 37 + 7)
        df     = add_derived_metrics(raw_df)
        df.insert(0, "symbol", symbol)
        df["date"] = df["date"].astype(str)

        df.to_sql("stock_data", conn, if_exists="append", index=False,
                  method="multi")
        print(f"  ✓ {symbol}: {len(df)} rows inserted")

    conn.commit()
    conn.close()
    print(f"\n✅ Database built at: {db_path}")


if __name__ == "__main__":
    if os.path.exists("stocks.db"):
        os.remove("stocks.db")
    build_database()
