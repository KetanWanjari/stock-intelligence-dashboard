"""
main.py
-------
Stock Data Intelligence Dashboard — FastAPI Backend
Author: JarNox Internship Assignment
"""

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from data_generator import build_database, COMPANIES


# ── DB helper ─────────────────────────────────────────────────────────────────

DB_PATH = "stocks.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ── Lifespan (startup) ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(DB_PATH):
        print("📦 Building database from mock data …")
        build_database(DB_PATH)
    else:
        print("✅ Database already exists — skipping generation.")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stock Data Intelligence Dashboard",
    description=(
        "A mini financial data platform exposing NSE stock data via REST APIs. "
        "Built for the JarNox Software Engineering Internship assignment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — /companies
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/companies",
    summary="List all available companies",
    tags=["Core"],
)
def get_companies():
    """Returns a list of all available companies with their metadata."""
    df = query_df("SELECT symbol, name, sector, base_price FROM companies ORDER BY symbol")
    return {"count": len(df), "companies": df.to_dict(orient="records")}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — /data/{symbol}
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/data/{symbol}",
    summary="Last N days of stock data",
    tags=["Core"],
)
def get_stock_data(symbol: str, days: int = Query(30, ge=1, le=400)):
    """
    Returns the last `days` trading days of OHLCV + derived metrics
    for the given stock symbol. Defaults to 30 days.
    """
    symbol = symbol.upper()
    _assert_symbol(symbol)

    df = query_df(
        """
        SELECT date, open, high, low, close, volume,
               daily_return, ma_7, ma_30, volatility_score, rsi_14, momentum_20d
        FROM stock_data
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT ?
        """,
        (symbol, days),
    )
    df = df.sort_values("date").reset_index(drop=True)
    return {
        "symbol": symbol,
        "days":   len(df),
        "data":   df.to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — /summary/{symbol}
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/summary/{symbol}",
    summary="52-week summary statistics",
    tags=["Core"],
)
def get_summary(symbol: str):
    """
    Returns 52-week high, low, average close, total trading days,
    best/worst single-day return, and average volume for the symbol.
    """
    symbol = symbol.upper()
    _assert_symbol(symbol)

    df = query_df(
        """
        SELECT date, open, high, low, close, volume, daily_return,
               volatility_score, rsi_14
        FROM stock_data
        WHERE symbol = ?
          AND date >= date('now', '-365 days')
        ORDER BY date
        """,
        (symbol,),
    )

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for this symbol.")

    latest_rsi = float(df["rsi_14"].iloc[-1]) if not df["rsi_14"].isnull().all() else None

    return {
        "symbol":            symbol,
        "name":              COMPANIES[symbol]["name"],
        "sector":            COMPANIES[symbol]["sector"],
        "trading_days":      len(df),
        "52w_high":          round(float(df["high"].max()), 2),
        "52w_low":           round(float(df["low"].min()), 2),
        "avg_close":         round(float(df["close"].mean()), 2),
        "latest_close":      round(float(df["close"].iloc[-1]), 2),
        "best_day_return":   round(float(df["daily_return"].max()), 4),
        "worst_day_return":  round(float(df["daily_return"].min()), 4),
        "avg_daily_return":  round(float(df["daily_return"].mean()), 4),
        "avg_volume":        int(df["volume"].mean()),
        "avg_volatility":    round(float(df["volatility_score"].mean()), 4),
        "latest_rsi":        round(latest_rsi, 2) if latest_rsi else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — /compare (BONUS)
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/compare",
    summary="Compare two stocks' performance",
    tags=["Bonus"],
)
def compare_stocks(
    symbol1: str = Query(..., description="First stock symbol, e.g. INFY"),
    symbol2: str = Query(..., description="Second stock symbol, e.g. TCS"),
    days: int    = Query(90, ge=7, le=400),
):
    """
    Compares two stocks over the last `days` days.
    Returns normalized prices (base 100), return correlation, and key stats.
    """
    s1, s2 = symbol1.upper(), symbol2.upper()
    _assert_symbol(s1)
    _assert_symbol(s2)

    def fetch(sym):
        return query_df(
            """
            SELECT date, close, daily_return, volatility_score
            FROM stock_data WHERE symbol = ?
            ORDER BY date DESC LIMIT ?
            """,
            (sym, days),
        ).sort_values("date").reset_index(drop=True)

    df1, df2 = fetch(s1), fetch(s2)

    # Align on common dates
    merged = pd.merge(df1, df2, on="date", suffixes=(f"_{s1}", f"_{s2}"))

    # Normalized price (base = 100 at start)
    merged[f"norm_{s1}"] = (merged[f"close_{s1}"] / merged[f"close_{s1}"].iloc[0] * 100).round(4)
    merged[f"norm_{s2}"] = (merged[f"close_{s2}"] / merged[f"close_{s2}"].iloc[0] * 100).round(4)

    corr = float(merged[f"daily_return_{s1}"].corr(merged[f"daily_return_{s2}"]))

    def perf(df, sym):
        start, end = df["close"].iloc[0], df["close"].iloc[-1]
        return {
            "symbol":          sym,
            "name":            COMPANIES[sym]["name"],
            "sector":          COMPANIES[sym]["sector"],
            "start_price":     round(start, 2),
            "end_price":       round(end, 2),
            "total_return_pct":round((end - start) / start * 100, 4),
            "avg_volatility":  round(float(df["volatility_score"].mean()), 4),
        }

    chart_data = merged[["date", f"norm_{s1}", f"norm_{s2}"]].to_dict(orient="records")

    return {
        "days":              len(merged),
        "correlation":       round(corr, 4),
        "stock1":            perf(df1.tail(len(merged)), s1),
        "stock2":            perf(df2.tail(len(merged)), s2),
        "chart_data":        chart_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5 — /gainers-losers (BONUS)
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/gainers-losers",
    summary="Top gainers and losers for the latest trading day",
    tags=["Bonus"],
)
def top_gainers_losers():
    """Returns the top-5 gainers and losers based on yesterday's daily return."""
    df = query_df(
        """
        SELECT s.symbol, c.name, c.sector, s.close, s.daily_return, s.volume
        FROM stock_data s
        JOIN companies c USING(symbol)
        WHERE s.date = (SELECT MAX(date) FROM stock_data)
        ORDER BY s.daily_return DESC
        """
    )
    return {
        "date":    query_df("SELECT MAX(date) as d FROM stock_data").iloc[0]["d"],
        "gainers": df.head(5).to_dict(orient="records"),
        "losers":  df.tail(5).iloc[::-1].to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 6 — /predict/{symbol} (BONUS — ML)
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/predict/{symbol}",
    summary="Simple ML-based 7-day price prediction",
    tags=["Bonus — ML"],
)
def predict_price(symbol: str, horizon: int = Query(7, ge=1, le=30)):
    """
    Uses a LinearRegression model trained on historical features to predict
    closing prices for the next `horizon` business days.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    symbol = symbol.upper()
    _assert_symbol(symbol)

    df = query_df(
        """
        SELECT date, close, daily_return, ma_7, ma_30,
               volatility_score, rsi_14, momentum_20d
        FROM stock_data WHERE symbol = ?
        ORDER BY date
        """,
        (symbol,),
    )

    df = df.dropna()
    if len(df) < 30:
        raise HTTPException(status_code=400, detail="Not enough data for prediction.")

    # Features: lag-based
    for lag in [1, 2, 3, 5]:
        df[f"close_lag{lag}"] = df["close"].shift(lag)
    df["target"] = df["close"].shift(-1)     # predict next-day close
    df = df.dropna()

    feature_cols = ["close", "ma_7", "ma_30", "volatility_score",
                    "rsi_14", "momentum_20d",
                    "close_lag1", "close_lag2", "close_lag3", "close_lag5"]

    X = df[feature_cols].values
    y = df["target"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model  = LinearRegression()
    model.fit(X_sc, y)

    # Predict iteratively for horizon days
    predictions = []
    last_row    = df[feature_cols].iloc[-1].values.copy()

    from pandas.tseries.offsets import BDay
    last_date = pd.to_datetime(df["date"].iloc[-1])

    for i in range(horizon):
        pred_price = float(model.predict(scaler.transform(last_row.reshape(1, -1)))[0])
        next_date  = last_date + BDay(i + 1)
        predictions.append({
            "date":            next_date.strftime("%Y-%m-%d"),
            "predicted_close": round(pred_price, 2),
        })
        # Slide lag features
        last_row[-1] = last_row[-2]     # lag5 ← lag3
        last_row[-2] = last_row[-3]     # lag3 ← lag2
        last_row[-3] = last_row[-4]     # lag2 ← lag1
        last_row[-4] = last_row[0]      # lag1 ← close
        last_row[0]  = pred_price       # close ← prediction

    actual_last_close = round(float(df["close"].iloc[-1]), 2)

    return {
        "symbol":           symbol,
        "last_known_close": actual_last_close,
        "model":            "LinearRegression (lag features + technical indicators)",
        "horizon_days":     horizon,
        "predictions":      predictions,
        "disclaimer":       "Predictions are generated by a simple ML model for demo purposes only.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 7 — / (Serve Dashboard HTML)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_dashboard():
    html_path = os.path.join("static", "index.html")
    if os.path.isfile(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard not found. Run the app with the static/ folder present.</h1>"


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _assert_symbol(symbol: str):
    if symbol not in COMPANIES:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found. Available: {list(COMPANIES.keys())}",
        )
