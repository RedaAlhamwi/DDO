"""
Local dashboard for the ai-trader data pipeline.

Lets you download data, generate features, add labels, and inspect the
result — all from a browser UI running on your own machine.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))
from data.download import fetch_ohlcv
from features.technical import build_features
from labels.target import add_targets

import ccxt

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="ai-trader pipeline", layout="wide")
st.title("ai-trader — data pipeline dashboard")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("1. Download data")
symbol = st.sidebar.selectbox("Symbol", ["BTC/USDT", "ETH/USDT"])
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h"], index=1)
years = st.sidebar.slider("Years of history", 0.25, 3.0, 1.0, step=0.25)

file_stub = symbol.replace("/", "") + f"_{timeframe}"
raw_path = RAW_DIR / f"{file_stub}.parquet"
processed_path = PROCESSED_DIR / f"{file_stub}.parquet"

if st.sidebar.button("Download from Binance"):
    from datetime import datetime, timedelta, timezone

    with st.spinner(f"Downloading {symbol} {timeframe}..."):
        exchange = ccxt.binance({"enableRateLimit": True})
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=365 * years)).timestamp() * 1000)
        df = fetch_ohlcv(exchange, symbol, timeframe, since_ms)
        df.to_parquet(raw_path, index=False)
    st.sidebar.success(f"Saved {len(df):,} candles")

st.sidebar.header("2. Build features + labels")
horizon = st.sidebar.number_input("Label horizon (candles ahead)", 1, 50, 4)
threshold = st.sidebar.number_input("Win threshold (e.g. 0.003 = 0.3%)", 0.0001, 0.05, 0.003, format="%.4f")
cost = st.sidebar.number_input("Assumed round-trip cost", 0.0, 0.02, 0.0015, format="%.4f")

if st.sidebar.button("Run features + labeling"):
    if not raw_path.exists():
        st.sidebar.error("Download data first.")
    else:
        with st.spinner("Building features..."):
            raw = pd.read_parquet(raw_path)
            feats = build_features(raw)
            labeled = add_targets(feats, horizon=horizon, threshold=threshold, cost=cost)
            labeled.to_parquet(processed_path, index=False)
        st.sidebar.success(f"Saved {len(labeled):,} rows, {labeled.shape[1]} columns")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if not processed_path.exists():
    st.info("Use the sidebar: download data, then run features + labeling, to see charts here.")
else:
    df = pd.read_parquet(processed_path).dropna(subset=["ema_200"]).reset_index(drop=True)

    st.subheader(f"{symbol} — {timeframe} — {len(df):,} candles")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.caption("Price with EMA 20/50/200")
        st.line_chart(df.set_index("timestamp")[["close", "ema_20", "ema_50", "ema_200"]])

        st.caption("RSI (14)")
        st.line_chart(df.set_index("timestamp")[["rsi_14"]])

        st.caption("ATR % of price (volatility)")
        st.line_chart(df.set_index("timestamp")[["atr_pct"]])

    with col2:
        if "target_class" in df.columns:
            st.caption("Label balance (target_class)")
            counts = df["target_class"].value_counts(dropna=True).sort_index()
            st.bar_chart(counts)
            win_rate = counts.get(1, 0) / counts.sum() if counts.sum() else 0
            st.metric("Positive class rate", f"{win_rate:.1%}")

        st.caption("Feature snapshot (last 5 rows)")
        show_cols = ["timestamp", "close", "rsi_14", "atr_pct", "volume_zscore_20"]
        if "target_class" in df.columns:
            show_cols.append("target_class")
        st.dataframe(df[show_cols].tail(5), use_container_width=True)

    st.caption("Raw feature table (last 200 rows)")
    st.dataframe(df.tail(200), use_container_width=True)
