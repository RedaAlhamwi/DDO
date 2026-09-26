"""
Phase 1 — Market data pipeline.

Downloads historical OHLCV candles from Binance (via ccxt, no API key needed
for public market data) and saves them as Parquet files.

Usage:
    python download.py --symbol BTC/USDT --timeframe 15m --years 2
    python download.py --symbol ETH/USDT --timeframe 15m --years 2
"""

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(exist_ok=True)


def fetch_ohlcv(exchange, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """Fetch all OHLCV candles from `since_ms` to now, paginating in
    batches (Binance caps each call at ~1000 candles)."""
    all_rows = []
    limit = 1000
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts == since_ms:
            break
        since_ms = last_ts + 1
        # be polite to the API — avoid rate-limit bans
        time.sleep(exchange.rateLimit / 1000)
        if last_ts > int(datetime.now(timezone.utc).timestamp() * 1000):
            break

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT", help="e.g. BTC/USDT, ETH/USDT")
    parser.add_argument("--timeframe", default="15m", help="e.g. 5m, 15m, 1h")
    parser.add_argument("--years", type=float, default=2.0, help="how many years of history")
    args = parser.parse_args()

    exchange = ccxt.binance({"enableRateLimit": True})
    since_dt = datetime.now(timezone.utc) - timedelta(days=365 * args.years)
    since_ms = int(since_dt.timestamp() * 1000)

    print(f"Downloading {args.symbol} {args.timeframe} since {since_dt.date()}...")
    df = fetch_ohlcv(exchange, args.symbol, args.timeframe, since_ms)

    out_name = args.symbol.replace("/", "") + f"_{args.timeframe}.parquet"
    out_path = DATA_DIR / out_name
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} candles to {out_path}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()
