# ai-trader — Phase 1–3 starter

This is the data pipeline, feature engineering, and target-definition
starter for the project (Phases 1–3 of the roadmap). No exchange API key
is needed yet — `download.py` uses Binance's public market-data endpoints.

## 1. Set up your environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download historical data (Phase 1)

```bash
cd data
python download.py --symbol BTC/USDT --timeframe 15m --years 2
python download.py --symbol ETH/USDT --timeframe 15m --years 2
```

This saves `data/raw/BTCUSDT_15m.parquet` and `data/raw/ETHUSDT_15m.parquet`.

## 3. Build features (Phase 2)

```bash
cd ../features
python technical.py ../data/raw/BTCUSDT_15m.parquet
```

This saves `data/processed/BTCUSDT_15m.parquet` with ~20 engineered
columns (momentum, volatility, trend, volume). Open it in a notebook and
sanity-check the values — plot RSI/EMA against price to confirm nothing
looks broken before moving on.

## 4. Add the prediction target (Phase 3)

```bash
cd ../labels
python target.py ../data/processed/BTCUSDT_15m.parquet
```

This adds `future_return` and `target_class` columns in place. Check
`target_class.value_counts()` — if it's wildly imbalanced (e.g. 95%/5%),
adjust `threshold` in `target.py` before training anything.

## Optional: run it as a dashboard instead

Once your venv is set up (`pip install -r requirements.txt` includes
Streamlit), you can skip the manual CLI steps 2–4 above and drive the whole
pipeline from a browser UI instead:

```bash
streamlit run app.py
```

This opens a local page (usually `http://localhost:8501`) where you can:
- pick symbol/timeframe/years and download from the sidebar
- run feature engineering + labeling with adjustable horizon/threshold/cost
- see price + EMA/RSI/ATR charts and the label balance update live

It's still just calling `download.py`, `technical.py`, and `target.py`
under the hood — the CLI scripts still work standalone if you prefer them.

## What's deliberately NOT here yet

No model training, no backtester, no live execution — those are Phases
4–6, and they depend on what you find when you actually look at this
data (label balance, feature distributions, any obvious data quality
issues). Get through steps 1–4, poke at the result, and we build the
next piece from what you actually see rather than guessing ahead.

## Notes

- `download.py` paginates through Binance's public OHLCV endpoint — it
  can take a few minutes for 2 years of 15m data (~70k candles) due to
  rate limiting.
- Everything is saved as Parquet, not a database — that's intentional
  for now. Move to PostgreSQL once you've validated the pipeline is
  worth persisting properly.
