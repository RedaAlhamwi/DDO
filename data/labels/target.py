"""
Phase 3 — Target definition.

Defines the prediction target as: will price move up by more than
`threshold` within the next `horizon` candles, net of an assumed
round-trip cost? This is a classification target; a regression version
(future_return) is also provided so you can compare both formulations.

Usage:
    from labels.target import add_targets
    labeled = add_targets(feats, horizon=4, threshold=0.003, cost=0.0015)
"""

import pandas as pd


def add_targets(
    df: pd.DataFrame,
    horizon: int = 4,
    threshold: float = 0.003,
    cost: float = 0.0015,
) -> pd.DataFrame:
    """
    horizon:   number of candles ahead to look (4 candles * 15m = 1 hour)
    threshold: minimum future return to count as a "win" (e.g. 0.003 = 0.3%)
    cost:      assumed round-trip cost (fees + slippage) to subtract before
               labeling — forces the model to only learn moves worth trading
    """
    out = df.copy()
    future_price = out["close"].shift(-horizon)
    out["future_return"] = future_price / out["close"] - 1

    net_return = out["future_return"] - cost
    out["target_class"] = (net_return > threshold).astype(int)

    # Rows near the end have no future price to look at — mark and drop later
    out.loc[out["future_return"].isna(), "target_class"] = pd.NA
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python target.py <path_to_processed_parquet>")
        sys.exit(1)

    path = Path(sys.argv[1])
    df = pd.read_parquet(path)
    labeled = add_targets(df)
    print(labeled["target_class"].value_counts(dropna=False))
    labeled.to_parquet(path, index=False)  # overwrite with labels added
    print(f"Added targets to {path}")
