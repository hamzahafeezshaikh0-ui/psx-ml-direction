import pandas as pd

FEATURES = ["ret_1", "ret_5", "vol_10", "vol_chg", "hl_range", "ma_dist"]


def make_features(prices):
    """prices: DataFrame with Open, High, Low, Close, Volume (oldest to newest).
    The feature at row t uses only rows t and earlier."""
    out = pd.DataFrame(index=prices.index)
    close = prices["Close"]
    out["ret_1"] = close.pct_change()                      # previous-session return
    out["ret_5"] = close.pct_change(5)                     # 5-session return
    out["vol_10"] = out["ret_1"].rolling(10).std()         # recent volatility
    out["vol_chg"] = prices["Volume"] / prices["Volume"].rolling(20).mean() - 1
    out["hl_range"] = (prices["High"] - prices["Low"]) / close
    out["ma_dist"] = close / close.rolling(20).mean() - 1  # distance from 20-day average
    return out
