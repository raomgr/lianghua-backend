from __future__ import annotations

import pandas as pd

from app.services.data_provider import BaseProvider

FEATURE_COLUMNS = [
    "ret_1d",
    "return_5d",
    "return_10d",
    "momentum_20",
    "momentum_60",
    "reversal_10",
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "volume_ratio_5",
    "volume_ratio_20",
    "turnover_rate",
    "turnover_ratio_5",
    "price_vs_ma_20",
    "price_vs_ma_60",
    "breakout_20",
    "close_position_20",
    "intraday_range",
    "atr_14",
]


def enrich_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    if "amount" not in frame.columns:
        frame["amount"] = frame["close"] * frame["volume"]
    if "turnover_rate" not in frame.columns:
        frame["turnover_rate"] = 0.0
    frame["ret_1d"] = frame["close"].pct_change()
    frame["return_5d"] = frame["close"].pct_change(5)
    frame["return_10d"] = frame["close"].pct_change(10)
    frame["momentum_20"] = frame["close"] / frame["close"].shift(20) - 1
    frame["momentum_60"] = frame["close"] / frame["close"].shift(60) - 1
    frame["reversal_10"] = -(frame["close"] / frame["close"].shift(10) - 1)
    frame["volatility_10"] = frame["ret_1d"].rolling(10).std().fillna(0)
    frame["volatility_20"] = frame["ret_1d"].rolling(20).std().fillna(0)
    frame["volatility_60"] = frame["ret_1d"].rolling(60).std().fillna(0)
    frame["volume_ma_5"] = frame["volume"].rolling(5).mean()
    frame["volume_ma_20"] = frame["volume"].rolling(20).mean()
    frame["volume_ratio_5"] = (frame["volume"] / frame["volume_ma_5"]).fillna(1)
    frame["volume_ratio_20"] = (frame["volume"] / frame["volume_ma_20"]).fillna(1)
    frame["turnover_ma_5"] = frame["turnover_rate"].rolling(5).mean()
    frame["turnover_ratio_5"] = (frame["turnover_rate"] / frame["turnover_ma_5"]).replace([pd.NA, pd.NaT], 0).fillna(1)
    frame["ma_20"] = frame["close"].rolling(20).mean()
    frame["ma_60"] = frame["close"].rolling(60).mean()
    frame["price_vs_ma_20"] = (frame["close"] / frame["ma_20"] - 1).fillna(0)
    frame["price_vs_ma_60"] = (frame["close"] / frame["ma_60"] - 1).fillna(0)
    frame["high_20"] = frame["high"].rolling(20).max()
    frame["low_20"] = frame["low"].rolling(20).min()
    frame["breakout_20"] = (frame["close"] / frame["high_20"] - 1).fillna(0)
    price_range = (frame["high_20"] - frame["low_20"]).replace(0, pd.NA)
    frame["close_position_20"] = ((frame["close"] - frame["low_20"]) / price_range).fillna(0.5)
    prev_close = frame["close"].shift(1)
    tr_components = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    frame["atr_14"] = tr_components.max(axis=1).rolling(14).mean().fillna(0) / frame["close"]
    frame["intraday_range"] = ((frame["high"] - frame["low"]) / frame["close"]).fillna(0)
    return frame


def build_factor_table(provider: BaseProvider) -> pd.DataFrame:
    rows = []
    for item in provider.get_universe():
        enriched = enrich_bars(provider.get_daily_bars(item.symbol, limit=90))
        latest = enriched.iloc[-1]
        rows.append(
            {
                "symbol": item.symbol,
                "name": item.name,
                "close": round(float(latest["close"]), 2),
                "ret_1d": float(latest["ret_1d"]),
                "return_5d": float(latest["return_5d"]),
                "return_10d": float(latest["return_10d"]),
                "momentum_20": float(latest["momentum_20"]),
                "momentum_60": float(latest["momentum_60"]),
                "reversal_10": float(latest["reversal_10"]),
                "volatility_10": float(latest["volatility_10"]),
                "volatility_20": float(latest["volatility_20"]),
                "volatility_60": float(latest["volatility_60"]),
                "volume_ratio_5": float(latest["volume_ratio_5"]),
                "volume_ratio_20": float(latest["volume_ratio_20"]),
                "turnover_rate": float(latest["turnover_rate"]),
                "turnover_ratio_5": float(latest["turnover_ratio_5"]),
                "price_vs_ma_20": float(latest["price_vs_ma_20"]),
                "price_vs_ma_60": float(latest["price_vs_ma_60"]),
                "breakout_20": float(latest["breakout_20"]),
                "close_position_20": float(latest["close_position_20"]),
                "intraday_range": float(latest["intraday_range"]),
                "atr_14": float(latest["atr_14"]),
            }
        )

    frame = pd.DataFrame(rows)
    return score_factors(frame)


def build_factor_table_from_histories(histories: dict[str, pd.DataFrame], names: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for symbol, bars in histories.items():
        if bars.empty or len(bars) < 25:
            continue

        enriched = enrich_bars(bars)
        latest = enriched.iloc[-1]
        rows.append(
            {
                "symbol": symbol,
                "name": (names or {}).get(symbol, symbol),
                "close": round(float(latest["close"]), 2),
                "ret_1d": float(latest["ret_1d"]),
                "return_5d": float(latest["return_5d"]),
                "return_10d": float(latest["return_10d"]),
                "momentum_20": float(latest["momentum_20"]),
                "momentum_60": float(latest["momentum_60"]),
                "reversal_10": float(latest["reversal_10"]),
                "volatility_10": float(latest["volatility_10"]),
                "volatility_20": float(latest["volatility_20"]),
                "volatility_60": float(latest["volatility_60"]),
                "volume_ratio_5": float(latest["volume_ratio_5"]),
                "volume_ratio_20": float(latest["volume_ratio_20"]),
                "turnover_rate": float(latest["turnover_rate"]),
                "turnover_ratio_5": float(latest["turnover_ratio_5"]),
                "price_vs_ma_20": float(latest["price_vs_ma_20"]),
                "price_vs_ma_60": float(latest["price_vs_ma_60"]),
                "breakout_20": float(latest["breakout_20"]),
                "close_position_20": float(latest["close_position_20"]),
                "intraday_range": float(latest["intraday_range"]),
                "atr_14": float(latest["atr_14"]),
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "name",
            "close",
            "ret_1d",
            "return_5d",
            "return_10d",
            "momentum_20",
            "momentum_60",
            "reversal_10",
            "volatility_10",
            "volatility_20",
            "volatility_60",
            "volume_ratio_5",
            "volume_ratio_20",
            "turnover_rate",
            "turnover_ratio_5",
            "price_vs_ma_20",
            "price_vs_ma_60",
            "breakout_20",
            "close_position_20",
            "intraday_range",
            "atr_14",
        ],
    )
    if frame.empty:
        return frame
    return score_factors(frame)


def build_training_dataset_from_histories(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str] | None = None,
    prediction_horizon: int = 5,
) -> pd.DataFrame:
    rows = []
    for symbol, bars in histories.items():
        if bars.empty or len(bars) < 40:
            continue

        enriched = enrich_bars(bars)
        enriched["future_return_5d"] = enriched["close"].shift(-prediction_horizon) / enriched["close"] - 1
        usable = enriched.dropna(subset=["future_return_5d", *FEATURE_COLUMNS]).copy()
        for _, row in usable.iterrows():
            rows.append(
                {
                    "symbol": symbol,
                    "name": (names or {}).get(symbol, symbol),
                    "trade_date": row["trade_date"],
                    "close": float(row["close"]),
                    "future_return_5d": float(row["future_return_5d"]),
                    **{feature: float(row[feature]) for feature in FEATURE_COLUMNS},
                }
            )

    columns = ["symbol", "name", "trade_date", "close", "future_return_5d", *FEATURE_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def score_factors(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    scored["momentum_rank"] = scored["momentum_20"].rank(pct=True)
    scored["trend_rank"] = scored["momentum_60"].rank(pct=True)
    scored["quality_rank"] = scored["return_5d"].rank(pct=True)
    scored["liquidity_rank"] = scored["volume_ratio_5"].rank(pct=True)
    scored["participation_rank"] = scored["turnover_ratio_5"].rank(pct=True)
    scored["structure_rank"] = scored["close_position_20"].rank(pct=True)
    scored["risk_rank"] = 1 - scored["volatility_20"].rank(pct=True)
    scored["score"] = (
        scored["momentum_rank"] * 0.22
        + scored["trend_rank"] * 0.18
        + scored["quality_rank"] * 0.16
        + scored["liquidity_rank"] * 0.12
        + scored["participation_rank"] * 0.10
        + scored["structure_rank"] * 0.10
        + scored["risk_rank"] * 0.12
    )
    return scored.sort_values("score", ascending=False).reset_index(drop=True)
