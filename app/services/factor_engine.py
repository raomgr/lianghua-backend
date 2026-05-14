from __future__ import annotations

import numpy as np
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
    "turnover_rate_f",
    "turnover_free_ratio_5",
    "pe_ttm_inv",
    "pb_inv",
    "log_total_mv",
    "log_circ_mv",
]


def _apply_price_basis(frame: pd.DataFrame) -> pd.DataFrame:
    price_frame = frame.copy()
    for raw_col, qfq_col in [
        ("open", "qfq_open"),
        ("high", "qfq_high"),
        ("low", "qfq_low"),
        ("close", "qfq_close"),
    ]:
        if qfq_col in price_frame.columns:
            adjusted = pd.to_numeric(price_frame[qfq_col], errors="coerce")
            price_frame[raw_col] = adjusted.fillna(price_frame[raw_col])
    return price_frame


def enrich_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = _apply_price_basis(bars)
    if "amount" not in frame.columns:
        frame["amount"] = frame["close"] * frame["volume"]
    if "turnover_rate" not in frame.columns:
        frame["turnover_rate"] = 0.0
    if "turnover_rate_f" not in frame.columns:
        frame["turnover_rate_f"] = frame["turnover_rate"]
    else:
        frame["turnover_rate_f"] = pd.to_numeric(frame["turnover_rate_f"], errors="coerce").fillna(frame["turnover_rate"])
    for column in ["pe_ttm", "pb", "total_mv", "circ_mv"]:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
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
    frame["turnover_f_ma_5"] = frame["turnover_rate_f"].rolling(5).mean()
    frame["turnover_free_ratio_5"] = (frame["turnover_rate_f"] / frame["turnover_f_ma_5"]).replace([pd.NA, pd.NaT], 0).fillna(1)
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
    positive_pe = frame["pe_ttm"].where(frame["pe_ttm"] > 0)
    positive_pb = frame["pb"].where(frame["pb"] > 0)
    positive_total_mv = frame["total_mv"].where(frame["total_mv"] > 0)
    positive_circ_mv = frame["circ_mv"].where(frame["circ_mv"] > 0)
    frame["pe_ttm_inv"] = (1 / positive_pe).fillna(0)
    frame["pb_inv"] = (1 / positive_pb).fillna(0)
    frame["log_total_mv"] = np.log(positive_total_mv).fillna(0)
    frame["log_circ_mv"] = np.log(positive_circ_mv).fillna(0)
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
                "turnover_rate_f": float(latest["turnover_rate_f"]),
                "turnover_free_ratio_5": float(latest["turnover_free_ratio_5"]),
                "pe_ttm_inv": float(latest["pe_ttm_inv"]),
                "pb_inv": float(latest["pb_inv"]),
                "log_total_mv": float(latest["log_total_mv"]),
                "log_circ_mv": float(latest["log_circ_mv"]),
            }
        )

    frame = pd.DataFrame(rows)
    return score_factors(frame)


def build_factor_table_from_histories(histories: dict[str, pd.DataFrame], names: dict[str, str] | None = None) -> pd.DataFrame:
    output_columns = [
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
        "turnover_rate_f",
        "turnover_free_ratio_5",
        "pe_ttm_inv",
        "pb_inv",
        "log_total_mv",
        "log_circ_mv",
        "score",
    ]
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
                "turnover_rate_f": float(latest["turnover_rate_f"]),
                "turnover_free_ratio_5": float(latest["turnover_free_ratio_5"]),
                "pe_ttm_inv": float(latest["pe_ttm_inv"]),
                "pb_inv": float(latest["pb_inv"]),
                "log_total_mv": float(latest["log_total_mv"]),
                "log_circ_mv": float(latest["log_circ_mv"]),
                "score": 0.0,
            }
        )

    frame = pd.DataFrame(rows, columns=output_columns)
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
    scored["free_participation_rank"] = scored["turnover_free_ratio_5"].rank(pct=True)
    scored["structure_rank"] = scored["close_position_20"].rank(pct=True)
    scored["risk_rank"] = 1 - scored["volatility_20"].rank(pct=True)
    scored["value_rank"] = (
        scored["pe_ttm_inv"].rank(pct=True) * 0.55
        + scored["pb_inv"].rank(pct=True) * 0.45
    )
    scored["size_rank"] = 1 - scored["log_total_mv"].rank(pct=True)
    scored["score"] = (
        scored["momentum_rank"] * 0.19
        + scored["trend_rank"] * 0.16
        + scored["quality_rank"] * 0.14
        + scored["liquidity_rank"] * 0.10
        + scored["participation_rank"] * 0.08
        + scored["free_participation_rank"] * 0.07
        + scored["structure_rank"] * 0.08
        + scored["risk_rank"] * 0.10
        + scored["value_rank"] * 0.05
        + scored["size_rank"] * 0.03
    )
    return scored.sort_values("score", ascending=False).reset_index(drop=True)
