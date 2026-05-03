from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from app.services.data_provider import BaseProvider
from app.services.factor_engine import build_factor_table, build_factor_table_from_histories, enrich_bars

REBALANCE_DAYS = 5
TRADING_COST_BPS = 8
SLIPPAGE_BPS = 5
TOP_N = 3


@dataclass
class BacktestConfig:
    rebalance_days: int = REBALANCE_DAYS
    top_n: int = TOP_N
    trading_cost_bps: float = TRADING_COST_BPS
    slippage_bps: float = SLIPPAGE_BPS


def _max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    drawdown = series / roll_max - 1
    return float(drawdown.min())


def _annualized_volatility(returns: pd.Series) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.std() * math.sqrt(252))


def _downside_volatility(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    downside = returns[returns < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(downside.std() * math.sqrt(252))


def _safe_calmar(annual_return: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return 0.0
    return float(annual_return / abs(max_drawdown))


def _safe_sortino(annual_return: float, returns: pd.Series) -> float:
    downside_vol = _downside_volatility(returns)
    if downside_vol == 0:
        return 0.0
    return float(annual_return / downside_vol)


def _information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0
    active_returns = portfolio_returns - benchmark_returns
    if active_returns.std() == 0:
        return 0.0
    return float((active_returns.mean() / active_returns.std()) * math.sqrt(252))


def _compute_beta_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    if portfolio_returns.empty or benchmark_returns.empty:
        return 0.0, 0.0
    benchmark_var = float(benchmark_returns.var())
    if benchmark_var == 0:
        return 0.0, 0.0
    covariance = float(portfolio_returns.cov(benchmark_returns))
    beta = covariance / benchmark_var
    alpha = float((portfolio_returns.mean() - beta * benchmark_returns.mean()) * 252)
    return float(beta), alpha


def _build_empty_backtest_payload(config: BacktestConfig) -> dict:
    return {
        "summary": {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "information_ratio": 0.0,
            "beta": 0.0,
            "alpha": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "benchmark_return": 0.0,
            "excess_return": 0.0,
            "avg_turnover": 0.0,
            "total_cost": 0.0,
            "rebalance_days": config.rebalance_days,
            "holdings_per_rebalance": config.top_n,
        },
        "equity_curve": [],
        "picks": [],
        "rebalances": [],
    }


def run_baseline_backtest(provider: BaseProvider, top_n: int = 3) -> dict:
    factor_table = build_factor_table(provider)
    picks = factor_table.head(top_n).copy()

    daily_returns = []
    benchmark_returns = []
    dates = None

    for _, row in picks.iterrows():
        bars = enrich_bars(provider.get_daily_bars(row["symbol"], limit=60))
        returns = bars["ret_1d"].fillna(0).reset_index(drop=True)
        daily_returns.append(returns)
        if dates is None:
            dates = bars["trade_date"].reset_index(drop=True)

    benchmark_symbol = factor_table.iloc[-1]["symbol"]
    benchmark_bars = enrich_bars(provider.get_daily_bars(benchmark_symbol, limit=60))
    benchmark_returns = benchmark_bars["ret_1d"].fillna(0).reset_index(drop=True)

    portfolio_ret = pd.concat(daily_returns, axis=1).mean(axis=1)
    equity = (1 + portfolio_ret).cumprod()
    benchmark = (1 + benchmark_returns).cumprod()
    annual_return = float((equity.iloc[-1] ** (252 / max(len(equity), 1))) - 1)
    annual_volatility = _annualized_volatility(portfolio_ret)
    max_drawdown = _max_drawdown(equity)
    benchmark_return = float(benchmark.iloc[-1] - 1)
    total_return = float(equity.iloc[-1] - 1)
    beta, alpha = _compute_beta_alpha(portfolio_ret, benchmark_returns)

    sharpe = 0.0
    if portfolio_ret.std() > 0:
        sharpe = float((portfolio_ret.mean() / portfolio_ret.std()) * math.sqrt(252))

    summary = {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "calmar": _safe_calmar(annual_return, max_drawdown),
        "sharpe": sharpe,
        "sortino": _safe_sortino(annual_return, portfolio_ret),
        "information_ratio": _information_ratio(portfolio_ret, benchmark_returns),
        "beta": beta,
        "alpha": alpha,
        "win_rate": float((portfolio_ret > 0).mean()),
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": float(total_return - benchmark_return),
        "avg_turnover": 0.0,
        "total_cost": 0.0,
        "rebalance_days": REBALANCE_DAYS,
        "holdings_per_rebalance": top_n,
    }

    equity_curve = [
        {
            "trade_date": dates.iloc[i],
            "equity": round(float(equity.iloc[i]), 4),
            "benchmark": round(float(benchmark.iloc[i]), 4),
        }
        for i in range(len(equity))
    ]

    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "picks": picks[
            ["symbol", "close", "return_5d", "momentum_20", "volatility_20", "volume_ratio_5", "score"]
        ].to_dict(orient="records"),
    }


def run_baseline_backtest_from_histories(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str] | None = None,
    config: BacktestConfig | None = None,
) -> dict:
    config = config or BacktestConfig()
    factor_table = build_factor_table_from_histories(histories, names=names)
    if factor_table.empty:
        return _build_empty_backtest_payload(config)

    usable_histories = {symbol: enrich_bars(frame.tail(120)) for symbol, frame in histories.items() if len(frame) >= 80}
    if not usable_histories:
        return _build_empty_backtest_payload(config)

    common_dates = sorted(set.intersection(*[set(frame["trade_date"]) for frame in usable_histories.values()]))
    common_dates = common_dates[-60:]
    date_index = {trade_date: idx for idx, trade_date in enumerate(common_dates)}

    returns_by_symbol: dict[str, pd.Series] = {}
    feature_rows = []
    for symbol, frame in usable_histories.items():
        aligned = frame[frame["trade_date"].isin(common_dates)].copy().reset_index(drop=True)
        if len(aligned) != len(common_dates):
            continue
        returns_by_symbol[symbol] = pd.Series(aligned["ret_1d"].fillna(0).values, index=common_dates)
        for _, row in aligned.iterrows():
            feature_rows.append(
                {
                    "trade_date": row["trade_date"],
                    "symbol": symbol,
                    "name": (names or {}).get(symbol, symbol),
                    "close": float(row["close"]),
                    "ret_1d": float(row["ret_1d"]),
                    "return_5d": float(row["return_5d"]),
                    "return_10d": float(row["return_10d"]),
                    "momentum_20": float(row["momentum_20"]),
                    "momentum_60": float(row["momentum_60"]),
                    "reversal_10": float(row["reversal_10"]),
                    "volatility_10": float(row["volatility_10"]),
                    "volatility_20": float(row["volatility_20"]),
                    "volatility_60": float(row["volatility_60"]),
                    "volume_ratio_5": float(row["volume_ratio_5"]),
                    "volume_ratio_20": float(row["volume_ratio_20"]),
                    "turnover_rate": float(row["turnover_rate"]),
                    "turnover_ratio_5": float(row["turnover_ratio_5"]),
                    "price_vs_ma_20": float(row["price_vs_ma_20"]),
                    "price_vs_ma_60": float(row["price_vs_ma_60"]),
                    "breakout_20": float(row["breakout_20"]),
                    "close_position_20": float(row["close_position_20"]),
                    "intraday_range": float(row["intraday_range"]),
                    "atr_14": float(row["atr_14"]),
                }
            )

    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.empty:
        raise RuntimeError("No aligned histories were available for backtest.")

    def rank_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
        ranked = snapshot.copy()
        score_inputs = [
            "return_5d",
            "momentum_20",
            "momentum_60",
            "volatility_20",
            "volume_ratio_5",
            "turnover_ratio_5",
            "close_position_20",
        ]
        ranked[score_inputs] = ranked[score_inputs].fillna(0.0)
        ranked["momentum_rank"] = ranked["momentum_20"].rank(pct=True)
        ranked["trend_rank"] = ranked["momentum_60"].rank(pct=True)
        ranked["quality_rank"] = ranked["return_5d"].rank(pct=True)
        ranked["liquidity_rank"] = ranked["volume_ratio_5"].rank(pct=True)
        ranked["participation_rank"] = ranked["turnover_ratio_5"].rank(pct=True)
        ranked["structure_rank"] = ranked["close_position_20"].rank(pct=True)
        ranked["risk_rank"] = 1 - ranked["volatility_20"].rank(pct=True)
        ranked["score"] = (
            ranked["momentum_rank"] * 0.22
            + ranked["trend_rank"] * 0.18
            + ranked["quality_rank"] * 0.16
            + ranked["liquidity_rank"] * 0.12
            + ranked["participation_rank"] * 0.10
            + ranked["structure_rank"] * 0.10
            + ranked["risk_rank"] * 0.12
        )
        return ranked.sort_values("score", ascending=False).reset_index(drop=True)

    rebalance_dates = common_dates[:: config.rebalance_days]
    positions_by_date: dict = {}
    previous_holdings: set[str] = set()
    cost_rate = (config.trading_cost_bps + config.slippage_bps) / 10000
    turnover_values: list[float] = []
    total_cost = 0.0
    rebalance_log: list[dict] = []

    for rebalance_date in rebalance_dates:
        snapshot = feature_frame[feature_frame["trade_date"] == rebalance_date].copy()
        ranked = rank_snapshot(snapshot)
        top_snapshot = ranked.head(config.top_n).copy()
        holdings = set(top_snapshot["symbol"].tolist())
        changed = previous_holdings.symmetric_difference(holdings)
        turnover = len(changed) / max(config.top_n * 2, 1)
        turnover_values.append(turnover)
        total_cost += turnover * cost_rate
        rebalance_log.append(
            {
                "trade_date": rebalance_date,
                "turnover": float(turnover),
                "estimated_cost": float(turnover * cost_rate),
                "holdings": [
                    {
                        "symbol": row["symbol"],
                        "name": row["name"],
                        "score": float(row["score"]),
                    }
                    for _, row in top_snapshot.iterrows()
                ],
            }
        )
        previous_holdings = holdings

        start_idx = date_index[rebalance_date]
        end_idx = min(start_idx + config.rebalance_days, len(common_dates))
        for idx in range(start_idx, end_idx):
            positions_by_date[common_dates[idx]] = list(holdings)

    portfolio_returns = []
    benchmark_symbol = sorted(returns_by_symbol.keys())[0]
    benchmark_returns = returns_by_symbol[benchmark_symbol].reindex(common_dates).fillna(0)

    for trade_date in common_dates:
        holdings = positions_by_date.get(trade_date, [])
        day_returns = [returns_by_symbol[symbol].loc[trade_date] for symbol in holdings if symbol in returns_by_symbol]
        gross_return = float(sum(day_returns) / len(day_returns)) if day_returns else 0.0
        day_cost = 0.0
        if trade_date in rebalance_dates:
            rebalance_idx = rebalance_dates.index(trade_date)
            day_cost = turnover_values[rebalance_idx] * cost_rate
        portfolio_returns.append(gross_return - day_cost)

    portfolio_ret = pd.Series(portfolio_returns, index=common_dates)
    equity = (1 + portfolio_ret).cumprod()
    benchmark = (1 + benchmark_returns).cumprod()

    sharpe = 0.0
    if portfolio_ret.std() > 0:
        sharpe = float((portfolio_ret.mean() / portfolio_ret.std()) * math.sqrt(252))

    annual_return = float((equity.iloc[-1] ** (252 / max(len(equity), 1))) - 1)
    annual_volatility = _annualized_volatility(portfolio_ret)
    max_drawdown = _max_drawdown(equity)
    benchmark_return = float(benchmark.iloc[-1] - 1)
    total_return = float(equity.iloc[-1] - 1)
    sortino = _safe_sortino(annual_return, portfolio_ret)
    information_ratio = _information_ratio(portfolio_ret, benchmark_returns)
    beta, alpha = _compute_beta_alpha(portfolio_ret, benchmark_returns)
    latest_snapshot = rank_snapshot(feature_frame[feature_frame["trade_date"] == common_dates[-1]].copy())
    summary = {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "calmar": _safe_calmar(annual_return, max_drawdown),
        "sharpe": sharpe,
        "sortino": sortino,
        "information_ratio": information_ratio,
        "beta": beta,
        "alpha": alpha,
        "win_rate": float((portfolio_ret > 0).mean()),
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": float(total_return - benchmark_return),
        "avg_turnover": float(sum(turnover_values) / len(turnover_values)) if turnover_values else 0.0,
        "total_cost": float(total_cost),
        "rebalance_days": config.rebalance_days,
        "holdings_per_rebalance": config.top_n,
    }

    equity_curve = [
        {
            "trade_date": common_dates[i],
            "equity": round(float(equity.iloc[i]), 4),
            "benchmark": round(float(benchmark.iloc[i]), 4),
        }
        for i in range(len(equity))
    ]

    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "picks": latest_snapshot.head(config.top_n)[
            [
                "symbol",
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
                "score",
            ]
        ].to_dict(orient="records"),
        "rebalances": rebalance_log[-8:],
    }


def _expand_int_options(base: int, step: int, width: int, floor: int, ceil: int) -> list[int]:
    values = {int(base)}
    for i in range(1, max(width, 1) + 1):
        values.add(int(base - step * i))
        values.add(int(base + step * i))
    return sorted({value for value in values if floor <= value <= ceil})


def _build_cost_options(base_cost: float, width: int) -> list[float]:
    if width <= 1:
        multipliers = [0.75, 1.0, 1.25]
    elif width == 2:
        multipliers = [0.6, 0.8, 1.0, 1.2, 1.4]
    else:
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    return sorted({round(max(base_cost * ratio, 0.0), 2) for ratio in multipliers})


def _summary_to_scan_point(summary: dict, config: BacktestConfig) -> dict:
    return {
        "rebalance_days": int(config.rebalance_days),
        "top_n": int(config.top_n),
        "trading_cost_bps": float(config.trading_cost_bps),
        "slippage_bps": float(config.slippage_bps),
        "annual_return": float(summary.get("annual_return", 0.0)),
        "sharpe": float(summary.get("sharpe", 0.0)),
        "max_drawdown": float(summary.get("max_drawdown", 0.0)),
        "calmar": float(summary.get("calmar", 0.0)),
        "total_return": float(summary.get("total_return", 0.0)),
        "excess_return": float(summary.get("excess_return", 0.0)),
        "sortino": float(summary.get("sortino", 0.0)),
        "information_ratio": float(summary.get("information_ratio", 0.0)),
        "avg_turnover": float(summary.get("avg_turnover", 0.0)),
        "total_cost": float(summary.get("total_cost", 0.0)),
    }


def run_backtest_sensitivity_from_histories(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str] | None = None,
    base_config: BacktestConfig | None = None,
    scan_width: int = 1,
) -> dict:
    config = base_config or BacktestConfig()
    width = min(max(int(scan_width), 1), 3)
    rebalance_options = _expand_int_options(config.rebalance_days, step=2, width=width, floor=1, ceil=20)
    top_n_options = _expand_int_options(config.top_n, step=1, width=width, floor=1, ceil=10)
    cost_options = _build_cost_options(config.trading_cost_bps, width=width)

    baseline_result = run_baseline_backtest_from_histories(histories, names=names, config=config)
    baseline_point = _summary_to_scan_point(baseline_result.get("summary", {}), config)

    rows: list[dict] = []
    for rebalance_days in rebalance_options:
        for top_n in top_n_options:
            for trading_cost_bps in cost_options:
                scan_config = BacktestConfig(
                    rebalance_days=rebalance_days,
                    top_n=top_n,
                    trading_cost_bps=trading_cost_bps,
                    slippage_bps=config.slippage_bps,
                )
                result = run_baseline_backtest_from_histories(histories, names=names, config=scan_config)
                rows.append(_summary_to_scan_point(result.get("summary", {}), scan_config))

    rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("sharpe", 0.0)),
            float(row.get("annual_return", 0.0)),
            float(row.get("calmar", 0.0)),
            float(row.get("total_return", 0.0)),
        ),
        reverse=True,
    )

    def pick_best(metric: str) -> dict:
        if not rows:
            return baseline_point
        return max(rows, key=lambda row: float(row.get(metric, 0.0)))

    return {
        "scan_id": f"scan-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scan_width": width,
        "rebalance_days_options": rebalance_options,
        "top_n_options": top_n_options,
        "trading_cost_options": cost_options,
        "slippage_bps": float(config.slippage_bps),
        "baseline": baseline_point,
        "rows": rows,
        "best_sharpe": pick_best("sharpe"),
        "best_annual_return": pick_best("annual_return"),
        "best_calmar": pick_best("calmar"),
    }


def _safe_mean_annualized_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.mean() * 252)


def _build_regime_stats(frame: pd.DataFrame) -> list[dict]:
    stats: list[dict] = []
    regimes = {
        "up": frame[frame["benchmark_ret"] > 0],
        "down": frame[frame["benchmark_ret"] < 0],
        "flat": frame[frame["benchmark_ret"] == 0],
    }
    for regime, subset in regimes.items():
        if subset.empty:
            stats.append(
                {
                    "regime": regime,
                    "days": 0,
                    "annualized_return": 0.0,
                    "annualized_excess_return": 0.0,
                    "win_rate": 0.0,
                    "sharpe": 0.0,
                }
            )
            continue
        sharpe = 0.0
        if subset["portfolio_ret"].std() > 0:
            sharpe = float((subset["portfolio_ret"].mean() / subset["portfolio_ret"].std()) * math.sqrt(252))
        stats.append(
            {
                "regime": regime,
                "days": int(len(subset)),
                "annualized_return": _safe_mean_annualized_return(subset["portfolio_ret"]),
                "annualized_excess_return": _safe_mean_annualized_return(subset["active_ret"]),
                "win_rate": float((subset["portfolio_ret"] > 0).mean()),
                "sharpe": sharpe,
            }
        )
    return stats


def run_backtest_stability_from_histories(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str] | None = None,
    base_config: BacktestConfig | None = None,
    rolling_window: int = 20,
) -> dict:
    config = base_config or BacktestConfig()
    window = min(max(int(rolling_window), 10), 40)
    result = run_baseline_backtest_from_histories(histories, names=names, config=config)
    curve = pd.DataFrame(result.get("equity_curve", []))
    if curve.empty:
        return {
            "analysis_id": f"stability-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rolling_window": window,
            "summary": {
                "best_month": "",
                "worst_month": "",
                "best_month_excess_return": 0.0,
                "worst_month_excess_return": 0.0,
                "positive_month_ratio": 0.0,
                "rolling_sharpe_mean": 0.0,
                "rolling_sharpe_p10": 0.0,
                "rolling_max_drawdown_worst": 0.0,
            },
            "rolling_points": [],
            "monthly_points": [],
            "regime_stats": [],
            "baseline_summary": result.get("summary", {}),
        }

    curve["trade_date"] = pd.to_datetime(curve["trade_date"], errors="coerce")
    curve = curve.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    curve["portfolio_ret"] = curve["equity"].pct_change().fillna(0.0)
    curve["benchmark_ret"] = curve["benchmark"].pct_change().fillna(0.0)
    curve["active_ret"] = curve["portfolio_ret"] - curve["benchmark_ret"]
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1

    rolling_mean = curve["portfolio_ret"].rolling(window).mean()
    rolling_std = curve["portfolio_ret"].rolling(window).std()
    rolling_sharpe = (rolling_mean / rolling_std).replace([float("inf"), float("-inf")], 0).fillna(0) * math.sqrt(252)
    rolling_volatility = rolling_std.fillna(0) * math.sqrt(252)
    rolling_excess = ((1 + curve["active_ret"]).rolling(window).apply(lambda x: x.prod(), raw=False) - 1).fillna(0)
    rolling_max_drawdown = curve["drawdown"].rolling(window).min().fillna(curve["drawdown"])

    rolling_points = [
        {
            "trade_date": row["trade_date"].date(),
            "rolling_sharpe": float(row["rolling_sharpe"]),
            "rolling_volatility": float(row["rolling_volatility"]),
            "rolling_excess_return": float(row["rolling_excess_return"]),
            "rolling_max_drawdown": float(row["rolling_max_drawdown"]),
        }
        for _, row in pd.DataFrame(
            {
                "trade_date": curve["trade_date"],
                "rolling_sharpe": rolling_sharpe,
                "rolling_volatility": rolling_volatility,
                "rolling_excess_return": rolling_excess,
                "rolling_max_drawdown": rolling_max_drawdown,
            }
        ).iterrows()
    ]

    monthly_rows: list[dict] = []
    for period, group in curve.groupby(curve["trade_date"].dt.to_period("M")):
        portfolio_return = float((1 + group["portfolio_ret"]).prod() - 1)
        benchmark_return = float((1 + group["benchmark_ret"]).prod() - 1)
        monthly_rows.append(
            {
                "month": str(period),
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "excess_return": float(portfolio_return - benchmark_return),
                "win_rate": float((group["portfolio_ret"] > 0).mean()),
                "max_drawdown": float(group["drawdown"].min()),
            }
        )
    monthly_rows = sorted(monthly_rows, key=lambda item: item["month"])

    if monthly_rows:
        best_month = max(monthly_rows, key=lambda item: item["excess_return"])
        worst_month = min(monthly_rows, key=lambda item: item["excess_return"])
        positive_month_ratio = float(sum(1 for item in monthly_rows if item["excess_return"] > 0) / len(monthly_rows))
    else:
        best_month = {"month": "", "excess_return": 0.0}
        worst_month = {"month": "", "excess_return": 0.0}
        positive_month_ratio = 0.0

    summary = {
        "best_month": best_month["month"],
        "worst_month": worst_month["month"],
        "best_month_excess_return": float(best_month["excess_return"]),
        "worst_month_excess_return": float(worst_month["excess_return"]),
        "positive_month_ratio": positive_month_ratio,
        "rolling_sharpe_mean": float(rolling_sharpe.mean()),
        "rolling_sharpe_p10": float(rolling_sharpe.quantile(0.1)),
        "rolling_max_drawdown_worst": float(rolling_max_drawdown.min()),
    }

    return {
        "analysis_id": f"stability-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rolling_window": window,
        "summary": summary,
        "rolling_points": rolling_points,
        "monthly_points": monthly_rows,
        "regime_stats": _build_regime_stats(curve),
        "baseline_summary": result.get("summary", {}),
    }


def _histogram_payload(values: np.ndarray, bins: int = 16) -> list[dict]:
    if values.size == 0:
        return []
    counts, edges = np.histogram(values, bins=bins)
    payload = []
    for index, count in enumerate(counts):
        payload.append(
            {
                "left": float(edges[index]),
                "right": float(edges[index + 1]),
                "count": int(count),
            }
        )
    return payload


def run_backtest_monte_carlo_from_histories(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str] | None = None,
    base_config: BacktestConfig | None = None,
    trials: int = 300,
    random_seed: int = 42,
) -> dict:
    config = base_config or BacktestConfig()
    n_trials = min(max(int(trials), 50), 1000)
    result = run_baseline_backtest_from_histories(histories, names=names, config=config)
    curve = pd.DataFrame(result.get("equity_curve", []))
    if curve.empty or len(curve) < 20:
        return {
            "analysis_id": f"mc-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trials": n_trials,
            "days": int(len(curve)),
            "summary": {
                "annual_return_p5": 0.0,
                "annual_return_p50": 0.0,
                "annual_return_p95": 0.0,
                "total_return_p5": 0.0,
                "total_return_p50": 0.0,
                "total_return_p95": 0.0,
                "max_drawdown_p50": 0.0,
                "max_drawdown_p95": 0.0,
                "loss_probability": 0.0,
                "under_benchmark_probability": 0.0,
            },
            "annual_return_histogram": [],
            "max_drawdown_histogram": [],
            "baseline_summary": result.get("summary", {}),
        }

    curve["portfolio_ret"] = curve["equity"].pct_change().fillna(0.0)
    curve["benchmark_ret"] = curve["benchmark"].pct_change().fillna(0.0)
    daily_returns = curve["portfolio_ret"].to_numpy(dtype=float)
    benchmark_total_return = float((1 + curve["benchmark_ret"]).prod() - 1)
    horizon = len(daily_returns)

    rng = np.random.default_rng(seed=random_seed)
    annual_returns = np.zeros(n_trials, dtype=float)
    total_returns = np.zeros(n_trials, dtype=float)
    max_drawdowns = np.zeros(n_trials, dtype=float)

    for trial_index in range(n_trials):
        sampled = rng.choice(daily_returns, size=horizon, replace=True)
        equity = np.cumprod(1 + sampled)
        total_return = float(equity[-1] - 1)
        annual_return = float((equity[-1] ** (252 / max(horizon, 1))) - 1)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / np.maximum(running_max, 1e-12) - 1
        max_drawdown = float(np.min(drawdown)) if drawdown.size else 0.0

        annual_returns[trial_index] = annual_return
        total_returns[trial_index] = total_return
        max_drawdowns[trial_index] = max_drawdown

    summary = {
        "annual_return_p5": float(np.quantile(annual_returns, 0.05)),
        "annual_return_p50": float(np.quantile(annual_returns, 0.50)),
        "annual_return_p95": float(np.quantile(annual_returns, 0.95)),
        "total_return_p5": float(np.quantile(total_returns, 0.05)),
        "total_return_p50": float(np.quantile(total_returns, 0.50)),
        "total_return_p95": float(np.quantile(total_returns, 0.95)),
        "max_drawdown_p50": float(np.quantile(max_drawdowns, 0.50)),
        "max_drawdown_p95": float(np.quantile(max_drawdowns, 0.95)),
        "loss_probability": float(np.mean(total_returns < 0)),
        "under_benchmark_probability": float(np.mean(total_returns < benchmark_total_return)),
    }

    return {
        "analysis_id": f"mc-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trials": n_trials,
        "days": int(horizon),
        "summary": summary,
        "annual_return_histogram": _histogram_payload(annual_returns * 100, bins=16),
        "max_drawdown_histogram": _histogram_payload(max_drawdowns * 100, bins=16),
        "baseline_summary": result.get("summary", {}),
    }


def _scenario_selection_score(summary: dict) -> float:
    sharpe = float(summary.get("sharpe", 0.0))
    annual_return = float(summary.get("annual_return", 0.0))
    max_drawdown = abs(float(summary.get("max_drawdown", 0.0)))
    information_ratio = float(summary.get("information_ratio", 0.0))
    return sharpe * 0.45 + annual_return * 0.35 + information_ratio * 0.15 - max_drawdown * 0.25


def run_backtest_scenarios_from_histories(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str] | None = None,
    base_config: BacktestConfig | None = None,
) -> dict:
    config = base_config or BacktestConfig()
    top_n = max(1, int(config.top_n))
    scenario_configs = [
        {
            "scenario_id": "current",
            "scenario_name": "当前参数",
            "description": "你当前在面板里设置的参数。",
            "config": BacktestConfig(
                rebalance_days=int(config.rebalance_days),
                top_n=top_n,
                trading_cost_bps=float(config.trading_cost_bps),
                slippage_bps=float(config.slippage_bps),
            ),
        },
        {
            "scenario_id": "conservative",
            "scenario_name": "稳健",
            "description": "更低换手，更关注回撤控制。",
            "config": BacktestConfig(
                rebalance_days=min(20, int(config.rebalance_days) + 3),
                top_n=min(10, top_n + 1),
                trading_cost_bps=float(config.trading_cost_bps),
                slippage_bps=float(config.slippage_bps),
            ),
        },
        {
            "scenario_id": "balanced",
            "scenario_name": "平衡",
            "description": "在收益与稳定之间折中。",
            "config": BacktestConfig(
                rebalance_days=max(1, int(config.rebalance_days)),
                top_n=min(10, max(2, top_n)),
                trading_cost_bps=float(config.trading_cost_bps),
                slippage_bps=float(config.slippage_bps),
            ),
        },
        {
            "scenario_id": "aggressive",
            "scenario_name": "进攻",
            "description": "更高调仓频率，追求高弹性收益。",
            "config": BacktestConfig(
                rebalance_days=max(1, int(config.rebalance_days) - 2),
                top_n=max(1, top_n - 1),
                trading_cost_bps=float(config.trading_cost_bps),
                slippage_bps=float(config.slippage_bps),
            ),
        },
        {
            "scenario_id": "low-cost",
            "scenario_name": "低成本",
            "description": "假设交易摩擦更低，观察策略上限。",
            "config": BacktestConfig(
                rebalance_days=max(1, int(config.rebalance_days)),
                top_n=top_n,
                trading_cost_bps=round(float(config.trading_cost_bps) * 0.6, 2),
                slippage_bps=round(float(config.slippage_bps) * 0.6, 2),
            ),
        },
    ]

    rows: list[dict] = []
    for entry in scenario_configs:
        scenario_result = run_baseline_backtest_from_histories(histories, names=names, config=entry["config"])
        summary = scenario_result.get("summary", {})
        rows.append(
            {
                "scenario_id": entry["scenario_id"],
                "scenario_name": entry["scenario_name"],
                "description": entry["description"],
                "rebalance_days": int(entry["config"].rebalance_days),
                "top_n": int(entry["config"].top_n),
                "trading_cost_bps": float(entry["config"].trading_cost_bps),
                "slippage_bps": float(entry["config"].slippage_bps),
                "annual_return": float(summary.get("annual_return", 0.0)),
                "sharpe": float(summary.get("sharpe", 0.0)),
                "max_drawdown": float(summary.get("max_drawdown", 0.0)),
                "calmar": float(summary.get("calmar", 0.0)),
                "total_return": float(summary.get("total_return", 0.0)),
                "excess_return": float(summary.get("excess_return", 0.0)),
                "sortino": float(summary.get("sortino", 0.0)),
                "information_ratio": float(summary.get("information_ratio", 0.0)),
                "avg_turnover": float(summary.get("avg_turnover", 0.0)),
                "total_cost": float(summary.get("total_cost", 0.0)),
                "score": _scenario_selection_score(summary),
            }
        )

    rows = sorted(rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    champion = rows[0] if rows else None
    return {
        "analysis_id": f"scenario-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows,
        "champion": champion,
    }
