from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.config import get_settings
from app.services.backtest import (
    REBALANCE_DAYS,
    SLIPPAGE_BPS,
    TOP_N,
    TRADING_COST_BPS,
    BacktestConfig,
    run_baseline_backtest_from_histories,
    run_backtest_monte_carlo_from_histories,
    run_backtest_scenarios_from_histories,
    run_backtest_sensitivity_from_histories,
    run_backtest_stability_from_histories,
)
from app.services.data_provider import (
    BaseProvider,
    MockAShareProvider,
    UniverseItem,
    get_common_universe_catalog,
    get_provider,
)
from app.services.factor_engine import build_factor_table_from_histories
from app.services.storage import MarketRepository

DEFAULT_SIGNAL_ACCOUNT_ID = "default"


class MarketService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider: BaseProvider | None = None
        self.provider_init_error = ""
        self.repo = MarketRepository()
        self.last_sync_warning = ""
        self.last_sync_error_detail = ""
        self.active_data_provider = self.settings.data_provider

    def _get_provider(self, provider: BaseProvider | None = None) -> BaseProvider:
        if provider is not None:
            return provider
        custom_universe = [
            UniverseItem(symbol=str(item["symbol"]).zfill(6), name=str(item["name"]))
            for item in self.repo.load_custom_universe()
        ]
        if not custom_universe:
            cached_universe = self.repo.load_universe(provider=self.settings.data_provider)
            if not cached_universe.empty:
                custom_universe = [
                    UniverseItem(symbol=str(row["symbol"]).zfill(6), name=str(row["name"]))
                    for _, row in cached_universe.iterrows()
                ]
        try:
            self.provider = get_provider(custom_universe=custom_universe)
            self.provider_init_error = ""
            return self.provider
        except Exception as exc:
            self.provider_init_error = str(exc)
            raise

    def _friendly_warning(self, exc: Exception, *, using_cache: bool = False) -> str:
        detail = str(exc).lower()
        provider_name = self.settings.data_provider
        if provider_name == "tushare":
            if any(keyword in detail for keyword in ["token", "积分", "permission", "权限", "missing"]):
                reason = "tushare token 缺失、无效，或当前账号没有对应接口权限。"
            elif "stock_basic" in detail and any(keyword in detail for keyword in ["频率超限", "rate limit", "频次"]):
                reason = "tushare 的 stock_basic 调用频率超限。请确认后端已经重启到最新代码，当前低权限模式已不再依赖该接口。"
            elif "no rows" in detail:
                reason = "tushare daily 接口返回空结果，当前账号可能没有日线权限，或请求区间内没有可用数据。"
            elif any(keyword in detail for keyword in ["proxy", "timed out", "name resolution", "failed to resolve"]):
                reason = "tushare 请求网络异常，当前环境无法稳定连接上游接口。"
            else:
                reason = "tushare 同步失败。"
        elif provider_name == "akshare":
            if any(keyword in detail for keyword in ["proxy", "timed out", "name resolution", "failed to resolve"]):
                reason = "akshare 上游行情接口当前不可达。"
            else:
                reason = "akshare 同步失败。"
        else:
            reason = f"{provider_name} 同步失败。"

        if using_cache:
            return f"{reason} 当前继续使用本地缓存数据。"
        return f"{reason} 系统已自动切换为本地模拟数据，这不影响你继续调试前后端流程。"

    def _mock_fallback(self, exc: Exception) -> dict:
        mock_provider = MockAShareProvider()
        self.active_data_provider = "mock-fallback"
        self.last_sync_warning = self._friendly_warning(exc, using_cache=False)
        self.last_sync_error_detail = str(exc)
        return self.repo.sync_provider_data(mock_provider, "mock-fallback")

    def _sync_with_provider(self, provider: BaseProvider | None = None, fail_if_empty: bool = False) -> dict:
        try:
            self.last_sync_warning = ""
            self.last_sync_error_detail = ""
            self.active_data_provider = self.settings.data_provider
            active_provider = self._get_provider(provider)
            return self.repo.sync_provider_data(active_provider, self.settings.data_provider)
        except Exception as exc:  # pragma: no cover - depends on external network/provider state
            if self.repo.has_data(provider=self.settings.data_provider):
                self.active_data_provider = self.settings.data_provider
                self.last_sync_warning = self._friendly_warning(exc, using_cache=True)
                self.last_sync_error_detail = str(exc)
                latest = self.repo.load_latest_sync(provider=self.settings.data_provider)
                run_at = self.repo.record_sync_run(
                    self.settings.data_provider,
                    latest["symbols_synced"],
                    latest["bars_written"],
                    self.last_sync_warning,
                )
                return {
                    "symbols_synced": latest["symbols_synced"],
                    "bars_written": latest["bars_written"],
                    "run_at": run_at,
                    "warning": self.last_sync_warning,
                }
            if self.settings.fallback_to_mock_on_data_error:
                return self._mock_fallback(exc)
            if fail_if_empty:
                raise RuntimeError(
                    "Initial sync failed for provider "
                    f"{self.settings.data_provider}. If you are using a real market data source, "
                    "check your network, token, or proxy settings, or switch DATA_PROVIDER back to mock first. "
                    f"Original error: {exc}"
                ) from exc
            self.last_sync_warning = f"Sync failed with provider {self.settings.data_provider}: {exc}"
            raise RuntimeError(
                "Sync failed for provider "
                f"{self.settings.data_provider}. If you are using a real market data source, "
                "the upstream data source may be unreachable, token authentication may have failed, "
                "or the current network may block the request. "
                f"Original error: {exc}"
            ) from exc

    def ensure_seed_data(self) -> None:
        if not self.repo.has_data(provider=self.active_data_provider):
            self._sync_with_provider(fail_if_empty=True)

    def refresh_market_data(self) -> dict:
        self.provider = None
        return self._sync_with_provider()

    def get_custom_universe(self) -> list[dict]:
        return self.repo.load_custom_universe()

    def configure_custom_universe(
        self,
        symbols: list[str] | None = None,
        items: list[dict] | None = None,
    ) -> list[dict]:
        normalized = []
        seen: set[str] = set()
        explicit_names: dict[str, str] = {}

        for item in items or []:
            raw_symbol = item.get("symbol", "")
            symbol = "".join(char for char in str(raw_symbol) if char.isdigit())
            if len(symbol) != 6 or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)
            explicit_name = str(item.get("name", "")).strip()
            if explicit_name and explicit_name != symbol:
                explicit_names[symbol] = explicit_name

        for raw_symbol in symbols or []:
            symbol = "".join(char for char in str(raw_symbol) if char.isdigit())
            if len(symbol) != 6 or symbol in seen:
                continue
            seen.add(symbol)
            normalized.append(symbol)

        known_names = self.repo.lookup_symbol_names(normalized)
        catalog_names = {item.symbol: item.name for item in get_common_universe_catalog()}
        resolved_items = [
            {
                "symbol": symbol,
                "name": explicit_names.get(symbol) or known_names.get(symbol) or catalog_names.get(symbol, symbol),
            }
            for symbol in normalized
        ]
        self.repo.save_custom_universe(resolved_items)
        self.provider = None
        return self.repo.load_custom_universe()

    def search_universe_candidates(self, query_text: str, limit: int = 12) -> list[dict]:
        normalized = query_text.strip()
        if not normalized:
            return []

        lower_query = normalized.lower()
        merged: list[dict] = []
        seen: set[str] = set()

        for item in self.repo.load_custom_universe():
            symbol = str(item["symbol"]).zfill(6)
            name = str(item["name"])
            if lower_query in symbol.lower() or lower_query in name.lower():
                if symbol not in seen:
                    merged.append({"symbol": symbol, "name": name})
                    seen.add(symbol)

        for item in self.repo.search_universe_candidates(normalized, limit=limit):
            symbol = str(item["symbol"]).zfill(6)
            name = str(item["name"])
            if symbol not in seen:
                merged.append({"symbol": symbol, "name": name})
                seen.add(symbol)

        for item in get_common_universe_catalog():
            if lower_query in item.symbol.lower() or lower_query in item.name.lower():
                if item.symbol not in seen:
                    merged.append({"symbol": item.symbol, "name": item.name})
                    seen.add(item.symbol)

        if len(merged) < limit and self.settings.data_provider.lower() == "tushare":
            try:
                provider = self._get_provider()
                for item in provider.search_universe(normalized, limit=limit * 3):
                    if item.symbol not in seen:
                        merged.append({"symbol": item.symbol, "name": item.name})
                        seen.add(item.symbol)
                    if len(merged) >= limit:
                        break
            except Exception:
                # Universe search should stay usable even if the upstream
                # catalog query is temporarily unavailable or rate limited.
                pass

        return merged[:limit]

    def _load_names(self) -> dict[str, str]:
        universe = self.repo.load_universe(provider=self.active_data_provider)
        if universe.empty:
            return {}
        return {str(row["symbol"]): str(row["name"]) for _, row in universe.iterrows()}

    def _load_histories(self, limit: int = 180) -> dict[str, pd.DataFrame]:
        self.ensure_seed_data()
        return self.repo.load_all_histories(limit=limit, provider=self.active_data_provider)

    def get_stock_snapshot(self) -> pd.DataFrame:
        histories = self._load_histories(limit=40)
        names = self._load_names()
        factors = self.get_factor_table()
        factor_cols = factors[["symbol", "momentum_20", "volatility_20", "score"]] if not factors.empty else pd.DataFrame()

        rows = []
        for symbol, bars in histories.items():
            if len(bars) < 2:
                continue
            latest = bars.iloc[-1]
            prev = bars.iloc[-2]
            rows.append(
                {
                    "symbol": symbol,
                    "name": names.get(symbol, symbol),
                    "latest_price": round(float(latest["close"]), 2),
                    "pct_change": round((float(latest["close"]) / float(prev["close"]) - 1) * 100, 2),
                    "volume": round(float(latest["volume"]), 2),
                }
            )

        snapshot = pd.DataFrame(rows)
        if snapshot.empty or factor_cols.empty:
            return snapshot
        return snapshot.merge(factor_cols, on="symbol", how="left")

    def get_stock_history(self, symbol: str, limit: int = 90) -> pd.DataFrame:
        self.ensure_seed_data()
        return self.repo.load_symbol_history(symbol, limit=limit, provider=self.active_data_provider)

    def get_signal_center(self, candidate_limit: int = 10) -> dict:
        from app.services.modeling import get_model_status, resolve_active_provider

        status = get_model_status()
        active_provider = resolve_active_provider(self.repo, self.settings.data_provider)
        latest_run = self.repo.load_latest_model_run(provider=active_provider)
        if not latest_run:
            return {
                "summary": {
                    "configured_provider": self.settings.data_provider,
                    "active_provider": active_provider,
                    "warnings": ["还没有可用的模型信号，请先同步数据并训练模型。"],
                },
                "review": {"model_run_id": 0, "status": "pending", "note": "", "updated_at": ""},
                "action_items": [],
                "target_positions": [],
                "top_candidates": [],
            }

        daily_settings = self.repo.load_paper_daily_settings(account_id=DEFAULT_SIGNAL_ACCOUNT_ID)
        top_n = int(daily_settings["top_n"]) if daily_settings else 3
        capital_fraction = float(daily_settings["capital_fraction"]) if daily_settings else 0.95
        max_position_weight = float(daily_settings["max_position_weight"]) if daily_settings else 0.35
        min_cash_buffer_ratio = float(daily_settings["min_cash_buffer_ratio"]) if daily_settings else 0.05
        max_turnover_ratio = float(daily_settings["max_turnover_ratio"]) if daily_settings else 1.0
        min_signal_return_pct = float(daily_settings["min_signal_return_pct"]) if daily_settings else 0.01
        min_liquidity_amount = float(daily_settings["min_liquidity_amount"]) if daily_settings else 30_000_000.0
        min_turnover_rate = float(daily_settings["min_turnover_rate"]) if daily_settings else 0.002

        predictions = self.repo.load_latest_predictions(
            limit=max(int(candidate_limit), top_n),
            provider=active_provider,
            model_name=latest_run.get("model_name"),
        )
        current_positions = self.repo.load_paper_positions(account_id=DEFAULT_SIGNAL_ACCOUNT_ID)
        current_symbols = [str(item["symbol"]) for item in current_positions]
        relevant_symbols = sorted(set([*current_symbols, *[str(item["symbol"]) for item in predictions]]))
        price_map = self.repo.load_latest_prices(relevant_symbols, provider=active_provider)
        latest_bar_map: dict[str, dict] = {}
        for symbol in relevant_symbols:
            bars = self.repo.load_symbol_history(symbol, limit=3, provider=active_provider)
            if bars.empty:
                continue
            latest = bars.iloc[-1]
            previous = bars.iloc[-2] if len(bars) > 1 else latest
            prev_close = float(previous.get("close", 0.0) or 0.0)
            latest_close = float(latest.get("close", 0.0) or 0.0)
            latest_bar_map[symbol] = {
                "change_pct": (latest_close / prev_close - 1.0) if prev_close > 0 else 0.0,
                "volume": float(latest.get("volume", 0.0) or 0.0),
                "amount": float(latest.get("amount", 0.0) or 0.0),
                "turnover_rate": float(latest.get("turnover_rate", 0.0) or 0.0),
            }
        if price_map:
            self.repo.refresh_paper_position_prices(price_map, account_id=DEFAULT_SIGNAL_ACCOUNT_ID)
            current_positions = self.repo.load_paper_positions(account_id=DEFAULT_SIGNAL_ACCOUNT_ID)

        account = self.repo.load_paper_account(account_id=DEFAULT_SIGNAL_ACCOUNT_ID)
        position_rows: list[dict] = []
        market_value = 0.0
        current_position_map: dict[str, dict] = {}
        for item in current_positions:
            last_price = float(price_map.get(str(item["symbol"]), item.get("last_price", 0.0)) or 0.0)
            quantity = int(item["quantity"])
            row = {
                **item,
                "last_price": last_price,
                "market_value": float(last_price * quantity),
            }
            market_value += row["market_value"]
            position_rows.append(row)
            current_position_map[str(item["symbol"])] = row

        cash = float(account["cash"])
        equity = float(cash + market_value)
        target_weight = min(max_position_weight, capital_fraction / max(top_n, 1))
        target_value = equity * target_weight if equity > 0 else 0.0

        skipped_target_signals: list[str] = []

        def is_symbol_hard_restricted(symbol: str, name: str) -> tuple[bool, str]:
            latest_bar = latest_bar_map.get(symbol, {})
            latest_volume = float(latest_bar.get("volume", 0.0) or 0.0)
            latest_amount = float(latest_bar.get("amount", 0.0) or 0.0)
            latest_price = float(price_map.get(symbol, 0.0) or 0.0)
            latest_change_pct = float(latest_bar.get("change_pct", 0.0) or 0.0)
            latest_turnover_rate = float(latest_bar.get("turnover_rate", 0.0) or 0.0)
            upper_name = name.upper()
            if "ST" in upper_name or "退" in name:
                return True, "存在 ST/退市风险"
            if latest_price <= 0:
                return True, "缺少可靠最新价"
            if latest_volume <= 0 or latest_amount <= 0:
                return True, "疑似停牌或当日无成交"
            if latest_change_pct >= 0.095:
                return True, f"接近涨停 {latest_change_pct:.2%}"
            if latest_amount < min_liquidity_amount and latest_turnover_rate < min_turnover_rate:
                return True, f"流动性不足（成交额 {latest_amount / 10_000:.0f} 万，换手率 {latest_turnover_rate:.2%}）"
            return False, ""

        filtered_predictions = []
        for item in predictions:
            symbol = str(item["symbol"])
            name = str(item["name"])
            blocked, reason = is_symbol_hard_restricted(symbol, name)
            if blocked:
                skipped_target_signals.append(f"{symbol} {name}：{reason}")
                continue
            filtered_predictions.append(item)
        predictions = filtered_predictions
        if not predictions:
            return {
                "summary": {
                    "configured_provider": self.settings.data_provider,
                    "active_provider": active_provider,
                    "warnings": ["当前信号候选都被盘前约束过滤，请检查行情状态或放宽股票池。", *skipped_target_signals[:5]],
                },
                "review": {"model_run_id": int(latest_run["id"]), "status": "pending", "note": "", "updated_at": ""},
                "action_items": [],
                "target_positions": [],
                "top_candidates": [],
            }

        suggestions: list[dict] = []
        target_rows: list[dict] = []
        target_symbols = [str(item["symbol"]) for item in predictions[:top_n]]
        exited_positions = [
            item for item in position_rows if str(item["symbol"]) not in target_symbols
        ]
        exited_positions = sorted(exited_positions, key=lambda item: float(item.get("market_value", 0.0)), reverse=True)
        exited_brief = "、".join(
            f"{str(item['symbol'])} {str(item['name'])}" for item in exited_positions[:3]
        )
        target_brief = "、".join(
            f"{str(item['symbol'])} {str(item['name'])}" for item in predictions[: min(top_n, 3)]
        )
        ranked_predictions = [item for item in predictions]
        ranked_prediction_map = {int(item["rank"]): item for item in ranked_predictions if item.get("rank") is not None}
        target_prediction_map = {str(item["symbol"]): item for item in predictions[:top_n]}

        def build_comparison_hint(
            *,
            action: str,
            symbol: str,
            rank: int,
            predicted_return: float,
            score: float,
        ) -> str:
            if action in {"买入", "加仓", "持有", "减仓"}:
                next_item = ranked_prediction_map.get(rank + 1)
                prev_item = ranked_prediction_map.get(rank - 1)
                if rank == 1 and next_item:
                    gap = predicted_return - float(next_item["predicted_return_5d"])
                    return (
                        f"当前是本轮最强信号，较第 2 名 {next_item['symbol']} {next_item['name']} "
                        f"高出 {gap:.2%}，Alpha Score 为 {score:.3f}。"
                    )
                if next_item:
                    gap = predicted_return - float(next_item["predicted_return_5d"])
                    return (
                        f"当前排在第 {rank}，领先下一名 {next_item['symbol']} {next_item['name']} "
                        f"{gap:.2%}，Alpha Score 为 {score:.3f}。"
                    )
                if prev_item:
                    gap = float(prev_item["predicted_return_5d"]) - predicted_return
                    return (
                        f"当前位于目标组合尾部，较上一名 {prev_item['symbol']} {prev_item['name']} "
                        f"低 {gap:.2%}，需要更关注盘中确认。"
                    )
            if action == "卖出":
                weakest_target = ranked_predictions[min(top_n - 1, len(ranked_predictions) - 1)] if ranked_predictions else None
                if weakest_target:
                    return (
                        f"这只股票已落出目标组合，当前腾出的仓位主要会让给 "
                        f"{weakest_target['symbol']} {weakest_target['name']} 等更强样本。"
                    )
            return ""

        def build_selection_reason(
            *,
            action: str,
            rank: int,
            predicted_return: float,
            score: float,
            current_weight: float,
            target_weight_value: float,
        ) -> str:
            rank_text = f"排名第 {rank}" if rank and rank < 999 else "当前不在主信号前列"
            edge_text = f"预期 5 日收益 {predicted_return:.2%}，Alpha Score {score:.3f}"
            if action == "买入":
                return f"{rank_text}，首次进入目标组合。{edge_text}。"
            if action == "加仓":
                return f"{rank_text}，仍在目标组合内，当前权重 {current_weight:.1%} 低于目标权重 {target_weight_value:.1%}。{edge_text}。"
            if action == "减仓":
                return f"{rank_text}，仍保留在目标组合，但当前权重 {current_weight:.1%} 高于目标权重 {target_weight_value:.1%}。"
            if action == "卖出":
                return "最新目标组合已不再包含这只股票，建议退出并为新的信号腾挪仓位。"
            return f"{rank_text}，当前仓位与目标仓位基本一致。{edge_text}。"

        def build_replacement_hint(*, action: str, symbol: str) -> str:
            if action in {"买入", "加仓"}:
                buy_index = max(rank_index_by_symbol.get(symbol, 0), 0)
                replaced = exited_positions[min(buy_index, len(exited_positions) - 1)] if exited_positions else None
                if replaced:
                    return f"这部分仓位主要替换自：{replaced['symbol']} {replaced['name']}。"
                return f"仓位腾挪主要来自：{exited_brief}。" if exited_brief else "当前没有明确的调出样本，更多是对强信号做集中配置。"
            if action == "卖出":
                return f"腾出仓位后，优先让位给：{target_brief}。" if target_brief else ""
            if action == "减仓":
                return "减仓后仍保留在目标组合，释放出的仓位会分配给更强信号。"
            return ""

        def build_risk_hint(
            *,
            action: str,
            price: float,
            predicted_return: float,
            current_quantity: int,
        ) -> str:
            if price <= 0:
                return "当前缺少可靠价格，执行前请先核对最新行情。"
            if action in {"买入", "加仓"} and predicted_return < min_signal_return_pct:
                return "预期收益边际不高，若盘中走势偏弱，可以适当降低优先级。"
            if action == "卖出" and current_quantity <= 0:
                return "当前仓位为空，确认是否已有场外处理。"
            if action in {"买入", "加仓"} and current_quantity == 0:
                return "这是新增开仓信号，优先确认是否符合你今天的风险预算。"
            return "执行前结合盘中流动性、涨跌幅和你自己的仓位约束再做最终确认。"

        def build_pretrade_constraint(
            *,
            action: str,
            current_quantity: int,
            target_quantity: int,
            sellable_quantity: int,
            buy_locked_quantity: int,
            price: float,
            predicted_return: float,
            latest_change_pct: float,
            latest_amount: float,
            latest_turnover_rate: float,
        ) -> tuple[str, str, list[str], int]:
            flags: list[str] = []
            executable_quantity = max(abs(int(target_quantity) - int(current_quantity)), 0)
            if price <= 0:
                return (
                    "blocked",
                    "缺少可靠最新价，先刷新行情后再决定是否执行。",
                    ["缺少最新价格"],
                    0,
                )

            def amount_brief(value: float) -> str:
                if value >= 100_000_000:
                    return f"{value / 100_000_000:.2f} 亿"
                if value >= 10_000:
                    return f"{value / 10_000:.0f} 万"
                return f"{value:.0f}"

            if action in {"卖出", "减仓"}:
                executable_quantity = min(executable_quantity or current_quantity, max(sellable_quantity, 0))
                if buy_locked_quantity > 0:
                    flags.append(f"T+1 锁定 {buy_locked_quantity} 股")
                if latest_change_pct <= -0.095:
                    flags.append(f"接近跌停 {latest_change_pct:.2%}")
                    return (
                        "blocked",
                        "该股票今日接近跌停，卖出成交不确定，建议盘中确认是否可执行。",
                        flags,
                        executable_quantity,
                    )
                if executable_quantity <= 0:
                    return (
                        "blocked",
                        "当前可卖数量为 0，今天不能按信号卖出，需等锁定仓位解锁。",
                        flags or ["可卖数量为 0"],
                        0,
                    )
                if executable_quantity < max(abs(int(target_quantity) - int(current_quantity)), 0):
                    flags.append(f"今日最多可卖 {executable_quantity} 股")
                    return (
                        "warning",
                        f"受 T+1 或可卖数量限制，今天最多只能卖出 {executable_quantity} 股。",
                        flags,
                        executable_quantity,
                    )
                if latest_amount > 0 and latest_amount < min_liquidity_amount:
                    flags.append(f"成交额偏低 {amount_brief(latest_amount)}")
                if latest_turnover_rate > 0 and latest_turnover_rate < min_turnover_rate:
                    flags.append(f"换手率偏低 {latest_turnover_rate:.2%}")
                if len(flags) > 0 and flags != [f"当前可卖 {sellable_quantity} 股，可按计划执行卖出动作。"]:
                    return (
                        "warning",
                        f"卖出侧存在盘前约束，当前可卖 {sellable_quantity} 股，建议结合流动性再确认。",
                        flags,
                        executable_quantity,
                    )
                return (
                    "normal",
                    f"当前可卖 {sellable_quantity} 股，可按计划执行卖出动作。",
                    flags or ["可按正常流程卖出"],
                    executable_quantity,
                )

            if action in {"买入", "加仓"}:
                if current_quantity == 0:
                    flags.append("新增开仓")
                if latest_change_pct >= 0.095:
                    flags.append(f"接近涨停 {latest_change_pct:.2%}")
                    return (
                        "blocked",
                        "该股票今日接近涨停，买入成交不确定，建议暂时跳过或等盘中确认。",
                        flags,
                        executable_quantity,
                    )
                if latest_change_pct >= 0.07:
                    flags.append(f"涨幅较大 {latest_change_pct:.2%}")
                if latest_amount > 0 and latest_amount < min_liquidity_amount:
                    flags.append(f"成交额偏低 {amount_brief(latest_amount)}")
                if latest_turnover_rate > 0 and latest_turnover_rate < min_turnover_rate:
                    flags.append(f"换手率偏低 {latest_turnover_rate:.2%}")
                if predicted_return < min_signal_return_pct:
                    flags.append("预期收益边际较低")
                    return (
                        "warning",
                        "这是低边际收益买入信号，建议只在仓位和风险预算允许时执行。",
                        flags,
                        executable_quantity,
                    )
                if len(flags) > (1 if "新增开仓" in flags else 0):
                    return (
                        "warning",
                        "买入侧存在盘前约束，建议先确认流动性和盘中涨幅，再决定是否执行。",
                        flags,
                        executable_quantity,
                    )
                return (
                    "normal",
                    "买入侧没有持仓约束，可结合资金和盘中流动性正常执行。",
                    flags or ["可按正常流程买入"],
                    executable_quantity,
                )

            return (
                "normal",
                "当前仓位与目标仓位接近，无需额外处理。",
                ["无需动作"],
                0,
            )

        def build_target_quantity(price: float) -> int:
            if price <= 0:
                return 0
            raw_quantity = int(target_value / price)
            return max((raw_quantity // 100) * 100, 0)

        eligible_predictions: list[dict] = []
        for item in predictions:
            symbol = str(item["symbol"])
            name = str(item["name"])
            restricted, reason = is_symbol_hard_restricted(symbol, name)
            if restricted:
                skipped_target_signals.append(f"{symbol} {name}：{reason}")
                continue
            if float(item["predicted_return_5d"]) < min_signal_return_pct:
                skipped_target_signals.append(
                    f"{symbol} {name}：预期收益 {float(item['predicted_return_5d']):.2%} 低于筛选阈值 {min_signal_return_pct:.2%}"
                )
                continue
            eligible_predictions.append(item)
            if len(eligible_predictions) >= top_n:
                break

        rank_index_by_symbol = {
            str(item["symbol"]): index
            for index, item in enumerate(eligible_predictions)
        }

        for item in eligible_predictions:
            symbol = str(item["symbol"])
            current = current_position_map.get(symbol)
            price = float(price_map.get(symbol, current.get("last_price", 0.0) if current else 0.0) or 0.0)
            latest_bar = latest_bar_map.get(symbol, {})
            current_quantity = int(current["quantity"]) if current else 0
            sellable_quantity = int(current.get("sellable_quantity", current_quantity)) if current else 0
            buy_locked_quantity = int(current.get("buy_locked_quantity", 0)) if current else 0
            current_value = float(current_quantity * price)
            current_weight = float(current_value / equity) if equity > 0 else 0.0
            target_quantity = build_target_quantity(price)
            delta_quantity = target_quantity - current_quantity
            action = "持有"
            note = "已在目标组合中，建议继续持有。"
            if current_quantity == 0 and target_quantity > 0:
                action = "买入"
                note = "模型新进前排信号，建议纳入目标组合。"
            elif delta_quantity > 0:
                action = "加仓"
                note = "当前仓位低于建议目标仓位。"
            elif delta_quantity < 0:
                action = "减仓"
                note = "当前仓位高于建议目标仓位。"
            constraint_level, execution_constraint, pretrade_flags, executable_quantity = build_pretrade_constraint(
                action=action,
                current_quantity=current_quantity,
                target_quantity=target_quantity,
                sellable_quantity=sellable_quantity,
                buy_locked_quantity=buy_locked_quantity,
                price=price,
                predicted_return=float(item["predicted_return_5d"]),
                latest_change_pct=float(latest_bar.get("change_pct", 0.0) or 0.0),
                latest_amount=float(latest_bar.get("amount", 0.0) or 0.0),
                latest_turnover_rate=float(latest_bar.get("turnover_rate", 0.0) or 0.0),
            )
            target_row = {
                "symbol": symbol,
                "name": str(item["name"]),
                "action": action,
                "predicted_return_5d": float(item["predicted_return_5d"]),
                "score": float(item["score"]),
                "rank": int(item["rank"]),
                "current_quantity": current_quantity,
                "target_quantity": target_quantity,
                "delta_quantity": delta_quantity,
                "last_price": price,
                "current_weight": current_weight,
                    "target_weight": target_weight,
                    "suggested_value": target_value,
                    "note": note,
                    "selection_reason": build_selection_reason(
                        action=action,
                        rank=int(item["rank"]),
                        predicted_return=float(item["predicted_return_5d"]),
                        score=float(item["score"]),
                        current_weight=current_weight,
                        target_weight_value=target_weight,
                    ),
                    "comparison_hint": build_comparison_hint(
                        action=action,
                        symbol=symbol,
                        rank=int(item["rank"]),
                        predicted_return=float(item["predicted_return_5d"]),
                        score=float(item["score"]),
                    ),
                    "replacement_hint": build_replacement_hint(action=action, symbol=symbol),
                    "risk_hint": build_risk_hint(
                        action=action,
                        price=price,
                        predicted_return=float(item["predicted_return_5d"]),
                        current_quantity=current_quantity,
                    ),
                    "sellable_quantity": sellable_quantity,
                    "buy_locked_quantity": buy_locked_quantity,
                    "executable_quantity": executable_quantity,
                    "constraint_level": constraint_level,
                    "execution_constraint": execution_constraint,
                    "pretrade_flags": pretrade_flags,
                }
            target_rows.append(target_row)
            suggestions.append(target_row)

        for item in position_rows:
            symbol = str(item["symbol"])
            if symbol in target_symbols:
                continue
            price = float(item["last_price"])
            latest_bar = latest_bar_map.get(symbol, {})
            current_quantity = int(item["quantity"])
            sellable_quantity = int(item.get("sellable_quantity", current_quantity))
            buy_locked_quantity = int(item.get("buy_locked_quantity", 0))
            current_value = float(current_quantity * price)
            current_weight = float(current_value / equity) if equity > 0 else 0.0
            constraint_level, execution_constraint, pretrade_flags, executable_quantity = build_pretrade_constraint(
                action="卖出",
                current_quantity=current_quantity,
                target_quantity=0,
                sellable_quantity=sellable_quantity,
                buy_locked_quantity=buy_locked_quantity,
                price=price,
                predicted_return=0.0,
                latest_change_pct=float(latest_bar.get("change_pct", 0.0) or 0.0),
                latest_amount=float(latest_bar.get("amount", 0.0) or 0.0),
                latest_turnover_rate=float(latest_bar.get("turnover_rate", 0.0) or 0.0),
            )
            suggestions.append(
                {
                    "symbol": symbol,
                    "name": str(item["name"]),
                    "action": "卖出",
                    "predicted_return_5d": 0.0,
                    "score": 0.0,
                    "rank": 999,
                    "current_quantity": current_quantity,
                    "target_quantity": 0,
                    "delta_quantity": -current_quantity,
                    "last_price": price,
                    "current_weight": current_weight,
                    "target_weight": 0.0,
                    "suggested_value": 0.0,
                    "note": "已不在最新目标组合中，可考虑卖出腾挪仓位。",
                    "selection_reason": build_selection_reason(
                        action="卖出",
                        rank=999,
                        predicted_return=0.0,
                        score=0.0,
                        current_weight=current_weight,
                        target_weight_value=0.0,
                    ),
                    "comparison_hint": build_comparison_hint(
                        action="卖出",
                        symbol=symbol,
                        rank=999,
                        predicted_return=0.0,
                        score=0.0,
                    ),
                    "replacement_hint": build_replacement_hint(action="卖出", symbol=symbol),
                    "risk_hint": build_risk_hint(
                        action="卖出",
                        price=price,
                        predicted_return=0.0,
                        current_quantity=current_quantity,
                    ),
                    "sellable_quantity": sellable_quantity,
                    "buy_locked_quantity": buy_locked_quantity,
                    "executable_quantity": executable_quantity,
                    "constraint_level": constraint_level,
                    "execution_constraint": execution_constraint,
                    "pretrade_flags": pretrade_flags,
                }
            )

        action_order = {"卖出": 0, "买入": 1, "加仓": 2, "减仓": 3, "持有": 4}
        suggestions = sorted(suggestions, key=lambda row: (action_order.get(str(row["action"]), 9), int(row["rank"]), str(row["symbol"])))

        review = self.repo.load_signal_review(int(latest_run["id"])) or {
            "model_run_id": int(latest_run["id"]),
            "status": "pending",
            "note": "",
            "execution_items": [],
            "execution_summary": {},
            "updated_at": "",
        }

        existing_execution_map = {
            str(item.get("symbol", "")).zfill(6): item
            for item in review.get("execution_items", [])
            if item.get("symbol")
        }
        merged_execution_items = []
        for item in suggestions:
            if item["action"] == "持有":
                continue
            existing = existing_execution_map.get(str(item["symbol"]))
            merged_execution_items.append(
                {
                    "symbol": str(item["symbol"]),
                    "name": str(item["name"]),
                    "action": str(item["action"]),
                    "planned_quantity": abs(int(item["delta_quantity"])) if item["action"] in {"卖出", "减仓"} else int(item["target_quantity"]),
                    "executed_quantity": int(existing.get("executed_quantity", 0)) if existing else 0,
                    "executed_price": float(existing.get("executed_price", 0.0)) if existing else 0.0,
                    "note": str(existing.get("note", "")) if existing else "",
                }
            )
        if merged_execution_items:
            review["execution_items"] = merged_execution_items
            review["execution_summary"] = {
                "items_count": len(merged_execution_items),
                "executed_items_count": sum(1 for item in merged_execution_items if item["executed_quantity"] > 0),
                "executed_buy_amount": round(
                    sum(
                        item["executed_quantity"] * item["executed_price"]
                        for item in merged_execution_items
                        if item["action"] in {"买入", "加仓"}
                    ),
                    2,
                ),
                "executed_sell_amount": round(
                    sum(
                        item["executed_quantity"] * item["executed_price"]
                        for item in merged_execution_items
                        if item["action"] in {"卖出", "减仓"}
                    ),
                    2,
                ),
            }

        warnings: list[str] = []
        if status.get("active_data_provider") != status.get("provider"):
            warnings.append(f"当前使用的是 {status.get('active_data_provider')} 数据，不是配置中的 {status.get('provider')}。")
        if status.get("notes"):
            warnings.append(str(status["notes"]))
        if review["status"] == "pending":
            warnings.append("这次信号还没有被你标记为已执行或已忽略。")
        if skipped_target_signals:
            preview = "；".join(skipped_target_signals[:3])
            suffix = " 等更多样本" if len(skipped_target_signals) > 3 else ""
            warnings.append(f"已有 {len(skipped_target_signals)} 只候选因盘前约束被跳过：{preview}{suffix}。")

        avg_return = (
            float(sum(float(item["predicted_return_5d"]) for item in predictions[:top_n]) / len(predictions[:top_n]))
            if predictions[:top_n]
            else 0.0
        )

        planned_buy_amount = round(
            sum(
                max(int(item.get("delta_quantity", 0)), 0) * float(item.get("last_price", 0.0) or 0.0)
                for item in suggestions
                if item["action"] in {"买入", "加仓"}
            ),
            2,
        )
        planned_sell_amount = round(
            sum(
                abs(min(int(item.get("delta_quantity", 0)), 0)) * float(item.get("last_price", 0.0) or 0.0)
                for item in suggestions
                if item["action"] in {"卖出", "减仓"}
            ),
            2,
        )
        estimated_cash_after_signal = round(cash + planned_sell_amount - planned_buy_amount, 2)
        min_cash_buffer_required = round(equity * min_cash_buffer_ratio, 2)
        estimated_turnover_ratio = (
            round((planned_buy_amount + planned_sell_amount) / equity, 4) if equity > 0 else 0.0
        )
        new_position_count = sum(1 for item in suggestions if item["action"] == "买入")
        exit_position_count = sum(1 for item in suggestions if item["action"] == "卖出")
        target_utilization_ratio = round((len(target_rows) / top_n), 4) if top_n > 0 else 0.0
        target_weights = [
            float(item.get("target_weight", 0.0) or 0.0)
            for item in target_rows
            if float(item.get("target_weight", 0.0) or 0.0) > 0
        ]
        sorted_target_weights = sorted(target_weights, reverse=True)
        max_target_position_weight = round(sorted_target_weights[0], 4) if sorted_target_weights else 0.0
        top_two_target_weight_share = round(sum(sorted_target_weights[:2]), 4) if sorted_target_weights else 0.0
        effective_holding_count = (
            round(1 / sum(weight * weight for weight in target_weights), 2)
            if target_weights and sum(weight * weight for weight in target_weights) > 0
            else 0.0
        )
        def classify_board_bucket(symbol: str) -> str:
            if symbol.startswith(("688", "689")):
                return "科创板"
            if symbol.startswith(("300", "301")):
                return "创业板"
            if symbol.startswith(("8", "4", "92")):
                return "北交所"
            if symbol.startswith(("600", "601", "603", "605")):
                return "沪主板"
            if symbol.startswith(("000", "001", "002", "003")):
                return "深主板"
            return "其他"

        def format_board_exposure(board_weights: dict[str, float]) -> list[str]:
            return [
                f"{name} {share:.0%}"
                for name, share in sorted(board_weights.items(), key=lambda pair: pair[1], reverse=True)
                if share > 0
            ]

        board_weights: dict[str, float] = {}
        for item in target_rows:
            weight = float(item.get("target_weight", 0.0) or 0.0)
            if weight <= 0:
                continue
            bucket = classify_board_bucket(str(item.get("symbol", "")))
            board_weights[bucket] = board_weights.get(bucket, 0.0) + weight

        current_board_weights: dict[str, float] = {}
        for item in position_rows:
            quantity = int(item.get("quantity", 0) or 0)
            if quantity <= 0:
                continue
            current_value = float(quantity) * float(item.get("last_price", 0.0) or 0.0)
            if equity <= 0 or current_value <= 0:
                continue
            bucket = classify_board_bucket(str(item.get("symbol", "")))
            current_board_weights[bucket] = current_board_weights.get(bucket, 0.0) + (current_value / equity)

        sorted_board_weights = sorted(board_weights.items(), key=lambda pair: pair[1], reverse=True)
        sorted_current_board_weights = sorted(current_board_weights.items(), key=lambda pair: pair[1], reverse=True)
        dominant_board_bucket = sorted_board_weights[0][0] if sorted_board_weights else ""
        dominant_board_share = round(sorted_board_weights[0][1], 4) if sorted_board_weights else 0.0
        board_exposure_breakdown = format_board_exposure(board_weights)
        current_board_exposure_breakdown = format_board_exposure(current_board_weights)

        top_entry_targets = [
            item for item in target_rows if str(item.get("action", "")) in {"买入", "加仓"}
        ]
        top_exit_positions = [
            item for item in suggestions if str(item.get("action", "")) == "卖出"
        ]
        entry_brief = "、".join(f"{item['symbol']} {item['name']}" for item in top_entry_targets[:3])
        exit_brief = "、".join(f"{item['symbol']} {item['name']}" for item in top_exit_positions[:3])
        portfolio_transition_summary = (
            f"本次计划退出 {exit_position_count} 只，新增 {new_position_count} 只，目标持仓维持 {len(target_rows)} 只。"
        )
        if exit_brief or entry_brief:
            transition_parts = [portfolio_transition_summary]
            if exit_brief:
                transition_parts.append(f"主要退出：{exit_brief}")
            if entry_brief:
                transition_parts.append(f"优先换入：{entry_brief}")
            portfolio_transition_summary = "；".join(transition_parts) + "。"

        before_top = sorted_current_board_weights[0] if sorted_current_board_weights else None
        after_top = sorted_board_weights[0] if sorted_board_weights else None
        if before_top and after_top:
            if before_top[0] == after_top[0]:
                board_shift_summary = (
                    f"主暴露板块仍为 {after_top[0]}，占比从 {before_top[1]:.0%} 变化到 {after_top[1]:.0%}。"
                )
            else:
                board_shift_summary = (
                    f"主暴露板块由 {before_top[0]} {before_top[1]:.0%} 切换到 {after_top[0]} {after_top[1]:.0%}。"
                )
        elif after_top:
            board_shift_summary = f"执行后组合主要暴露在 {after_top[0]}，占比约 {after_top[1]:.0%}。"
        elif before_top:
            board_shift_summary = f"当前组合主要暴露在 {before_top[0]}，占比约 {before_top[1]:.0%}。"
        else:
            board_shift_summary = "当前没有足够的持仓暴露数据。"

        portfolio_constraint_level = "normal"
        portfolio_constraint_note = "组合层风险约束正常。"
        if estimated_cash_after_signal < min_cash_buffer_required:
            portfolio_constraint_level = "warning"
            portfolio_constraint_note = (
                f"执行后预计现金 {estimated_cash_after_signal:,.0f}，低于最低缓冲 {min_cash_buffer_required:,.0f}。"
            )
            warnings.append(portfolio_constraint_note)
        elif estimated_turnover_ratio > max_turnover_ratio:
            portfolio_constraint_level = "warning"
            portfolio_constraint_note = (
                f"预计换手率 {estimated_turnover_ratio:.0%}，高于上限 {max_turnover_ratio:.0%}。"
            )
            warnings.append(portfolio_constraint_note)
        elif target_utilization_ratio < 0.8:
            portfolio_constraint_level = "warning"
            portfolio_constraint_note = (
                f"目标仓位只完成 {target_utilization_ratio:.0%}，当前候选经过滤后不足，组合分散度偏低。"
            )
            warnings.append(portfolio_constraint_note)
        elif top_two_target_weight_share >= 0.6 and effective_holding_count < max(float(top_n - 1), 2.5):
            portfolio_constraint_level = "warning"
            portfolio_constraint_note = (
                f"前两大目标持仓占比 {top_two_target_weight_share:.0%}，有效持仓约 {effective_holding_count:.2f} 只，组合集中度偏高。"
            )
            warnings.append(portfolio_constraint_note)
        elif dominant_board_share >= 0.8 and dominant_board_bucket:
            portfolio_constraint_level = "warning"
            portfolio_constraint_note = (
                f"{dominant_board_bucket} 占目标组合 {dominant_board_share:.0%}，板块暴露过于集中。"
            )
            warnings.append(portfolio_constraint_note)
        elif estimated_turnover_ratio > max_turnover_ratio * 0.8:
            portfolio_constraint_note = (
                f"预计换手率 {estimated_turnover_ratio:.0%}，接近上限 {max_turnover_ratio:.0%}。"
            )
        elif new_position_count >= max(top_n - 1, 2):
            portfolio_constraint_note = (
                f"本次预计新增 {new_position_count} 只、退出 {exit_position_count} 只，组合变化较大，建议加强人工复核。"
            )
        elif top_two_target_weight_share >= 0.55:
            portfolio_constraint_note = (
                f"前两大目标持仓占比 {top_two_target_weight_share:.0%}，有效持仓约 {effective_holding_count:.2f} 只，建议关注组合集中度。"
            )
        elif dominant_board_share >= 0.65 and dominant_board_bucket:
            portfolio_constraint_note = (
                f"{dominant_board_bucket} 占目标组合 {dominant_board_share:.0%}，建议关注板块暴露是否过于集中。"
            )

        data_quality_level = "normal"
        data_quality_note = "最近同步和信号生成状态正常。"
        data_quality_checks: list[str] = []
        last_sync_at = str(status.get("last_sync_at", "") or "")
        signal_trade_date = str(predictions[0]["trade_date"]) if predictions else ""
        universe_size = int(status.get("universe_size", 0) or 0)
        total_bars = int(status.get("total_bars", 0) or 0)
        bars_per_symbol = round((total_bars / universe_size), 1) if universe_size > 0 else 0.0

        if not last_sync_at:
            data_quality_level = "warning"
            data_quality_note = "尚未记录最近同步时间，请先确认数据是否已经手动刷新。"
            data_quality_checks.append("最近同步时间缺失")
        else:
            try:
                sync_date = datetime.fromisoformat(last_sync_at).date().isoformat()
                if signal_trade_date and sync_date < signal_trade_date:
                    data_quality_level = "warning"
                    data_quality_note = f"最近同步日期 {sync_date} 早于信号交易日 {signal_trade_date}，请确认数据是否已更新。"
                    data_quality_checks.append("同步日期早于信号交易日")
            except ValueError:
                data_quality_checks.append("最近同步时间格式异常")

        if universe_size <= 0:
            data_quality_level = "warning"
            data_quality_note = "当前股票池为空，信号和训练结果可能不完整。"
            data_quality_checks.append("股票池为空")
        elif universe_size < top_n:
            data_quality_level = "warning"
            data_quality_note = f"当前股票池只有 {universe_size} 只，低于目标持仓数 {top_n}。"
            data_quality_checks.append("股票池规模不足")

        if total_bars <= 0:
            data_quality_level = "warning"
            data_quality_note = "历史K线为空，当前信号缺少可靠行情基础。"
            data_quality_checks.append("历史K线为空")
        elif bars_per_symbol < 60:
            data_quality_level = "warning"
            data_quality_note = f"平均每只股票仅有 {bars_per_symbol:.0f} 条K线，历史样本偏浅。"
            data_quality_checks.append("历史样本偏浅")

        if len(predictions) < top_n:
            data_quality_level = "warning"
            data_quality_note = f"当前仅生成 {len(predictions)} 只候选，低于目标持仓数 {top_n}。"
            data_quality_checks.append("候选数量不足")

        if status.get("active_data_provider") != status.get("provider"):
            if data_quality_level == "normal":
                data_quality_note = (
                    f"当前使用 {status.get('active_data_provider')} 数据，和配置数据源 {status.get('provider')} 不一致。"
                )
            data_quality_checks.append("数据源已回退")

        enriched_candidates = []
        for item in predictions:
            symbol = str(item["symbol"])
            enriched_candidates.append(
                {
                    **item,
                    "last_price": float(price_map.get(symbol, 0.0) or 0.0),
                }
            )

        return {
            "summary": {
                "model_run_id": int(latest_run["id"]),
                "model_name": str(latest_run.get("model_name", "")),
                "active_provider": active_provider,
                "configured_provider": self.settings.data_provider,
                "data_status_note": str(status.get("notes", "")),
                "signal_trade_date": str(predictions[0]["trade_date"]) if predictions else "",
                "generated_at": str(latest_run.get("run_at", "")),
                "last_sync_at": str(status.get("last_sync_at", "") or ""),
                "universe_size": int(status.get("universe_size", 0) or 0),
                "total_bars": int(status.get("total_bars", 0) or 0),
                "top_n": top_n,
                "candidate_count": len(predictions),
                "current_position_count": len(position_rows),
                "target_position_count": len(target_rows),
                "account_equity": equity,
                "account_cash": cash,
                "capital_fraction": capital_fraction,
                "target_weight_per_position": target_weight,
                "avg_predicted_return_5d": avg_return,
                "estimated_turnover_count": sum(1 for item in suggestions if item["action"] != "持有"),
                "planned_buy_amount": planned_buy_amount,
                "planned_sell_amount": planned_sell_amount,
                "estimated_cash_after_signal": estimated_cash_after_signal,
                "min_cash_buffer_required": min_cash_buffer_required,
                "estimated_turnover_ratio": estimated_turnover_ratio,
                "max_turnover_ratio": max_turnover_ratio,
                "target_utilization_ratio": target_utilization_ratio,
                "new_position_count": new_position_count,
                "exit_position_count": exit_position_count,
                "max_target_position_weight": max_target_position_weight,
                "top_two_target_weight_share": top_two_target_weight_share,
                "effective_holding_count": effective_holding_count,
                "current_board_exposure_breakdown": current_board_exposure_breakdown,
                "dominant_board_bucket": dominant_board_bucket,
                "dominant_board_share": dominant_board_share,
                "board_exposure_breakdown": board_exposure_breakdown,
                "portfolio_transition_summary": portfolio_transition_summary,
                "board_shift_summary": board_shift_summary,
                "portfolio_constraint_level": portfolio_constraint_level,
                "portfolio_constraint_note": portfolio_constraint_note,
                "data_quality_level": data_quality_level,
                "data_quality_note": data_quality_note,
                "data_quality_checks": data_quality_checks,
                "review_status": str(review["status"]),
                "warnings": warnings,
            },
            "review": review,
            "action_items": suggestions,
            "target_positions": target_rows,
            "top_candidates": enriched_candidates,
        }

    def get_signal_history(self, limit: int = 12) -> list[dict]:
        from app.services.modeling import resolve_active_provider

        active_provider = resolve_active_provider(self.repo, self.settings.data_provider)
        return self.repo.load_recent_signal_batches(limit=limit, top_n=5, provider=active_provider)

    def save_signal_review(
        self,
        model_run_id: int,
        status: str,
        note: str = "",
        execution_items: list[dict] | None = None,
    ) -> dict:
        return self.repo.save_signal_review(
            model_run_id=model_run_id,
            status=status,
            note=note,
            execution_items=execution_items,
        )

    def get_factor_table(self) -> pd.DataFrame:
        histories = self._load_histories(limit=90)
        names = self._load_names()
        return build_factor_table_from_histories(histories, names=names)

    def get_training_inputs(self, limit: int = 220) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        histories = self._load_histories(limit=limit)
        names = self._load_names()
        return histories, names

    def get_backtest(
        self,
        *,
        rebalance_days: int | None = None,
        top_n: int | None = None,
        trading_cost_bps: float | None = None,
        slippage_bps: float | None = None,
    ) -> dict:
        histories = self._load_histories(limit=90)
        names = self._load_names()
        config = BacktestConfig(
            rebalance_days=rebalance_days or REBALANCE_DAYS,
            top_n=top_n or TOP_N,
            trading_cost_bps=trading_cost_bps if trading_cost_bps is not None else TRADING_COST_BPS,
            slippage_bps=slippage_bps if slippage_bps is not None else SLIPPAGE_BPS,
        )
        return run_baseline_backtest_from_histories(histories, names=names, config=config)

    def get_backtest_sensitivity(
        self,
        *,
        rebalance_days: int | None = None,
        top_n: int | None = None,
        trading_cost_bps: float | None = None,
        slippage_bps: float | None = None,
        scan_width: int = 1,
    ) -> dict:
        histories = self._load_histories(limit=120)
        names = self._load_names()
        config = BacktestConfig(
            rebalance_days=rebalance_days or REBALANCE_DAYS,
            top_n=top_n or TOP_N,
            trading_cost_bps=trading_cost_bps if trading_cost_bps is not None else TRADING_COST_BPS,
            slippage_bps=slippage_bps if slippage_bps is not None else SLIPPAGE_BPS,
        )
        return run_backtest_sensitivity_from_histories(histories, names=names, base_config=config, scan_width=scan_width)

    def get_backtest_stability(
        self,
        *,
        rebalance_days: int | None = None,
        top_n: int | None = None,
        trading_cost_bps: float | None = None,
        slippage_bps: float | None = None,
        rolling_window: int = 20,
    ) -> dict:
        histories = self._load_histories(limit=120)
        names = self._load_names()
        config = BacktestConfig(
            rebalance_days=rebalance_days or REBALANCE_DAYS,
            top_n=top_n or TOP_N,
            trading_cost_bps=trading_cost_bps if trading_cost_bps is not None else TRADING_COST_BPS,
            slippage_bps=slippage_bps if slippage_bps is not None else SLIPPAGE_BPS,
        )
        return run_backtest_stability_from_histories(
            histories,
            names=names,
            base_config=config,
            rolling_window=rolling_window,
        )

    def get_backtest_monte_carlo(
        self,
        *,
        rebalance_days: int | None = None,
        top_n: int | None = None,
        trading_cost_bps: float | None = None,
        slippage_bps: float | None = None,
        trials: int = 300,
    ) -> dict:
        histories = self._load_histories(limit=120)
        names = self._load_names()
        config = BacktestConfig(
            rebalance_days=rebalance_days or REBALANCE_DAYS,
            top_n=top_n or TOP_N,
            trading_cost_bps=trading_cost_bps if trading_cost_bps is not None else TRADING_COST_BPS,
            slippage_bps=slippage_bps if slippage_bps is not None else SLIPPAGE_BPS,
        )
        return run_backtest_monte_carlo_from_histories(
            histories,
            names=names,
            base_config=config,
            trials=trials,
        )

    def get_backtest_scenarios(
        self,
        *,
        rebalance_days: int | None = None,
        top_n: int | None = None,
        trading_cost_bps: float | None = None,
        slippage_bps: float | None = None,
    ) -> dict:
        histories = self._load_histories(limit=120)
        names = self._load_names()
        config = BacktestConfig(
            rebalance_days=rebalance_days or REBALANCE_DAYS,
            top_n=top_n or TOP_N,
            trading_cost_bps=trading_cost_bps if trading_cost_bps is not None else TRADING_COST_BPS,
            slippage_bps=slippage_bps if slippage_bps is not None else SLIPPAGE_BPS,
        )
        return run_backtest_scenarios_from_histories(
            histories,
            names=names,
            base_config=config,
        )

    def get_market_stats(self) -> dict:
        universe = self.repo.load_universe(provider=self.active_data_provider)
        latest = self.repo.load_latest_sync(provider=self.active_data_provider)
        return {
            "active_data_provider": self.active_data_provider,
            "universe_size": 0 if universe.empty else len(universe),
            "total_bars": self.repo.load_bar_count(provider=self.active_data_provider),
            "last_sync": latest,
            "warning": latest.get("note", "") if latest.get("note") != "sync complete" else "",
        }
