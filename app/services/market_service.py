from __future__ import annotations

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

        predictions = self.repo.load_latest_predictions(
            limit=max(int(candidate_limit), top_n),
            provider=active_provider,
            model_name=latest_run.get("model_name"),
        )
        current_positions = self.repo.load_paper_positions(account_id=DEFAULT_SIGNAL_ACCOUNT_ID)
        current_symbols = [str(item["symbol"]) for item in current_positions]
        relevant_symbols = sorted(set([*current_symbols, *[str(item["symbol"]) for item in predictions]]))
        price_map = self.repo.load_latest_prices(relevant_symbols, provider=active_provider)
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

        suggestions: list[dict] = []
        target_rows: list[dict] = []
        target_symbols = [str(item["symbol"]) for item in predictions[:top_n]]

        def build_target_quantity(price: float) -> int:
            if price <= 0:
                return 0
            raw_quantity = int(target_value / price)
            return max((raw_quantity // 100) * 100, 0)

        for item in predictions[:top_n]:
            symbol = str(item["symbol"])
            current = current_position_map.get(symbol)
            price = float(price_map.get(symbol, current.get("last_price", 0.0) if current else 0.0) or 0.0)
            current_quantity = int(current["quantity"]) if current else 0
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
            }
            target_rows.append(target_row)
            suggestions.append(target_row)

        for item in position_rows:
            symbol = str(item["symbol"])
            if symbol in target_symbols:
                continue
            price = float(item["last_price"])
            current_quantity = int(item["quantity"])
            current_value = float(current_quantity * price)
            current_weight = float(current_value / equity) if equity > 0 else 0.0
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

        avg_return = (
            float(sum(float(item["predicted_return_5d"]) for item in predictions[:top_n]) / len(predictions[:top_n]))
            if predictions[:top_n]
            else 0.0
        )

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
                "signal_trade_date": str(predictions[0]["trade_date"]) if predictions else "",
                "generated_at": str(latest_run.get("run_at", "")),
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
