from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.services.market_service import MarketService
from app.services.modeling import resolve_active_provider
from app.services.storage import MarketRepository
from app.services.training import train_local_model

DEFAULT_ACCOUNT_ID = "default"
DEFAULT_INITIAL_CASH = 1_000_000.0


@dataclass
class PaperRebalanceConfig:
    top_n: int = 3
    capital_fraction: float = 0.95
    board_lot: int = 100
    max_position_weight: float = 0.35
    min_cash_buffer_ratio: float = 0.05
    max_turnover_ratio: float = 1.0
    stop_loss_pct: float = 0.1
    take_profit_pct: float = 0.2
    fill_ratio: float = 1.0
    max_drawdown_limit: float = 0.18
    max_equity_change_limit: float = 0.04
    min_signal_return_pct: float = 0.01
    min_liquidity_amount: float = 30_000_000.0
    min_turnover_rate: float = 0.002


class PaperTradingService:
    def __init__(self, market: MarketService) -> None:
        self.market = market
        self.repo = MarketRepository()

    def _latest_price_map(self, symbols: list[str]) -> dict[str, float]:
        provider = resolve_active_provider(self.repo)
        return self.repo.load_latest_prices(symbols, provider=provider)

    def _compute_account_totals(self, account: dict, positions: list[dict]) -> dict:
        cash = float(account["cash"])
        market_value = float(sum(item["market_value"] for item in positions))
        equity = cash + market_value
        initial_cash = float(account["initial_cash"])
        return {
            "cash": cash,
            "market_value": market_value,
            "equity": equity,
            "total_pnl": float(equity - initial_cash),
            "total_return": float((equity / initial_cash - 1) if initial_cash else 0.0),
            "position_count": len(positions),
        }

    def _mark_positions(self, positions: list[dict]) -> list[dict]:
        price_map = self._latest_price_map([str(item["symbol"]) for item in positions])
        if price_map:
            self.repo.refresh_paper_position_prices(price_map)
            positions = self.repo.load_paper_positions(account_id=DEFAULT_ACCOUNT_ID)

        enriched = []
        for item in positions:
            quantity = int(item["quantity"])
            avg_cost = float(item["avg_cost"])
            last_price = float(item["last_price"])
            market_value = quantity * last_price
            unrealized_pnl = quantity * (last_price - avg_cost)
            enriched.append(
                {
                    **item,
                    "quantity": quantity,
                    "avg_cost": avg_cost,
                    "last_price": last_price,
                    "market_value": float(market_value),
                    "unrealized_pnl": float(unrealized_pnl),
                    "unrealized_pnl_pct": float((last_price / avg_cost - 1) if avg_cost else 0.0),
                }
            )
        return enriched

    def _hydrate_preview_plan(self, plan: dict | None) -> dict | None:
        if not plan:
            return None
        preview = dict(plan.get("preview", {}))
        preview["plan_id"] = int(plan["id"])
        preview["created_at"] = str(plan["created_at"])
        preview["status"] = str(plan["status"])
        return preview

    def _resolve_filled_quantity(self, planned_quantity: int, board_lot: int, fill_ratio: float) -> int:
        if planned_quantity <= 0 or fill_ratio <= 0:
            return 0
        if fill_ratio >= 1:
            return int(planned_quantity)
        raw_quantity = int(planned_quantity * fill_ratio)
        rounded_quantity = (raw_quantity // max(board_lot, 1)) * max(board_lot, 1)
        if rounded_quantity <= 0 and planned_quantity >= board_lot:
            rounded_quantity = min(board_lot, planned_quantity)
        return int(min(rounded_quantity, planned_quantity))

    def _current_trade_date(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _compute_risk_metrics(self, equity: float, equity_curve: list[dict]) -> dict:
        historical_equities = [float(item.get("equity", 0.0)) for item in equity_curve if float(item.get("equity", 0.0)) > 0]
        peak_equity = max([equity, *historical_equities], default=equity)
        current_drawdown = float(max((peak_equity - equity) / peak_equity, 0.0)) if peak_equity > 0 else 0.0
        latest_reference = historical_equities[-1] if historical_equities else equity
        latest_equity_change = float((equity / latest_reference - 1.0)) if latest_reference > 0 else 0.0
        return {
            "peak_equity": float(peak_equity),
            "current_drawdown": current_drawdown,
            "latest_equity_change": latest_equity_change,
        }

    def _log_risk_event(
        self,
        *,
        event_type: str,
        severity: str,
        title: str,
        details: dict,
        note: str = "",
    ) -> None:
        self.repo.record_paper_risk_event(
            account_id=DEFAULT_ACCOUNT_ID,
            event_type=event_type,
            severity=severity,
            title=title,
            details=details,
            note=note,
        )

    def _record_execution_report(
        self,
        *,
        report_type: str,
        title: str,
        account: dict,
        positions: list[dict],
        note: str = "",
        execution: dict | None = None,
        extra: dict | None = None,
    ) -> None:
        summary = {
            "equity": float(account.get("equity", 0.0)),
            "cash": float(account.get("cash", 0.0)),
            "market_value": float(account.get("market_value", 0.0)),
            "total_return": float(account.get("total_return", 0.0)),
            "total_pnl": float(account.get("total_pnl", 0.0)),
            "position_count": int(account.get("position_count", len(positions))),
            "peak_equity": float(account.get("peak_equity", 0.0)),
            "current_drawdown": float(account.get("current_drawdown", 0.0)),
            "latest_equity_change": float(account.get("latest_equity_change", 0.0)),
            "holdings": [
                {
                    "symbol": str(item.get("symbol", "")),
                    "name": str(item.get("name", "")),
                    "quantity": int(item.get("quantity", 0)),
                    "market_value": float(item.get("market_value", 0.0)),
                    "unrealized_pnl_pct": float(item.get("unrealized_pnl_pct", 0.0)),
                }
                for item in positions[:8]
            ],
        }
        if execution:
            summary["execution"] = execution
        if extra:
            summary.update(extra)
        self.repo.record_paper_execution_report(
            account_id=DEFAULT_ACCOUNT_ID,
            report_type=report_type,
            title=title,
            summary=summary,
            note=note,
        )

    def get_account_snapshot(self) -> dict:
        self.repo.unlock_paper_t1_positions(account_id=DEFAULT_ACCOUNT_ID, today=self._current_trade_date())
        account = self.repo.load_paper_account(account_id=DEFAULT_ACCOUNT_ID)
        positions = self._mark_positions(self.repo.load_paper_positions(account_id=DEFAULT_ACCOUNT_ID))
        orders = self.repo.load_paper_orders(account_id=DEFAULT_ACCOUNT_ID, limit=20)
        rebalances = self.repo.load_paper_rebalance_runs(account_id=DEFAULT_ACCOUNT_ID, limit=12)
        equity_curve = self.repo.load_paper_equity_curve(account_id=DEFAULT_ACCOUNT_ID, limit=120)
        risk_events = self.repo.load_paper_risk_events(account_id=DEFAULT_ACCOUNT_ID, limit=16)
        reports = self.repo.load_paper_execution_reports(account_id=DEFAULT_ACCOUNT_ID, limit=12)
        daily_settings = self.repo.load_paper_daily_settings(account_id=DEFAULT_ACCOUNT_ID)
        daily_runs = self.repo.load_paper_daily_runs(account_id=DEFAULT_ACCOUNT_ID, limit=12)
        predictions = self.repo.load_latest_predictions(limit=10, provider=resolve_active_provider(self.repo))
        active_plan = self.repo.load_latest_pending_paper_plan(account_id=DEFAULT_ACCOUNT_ID)
        totals = self._compute_account_totals(account, positions)
        risk_metrics = self._compute_risk_metrics(float(totals["equity"]), equity_curve)

        return {
            "account": {
                **account,
                **totals,
                **risk_metrics,
            },
            "positions": positions,
            "orders": orders,
            "equity_curve": equity_curve,
            "rebalances": rebalances,
            "risk_events": risk_events,
            "reports": reports,
            "daily_settings": daily_settings,
            "daily_runs": daily_runs,
            "signals": predictions,
            "preview": self._hydrate_preview_plan(active_plan),
        }

    def update_daily_settings(
        self,
        *,
        enabled: bool,
        run_time: str,
        auto_sync: bool,
        auto_train: bool,
        auto_rebalance: bool,
        top_n: int,
        capital_fraction: float,
        max_position_weight: float,
        min_cash_buffer_ratio: float,
        max_turnover_ratio: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        fill_ratio: float,
        max_drawdown_limit: float,
        max_equity_change_limit: float,
        min_signal_return_pct: float,
        min_liquidity_amount: float,
        min_turnover_rate: float,
    ) -> dict:
        normalized = self.repo.update_paper_daily_settings(
            account_id=DEFAULT_ACCOUNT_ID,
            enabled=bool(enabled),
            run_time=str(run_time or "15:10"),
            auto_sync=bool(auto_sync),
            auto_train=bool(auto_train),
            auto_rebalance=bool(auto_rebalance),
            top_n=max(1, min(int(top_n), 10)),
            capital_fraction=max(0.1, min(float(capital_fraction), 1.0)),
            max_position_weight=max(0.05, min(float(max_position_weight), 1.0)),
            min_cash_buffer_ratio=max(0.0, min(float(min_cash_buffer_ratio), 0.8)),
            max_turnover_ratio=max(0.0, min(float(max_turnover_ratio), 2.0)),
            stop_loss_pct=max(0.0, min(float(stop_loss_pct), 0.5)),
            take_profit_pct=max(0.0, min(float(take_profit_pct), 1.0)),
            fill_ratio=max(0.0, min(float(fill_ratio), 1.0)),
            max_drawdown_limit=max(0.0, min(float(max_drawdown_limit), 0.8)),
            max_equity_change_limit=max(0.0, min(float(max_equity_change_limit), 0.5)),
            min_signal_return_pct=max(0.0, min(float(min_signal_return_pct), 0.2)),
            min_liquidity_amount=max(0.0, float(min_liquidity_amount)),
            min_turnover_rate=max(0.0, min(float(min_turnover_rate), 0.1)),
        )
        snapshot = self.get_account_snapshot()
        snapshot["daily_settings"] = normalized
        return snapshot

    def run_daily_cycle(
        self,
        *,
        auto_sync: bool | None = None,
        auto_train: bool | None = None,
        auto_rebalance: bool | None = None,
        trigger_source: str = "manual",
        retry_attempt: int = 0,
        start_from_step: str = "sync",
    ) -> dict:
        settings = self.repo.load_paper_daily_settings(account_id=DEFAULT_ACCOUNT_ID)
        ordered_steps = ["sync", "train", "rebalance"]
        start_index = ordered_steps.index(start_from_step) if start_from_step in ordered_steps else 0
        sync_enabled = (settings["auto_sync"] if auto_sync is None else bool(auto_sync)) and start_index <= 0
        train_enabled = (settings["auto_train"] if auto_train is None else bool(auto_train)) and start_index <= 1
        rebalance_enabled = (settings["auto_rebalance"] if auto_rebalance is None else bool(auto_rebalance)) and start_index <= 2
        steps: list[dict] = []
        status = "success"

        def make_step(step: str) -> dict:
            return {
                "step": step,
                "status": "running",
                "message": "",
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": "",
                "duration_ms": 0,
                "error_type": "",
            }

        def finish_step(step: dict, *, status_value: str, message: str, error_type: str = "") -> None:
            finished_at = datetime.now()
            started_at = datetime.strptime(step["started_at"], "%Y-%m-%d %H:%M:%S")
            step["status"] = status_value
            step["message"] = message
            step["finished_at"] = finished_at.strftime("%Y-%m-%d %H:%M:%S")
            step["duration_ms"] = max(int((finished_at - started_at).total_seconds() * 1000), 0)
            step["error_type"] = error_type

        if sync_enabled:
            step = make_step("sync")
            try:
                result = self.market.refresh_market_data()
                finish_step(
                    step,
                    status_value="success",
                    message=result.get("warning") or f"同步 {result.get('symbols_synced', 0)} 只股票，写入 {result.get('bars_written', 0)} 条K线。",
                )
                steps.append(step)
                if result.get("warning"):
                    status = "partial"
                    self._log_risk_event(
                        event_type="daily_sync_warning",
                        severity="warning",
                        title="日更同步带告警完成",
                        details=result,
                        note=str(result.get("warning", "")),
                    )
            except Exception as exc:
                status = "failed"
                finish_step(step, status_value="failed", message=str(exc), error_type=type(exc).__name__)
                steps.append(step)
                self._log_risk_event(
                    event_type="daily_sync_failed",
                    severity="error",
                    title="日更同步失败",
                    details={"error": str(exc)},
                    note="Daily cycle sync step failed.",
                )

        if train_enabled:
            step = make_step("train")
            try:
                provider_name = resolve_active_provider(self.repo)
                histories = self.repo.load_all_histories(limit=240, provider=provider_name)
                names = self.repo.lookup_symbol_names(list(histories.keys()))
                result = train_local_model(
                    histories,
                    names,
                    provider_name=provider_name,
                    configured_provider=self.market.settings.data_provider,
                )
                finish_step(
                    step,
                    status_value="success",
                    message=f"训练完成，当前主模型 {result.get('model_name', '-')}",
                )
                steps.append(step)
            except Exception as exc:
                status = "failed" if status != "partial" else "failed"
                finish_step(step, status_value="failed", message=str(exc), error_type=type(exc).__name__)
                steps.append(step)
                self._log_risk_event(
                    event_type="daily_train_failed",
                    severity="error",
                    title="日更训练失败",
                    details={"error": str(exc)},
                    note="Daily cycle training step failed.",
                )

        execution_meta: dict | None = None
        if rebalance_enabled:
            step = make_step("rebalance")
            try:
                daily_snapshot = self.execute_rebalance(
                    top_n=int(settings["top_n"]),
                    capital_fraction=float(settings["capital_fraction"]),
                    max_position_weight=float(settings["max_position_weight"]),
                    min_cash_buffer_ratio=float(settings["min_cash_buffer_ratio"]),
                    max_turnover_ratio=float(settings["max_turnover_ratio"]),
                    stop_loss_pct=float(settings["stop_loss_pct"]),
                    take_profit_pct=float(settings["take_profit_pct"]),
                    fill_ratio=float(settings["fill_ratio"]),
                    max_drawdown_limit=float(settings["max_drawdown_limit"]),
                    max_equity_change_limit=float(settings["max_equity_change_limit"]),
                )
                execution_meta = daily_snapshot.get("execution")
                finish_step(
                    step,
                    status_value="success",
                    message=f"调仓完成，生成 {int(execution_meta.get('orders_created', 0) if execution_meta else 0)} 笔订单。",
                )
                steps.append(step)
            except Exception as exc:
                if "阻断" in str(exc) or "风控" in str(exc):
                    status = "partial" if status == "success" else status
                    finish_step(step, status_value="blocked", message=str(exc), error_type=type(exc).__name__)
                    steps.append(step)
                    self._log_risk_event(
                        event_type="daily_rebalance_blocked",
                        severity="warning",
                        title="日更调仓被风控阻断",
                        details={"error": str(exc)},
                        note="Daily cycle rebalance blocked by controls.",
                    )
                else:
                    status = "failed"
                    finish_step(step, status_value="failed", message=str(exc), error_type=type(exc).__name__)
                    steps.append(step)
                    self._log_risk_event(
                        event_type="daily_rebalance_failed",
                        severity="error",
                        title="日更调仓失败",
                        details={"error": str(exc)},
                        note="Daily cycle rebalance failed.",
                    )

        note = f"Daily cycle finished via {trigger_source} from {start_from_step}."
        self.repo.record_paper_daily_run(
            account_id=DEFAULT_ACCOUNT_ID,
            run_date=self._current_trade_date(),
            status=status,
            steps=steps,
            note=note,
        )
        snapshot = self.get_account_snapshot()
        self._record_execution_report(
            report_type="daily-cycle",
            title="日更流程已执行",
            account=snapshot["account"],
            positions=snapshot["positions"],
            note=note,
            execution=execution_meta,
            extra={
                "cycle_status": status,
                "steps": steps,
                "trigger_source": trigger_source,
                "retry_attempt": retry_attempt,
                "start_from_step": start_from_step,
            },
        )
        return self.get_account_snapshot()

    def reset_account(self, initial_cash: float = DEFAULT_INITIAL_CASH) -> dict:
        account = self.repo.reset_paper_account(account_id=DEFAULT_ACCOUNT_ID, initial_cash=initial_cash)
        self.repo.record_paper_equity_snapshot(
            account_id=DEFAULT_ACCOUNT_ID,
            cash=float(account["cash"]),
            market_value=0.0,
            equity=float(account["cash"]),
            note="Account reset.",
        )
        snapshot = self.get_account_snapshot()
        self._record_execution_report(
            report_type="reset",
            title="模拟账户已重置",
            account=snapshot["account"],
            positions=snapshot["positions"],
            note="Paper account reset to new initial capital.",
            extra={"initial_cash": float(account["cash"])},
        )
        return self.get_account_snapshot()

    def _build_rebalance_plan(
        self,
        *,
        top_n: int = 3,
        capital_fraction: float = 0.95,
        max_position_weight: float = 0.35,
        min_cash_buffer_ratio: float = 0.05,
        max_turnover_ratio: float = 1.0,
        stop_loss_pct: float = 0.1,
        take_profit_pct: float = 0.2,
        fill_ratio: float = 1.0,
        max_drawdown_limit: float = 0.18,
        max_equity_change_limit: float = 0.04,
        min_signal_return_pct: float = 0.01,
        min_liquidity_amount: float = 30_000_000.0,
        min_turnover_rate: float = 0.002,
    ) -> dict:
        self.repo.unlock_paper_t1_positions(account_id=DEFAULT_ACCOUNT_ID, today=self._current_trade_date())
        account = self.repo.load_paper_account(account_id=DEFAULT_ACCOUNT_ID)
        equity_curve = self.repo.load_paper_equity_curve(account_id=DEFAULT_ACCOUNT_ID, limit=120)
        config = PaperRebalanceConfig(
            top_n=max(1, min(int(top_n), 10)),
            capital_fraction=max(0.1, min(float(capital_fraction), 1.0)),
            max_position_weight=max(0.05, min(float(max_position_weight), 1.0)),
            min_cash_buffer_ratio=max(0.0, min(float(min_cash_buffer_ratio), 0.8)),
            max_turnover_ratio=max(0.0, min(float(max_turnover_ratio), 2.0)),
            stop_loss_pct=max(0.0, min(float(stop_loss_pct), 0.5)),
            take_profit_pct=max(0.0, min(float(take_profit_pct), 1.0)),
            fill_ratio=max(0.0, min(float(fill_ratio), 1.0)),
            max_drawdown_limit=max(0.0, min(float(max_drawdown_limit), 0.8)),
            max_equity_change_limit=max(0.0, min(float(max_equity_change_limit), 0.5)),
            min_signal_return_pct=max(0.0, min(float(min_signal_return_pct), 0.2)),
            min_liquidity_amount=max(0.0, float(min_liquidity_amount)),
            min_turnover_rate=max(0.0, min(float(min_turnover_rate), 0.1)),
        )
        prediction_pool = self.repo.load_latest_predictions(
            limit=max(config.top_n * 3, config.top_n + 10),
            provider=resolve_active_provider(self.repo),
        )
        if not prediction_pool:
            raise RuntimeError("当前没有可执行的模型信号，请先完成一次模型训练。")

        current_positions = {
            str(item["symbol"]): item
            for item in self.repo.load_paper_positions(account_id=DEFAULT_ACCOUNT_ID)
        }
        signal_pool_symbols = [str(item["symbol"]) for item in prediction_pool]
        relevant_symbols = sorted(set([*signal_pool_symbols, *current_positions.keys()]))
        price_map = self._latest_price_map(relevant_symbols)
        latest_bar_map: dict[str, dict] = {}
        provider = resolve_active_provider(self.repo)
        for symbol in relevant_symbols:
            bars = self.repo.load_symbol_history(symbol, limit=3, provider=provider)
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
        missing = [symbol for symbol in signal_pool_symbols[: config.top_n] if symbol not in price_map]
        if missing:
            raise RuntimeError(f"缺少这些股票的最新价格，无法执行模拟调仓：{', '.join(missing)}")

        cash = float(account["cash"])
        current_market_value = 0.0
        forced_exit_reasons: dict[str, str] = {}
        locked_position_reasons: dict[str, str] = {}
        for symbol, position in current_positions.items():
            last_price = float(price_map.get(symbol, position.get("last_price", 0.0)))
            current_market_value += int(position["quantity"]) * last_price
            avg_cost = float(position.get("avg_cost", 0.0))
            pnl_pct = (last_price / avg_cost - 1) if avg_cost else 0.0
            if config.stop_loss_pct > 0 and pnl_pct <= -config.stop_loss_pct:
                forced_exit_reasons[symbol] = f"Stop loss triggered at {pnl_pct:.2%}."
            elif config.take_profit_pct > 0 and pnl_pct >= config.take_profit_pct:
                forced_exit_reasons[symbol] = f"Take profit triggered at {pnl_pct:.2%}."
            buy_locked_quantity = int(position.get("buy_locked_quantity", 0))
            if buy_locked_quantity > 0:
                locked_position_reasons[symbol] = f"T+1 lock: {buy_locked_quantity} shares are not sellable today."

        predictions = []
        filtered_signal_notes: list[str] = []
        for item in prediction_pool:
            symbol = str(item["symbol"])
            name = str(item["name"])
            if symbol in forced_exit_reasons:
                continue
            latest_bar = latest_bar_map.get(symbol, {})
            latest_price = float(price_map.get(symbol, 0.0) or 0.0)
            latest_volume = float(latest_bar.get("volume", 0.0) or 0.0)
            latest_amount = float(latest_bar.get("amount", 0.0) or 0.0)
            latest_turnover_rate = float(latest_bar.get("turnover_rate", 0.0) or 0.0)
            latest_change_pct = float(latest_bar.get("change_pct", 0.0) or 0.0)
            predicted_return = float(item.get("predicted_return_5d", 0.0) or 0.0)
            upper_name = name.upper()
            if "ST" in upper_name or "退" in name:
                filtered_signal_notes.append(f"{symbol} {name}：存在 ST/退市风险")
                continue
            if latest_price <= 0:
                filtered_signal_notes.append(f"{symbol} {name}：缺少可靠最新价")
                continue
            if latest_volume <= 0 or latest_amount <= 0:
                filtered_signal_notes.append(f"{symbol} {name}：疑似停牌或当日无成交")
                continue
            if latest_change_pct >= 0.095:
                filtered_signal_notes.append(f"{symbol} {name}：接近涨停 {latest_change_pct:.2%}")
                continue
            if latest_amount < config.min_liquidity_amount and latest_turnover_rate < config.min_turnover_rate:
                filtered_signal_notes.append(
                    f"{symbol} {name}：流动性不足（成交额 {latest_amount / 10_000:.0f} 万，换手率 {latest_turnover_rate:.2%}）"
                )
                continue
            if predicted_return < config.min_signal_return_pct:
                filtered_signal_notes.append(
                    f"{symbol} {name}：预期收益 {predicted_return:.2%} 低于筛选阈值 {config.min_signal_return_pct:.2%}"
                )
                continue
            predictions.append(item)
            if len(predictions) >= config.top_n:
                break
        if not predictions:
            raise RuntimeError("当前可用信号都被风险规则过滤，暂时没有可执行的目标仓位。")

        signal_symbols = [str(item["symbol"]) for item in predictions]

        equity = cash + current_market_value
        gross_target_capital = equity * config.capital_fraction
        reserve_cash = equity * config.min_cash_buffer_ratio
        investable_capital = max(min(gross_target_capital, max(equity - reserve_cash, 0.0)), 0.0)
        equal_weight_target = investable_capital / max(len(predictions), 1)
        capped_target_value = min(equal_weight_target, equity * config.max_position_weight)
        target_value_per_symbol = capped_target_value

        plan_orders: list[dict] = []
        planned_sell_notional = 0.0
        planned_buy_notional = 0.0
        warnings: list[str] = []

        for symbol, position in current_positions.items():
            quantity = int(position["quantity"])
            sellable_quantity = int(position.get("sellable_quantity", quantity))
            price = float(price_map.get(symbol, position.get("last_price", 0.0)))
            if quantity <= 0 or price <= 0:
                continue
            if symbol not in signal_symbols or symbol in forced_exit_reasons:
                executable_quantity = quantity if symbol in forced_exit_reasons and sellable_quantity > 0 else min(quantity, sellable_quantity)
                if executable_quantity <= 0:
                    warnings.append(f"{symbol} 当前持仓受 T+1 约束，今日不可卖出。")
                    continue
                notional = executable_quantity * price
                base_reason = forced_exit_reasons.get(symbol, "Signal exited from top picks.")
                reason = base_reason if executable_quantity == quantity else f"{base_reason} {locked_position_reasons.get(symbol, '')}".strip()
                plan_orders.append(
                    {
                        "symbol": symbol,
                        "name": str(position["name"]),
                        "side": "sell",
                        "current_quantity": quantity,
                        "target_quantity": quantity - executable_quantity,
                        "planned_quantity": executable_quantity,
                        "price": price,
                        "notional": notional,
                        "reason": reason,
                    }
                )
                planned_sell_notional += notional

        for signal in predictions:
            symbol = str(signal["symbol"])
            name = str(signal["name"])
            price = float(price_map[symbol])
            current_quantity = int(current_positions.get(symbol, {}).get("quantity", 0))
            current_sellable_quantity = int(current_positions.get(symbol, {}).get("sellable_quantity", current_quantity))
            target_quantity = int(target_value_per_symbol / price / config.board_lot) * config.board_lot
            delta = target_quantity - current_quantity
            if delta == 0:
                continue
            side = "buy" if delta > 0 else "sell"
            planned_quantity = abs(delta)
            reason = "Adjust to target weight."
            if side == "sell":
                planned_quantity = min(planned_quantity, current_sellable_quantity)
                if planned_quantity <= 0:
                    warnings.append(f"{symbol} 目标需要减仓，但可卖数量为 0，已跳过。")
                    continue
                if planned_quantity < abs(delta):
                    reason = f"Adjust to target weight. {locked_position_reasons.get(symbol, 'T+1 lock reduced sell size.')}".strip()
            notional = planned_quantity * price
            plan_orders.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "side": side,
                    "current_quantity": current_quantity,
                    "target_quantity": target_quantity,
                    "planned_quantity": planned_quantity,
                    "price": price,
                    "notional": notional,
                    "reason": reason,
                }
            )
            if side == "buy":
                planned_buy_notional += notional
            else:
                planned_sell_notional += notional

        turnover_ratio = float((planned_buy_notional + planned_sell_notional) / equity) if equity > 0 else 0.0
        blocked = False
        risk_metrics = self._compute_risk_metrics(equity, equity_curve)
        if turnover_ratio > config.max_turnover_ratio:
            warnings.append(f"预计换手 {turnover_ratio:.2f} 超过上限 {config.max_turnover_ratio:.2f}")
            blocked = True
        estimated_cash_after = cash + planned_sell_notional - planned_buy_notional
        min_cash_after = equity * config.min_cash_buffer_ratio
        if estimated_cash_after < min_cash_after:
            warnings.append("执行后现金缓冲不足，低于设定的最低现金比例")
            blocked = True
        if forced_exit_reasons:
            warnings.append(f"本次有 {len(forced_exit_reasons)} 个持仓触发止损/止盈，将被强制调出。")
        if locked_position_reasons:
            warnings.append(f"本次有 {len(locked_position_reasons)} 个持仓存在 T+1 锁定，卖出会受到限制。")
        if filtered_signal_notes:
            warnings.extend(filtered_signal_notes[:5])
            if len(filtered_signal_notes) > 5:
                warnings.append(f"另有 {len(filtered_signal_notes) - 5} 只候选因筛选阈值被跳过。")
        blocked_by_risk = False
        if risk_metrics["current_drawdown"] >= config.max_drawdown_limit:
            warnings.append(
                f"当前账户回撤 {risk_metrics['current_drawdown']:.2%} 已超过阈值 {config.max_drawdown_limit:.2%}，已阻断新调仓。"
            )
            blocked = True
            blocked_by_risk = True
        if risk_metrics["latest_equity_change"] <= -config.max_equity_change_limit:
            warnings.append(
                f"最近权益变化 {risk_metrics['latest_equity_change']:.2%} 低于阈值 -{config.max_equity_change_limit:.2%}，已阻断新调仓。"
            )
            blocked = True
            blocked_by_risk = True

        estimated_market_value_after = max(equity - estimated_cash_after, 0.0)
        holdings_preview = []
        preview_symbols = sorted(set([*current_positions.keys(), *signal_symbols]))
        for symbol in preview_symbols:
            current_position = current_positions.get(symbol, {})
            current_quantity = int(current_position.get("quantity", 0))
            target_quantity = current_quantity
            reason = "Hold current position."
            action = "hold"
            if symbol in forced_exit_reasons:
                target_quantity = 0
                action = "risk_exit"
                reason = forced_exit_reasons[symbol]
            else:
                for order in plan_orders:
                    if str(order["symbol"]) == symbol:
                        target_quantity = int(order["target_quantity"])
                        reason = str(order["reason"])
                        if target_quantity == 0:
                            action = "exit"
                        elif current_quantity == 0:
                            action = "entry"
                        elif target_quantity > current_quantity:
                            action = "add"
                        elif target_quantity < current_quantity:
                            action = "trim"
                        break
                else:
                    if current_quantity == 0 and symbol in signal_symbols:
                        target_quantity = int(target_value_per_symbol / float(price_map[symbol]) / config.board_lot) * config.board_lot
                        action = "entry" if target_quantity > 0 else "watch"
                        reason = "Target signal selected."
                    elif current_quantity > 0:
                        action = "hold"
            price = float(price_map.get(symbol, current_position.get("last_price", 0.0)))
            current_value = current_quantity * price
            target_value = target_quantity * price
            avg_cost = float(current_position.get("avg_cost", 0.0))
            unrealized_pnl_pct = (price / avg_cost - 1) if avg_cost else None
            holdings_preview.append(
                {
                    "symbol": symbol,
                    "name": str(current_position.get("name") or next((item["name"] for item in predictions if str(item["symbol"]) == symbol), symbol)),
                    "current_quantity": current_quantity,
                    "target_quantity": target_quantity,
                    "delta_quantity": target_quantity - current_quantity,
                    "current_value": float(current_value),
                    "target_value": float(target_value),
                    "price": price,
                    "weight_before": float((current_value / equity) if equity > 0 else 0.0),
                    "weight_after": float((target_value / estimated_market_value_after) if estimated_market_value_after > 0 else 0.0),
                    "unrealized_pnl_pct": float(unrealized_pnl_pct) if unrealized_pnl_pct is not None else None,
                    "action": action,
                    "reason": reason,
                }
            )

        return {
            "account": account,
            "config": {
                "top_n": config.top_n,
                "capital_fraction": config.capital_fraction,
                "max_position_weight": config.max_position_weight,
                "min_cash_buffer_ratio": config.min_cash_buffer_ratio,
                "max_turnover_ratio": config.max_turnover_ratio,
                "stop_loss_pct": config.stop_loss_pct,
                "take_profit_pct": config.take_profit_pct,
                "fill_ratio": config.fill_ratio,
                "max_drawdown_limit": config.max_drawdown_limit,
                "max_equity_change_limit": config.max_equity_change_limit,
                "min_signal_return_pct": config.min_signal_return_pct,
                "min_liquidity_amount": config.min_liquidity_amount,
                "min_turnover_rate": config.min_turnover_rate,
                "board_lot": config.board_lot,
            },
            "signals": predictions,
            "plan_orders": plan_orders,
            "holdings_preview": sorted(
                holdings_preview,
                key=lambda item: (
                    0 if item["action"] in {"risk_exit", "entry", "add", "trim", "exit"} else 1,
                    -abs(item["delta_quantity"]),
                    item["symbol"],
                ),
            ),
            "summary": {
                "equity": equity,
                "cash_before": cash,
                "market_value_before": current_market_value,
                "planned_buy_notional": planned_buy_notional,
                "planned_sell_notional": planned_sell_notional,
                "estimated_cash_after": estimated_cash_after,
                "estimated_position_ratio": float(((equity - estimated_cash_after) / equity) if equity > 0 else 0.0),
                "turnover_ratio": turnover_ratio,
                "blocked": blocked,
                "warnings": warnings,
                "target_symbols": signal_symbols,
                "forced_exit_count": len(forced_exit_reasons),
                "estimated_position_count_after": sum(1 for item in holdings_preview if int(item["target_quantity"]) > 0),
                "locked_position_count": len(locked_position_reasons),
                "blocked_by_risk": blocked_by_risk,
                "peak_equity": risk_metrics["peak_equity"],
                "current_drawdown": risk_metrics["current_drawdown"],
                "latest_equity_change": risk_metrics["latest_equity_change"],
            },
            "price_map": price_map,
            "current_positions": current_positions,
        }

    def preview_rebalance(
        self,
        *,
        preview_id: int | None = None,
        top_n: int = 3,
        capital_fraction: float = 0.95,
        max_position_weight: float = 0.35,
        min_cash_buffer_ratio: float = 0.05,
        max_turnover_ratio: float = 1.0,
        stop_loss_pct: float = 0.1,
        take_profit_pct: float = 0.2,
        fill_ratio: float = 1.0,
        max_drawdown_limit: float = 0.18,
        max_equity_change_limit: float = 0.04,
        min_signal_return_pct: float = 0.01,
        min_liquidity_amount: float = 30_000_000.0,
        min_turnover_rate: float = 0.002,
    ) -> dict:
        if preview_id is not None:
            existing_plan = self.repo.load_paper_rebalance_plan(preview_id, account_id=DEFAULT_ACCOUNT_ID)
            if not existing_plan:
                raise RuntimeError("未找到对应的调仓预览计划。")
            if existing_plan.get("status") != "pending":
                raise RuntimeError("这份调仓预览计划已经失效，请重新生成。")
            snapshot = self.get_account_snapshot()
            snapshot["preview"] = self._hydrate_preview_plan(existing_plan)
            return snapshot

        plan = self._build_rebalance_plan(
            top_n=top_n,
            capital_fraction=capital_fraction,
            max_position_weight=max_position_weight,
            min_cash_buffer_ratio=min_cash_buffer_ratio,
            max_turnover_ratio=max_turnover_ratio,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            fill_ratio=fill_ratio,
            max_drawdown_limit=max_drawdown_limit,
            max_equity_change_limit=max_equity_change_limit,
            min_signal_return_pct=min_signal_return_pct,
            min_liquidity_amount=min_liquidity_amount,
            min_turnover_rate=min_turnover_rate,
        )
        stored_plan = self.repo.create_paper_rebalance_plan(
            account_id=DEFAULT_ACCOUNT_ID,
            config=plan["config"],
            preview={
                "config": plan["config"],
                "orders": plan["plan_orders"],
                "holdings": plan["holdings_preview"],
                "summary": plan["summary"],
            },
            note="Preview generated and waiting for confirmation.",
        )
        if plan["summary"]["blocked"]:
            self._log_risk_event(
                event_type="preview_blocked",
                severity="error" if plan["summary"].get("blocked_by_risk") else "warning",
                title="调仓预览被风控阻断",
                details={
                    "target_symbols": list(plan["summary"].get("target_symbols", [])),
                    "warnings": list(plan["summary"].get("warnings", [])),
                    "turnover_ratio": float(plan["summary"].get("turnover_ratio", 0.0)),
                    "estimated_cash_after": float(plan["summary"].get("estimated_cash_after", 0.0)),
                    "current_drawdown": float(plan["summary"].get("current_drawdown", 0.0)),
                    "latest_equity_change": float(plan["summary"].get("latest_equity_change", 0.0)),
                },
                note="Preview generated but blocked.",
            )
        elif plan["summary"].get("forced_exit_count", 0) > 0 or plan["summary"].get("locked_position_count", 0) > 0:
            self._log_risk_event(
                event_type="preview_warning",
                severity="warning",
                title="调仓预览包含风险提示",
                details={
                    "forced_exit_count": int(plan["summary"].get("forced_exit_count", 0)),
                    "locked_position_count": int(plan["summary"].get("locked_position_count", 0)),
                    "warnings": list(plan["summary"].get("warnings", [])),
                    "target_symbols": list(plan["summary"].get("target_symbols", [])),
                },
                note="Preview generated with warnings.",
            )
        snapshot = self.get_account_snapshot()
        snapshot["preview"] = self._hydrate_preview_plan(stored_plan)
        return snapshot

    def execute_rebalance(
        self,
        preview_id: int | None = None,
        top_n: int = 3,
        capital_fraction: float = 0.95,
        max_position_weight: float = 0.35,
        min_cash_buffer_ratio: float = 0.05,
        max_turnover_ratio: float = 1.0,
        stop_loss_pct: float = 0.1,
        take_profit_pct: float = 0.2,
        fill_ratio: float = 1.0,
        max_drawdown_limit: float = 0.18,
        max_equity_change_limit: float = 0.04,
        min_signal_return_pct: float = 0.01,
        min_liquidity_amount: float = 30_000_000.0,
        min_turnover_rate: float = 0.002,
    ) -> dict:
        self.repo.unlock_paper_t1_positions(account_id=DEFAULT_ACCOUNT_ID, today=self._current_trade_date())
        plan_id: int | None = None
        if preview_id is not None:
            stored_plan = self.repo.load_paper_rebalance_plan(preview_id, account_id=DEFAULT_ACCOUNT_ID)
            if not stored_plan:
                raise RuntimeError("未找到要执行的调仓预览计划。")
            if stored_plan.get("status") != "pending":
                raise RuntimeError("这份调仓预览计划已经执行或被放弃，请重新生成。")
            preview = dict(stored_plan.get("preview", {}))
            plan = {
                "account": self.repo.load_paper_account(account_id=DEFAULT_ACCOUNT_ID),
                "config": dict(stored_plan.get("config", {})),
                "plan_orders": list(preview.get("orders", [])),
                "holdings_preview": list(preview.get("holdings", [])),
                "summary": dict(preview.get("summary", {})),
                "current_positions": {
                    str(item["symbol"]): item
                    for item in self.repo.load_paper_positions(account_id=DEFAULT_ACCOUNT_ID)
                },
            }
            plan_id = int(stored_plan["id"])
        else:
            plan = self._build_rebalance_plan(
                top_n=top_n,
                capital_fraction=capital_fraction,
                max_position_weight=max_position_weight,
                min_cash_buffer_ratio=min_cash_buffer_ratio,
                max_turnover_ratio=max_turnover_ratio,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                fill_ratio=fill_ratio,
                max_drawdown_limit=max_drawdown_limit,
                max_equity_change_limit=max_equity_change_limit,
                min_signal_return_pct=min_signal_return_pct,
                min_liquidity_amount=min_liquidity_amount,
                min_turnover_rate=min_turnover_rate,
            )
        if plan["summary"]["blocked"]:
            raise RuntimeError("当前调仓计划触发风控限制，请先调整参数或降低风险后再执行。")

        cash = float(plan["account"]["cash"])
        current_positions = {
            str(item["symbol"]): item
            for item in self.repo.load_paper_positions(account_id=DEFAULT_ACCOUNT_ID)
        }
        execution_fill_ratio = float(plan["config"].get("fill_ratio", 1.0))
        board_lot = int(plan["config"].get("board_lot", 100))
        orders = []
        filled_orders = 0
        partial_orders = 0
        cancelled_orders = 0
        total_planned_quantity = 0
        total_filled_quantity = 0
        for item in plan["plan_orders"]:
            symbol = str(item["symbol"])
            name = str(item["name"])
            side = str(item["side"])
            quantity = int(item["planned_quantity"])
            price = float(item["price"])
            if quantity <= 0:
                continue
            total_planned_quantity += quantity
            filled_quantity = self._resolve_filled_quantity(quantity, board_lot, execution_fill_ratio)
            remaining_quantity = max(quantity - filled_quantity, 0)
            order_status = "filled"
            if filled_quantity == 0:
                order_status = "cancelled"
                cancelled_orders += 1
            elif filled_quantity < quantity:
                order_status = "partial"
                partial_orders += 1
            else:
                filled_orders += 1
            filled_notional = float(filled_quantity * price)
            total_filled_quantity += filled_quantity

            if filled_quantity <= 0:
                self.repo.record_paper_order(
                    account_id=DEFAULT_ACCOUNT_ID,
                    symbol=symbol,
                    name=name,
                    side=side,
                    quantity=quantity,
                    filled_quantity=0,
                    remaining_quantity=remaining_quantity,
                    fill_ratio=0.0,
                    price=price,
                    notional=0.0,
                    status=order_status,
                    note=f"{item['reason']} Partial-fill simulator left this order unfilled.",
                )
                orders.append({**item, "filled_quantity": 0, "remaining_quantity": remaining_quantity, "status": order_status, "filled_notional": 0.0})
                continue

            if side == "sell":
                cash += filled_notional
                existing = current_positions.get(symbol)
                current_qty = int(existing["quantity"]) if existing else 0
                current_sellable_qty = int(existing.get("sellable_quantity", current_qty)) if existing else 0
                if current_sellable_qty < filled_quantity:
                    filled_quantity = current_sellable_qty
                    remaining_quantity = max(quantity - filled_quantity, 0)
                    filled_notional = float(filled_quantity * price)
                current_buy_locked_qty = int(existing.get("buy_locked_quantity", 0)) if existing else 0
                current_buy_locked_at = existing.get("buy_locked_at") if existing else None
                remaining_qty = max(current_qty - filled_quantity, 0)
                next_sellable_qty = max(current_sellable_qty - filled_quantity, 0)
                self.repo.record_paper_order(
                    account_id=DEFAULT_ACCOUNT_ID,
                    symbol=symbol,
                    name=name,
                    side="sell",
                    quantity=quantity,
                    filled_quantity=filled_quantity,
                    remaining_quantity=remaining_quantity,
                    fill_ratio=float(filled_quantity / max(quantity, 1)),
                    price=price,
                    notional=filled_notional,
                    status=order_status,
                    note=item["reason"],
                )
                if remaining_qty > 0 and existing:
                    self.repo.upsert_paper_position(
                        account_id=DEFAULT_ACCOUNT_ID,
                        symbol=symbol,
                        name=name,
                        quantity=remaining_qty,
                        avg_cost=float(existing["avg_cost"]),
                        last_price=price,
                        sellable_quantity=next_sellable_qty,
                        buy_locked_quantity=min(current_buy_locked_qty, remaining_qty),
                        buy_locked_at=current_buy_locked_at,
                    )
                    current_positions[symbol] = {
                        **existing,
                        "quantity": remaining_qty,
                        "sellable_quantity": next_sellable_qty,
                        "buy_locked_quantity": min(current_buy_locked_qty, remaining_qty),
                        "buy_locked_at": current_buy_locked_at,
                        "last_price": price,
                    }
                else:
                    self.repo.delete_paper_position(DEFAULT_ACCOUNT_ID, symbol)
                    current_positions.pop(symbol, None)
            else:
                cash -= filled_notional
                existing = current_positions.get(symbol)
                current_qty = int(existing["quantity"]) if existing else 0
                current_sellable_qty = int(existing.get("sellable_quantity", current_qty)) if existing else 0
                current_buy_locked_qty = int(existing.get("buy_locked_quantity", 0)) if existing else 0
                new_quantity = current_qty + filled_quantity
                if existing:
                    new_avg_cost = ((current_qty * float(existing["avg_cost"])) + filled_notional) / max(new_quantity, 1)
                else:
                    new_avg_cost = price
                next_buy_locked_qty = current_buy_locked_qty + filled_quantity
                self.repo.record_paper_order(
                    account_id=DEFAULT_ACCOUNT_ID,
                    symbol=symbol,
                    name=name,
                    side="buy",
                    quantity=quantity,
                    filled_quantity=filled_quantity,
                    remaining_quantity=remaining_quantity,
                    fill_ratio=float(filled_quantity / max(quantity, 1)),
                    price=price,
                    notional=filled_notional,
                    status=order_status,
                    note=item["reason"],
                )
                self.repo.upsert_paper_position(
                    account_id=DEFAULT_ACCOUNT_ID,
                    symbol=symbol,
                    name=name,
                    quantity=new_quantity,
                    avg_cost=new_avg_cost,
                    last_price=price,
                    sellable_quantity=current_sellable_qty,
                    buy_locked_quantity=next_buy_locked_qty,
                    buy_locked_at=self._current_trade_date(),
                )
                current_positions[symbol] = {
                    "symbol": symbol,
                    "name": name,
                    "quantity": new_quantity,
                    "sellable_quantity": current_sellable_qty,
                    "buy_locked_quantity": next_buy_locked_qty,
                    "buy_locked_at": self._current_trade_date(),
                    "avg_cost": new_avg_cost,
                    "last_price": price,
                }
            orders.append(
                {
                    **item,
                    "filled_quantity": filled_quantity,
                    "remaining_quantity": remaining_quantity,
                    "status": order_status,
                    "filled_notional": filled_notional,
                }
            )

        account = self.repo.update_paper_account_cash(cash, account_id=DEFAULT_ACCOUNT_ID)
        positions = self._mark_positions(self.repo.load_paper_positions(account_id=DEFAULT_ACCOUNT_ID))
        totals = self._compute_account_totals(account, positions)
        self.repo.record_paper_equity_snapshot(
            account_id=DEFAULT_ACCOUNT_ID,
            cash=totals["cash"],
            market_value=totals["market_value"],
            equity=totals["equity"],
            note="Manual paper rebalance executed.",
        )
        self.repo.record_paper_rebalance_run(
            account_id=DEFAULT_ACCOUNT_ID,
            top_n=int(plan["config"]["top_n"]),
            capital_fraction=float(plan["config"]["capital_fraction"]),
            max_position_weight=float(plan["config"]["max_position_weight"]),
            min_cash_buffer_ratio=float(plan["config"]["min_cash_buffer_ratio"]),
            max_turnover_ratio=float(plan["config"]["max_turnover_ratio"]),
            orders_created=len(orders),
            turnover_ratio=float(plan["summary"]["turnover_ratio"]),
            target_symbols=list(plan["summary"]["target_symbols"]),
            note="Confirmed paper rebalance executed.",
        )
        if plan_id is not None:
            self.repo.update_paper_rebalance_plan_status(
                plan_id,
                account_id=DEFAULT_ACCOUNT_ID,
                status="executed",
                note="Preview confirmed and executed.",
            )
        snapshot = self.get_account_snapshot()
        snapshot["execution"] = {
            "orders_created": len(orders),
            "filled_orders": filled_orders,
            "partial_orders": partial_orders,
            "cancelled_orders": cancelled_orders,
            "effective_fill_ratio": float((total_filled_quantity / total_planned_quantity) if total_planned_quantity > 0 else 0.0),
            "target_symbols": plan["summary"]["target_symbols"],
            "capital_fraction": plan["config"]["capital_fraction"],
            "top_n": plan["config"]["top_n"],
            "stop_loss_pct": plan["config"]["stop_loss_pct"],
            "take_profit_pct": plan["config"]["take_profit_pct"],
            "fill_ratio": execution_fill_ratio,
            "plan_id": plan_id,
        }
        self._record_execution_report(
            report_type="rebalance",
            title="模拟调仓已执行",
            account=snapshot["account"],
            positions=snapshot["positions"],
            note="Confirmed paper rebalance executed.",
            execution=snapshot["execution"],
            extra={
                "target_symbols": list(plan["summary"].get("target_symbols", [])),
                "turnover_ratio": float(plan["summary"].get("turnover_ratio", 0.0)),
                "warnings": list(plan["summary"].get("warnings", [])),
            },
        )
        if partial_orders > 0 or cancelled_orders > 0:
            self._log_risk_event(
                event_type="execution_slippage",
                severity="warning",
                title="调仓执行存在未完全成交",
                details={
                    "filled_orders": filled_orders,
                    "partial_orders": partial_orders,
                    "cancelled_orders": cancelled_orders,
                    "effective_fill_ratio": snapshot["execution"]["effective_fill_ratio"],
                    "target_symbols": list(plan["summary"].get("target_symbols", [])),
                },
                note="Execution completed with partial or cancelled orders.",
            )
        return self.get_account_snapshot() | {"execution": snapshot["execution"]}

    def retry_order(self, *, order_id: int, fill_ratio: float = 1.0) -> dict:
        self.repo.unlock_paper_t1_positions(account_id=DEFAULT_ACCOUNT_ID, today=self._current_trade_date())
        order = self.repo.load_paper_order(order_id, account_id=DEFAULT_ACCOUNT_ID)
        if not order:
            raise RuntimeError("未找到要重试的订单。")
        if str(order.get("status")) not in {"partial", "cancelled"}:
            raise RuntimeError("只有部分成交或未成交订单可以重试。")
        remaining_quantity = int(order.get("remaining_quantity", 0))
        if remaining_quantity <= 0:
            raise RuntimeError("这笔订单没有待处理余量。")

        ratio = max(0.0, min(float(fill_ratio), 1.0))
        latest_price = self._latest_price_map([str(order["symbol"])]).get(str(order["symbol"]), float(order["price"]))
        board_lot = 100
        filled_quantity = self._resolve_filled_quantity(remaining_quantity, board_lot, ratio)
        positions = {str(item["symbol"]): item for item in self.repo.load_paper_positions(account_id=DEFAULT_ACCOUNT_ID)}
        position = positions.get(str(order["symbol"]))
        if str(order["side"]) == "sell":
            sellable_quantity = int(position.get("sellable_quantity", position.get("quantity", 0))) if position else 0
            filled_quantity = min(filled_quantity, sellable_quantity)
        next_remaining = max(remaining_quantity - filled_quantity, 0)
        status = "filled"
        if filled_quantity <= 0:
            status = "cancelled"
        elif next_remaining > 0:
            status = "partial"
        filled_notional = float(filled_quantity * latest_price)

        account = self.repo.load_paper_account(account_id=DEFAULT_ACCOUNT_ID)
        cash = float(account["cash"])
        if str(order["side"]) == "sell":
            if filled_quantity > 0 and position:
                current_qty = int(position.get("quantity", 0))
                current_sellable = int(position.get("sellable_quantity", current_qty))
                current_locked = int(position.get("buy_locked_quantity", 0))
                remaining_position_qty = max(current_qty - filled_quantity, 0)
                remaining_sellable = max(current_sellable - filled_quantity, 0)
                cash += filled_notional
                if remaining_position_qty > 0:
                    self.repo.upsert_paper_position(
                        account_id=DEFAULT_ACCOUNT_ID,
                        symbol=str(order["symbol"]),
                        name=str(order["name"]),
                        quantity=remaining_position_qty,
                        avg_cost=float(position["avg_cost"]),
                        last_price=float(latest_price),
                        sellable_quantity=remaining_sellable,
                        buy_locked_quantity=min(current_locked, remaining_position_qty),
                        buy_locked_at=position.get("buy_locked_at"),
                    )
                else:
                    self.repo.delete_paper_position(DEFAULT_ACCOUNT_ID, str(order["symbol"]))
        else:
            cash -= filled_notional
            current_qty = int(position.get("quantity", 0)) if position else 0
            current_sellable = int(position.get("sellable_quantity", current_qty)) if position else 0
            current_locked = int(position.get("buy_locked_quantity", 0)) if position else 0
            new_quantity = current_qty + filled_quantity
            if filled_quantity > 0:
                if position:
                    new_avg_cost = ((current_qty * float(position["avg_cost"])) + filled_notional) / max(new_quantity, 1)
                else:
                    new_avg_cost = float(latest_price)
                self.repo.upsert_paper_position(
                    account_id=DEFAULT_ACCOUNT_ID,
                    symbol=str(order["symbol"]),
                    name=str(order["name"]),
                    quantity=new_quantity,
                    avg_cost=new_avg_cost,
                    last_price=float(latest_price),
                    sellable_quantity=current_sellable,
                    buy_locked_quantity=current_locked + filled_quantity,
                    buy_locked_at=self._current_trade_date(),
                )

        if filled_quantity > 0:
            self.repo.update_paper_account_cash(cash, account_id=DEFAULT_ACCOUNT_ID)
            self.repo.record_paper_order(
                account_id=DEFAULT_ACCOUNT_ID,
                symbol=str(order["symbol"]),
                name=str(order["name"]),
                side=str(order["side"]),
                quantity=remaining_quantity,
                filled_quantity=filled_quantity,
                remaining_quantity=next_remaining,
                fill_ratio=float((filled_quantity / max(remaining_quantity, 1))),
                price=float(latest_price),
                notional=filled_notional,
                status=status,
                source="paper-retry",
                note=f"Retry for order #{order_id}.",
            )
        else:
            self.repo.record_paper_order(
                account_id=DEFAULT_ACCOUNT_ID,
                symbol=str(order["symbol"]),
                name=str(order["name"]),
                side=str(order["side"]),
                quantity=remaining_quantity,
                filled_quantity=0,
                remaining_quantity=remaining_quantity,
                fill_ratio=0.0,
                price=float(latest_price),
                notional=0.0,
                status="cancelled",
                source="paper-retry",
                note=f"Retry for order #{order_id} produced no fill.",
            )
        self.repo.update_paper_order_status(
            order_id,
            account_id=DEFAULT_ACCOUNT_ID,
            status=f"{order['status']}_retried",
            note=f"{order.get('note', '')} Retried via follow-up order.",
        )
        snapshot = self.get_account_snapshot()
        snapshot["execution"] = {
            "orders_created": 1,
            "filled_orders": 1 if status == "filled" else 0,
            "partial_orders": 1 if status == "partial" else 0,
            "cancelled_orders": 1 if status == "cancelled" else 0,
            "effective_fill_ratio": float((filled_quantity / max(remaining_quantity, 1))),
            "target_symbols": [str(order["symbol"])],
            "capital_fraction": 0.0,
            "top_n": 0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "fill_ratio": ratio,
            "plan_id": None,
        }
        self._record_execution_report(
            report_type="retry",
            title=f"订单 #{order_id} 已重试",
            account=snapshot["account"],
            positions=snapshot["positions"],
            note="Retry executed for remaining quantity.",
            execution=snapshot["execution"],
            extra={
                "order_id": int(order_id),
                "symbol": str(order["symbol"]),
                "side": str(order["side"]),
                "remaining_quantity_before": remaining_quantity,
                "remaining_quantity_after": next_remaining,
            },
        )
        if status != "filled":
            self._log_risk_event(
                event_type="retry_incomplete",
                severity="warning",
                title="订单重试后仍未完全成交",
                details={
                    "order_id": int(order_id),
                    "symbol": str(order["symbol"]),
                    "status": status,
                    "filled_quantity": int(filled_quantity),
                    "remaining_quantity": int(next_remaining),
                    "fill_ratio": ratio,
                },
                note="Retry still incomplete.",
            )
        return self.get_account_snapshot() | {"execution": snapshot["execution"]}

    def cancel_order_remainder(self, *, order_id: int) -> dict:
        order = self.repo.load_paper_order(order_id, account_id=DEFAULT_ACCOUNT_ID)
        if not order:
            raise RuntimeError("未找到要撤销余量的订单。")
        if str(order.get("status")) not in {"partial", "cancelled"}:
            raise RuntimeError("只有部分成交或未成交订单可以撤销余量。")
        self.repo.update_paper_order_status(
            order_id,
            account_id=DEFAULT_ACCOUNT_ID,
            status=f"{order['status']}_closed",
            note=f"{order.get('note', '')} Remaining quantity was manually closed.",
        )
        snapshot = self.get_account_snapshot()
        snapshot["execution"] = {
            "orders_created": 0,
            "filled_orders": 0,
            "partial_orders": 0,
            "cancelled_orders": 1,
            "effective_fill_ratio": 0.0,
            "target_symbols": [str(order["symbol"])],
            "capital_fraction": 0.0,
            "top_n": 0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "fill_ratio": 0.0,
            "plan_id": None,
        }
        self._record_execution_report(
            report_type="cancel",
            title=f"订单 #{order_id} 余量已关闭",
            account=snapshot["account"],
            positions=snapshot["positions"],
            note="Remaining quantity manually closed.",
            execution=snapshot["execution"],
            extra={
                "order_id": int(order_id),
                "symbol": str(order["symbol"]),
                "side": str(order["side"]),
                "remaining_quantity": int(order.get("remaining_quantity", 0)),
            },
        )
        self._log_risk_event(
            event_type="manual_cancel",
            severity="info",
            title="订单余量被手动关闭",
            details={
                "order_id": int(order_id),
                "symbol": str(order["symbol"]),
                "remaining_quantity": int(order.get("remaining_quantity", 0)),
            },
            note="Operator manually closed remaining quantity.",
        )
        return self.get_account_snapshot() | {"execution": snapshot["execution"]}

    def reject_rebalance_preview(self, *, preview_id: int) -> dict:
        plan = self.repo.load_paper_rebalance_plan(preview_id, account_id=DEFAULT_ACCOUNT_ID)
        if not plan:
            raise RuntimeError("未找到要放弃的调仓预览计划。")
        if plan.get("status") != "pending":
            raise RuntimeError("这份调仓预览计划已经不是待执行状态。")
        self.repo.update_paper_rebalance_plan_status(
            preview_id,
            account_id=DEFAULT_ACCOUNT_ID,
            status="rejected",
            note="Preview manually rejected.",
        )
        snapshot = self.get_account_snapshot()
        snapshot["execution"] = {
            "orders_created": 0,
            "filled_orders": 0,
            "partial_orders": 0,
            "cancelled_orders": 0,
            "effective_fill_ratio": 0.0,
            "target_symbols": list(plan.get("preview", {}).get("summary", {}).get("target_symbols", [])),
            "capital_fraction": float(plan.get("config", {}).get("capital_fraction", 0.0)),
            "top_n": int(plan.get("config", {}).get("top_n", 0)),
            "stop_loss_pct": float(plan.get("config", {}).get("stop_loss_pct", 0.0)),
            "take_profit_pct": float(plan.get("config", {}).get("take_profit_pct", 0.0)),
            "fill_ratio": float(plan.get("config", {}).get("fill_ratio", 0.0)),
            "plan_id": int(preview_id),
        }
        self._record_execution_report(
            report_type="reject",
            title=f"计划单 #{preview_id} 已放弃",
            account=snapshot["account"],
            positions=snapshot["positions"],
            note="Preview manually rejected.",
            execution=snapshot["execution"],
            extra={
                "warnings": list(plan.get("preview", {}).get("summary", {}).get("warnings", [])),
                "target_symbols": list(plan.get("preview", {}).get("summary", {}).get("target_symbols", [])),
            },
        )
        return self.get_account_snapshot() | {"execution": snapshot["execution"]}
