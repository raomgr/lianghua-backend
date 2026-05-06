from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import (
    BacktestResponse,
    BacktestMonteCarloResponse,
    BacktestScenarioResponse,
    BacktestSensitivityResponse,
    BacktestStabilityResponse,
    CustomUniverseItem,
    CustomUniverseRequest,
    CustomUniverseResponse,
    FactorResult,
    ModelCompareItem,
    ModelDetailResponse,
    ModelFeatureImportance,
    ModelPrediction,
    PaperOrderActionRequest,
    PaperDailyRunRequest,
    PaperDailySettingsRequest,
    PaperPreviewActionRequest,
    SignalCenterResponse,
    SignalHistoryItem,
    SignalReview,
    SignalReviewRequest,
    ModelRunInfo,
    ModelStatus,
    PaperRebalanceRequest,
    PaperResetRequest,
    PaperTradingSnapshot,
    PriceBar,
    StockSnapshot,
    TrainResponse,
    UpdateResponse,
)
from app.services.market_service import MarketService
from app.services.paper_trading import PaperTradingService
from app.services.paper_scheduler import PaperDailyScheduler
from app.services.modeling import get_model_status, resolve_active_provider
from app.services.storage import MarketRepository
from app.services.training import train_local_model

settings = get_settings()
app = FastAPI(title=settings.app_name)
market = MarketService()
paper = PaperTradingService(market)
scheduler = PaperDailyScheduler(paper)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def with_scheduler_status(snapshot: dict) -> dict:
    return {**snapshot, "scheduler_status": scheduler.get_status()}


@app.on_event("startup")
async def startup_scheduler() -> None:
    await scheduler.start()


@app.on_event("shutdown")
async def shutdown_scheduler() -> None:
    await scheduler.stop()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "provider": settings.data_provider,
        "universe_source": settings.universe_source,
        "universe_index_code": settings.universe_index_code,
        "universe_index_name": settings.universe_index_name,
    }


@app.get("/api/stocks", response_model=list[StockSnapshot])
def list_stocks() -> list[StockSnapshot]:
    snapshot = market.get_stock_snapshot()
    return [StockSnapshot(**row) for row in snapshot.to_dict(orient="records")]


@app.get("/api/stocks/{symbol}/history", response_model=list[PriceBar])
def stock_history(symbol: str, limit: int = Query(240, ge=30, le=600)) -> list[PriceBar]:
    bars = market.get_stock_history(symbol, limit=limit)
    return [PriceBar(**row) for row in bars.to_dict(orient="records")]


@app.get("/api/factors", response_model=list[FactorResult])
def factors() -> list[FactorResult]:
    frame = market.get_factor_table()
    payload = frame[
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
    ]
    return [FactorResult(**row) for row in payload.to_dict(orient="records")]


@app.get("/api/backtest", response_model=BacktestResponse)
def backtest(
    rebalance_days: int = Query(5, ge=1, le=20),
    top_n: int = Query(3, ge=1, le=10),
    trading_cost_bps: float = Query(8.0, ge=0, le=100),
    slippage_bps: float = Query(5.0, ge=0, le=100),
) -> BacktestResponse:
    return BacktestResponse(
        **market.get_backtest(
            rebalance_days=rebalance_days,
            top_n=top_n,
            trading_cost_bps=trading_cost_bps,
            slippage_bps=slippage_bps,
        )
    )


@app.get("/api/backtest/sensitivity", response_model=BacktestSensitivityResponse)
def backtest_sensitivity(
    rebalance_days: int = Query(5, ge=1, le=20),
    top_n: int = Query(3, ge=1, le=10),
    trading_cost_bps: float = Query(8.0, ge=0, le=100),
    slippage_bps: float = Query(5.0, ge=0, le=100),
    scan_width: int = Query(1, ge=1, le=3),
) -> BacktestSensitivityResponse:
    return BacktestSensitivityResponse(
        **market.get_backtest_sensitivity(
            rebalance_days=rebalance_days,
            top_n=top_n,
            trading_cost_bps=trading_cost_bps,
            slippage_bps=slippage_bps,
            scan_width=scan_width,
        )
    )


@app.get("/api/backtest/stability", response_model=BacktestStabilityResponse)
def backtest_stability(
    rebalance_days: int = Query(5, ge=1, le=20),
    top_n: int = Query(3, ge=1, le=10),
    trading_cost_bps: float = Query(8.0, ge=0, le=100),
    slippage_bps: float = Query(5.0, ge=0, le=100),
    rolling_window: int = Query(20, ge=10, le=40),
) -> BacktestStabilityResponse:
    return BacktestStabilityResponse(
        **market.get_backtest_stability(
            rebalance_days=rebalance_days,
            top_n=top_n,
            trading_cost_bps=trading_cost_bps,
            slippage_bps=slippage_bps,
            rolling_window=rolling_window,
        )
    )


@app.get("/api/backtest/montecarlo", response_model=BacktestMonteCarloResponse)
def backtest_monte_carlo(
    rebalance_days: int = Query(5, ge=1, le=20),
    top_n: int = Query(3, ge=1, le=10),
    trading_cost_bps: float = Query(8.0, ge=0, le=100),
    slippage_bps: float = Query(5.0, ge=0, le=100),
    trials: int = Query(300, ge=50, le=1000),
) -> BacktestMonteCarloResponse:
    return BacktestMonteCarloResponse(
        **market.get_backtest_monte_carlo(
            rebalance_days=rebalance_days,
            top_n=top_n,
            trading_cost_bps=trading_cost_bps,
            slippage_bps=slippage_bps,
            trials=trials,
        )
    )


@app.get("/api/backtest/scenarios", response_model=BacktestScenarioResponse)
def backtest_scenarios(
    rebalance_days: int = Query(5, ge=1, le=20),
    top_n: int = Query(3, ge=1, le=10),
    trading_cost_bps: float = Query(8.0, ge=0, le=100),
    slippage_bps: float = Query(5.0, ge=0, le=100),
) -> BacktestScenarioResponse:
    return BacktestScenarioResponse(
        **market.get_backtest_scenarios(
            rebalance_days=rebalance_days,
            top_n=top_n,
            trading_cost_bps=trading_cost_bps,
            slippage_bps=slippage_bps,
        )
    )


@app.get("/api/model/status", response_model=ModelStatus)
def model_status() -> ModelStatus:
    return ModelStatus(**get_model_status())


@app.get("/api/paper/account", response_model=PaperTradingSnapshot)
def paper_account() -> PaperTradingSnapshot:
    return PaperTradingSnapshot(**with_scheduler_status(paper.get_account_snapshot()))


@app.post("/api/paper/reset", response_model=PaperTradingSnapshot)
def paper_reset(payload: PaperResetRequest) -> PaperTradingSnapshot:
    return PaperTradingSnapshot(**with_scheduler_status(paper.reset_account(initial_cash=payload.initial_cash)))


@app.post("/api/paper/rebalance/preview", response_model=PaperTradingSnapshot)
def paper_rebalance_preview(payload: PaperRebalanceRequest) -> PaperTradingSnapshot:
    return PaperTradingSnapshot(
        **with_scheduler_status(paper.preview_rebalance(
            preview_id=payload.preview_id,
            top_n=payload.top_n,
            capital_fraction=payload.capital_fraction,
            max_position_weight=payload.max_position_weight,
            min_cash_buffer_ratio=payload.min_cash_buffer_ratio,
            max_turnover_ratio=payload.max_turnover_ratio,
            stop_loss_pct=payload.stop_loss_pct,
            take_profit_pct=payload.take_profit_pct,
            fill_ratio=payload.fill_ratio,
            max_drawdown_limit=payload.max_drawdown_limit,
            max_equity_change_limit=payload.max_equity_change_limit,
        ))
    )


@app.post("/api/paper/rebalance", response_model=PaperTradingSnapshot)
def paper_rebalance(payload: PaperRebalanceRequest) -> PaperTradingSnapshot:
    return PaperTradingSnapshot(
        **with_scheduler_status(paper.execute_rebalance(
            preview_id=payload.preview_id,
            top_n=payload.top_n,
            capital_fraction=payload.capital_fraction,
            max_position_weight=payload.max_position_weight,
            min_cash_buffer_ratio=payload.min_cash_buffer_ratio,
            max_turnover_ratio=payload.max_turnover_ratio,
            stop_loss_pct=payload.stop_loss_pct,
            take_profit_pct=payload.take_profit_pct,
            fill_ratio=payload.fill_ratio,
            max_drawdown_limit=payload.max_drawdown_limit,
            max_equity_change_limit=payload.max_equity_change_limit,
        ))
    )


@app.post("/api/paper/rebalance/reject", response_model=PaperTradingSnapshot)
def paper_rebalance_reject(payload: PaperPreviewActionRequest) -> PaperTradingSnapshot:
    return PaperTradingSnapshot(**with_scheduler_status(paper.reject_rebalance_preview(preview_id=payload.preview_id)))


@app.post("/api/paper/orders/retry", response_model=PaperTradingSnapshot)
def paper_order_retry(payload: PaperOrderActionRequest) -> PaperTradingSnapshot:
    return PaperTradingSnapshot(**with_scheduler_status(paper.retry_order(order_id=payload.order_id, fill_ratio=payload.fill_ratio)))


@app.post("/api/paper/orders/cancel", response_model=PaperTradingSnapshot)
def paper_order_cancel(payload: PaperOrderActionRequest) -> PaperTradingSnapshot:
    return PaperTradingSnapshot(**with_scheduler_status(paper.cancel_order_remainder(order_id=payload.order_id)))


@app.post("/api/paper/daily/settings", response_model=PaperTradingSnapshot)
def paper_daily_settings(payload: PaperDailySettingsRequest) -> PaperTradingSnapshot:
    return PaperTradingSnapshot(
        **with_scheduler_status(paper.update_daily_settings(
            enabled=payload.enabled,
            run_time=payload.run_time,
            auto_sync=payload.auto_sync,
            auto_train=payload.auto_train,
            auto_rebalance=payload.auto_rebalance,
            top_n=payload.top_n,
            capital_fraction=payload.capital_fraction,
            max_position_weight=payload.max_position_weight,
            min_cash_buffer_ratio=payload.min_cash_buffer_ratio,
            max_turnover_ratio=payload.max_turnover_ratio,
            stop_loss_pct=payload.stop_loss_pct,
            take_profit_pct=payload.take_profit_pct,
            fill_ratio=payload.fill_ratio,
            max_drawdown_limit=payload.max_drawdown_limit,
            max_equity_change_limit=payload.max_equity_change_limit,
        ))
    )


@app.post("/api/paper/daily/run", response_model=PaperTradingSnapshot)
def paper_daily_run(payload: PaperDailyRunRequest) -> PaperTradingSnapshot:
    return PaperTradingSnapshot(
        **with_scheduler_status(paper.run_daily_cycle(
            auto_sync=payload.auto_sync,
            auto_train=payload.auto_train,
            auto_rebalance=payload.auto_rebalance,
            trigger_source="manual-rerun" if payload.start_from_step != "sync" else "manual",
            start_from_step=payload.start_from_step,
        ))
    )


@app.get("/api/universe/custom", response_model=list[CustomUniverseItem])
def get_custom_universe() -> list[CustomUniverseItem]:
    items = market.get_custom_universe()
    return [CustomUniverseItem(**item) for item in items]


@app.get("/api/universe/search", response_model=list[CustomUniverseItem])
def search_universe(q: str = Query("", min_length=1, max_length=20)) -> list[CustomUniverseItem]:
    items = market.search_universe_candidates(q, limit=12)
    return [CustomUniverseItem(**item) for item in items]


@app.post("/api/universe/custom", response_model=CustomUniverseResponse)
def set_custom_universe(payload: CustomUniverseRequest) -> CustomUniverseResponse:
    items = market.configure_custom_universe(payload.symbols, [item.model_dump() for item in payload.items])
    message = (
        "已恢复默认股票池。"
        if not items
        else f"已保存 {len(items)} 只自定义股票，下一次同步将使用这批代码。"
    )
    return CustomUniverseResponse(
        status="success",
        message=message,
        items=[CustomUniverseItem(**item) for item in items],
    )


@app.get("/api/model/predictions", response_model=list[ModelPrediction])
def model_predictions() -> list[ModelPrediction]:
    repo = MarketRepository()
    predictions = repo.load_latest_predictions(limit=10, provider=resolve_active_provider(repo))
    return [ModelPrediction(**row) for row in predictions]


@app.get("/api/signals/center", response_model=SignalCenterResponse)
def signal_center() -> SignalCenterResponse:
    return SignalCenterResponse(**market.get_signal_center(candidate_limit=10))


@app.get("/api/signals/history", response_model=list[SignalHistoryItem])
def signal_history(limit: int = Query(12, ge=3, le=40)) -> list[SignalHistoryItem]:
    return [SignalHistoryItem(**row) for row in market.get_signal_history(limit=limit)]


@app.post("/api/signals/review", response_model=SignalReview)
def signal_review(payload: SignalReviewRequest) -> SignalReview:
    return SignalReview(
        **market.save_signal_review(
            payload.model_run_id,
            payload.status,
            payload.note,
            execution_items=[item.model_dump() for item in payload.execution_items],
        )
    )


@app.get("/api/model/detail", response_model=ModelDetailResponse)
def model_detail() -> ModelDetailResponse:
    repo = MarketRepository()
    latest = repo.load_latest_model_run(provider=resolve_active_provider(repo))
    feature_importance = latest.get("coefficients", {}).get("feature_importance", {})
    top_features = [
        ModelFeatureImportance(feature=feature, importance=value)
        for feature, value in list(feature_importance.items())[:8]
    ]
    return ModelDetailResponse(
        model_name=latest.get("model_name", "unavailable"),
        provider=latest.get("provider", settings.data_provider),
        run_at=latest.get("run_at", ""),
        validation_ic=latest.get("validation_ic", 0.0),
        validation_directional_accuracy=latest.get("validation_directional_accuracy", 0.0),
        train_rows=latest.get("train_rows", 0),
        validation_rows=latest.get("validation_rows", 0),
        metrics=latest.get("metrics", {}),
        top_features=top_features,
        note=latest.get("note", "No model run is available yet."),
    )


@app.get("/api/model/compare", response_model=list[ModelCompareItem])
def model_compare() -> list[ModelCompareItem]:
    repo = MarketRepository()
    rows = repo.load_latest_model_comparison(provider=resolve_active_provider(repo), limit=6)
    return [ModelCompareItem(**row) for row in rows]


@app.get("/api/model/runs", response_model=list[ModelRunInfo])
def model_runs() -> list[ModelRunInfo]:
    repo = MarketRepository()
    return [ModelRunInfo(**row) for row in repo.load_recent_model_runs(limit=6, provider=resolve_active_provider(repo))]


@app.post("/api/model/train", response_model=TrainResponse)
def train_model() -> TrainResponse:
    histories, names = market.get_training_inputs(limit=220)
    result = train_local_model(
        histories,
        names,
        provider_name=market.active_data_provider,
        configured_provider=settings.data_provider,
    )
    return TrainResponse(**result)


@app.post("/api/tasks/update", response_model=UpdateResponse)
def trigger_update() -> UpdateResponse:
    sync = market.refresh_market_data()
    if market.active_data_provider != settings.data_provider:
        message = (
            f"已切换到 {market.active_data_provider}，当前更新 {sync['symbols_synced']} 只股票，"
            f"写入 {sync['bars_written']} 条K线。"
        )
    else:
        message = f"数据同步完成，共更新 {sync['symbols_synced']} 只股票，写入 {sync['bars_written']} 条K线。"
    return UpdateResponse(
        status="success",
        message=message,
        symbols_synced=sync["symbols_synced"],
        bars_written=sync["bars_written"],
    )
