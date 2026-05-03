from datetime import date

from pydantic import BaseModel, Field


class StockSnapshot(BaseModel):
    symbol: str
    name: str
    latest_price: float
    pct_change: float
    volume: float
    momentum_20: float | None = None
    volatility_20: float | None = None
    score: float | None = None


class PriceBar(BaseModel):
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float | None = None
    turnover_rate: float | None = None


class FactorResult(BaseModel):
    symbol: str
    close: float
    ret_1d: float | None = None
    return_5d: float
    return_10d: float | None = None
    momentum_20: float
    momentum_60: float | None = None
    reversal_10: float | None = None
    volatility_10: float | None = None
    volatility_20: float
    volatility_60: float | None = None
    volume_ratio_5: float
    volume_ratio_20: float | None = None
    turnover_rate: float | None = None
    turnover_ratio_5: float | None = None
    price_vs_ma_20: float | None = None
    price_vs_ma_60: float | None = None
    breakout_20: float | None = None
    close_position_20: float | None = None
    intraday_range: float | None = None
    atr_14: float | None = None
    score: float = Field(description="Baseline alpha score")


class MetricPoint(BaseModel):
    trade_date: date
    equity: float
    benchmark: float


class BacktestSummary(BaseModel):
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    calmar: float
    sharpe: float
    sortino: float = 0.0
    information_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    win_rate: float
    total_return: float
    benchmark_return: float
    excess_return: float
    avg_turnover: float
    total_cost: float
    rebalance_days: int
    holdings_per_rebalance: int


class RebalanceHolding(BaseModel):
    symbol: str
    name: str
    score: float


class RebalanceRecord(BaseModel):
    trade_date: date
    turnover: float
    estimated_cost: float
    holdings: list[RebalanceHolding]


class BacktestResponse(BaseModel):
    summary: BacktestSummary
    equity_curve: list[MetricPoint]
    picks: list[FactorResult]
    rebalances: list[RebalanceRecord]


class BacktestSensitivityPoint(BaseModel):
    rebalance_days: int
    top_n: int
    trading_cost_bps: float
    slippage_bps: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    calmar: float
    total_return: float
    excess_return: float
    sortino: float = 0.0
    information_ratio: float = 0.0
    avg_turnover: float = 0.0
    total_cost: float = 0.0


class BacktestSensitivityResponse(BaseModel):
    scan_id: str
    generated_at: str
    scan_width: int = 1
    rebalance_days_options: list[int]
    top_n_options: list[int]
    trading_cost_options: list[float]
    slippage_bps: float
    baseline: BacktestSensitivityPoint
    rows: list[BacktestSensitivityPoint]
    best_sharpe: BacktestSensitivityPoint
    best_annual_return: BacktestSensitivityPoint
    best_calmar: BacktestSensitivityPoint


class BacktestRollingPoint(BaseModel):
    trade_date: date
    rolling_sharpe: float
    rolling_volatility: float
    rolling_excess_return: float
    rolling_max_drawdown: float


class BacktestMonthlyPoint(BaseModel):
    month: str
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    win_rate: float
    max_drawdown: float


class BacktestRegimeStat(BaseModel):
    regime: str
    days: int
    annualized_return: float
    annualized_excess_return: float
    win_rate: float
    sharpe: float


class BacktestStabilitySummary(BaseModel):
    best_month: str
    worst_month: str
    best_month_excess_return: float
    worst_month_excess_return: float
    positive_month_ratio: float
    rolling_sharpe_mean: float
    rolling_sharpe_p10: float
    rolling_max_drawdown_worst: float


class BacktestStabilityResponse(BaseModel):
    analysis_id: str
    generated_at: str
    rolling_window: int
    summary: BacktestStabilitySummary
    rolling_points: list[BacktestRollingPoint]
    monthly_points: list[BacktestMonthlyPoint]
    regime_stats: list[BacktestRegimeStat]
    baseline_summary: BacktestSummary


class BacktestHistogramBin(BaseModel):
    left: float
    right: float
    count: int


class BacktestMonteCarloSummary(BaseModel):
    annual_return_p5: float
    annual_return_p50: float
    annual_return_p95: float
    total_return_p5: float
    total_return_p50: float
    total_return_p95: float
    max_drawdown_p50: float
    max_drawdown_p95: float
    loss_probability: float
    under_benchmark_probability: float


class BacktestMonteCarloResponse(BaseModel):
    analysis_id: str
    generated_at: str
    trials: int
    days: int
    summary: BacktestMonteCarloSummary
    annual_return_histogram: list[BacktestHistogramBin]
    max_drawdown_histogram: list[BacktestHistogramBin]
    baseline_summary: BacktestSummary


class BacktestScenarioItem(BaseModel):
    scenario_id: str
    scenario_name: str
    description: str
    rebalance_days: int
    top_n: int
    trading_cost_bps: float
    slippage_bps: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    calmar: float
    total_return: float
    excess_return: float
    sortino: float
    information_ratio: float
    avg_turnover: float
    total_cost: float
    score: float


class BacktestScenarioResponse(BaseModel):
    analysis_id: str
    generated_at: str
    rows: list[BacktestScenarioItem]
    champion: BacktestScenarioItem | None = None


class PaperSignal(BaseModel):
    symbol: str
    name: str
    trade_date: date
    predicted_return_5d: float
    score: float
    rank: int
    model_name: str
    run_at: str


class PaperAccount(BaseModel):
    account_id: str
    name: str
    initial_cash: float
    cash: float
    created_at: str
    updated_at: str
    market_value: float
    equity: float
    total_pnl: float
    total_return: float
    position_count: int
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    latest_equity_change: float = 0.0


class PaperEquityPoint(BaseModel):
    id: int
    account_id: str
    snapshot_at: str
    cash: float
    market_value: float
    equity: float
    note: str


class PaperPosition(BaseModel):
    account_id: str
    symbol: str
    name: str
    quantity: int
    sellable_quantity: int
    buy_locked_quantity: int = 0
    buy_locked_at: str | None = None
    avg_cost: float
    last_price: float
    updated_at: str
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class PaperOrder(BaseModel):
    id: int
    account_id: str
    created_at: str
    symbol: str
    name: str
    side: str
    quantity: int
    filled_quantity: int
    remaining_quantity: int
    fill_ratio: float
    price: float
    notional: float
    status: str
    source: str
    note: str


class PaperExecution(BaseModel):
    orders_created: int
    filled_orders: int = 0
    partial_orders: int = 0
    cancelled_orders: int = 0
    effective_fill_ratio: float = 0.0
    target_symbols: list[str]
    capital_fraction: float
    top_n: int
    stop_loss_pct: float
    take_profit_pct: float
    fill_ratio: float = 1.0
    plan_id: int | None = None


class PaperRebalanceLog(BaseModel):
    id: int
    account_id: str
    created_at: str
    top_n: int
    capital_fraction: float
    max_position_weight: float
    min_cash_buffer_ratio: float
    max_turnover_ratio: float
    orders_created: int
    turnover_ratio: float
    note: str
    target_symbols: list[str]


class PaperRiskEvent(BaseModel):
    id: int
    account_id: str
    created_at: str
    event_type: str
    severity: str
    title: str
    details: dict = Field(default_factory=dict)
    note: str


class PaperExecutionReport(BaseModel):
    id: int
    account_id: str
    created_at: str
    report_type: str
    title: str
    summary: dict = Field(default_factory=dict)
    note: str


class PaperDailySettings(BaseModel):
    account_id: str
    enabled: bool = False
    run_time: str = "15:10"
    auto_sync: bool = True
    auto_train: bool = True
    auto_rebalance: bool = True
    top_n: int = 3
    capital_fraction: float = 0.95
    max_position_weight: float = 0.35
    min_cash_buffer_ratio: float = 0.05
    max_turnover_ratio: float = 1.0
    stop_loss_pct: float = 0.1
    take_profit_pct: float = 0.2
    fill_ratio: float = 1.0
    max_drawdown_limit: float = 0.18
    max_equity_change_limit: float = 0.04
    updated_at: str


class PaperDailyRun(BaseModel):
    id: int
    account_id: str
    run_date: str
    created_at: str
    status: str
    steps: list[dict] = Field(default_factory=list)
    note: str


class PaperSchedulerStatus(BaseModel):
    loop_running: bool = False
    enabled: bool = False
    last_checked_at: str = ""
    last_triggered_at: str = ""
    last_completed_at: str = ""
    last_outcome: str = ""
    retry_from_step: str = ""
    next_run_at: str = ""
    next_retry_at: str = ""
    retry_attempt: int = 0
    max_retry_attempts: int = 0
    note: str = ""


class PaperPlannedOrder(BaseModel):
    symbol: str
    name: str
    side: str
    current_quantity: int
    target_quantity: int
    planned_quantity: int
    price: float
    notional: float
    reason: str


class PaperHoldingDelta(BaseModel):
    symbol: str
    name: str
    current_quantity: int
    target_quantity: int
    delta_quantity: int
    current_value: float
    target_value: float
    price: float
    weight_before: float
    weight_after: float
    unrealized_pnl_pct: float | None = None
    action: str
    reason: str


class PaperPreviewSummary(BaseModel):
    equity: float
    cash_before: float
    market_value_before: float
    planned_buy_notional: float
    planned_sell_notional: float
    estimated_cash_after: float
    estimated_position_ratio: float
    turnover_ratio: float
    blocked: bool
    warnings: list[str]
    target_symbols: list[str]
    forced_exit_count: int
    estimated_position_count_after: int
    locked_position_count: int = 0
    blocked_by_risk: bool = False
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    latest_equity_change: float = 0.0


class PaperPreviewConfig(BaseModel):
    top_n: int
    capital_fraction: float
    max_position_weight: float
    min_cash_buffer_ratio: float
    max_turnover_ratio: float
    stop_loss_pct: float
    take_profit_pct: float
    fill_ratio: float
    max_drawdown_limit: float
    max_equity_change_limit: float
    board_lot: int


class PaperRebalancePreview(BaseModel):
    plan_id: int | None = None
    created_at: str | None = None
    status: str = "pending"
    config: PaperPreviewConfig
    orders: list[PaperPlannedOrder]
    holdings: list[PaperHoldingDelta]
    summary: PaperPreviewSummary


class PaperTradingSnapshot(BaseModel):
    account: PaperAccount
    positions: list[PaperPosition]
    orders: list[PaperOrder]
    equity_curve: list[PaperEquityPoint] = []
    rebalances: list[PaperRebalanceLog] = []
    risk_events: list[PaperRiskEvent] = []
    reports: list[PaperExecutionReport] = []
    daily_settings: PaperDailySettings | None = None
    daily_runs: list[PaperDailyRun] = []
    scheduler_status: PaperSchedulerStatus | None = None
    signals: list[PaperSignal]
    execution: PaperExecution | None = None
    preview: PaperRebalancePreview | None = None


class PaperResetRequest(BaseModel):
    initial_cash: float = 1_000_000.0


class PaperRebalanceRequest(BaseModel):
    preview_id: int | None = None
    top_n: int = 3
    capital_fraction: float = 0.95
    max_position_weight: float = 0.35
    min_cash_buffer_ratio: float = 0.05
    max_turnover_ratio: float = 1.0
    stop_loss_pct: float = 0.1
    take_profit_pct: float = 0.2
    fill_ratio: float = 1.0
    max_drawdown_limit: float = 0.18
    max_equity_change_limit: float = 0.04


class PaperPreviewActionRequest(BaseModel):
    preview_id: int


class PaperOrderActionRequest(BaseModel):
    order_id: int
    fill_ratio: float = 1.0


class PaperDailySettingsRequest(BaseModel):
    enabled: bool = False
    run_time: str = "15:10"
    auto_sync: bool = True
    auto_train: bool = True
    auto_rebalance: bool = True
    top_n: int = 3
    capital_fraction: float = 0.95
    max_position_weight: float = 0.35
    min_cash_buffer_ratio: float = 0.05
    max_turnover_ratio: float = 1.0
    stop_loss_pct: float = 0.1
    take_profit_pct: float = 0.2
    fill_ratio: float = 1.0
    max_drawdown_limit: float = 0.18
    max_equity_change_limit: float = 0.04


class PaperDailyRunRequest(BaseModel):
    auto_sync: bool | None = None
    auto_train: bool | None = None
    auto_rebalance: bool | None = None
    start_from_step: str = "sync"


class ModelStatus(BaseModel):
    provider: str
    active_data_provider: str | None = None
    universe_source: str | None = None
    universe_index_code: str | None = None
    universe_index_name: str | None = None
    status: str
    last_train_at: str
    last_sync_at: str | None = None
    next_action: str
    notes: str
    universe_size: int = 0
    custom_universe_size: int = 0
    total_bars: int = 0
    latest_model_name: str | None = None
    validation_ic: float | None = None
    validation_directional_accuracy: float | None = None


class ModelPrediction(BaseModel):
    symbol: str
    name: str
    trade_date: date
    predicted_return_5d: float
    score: float
    rank: int
    model_name: str
    run_at: str


class SignalReview(BaseModel):
    model_run_id: int
    status: str = "pending"
    note: str = ""
    updated_at: str = ""


class SignalSuggestion(BaseModel):
    symbol: str
    name: str
    action: str
    predicted_return_5d: float
    score: float
    rank: int
    current_quantity: int
    target_quantity: int
    delta_quantity: int
    last_price: float
    current_weight: float
    target_weight: float
    suggested_value: float
    note: str


class SignalCenterSummary(BaseModel):
    model_run_id: int = 0
    model_name: str = ""
    active_provider: str = ""
    configured_provider: str = ""
    signal_trade_date: str = ""
    generated_at: str = ""
    top_n: int = 0
    candidate_count: int = 0
    current_position_count: int = 0
    target_position_count: int = 0
    account_equity: float = 0.0
    account_cash: float = 0.0
    capital_fraction: float = 0.0
    target_weight_per_position: float = 0.0
    avg_predicted_return_5d: float = 0.0
    estimated_turnover_count: int = 0
    review_status: str = "pending"
    warnings: list[str] = Field(default_factory=list)


class SignalCenterResponse(BaseModel):
    summary: SignalCenterSummary
    review: SignalReview
    action_items: list[SignalSuggestion] = Field(default_factory=list)
    target_positions: list[SignalSuggestion] = Field(default_factory=list)
    top_candidates: list[ModelPrediction] = Field(default_factory=list)


class SignalHistoryItem(BaseModel):
    model_run_id: int
    model_name: str
    provider: str
    generated_at: str
    signal_trade_date: str
    top_symbols: list[str] = Field(default_factory=list)
    top_names: list[str] = Field(default_factory=list)
    avg_predicted_return_5d: float = 0.0
    best_predicted_return_5d: float = 0.0
    review_status: str = "pending"
    review_note: str = ""
    review_updated_at: str = ""


class SignalReviewRequest(BaseModel):
    model_run_id: int
    status: str = "pending"
    note: str = ""


class ModelFeatureImportance(BaseModel):
    feature: str
    importance: float


class ModelRunInfo(BaseModel):
    model_name: str
    run_at: str
    validation_ic: float
    validation_directional_accuracy: float
    train_rows: int
    validation_rows: int
    note: str


class ModelCompareItem(BaseModel):
    model_name: str
    provider: str
    run_at: str
    validation_ic: float
    validation_directional_accuracy: float
    walk_forward_mean_ic: float
    walk_forward_positive_ic_ratio: float
    walk_forward_mean_long_short_return: float
    note: str
    is_champion: bool = False


class ModelDetailResponse(BaseModel):
    model_name: str
    provider: str
    run_at: str
    validation_ic: float
    validation_directional_accuracy: float
    train_rows: int
    validation_rows: int
    metrics: dict
    top_features: list[ModelFeatureImportance]
    note: str


class TrainResponse(BaseModel):
    status: str
    message: str
    model_name: str
    train_rows: int
    validation_rows: int
    validation_ic: float
    validation_directional_accuracy: float
    comparison: list[ModelCompareItem] = []


class UpdateResponse(BaseModel):
    status: str
    message: str
    symbols_synced: int = 0
    bars_written: int = 0


class CustomUniverseItem(BaseModel):
    symbol: str
    name: str


class CustomUniverseRequest(BaseModel):
    symbols: list[str] = []
    items: list[CustomUniverseItem] = []


class CustomUniverseResponse(BaseModel):
    status: str
    message: str
    items: list[CustomUniverseItem]
