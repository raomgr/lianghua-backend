from __future__ import annotations

from datetime import datetime

from app.config import get_settings
from app.services.storage import MarketRepository


def _status_note(configured_provider: str, active_provider: str, raw_note: str) -> str:
    if active_provider != configured_provider:
        return (
            f"{configured_provider} 当前不可用，页面正在使用 {active_provider} 数据。"
            " 你可以继续本地开发；等网络恢复后再切回真实数据同步。"
        )
    if raw_note and raw_note != "sync complete":
        return (
            f"{configured_provider} 最新一次同步失败，当前继续使用本地缓存数据。"
            " 如需刷新真实数据，请检查网络或代理配置后重试。"
        )
    return "Current version uses rule-based factor ranking and stores market history locally for iterative upgrades."


def resolve_active_provider(repo: MarketRepository, preferred_provider: str | None = None) -> str:
    settings = get_settings()
    configured_provider = preferred_provider or settings.data_provider
    if repo.has_data(provider=configured_provider):
        return configured_provider
    if repo.has_data(provider="mock-fallback"):
        return "mock-fallback"
    if repo.load_latest_model_run(provider=configured_provider):
        return configured_provider
    if repo.load_latest_model_run(provider="mock-fallback"):
        return "mock-fallback"
    return configured_provider


def get_model_status() -> dict:
    settings = get_settings()
    repo = MarketRepository()
    configured_provider = settings.data_provider
    active_provider = resolve_active_provider(repo, configured_provider)
    configured_sync = repo.load_latest_sync(provider=configured_provider)
    active_sync = repo.load_latest_sync(provider=active_provider)
    latest_model = repo.load_latest_model_run(provider=active_provider)
    note = _status_note(configured_provider, active_provider, configured_sync.get("note", ""))
    return {
        "provider": configured_provider,
        "active_data_provider": active_provider,
        "universe_source": settings.universe_source,
        "universe_index_code": settings.universe_index_code,
        "universe_index_name": settings.universe_index_name,
        "status": "trained-local-model" if latest_model else "baseline-ready",
        "last_train_at": latest_model.get("run_at", "") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_sync_at": active_sync.get("run_at"),
        "next_action": "extend to gradient boosting model and rolling walk-forward retrain",
        "notes": note,
        "universe_size": len(repo.load_universe(provider=active_provider)),
        "custom_universe_size": len(repo.load_custom_universe()),
        "total_bars": repo.load_bar_count(provider=active_provider),
        "latest_model_name": latest_model.get("model_name"),
        "validation_ic": latest_model.get("validation_ic"),
        "validation_directional_accuracy": latest_model.get("validation_directional_accuracy"),
    }
