from datetime import datetime

from app.config import get_settings
from app.services.market_service import MarketService


def main() -> None:
    settings = get_settings()
    service = MarketService()
    sync = service.refresh_market_data()
    factors = service.get_factor_table()
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
        f"provider={settings.data_provider} active_provider={service.active_data_provider}"
    )
    print(
        f"synced symbols={sync['symbols_synced']} bars={sync['bars_written']} "
        f"db={settings.database_url}"
    )
    if service.last_sync_warning:
        print(f"warning={service.last_sync_warning}")
    if service.last_sync_error_detail:
        print(f"error_detail={service.last_sync_error_detail}")
    if not factors.empty:
        print(factors[["symbol", "name", "score"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
