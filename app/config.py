from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "A-Share Quant API"
    data_provider: str = "mock"
    frontend_origin: str = "http://localhost:5173"
    frontend_origins: str = ""
    data_dir: str = "data"
    database_url: str = "data/ashare_quant.db"
    tushare_token: str = ""
    universe_limit: int = 20
    universe_source: str = "index"
    universe_index_code: str = "000300"
    universe_index_name: str = "沪深300"
    akshare_disable_env_proxy: bool = True
    fallback_to_mock_on_data_error: bool = True

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def data_path(self) -> Path:
        raw = Path(self.data_dir)
        return raw if raw.is_absolute() else BACKEND_DIR / raw

    @property
    def database_path(self) -> Path:
        raw = Path(self.database_url)
        return raw if raw.is_absolute() else BACKEND_DIR / raw

    @property
    def cors_origins(self) -> list[str]:
        configured = [item.strip() for item in self.frontend_origins.split(",") if item.strip()]
        if not configured:
            configured = [self.frontend_origin]

        origins: list[str] = []
        for item in [*configured, "http://localhost:3000", "http://localhost:5173"]:
            if item not in origins:
                origins.append(item)
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()
