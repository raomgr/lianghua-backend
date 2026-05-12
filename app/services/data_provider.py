from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import requests

from app.config import get_settings

try:
    import akshare as ak
except Exception:  # pragma: no cover - optional dependency at runtime
    ak = None

try:
    import tushare as ts
except Exception:  # pragma: no cover - optional dependency at runtime
    ts = None


@dataclass
class UniverseItem:
    symbol: str
    name: str


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "a_share_catalog.json"


def _load_catalog() -> list[UniverseItem]:
    if not CATALOG_PATH.exists():
        return []
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        raw_items = json.load(handle)
    return [
        UniverseItem(symbol=str(item["symbol"]).zfill(6), name=str(item["name"]))
        for item in raw_items
        if item.get("symbol") and item.get("name")
    ]


COMMON_A_SHARE_CATALOG = _load_catalog()
_TUSHARE_STOCK_BASIC_CACHE: dict[str, object] = {"loaded_at": 0.0, "items": []}


class BaseProvider:
    def get_universe(self) -> list[UniverseItem]:
        raise NotImplementedError

    def get_daily_bars(self, symbol: str, limit: int = 120) -> pd.DataFrame:
        raise NotImplementedError

    def get_universe_snapshot(self) -> pd.DataFrame:
        frames = []
        for item in self.get_universe():
            bars = self.get_daily_bars(item.symbol, limit=40)
            latest = bars.iloc[-1]
            prev = bars.iloc[-2]
            frames.append(
                {
                    "symbol": item.symbol,
                    "name": item.name,
                    "latest_price": round(float(latest["close"]), 2),
                    "pct_change": round((float(latest["close"]) / float(prev["close"]) - 1) * 100, 2),
                    "volume": round(float(latest["volume"]), 2),
                }
            )
        return pd.DataFrame(frames)

    def search_universe(self, query_text: str, limit: int = 12) -> list[UniverseItem]:
        return []


def _symbol_to_ts_code(symbol: str) -> str:
    clean = str(symbol).strip().upper()
    if "." in clean:
        return clean
    if clean.startswith(("600", "601", "603", "605", "688", "900")):
        return f"{clean}.SH"
    return f"{clean}.SZ"


def _normalize_tushare_index_code(index_code: str) -> str:
    clean = str(index_code).strip().upper()
    if "." in clean:
        return clean

    explicit_map = {
        "000300": "000300.SH",
        "000905": "000905.SH",
        "000852": "000852.SH",
        "000016": "000016.SH",
        "399300": "399300.SZ",
        "399905": "399905.SZ",
        "399006": "399006.SZ",
    }
    if clean in explicit_map:
        return explicit_map[clean]
    if clean.startswith(("000", "001", "5", "6")):
        return f"{clean}.SH"
    return f"{clean}.SZ"


def get_common_universe_catalog() -> list[UniverseItem]:
    return COMMON_A_SHARE_CATALOG.copy()


class MockAShareProvider(BaseProvider):
    def __init__(self, custom_universe: list[UniverseItem] | None = None) -> None:
        self._universe = custom_universe or [
            UniverseItem(symbol="600519", name="贵州茅台"),
            UniverseItem(symbol="000001", name="平安银行"),
            UniverseItem(symbol="300750", name="宁德时代"),
            UniverseItem(symbol="601318", name="中国平安"),
            UniverseItem(symbol="600036", name="招商银行"),
        ]

    def get_universe(self) -> list[UniverseItem]:
        return self._universe

    def get_daily_bars(self, symbol: str, limit: int = 120) -> pd.DataFrame:
        seed = sum(ord(char) for char in symbol)
        rng = np.random.default_rng(seed)
        start_price = 20 + (seed % 150)
        dates = pd.date_range(end=datetime.today(), periods=limit, freq="B")

        drift = rng.normal(0.0008, 0.01, size=limit)
        close = start_price * np.cumprod(1 + drift)
        open_ = close * (1 + rng.normal(0, 0.003, size=limit))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.015, size=limit))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.015, size=limit))
        volume = rng.integers(2_000_000, 25_000_000, size=limit).astype(float)
        amount = (close * volume).astype(float)
        turnover_rate = rng.uniform(0.2, 4.8, size=limit)

        return pd.DataFrame(
            {
                "trade_date": dates.date,
                "open": open_.round(2),
                "high": high.round(2),
                "low": low.round(2),
                "close": close.round(2),
                "volume": volume,
                "amount": amount.round(2),
                "turnover_rate": turnover_rate.round(4),
            }
        )


class AkshareProvider(BaseProvider):
    def __init__(self, custom_universe: list[UniverseItem] | None = None) -> None:
        if ak is None:
            raise RuntimeError("AKShare is not installed.")
        self.settings = get_settings()
        self.custom_universe = custom_universe or []

    @contextmanager
    def _proxy_guard(self):
        if not self.settings.akshare_disable_env_proxy:
            yield
            return

        proxy_keys = [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ]
        backup = {key: os.environ.get(key) for key in proxy_keys}
        original_session_init = requests.sessions.Session.__init__

        def patched_session_init(session, *args, **kwargs):
            original_session_init(session, *args, **kwargs)
            session.trust_env = False

        try:
            for key in proxy_keys:
                os.environ.pop(key, None)
            requests.sessions.Session.__init__ = patched_session_init
            yield
        finally:
            requests.sessions.Session.__init__ = original_session_init
            for key, value in backup.items():
                if value is not None:
                    os.environ[key] = value

    def _get_index_universe(self) -> list[UniverseItem]:
        with self._proxy_guard():
            frame = ak.index_stock_cons_csindex(symbol=self.settings.universe_index_code)
        sample = frame[["成分券代码", "成分券名称"]].dropna().drop_duplicates().head(self.settings.universe_limit)
        return [
            UniverseItem(symbol=str(row["成分券代码"]).zfill(6), name=str(row["成分券名称"]))
            for _, row in sample.iterrows()
        ]

    def _get_spot_universe(self) -> list[UniverseItem]:
        with self._proxy_guard():
            spot = ak.stock_zh_a_spot_em()
        sample = spot.head(self.settings.universe_limit)
        return [
            UniverseItem(symbol=str(row["代码"]).zfill(6), name=str(row["名称"]))
            for _, row in sample.iterrows()
        ]

    def get_universe(self) -> list[UniverseItem]:
        if self.custom_universe:
            return self.custom_universe[: self.settings.universe_limit]
        if self.settings.universe_source.lower() == "index":
            try:
                return self._get_index_universe()
            except Exception:
                return self._get_spot_universe()
        return self._get_spot_universe()

    def get_daily_bars(self, symbol: str, limit: int = 120) -> pd.DataFrame:
        end = datetime.today()
        start = end - timedelta(days=limit * 2)
        with self._proxy_guard():
            frame = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            )
        renamed = frame.rename(
            columns={
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover_rate",
            }
        )
        renamed["trade_date"] = pd.to_datetime(renamed["trade_date"]).dt.date
        if "amount" not in renamed.columns:
            renamed["amount"] = renamed["close"] * renamed["volume"]
        if "turnover_rate" not in renamed.columns:
            renamed["turnover_rate"] = 0.0
        return renamed[
            ["trade_date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"]
        ].tail(limit)


class TushareProvider(BaseProvider):
    def __init__(self) -> None:
        if ts is None:
            raise RuntimeError("Tushare is not installed.")
        self.settings = get_settings()
        if not self.settings.tushare_token:
            raise RuntimeError("Tushare token is missing. Please set TUSHARE_TOKEN in backend/.env.")
        self.pro = ts.pro_api(self.settings.tushare_token)
        self.custom_universe: list[UniverseItem] = []

    def with_custom_universe(self, custom_universe: list[UniverseItem] | None) -> "TushareProvider":
        self.custom_universe = custom_universe or []
        return self

    def _get_basic_universe(self) -> list[UniverseItem]:
        if self.custom_universe:
            return self.custom_universe[: self.settings.universe_limit]
        # Keep a low-permission compatible default universe so Tushare users can
        # still sync/train before building a custom pool in the UI.
        return get_common_universe_catalog()[: self.settings.universe_limit]

    def get_universe(self) -> list[UniverseItem]:
        return self._get_basic_universe()

    def _load_stock_basic_catalog(self) -> list[UniverseItem]:
        loaded_at = float(_TUSHARE_STOCK_BASIC_CACHE.get("loaded_at", 0.0) or 0.0)
        cached_items = _TUSHARE_STOCK_BASIC_CACHE.get("items", [])
        if cached_items and (time() - loaded_at) < 12 * 60 * 60:
            return list(cached_items)

        frame = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="symbol,name",
        )
        if frame is None or frame.empty:
            return []

        items = [
            UniverseItem(symbol=str(row["symbol"]).zfill(6), name=str(row["name"]).strip())
            for _, row in frame.iterrows()
            if row.get("symbol") and row.get("name")
        ]
        _TUSHARE_STOCK_BASIC_CACHE["loaded_at"] = time()
        _TUSHARE_STOCK_BASIC_CACHE["items"] = items
        return list(items)

    def search_universe(self, query_text: str, limit: int = 12) -> list[UniverseItem]:
        normalized = query_text.strip().lower()
        if not normalized:
            return []

        items = self._load_stock_basic_catalog()
        if not items:
            return []

        def match_score(item: UniverseItem) -> tuple[int, str]:
            symbol = item.symbol.lower()
            name = item.name.lower()
            if symbol == normalized or name == normalized:
                return (0, item.symbol)
            if symbol.startswith(normalized):
                return (1, item.symbol)
            if name.startswith(normalized):
                return (2, item.symbol)
            if normalized in symbol:
                return (3, item.symbol)
            if normalized in name:
                return (4, item.symbol)
            return (99, item.symbol)

        matched = [item for item in items if normalized in item.symbol.lower() or normalized in item.name.lower()]
        matched.sort(key=match_score)
        return matched[:limit]

    def get_daily_bars(self, symbol: str, limit: int = 120) -> pd.DataFrame:
        end = datetime.today()
        start = end - timedelta(days=limit * 3)
        ts_code = _symbol_to_ts_code(symbol)

        frame = self.pro.daily(
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if frame is None or frame.empty:
            raise RuntimeError(f"Tushare daily returned no rows for {ts_code}.")

        renamed = frame.rename(
            columns={
                "trade_date": "trade_date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "amount": "amount",
            }
        )
        renamed["trade_date"] = pd.to_datetime(renamed["trade_date"], format="%Y%m%d").dt.date

        if "amount" not in renamed.columns:
            renamed["amount"] = 0.0
        # Tushare `daily.amount` is reported in thousand CNY, while the rest of
        # the app formats成交额 as plain CNY. Normalize it here so downstream UI
        # and storage can use a single yuan-based convention.
        renamed["amount"] = renamed["amount"].fillna(0.0).astype(float) * 1000.0
        renamed["turnover_rate"] = 0.0
        renamed = renamed.sort_values("trade_date").reset_index(drop=True)
        return renamed[
            ["trade_date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"]
        ].tail(limit)


def get_provider(custom_universe: list[UniverseItem] | None = None) -> BaseProvider:
    settings = get_settings()
    provider_name = settings.data_provider.lower()
    if provider_name == "akshare":
        return AkshareProvider(custom_universe=custom_universe)
    if provider_name == "tushare":
        return TushareProvider().with_custom_universe(custom_universe)
    return MockAShareProvider(custom_universe=custom_universe)
