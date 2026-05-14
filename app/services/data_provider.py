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
_TUSHARE_STOCK_BASIC_CACHE: dict[str, object] = {"loaded_at": 0.0, "frame": pd.DataFrame()}
_TUSHARE_STK_LIMIT_CACHE: dict[str, pd.DataFrame] = {}
_TUSHARE_SUSPEND_CACHE: dict[str, pd.DataFrame] = {}


def _merge_universe_items(base_items: list[UniverseItem], extra_items: list[UniverseItem] | None = None) -> list[UniverseItem]:
    merged: list[UniverseItem] = []
    seen: set[str] = set()

    for item in [*(base_items or []), *(extra_items or [])]:
        symbol = str(item.symbol).zfill(6)
        if symbol in seen:
            continue
        merged.append(UniverseItem(symbol=symbol, name=str(item.name)))
        seen.add(symbol)

    return merged


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

    def get_stock_basics(self, symbols: list[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["symbol", "name", "area", "industry", "market", "exchange", "list_date"]
        )

    def get_daily_basics(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["trade_date", "turnover_rate_f", "pe_ttm", "pb", "total_mv", "circ_mv"]
        )

    def get_price_limits(self, symbols: list[str], trade_date: datetime | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "trade_date", "up_limit", "down_limit", "pre_close"])

    def get_suspensions(self, trade_date: datetime | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "trade_date", "suspend_type", "suspend_timing"])

    def get_adj_factors(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return pd.DataFrame(columns=["trade_date", "adj_factor"])

    def get_moneyflow(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "buy_lg_amount",
                "sell_lg_amount",
                "buy_elg_amount",
                "sell_elg_amount",
                "net_mf_amount",
            ]
        )

    def get_financial_indicators(self, symbols: list[str] | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "symbol",
                "fin_ann_date",
                "fin_end_date",
                "roe_dt",
                "grossprofit_margin",
                "debt_to_assets",
                "ocfps",
            ]
        )


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
        base_universe = [
            UniverseItem(symbol="600519", name="贵州茅台"),
            UniverseItem(symbol="000001", name="平安银行"),
            UniverseItem(symbol="300750", name="宁德时代"),
            UniverseItem(symbol="601318", name="中国平安"),
            UniverseItem(symbol="600036", name="招商银行"),
        ]
        self._universe = _merge_universe_items(base_universe, custom_universe)

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
        if self.settings.universe_source.lower() == "index":
            try:
                base_universe = self._get_index_universe()
            except Exception:
                base_universe = self._get_spot_universe()
        else:
            base_universe = self._get_spot_universe()
        return _merge_universe_items(base_universe, self.custom_universe)

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
        # Keep a low-permission compatible default universe so Tushare users can
        # still sync/train before building a custom pool in the UI.
        return get_common_universe_catalog()[: self.settings.universe_limit]

    def get_universe(self) -> list[UniverseItem]:
        return _merge_universe_items(self._get_basic_universe(), self.custom_universe)

    def _load_stock_basic_frame(self) -> pd.DataFrame:
        loaded_at = float(_TUSHARE_STOCK_BASIC_CACHE.get("loaded_at", 0.0) or 0.0)
        cached_frame = _TUSHARE_STOCK_BASIC_CACHE.get("frame")
        if isinstance(cached_frame, pd.DataFrame) and not cached_frame.empty and (time() - loaded_at) < 12 * 60 * 60:
            return cached_frame.copy()

        frame = self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="symbol,name,area,industry,market,exchange,list_date",
        )
        if frame is None or frame.empty:
            return pd.DataFrame()

        normalized = frame.rename(columns=str).copy()
        normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
        normalized["name"] = normalized["name"].astype(str).str.strip()
        for column in ["area", "industry", "market", "exchange", "list_date"]:
            if column not in normalized.columns:
                normalized[column] = ""
            normalized[column] = normalized[column].fillna("").astype(str).str.strip()

        _TUSHARE_STOCK_BASIC_CACHE["loaded_at"] = time()
        _TUSHARE_STOCK_BASIC_CACHE["frame"] = normalized
        return normalized.copy()

    def _load_stock_basic_catalog(self) -> list[UniverseItem]:
        frame = self._load_stock_basic_frame()
        if frame.empty:
            return []
        return [
            UniverseItem(symbol=str(row["symbol"]).zfill(6), name=str(row["name"]).strip())
            for _, row in frame.iterrows()
            if row.get("symbol") and row.get("name")
        ]

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

    def get_stock_basics(self, symbols: list[str] | None = None) -> pd.DataFrame:
        frame = self._load_stock_basic_frame()
        if frame.empty:
            return frame
        if symbols:
            symbol_set = {str(symbol).zfill(6) for symbol in symbols}
            frame = frame[frame["symbol"].isin(symbol_set)]
        return frame[
            ["symbol", "name", "area", "industry", "market", "exchange", "list_date"]
        ].drop_duplicates("symbol")

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

    def get_daily_basics(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        ts_code = _symbol_to_ts_code(symbol)
        frame = self.pro.daily_basic(
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields="trade_date,turnover_rate_f,pe_ttm,pb,total_mv,circ_mv",
        )
        if frame is None or frame.empty:
            return pd.DataFrame(
                columns=["trade_date", "turnover_rate_f", "pe_ttm", "pb", "total_mv", "circ_mv"]
            )

        renamed = frame.rename(columns=str).copy()
        renamed["trade_date"] = pd.to_datetime(renamed["trade_date"], format="%Y%m%d").dt.date
        for column in ["turnover_rate_f", "pe_ttm", "pb", "total_mv", "circ_mv"]:
            if column not in renamed.columns:
                renamed[column] = np.nan
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
        renamed["total_mv"] = renamed["total_mv"] * 10_000.0
        renamed["circ_mv"] = renamed["circ_mv"] * 10_000.0
        return renamed[
            ["trade_date", "turnover_rate_f", "pe_ttm", "pb", "total_mv", "circ_mv"]
        ].sort_values("trade_date")

    def get_price_limits(self, symbols: list[str], trade_date: datetime | None = None) -> pd.DataFrame:
        if not symbols or trade_date is None:
            return pd.DataFrame(columns=["symbol", "trade_date", "up_limit", "down_limit", "pre_close"])

        trade_date_text = trade_date.strftime("%Y%m%d")
        cached = _TUSHARE_STK_LIMIT_CACHE.get(trade_date_text)
        if cached is None:
            frame = self.pro.stk_limit(
                trade_date=trade_date_text,
                fields="ts_code,trade_date,pre_close,up_limit,down_limit",
            )
            if frame is None or frame.empty:
                cached = pd.DataFrame(columns=["symbol", "trade_date", "up_limit", "down_limit", "pre_close"])
            else:
                cached = frame.rename(columns={"ts_code": "symbol"}).copy()
                cached["symbol"] = (
                    cached["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
                )
                cached["trade_date"] = pd.to_datetime(
                    cached["trade_date"], format="%Y%m%d", errors="coerce"
                ).dt.date
                for column in ["pre_close", "up_limit", "down_limit"]:
                    cached[column] = pd.to_numeric(cached[column], errors="coerce")
                cached = cached[["symbol", "trade_date", "up_limit", "down_limit", "pre_close"]]
            _TUSHARE_STK_LIMIT_CACHE[trade_date_text] = cached

        symbol_set = {str(symbol).zfill(6) for symbol in symbols}
        return cached[cached["symbol"].isin(symbol_set)].copy()

    def get_suspensions(self, trade_date: datetime | None = None) -> pd.DataFrame:
        if trade_date is None:
            return pd.DataFrame(columns=["symbol", "trade_date", "suspend_type", "suspend_timing"])

        trade_date_text = trade_date.strftime("%Y%m%d")
        cached = _TUSHARE_SUSPEND_CACHE.get(trade_date_text)
        if cached is None:
            frame = self.pro.suspend_d(
                trade_date=trade_date_text,
                fields="ts_code,trade_date,suspend_timing,suspend_type",
            )
            if frame is None or frame.empty:
                cached = pd.DataFrame(columns=["symbol", "trade_date", "suspend_type", "suspend_timing"])
            else:
                cached = frame.rename(columns={"ts_code": "symbol"}).copy()
                cached["symbol"] = (
                    cached["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
                )
                cached["trade_date"] = pd.to_datetime(
                    cached["trade_date"], format="%Y%m%d", errors="coerce"
                ).dt.date
                for column in ["suspend_type", "suspend_timing"]:
                    if column not in cached.columns:
                        cached[column] = ""
                    cached[column] = cached[column].fillna("").astype(str).str.strip()
                cached = cached[["symbol", "trade_date", "suspend_type", "suspend_timing"]]
            _TUSHARE_SUSPEND_CACHE[trade_date_text] = cached

        return cached.copy()

    def get_adj_factors(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        ts_code = _symbol_to_ts_code(symbol)
        frame = self.pro.adj_factor(
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["trade_date", "adj_factor"])

        renamed = frame.rename(columns=str).copy()
        renamed["trade_date"] = pd.to_datetime(renamed["trade_date"], format="%Y%m%d").dt.date
        renamed["adj_factor"] = pd.to_numeric(renamed["adj_factor"], errors="coerce")
        return renamed[["trade_date", "adj_factor"]].sort_values("trade_date")

    def get_moneyflow(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        ts_code = _symbol_to_ts_code(symbol)
        frame = self.pro.moneyflow(
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields="trade_date,buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount,net_mf_amount",
        )
        if frame is None or frame.empty:
            return pd.DataFrame(
                columns=[
                    "trade_date",
                    "buy_lg_amount",
                    "sell_lg_amount",
                    "buy_elg_amount",
                    "sell_elg_amount",
                    "net_mf_amount",
                ]
            )

        renamed = frame.rename(columns=str).copy()
        renamed["trade_date"] = pd.to_datetime(renamed["trade_date"], format="%Y%m%d").dt.date
        for column in [
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
            "net_mf_amount",
        ]:
            if column not in renamed.columns:
                renamed[column] = np.nan
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce").fillna(0.0) * 10_000.0
        return renamed[
            [
                "trade_date",
                "buy_lg_amount",
                "sell_lg_amount",
                "buy_elg_amount",
                "sell_elg_amount",
                "net_mf_amount",
            ]
        ].sort_values("trade_date")

    def get_financial_indicators(self, symbols: list[str] | None = None) -> pd.DataFrame:
        symbol_list = [str(symbol).zfill(6) for symbol in symbols or []]
        if not symbol_list:
            symbol_list = [item.symbol for item in self.get_universe()]

        rows: list[dict[str, object]] = []
        for symbol in symbol_list:
            ts_code = _symbol_to_ts_code(symbol)
            frame = self.pro.fina_indicator(
                ts_code=ts_code,
                fields="ts_code,ann_date,end_date,roe_dt,grossprofit_margin,debt_to_assets,ocfps",
            )
            if frame is None or frame.empty:
                continue

            renamed = frame.rename(columns=str).copy()
            for column in ["ann_date", "end_date"]:
                if column not in renamed.columns:
                    renamed[column] = ""
                renamed[column] = renamed[column].fillna("").astype(str).str.strip()
            for column in ["roe_dt", "grossprofit_margin", "debt_to_assets", "ocfps"]:
                if column not in renamed.columns:
                    renamed[column] = np.nan
                renamed[column] = pd.to_numeric(renamed[column], errors="coerce")

            renamed = renamed.sort_values(["ann_date", "end_date"], ascending=False, na_position="last")
            latest = renamed.iloc[0]
            rows.append(
                {
                    "symbol": symbol,
                    "fin_ann_date": str(latest.get("ann_date", "") or ""),
                    "fin_end_date": str(latest.get("end_date", "") or ""),
                    "roe_dt": float(latest["roe_dt"]) if pd.notna(latest.get("roe_dt")) else None,
                    "grossprofit_margin": float(latest["grossprofit_margin"])
                    if pd.notna(latest.get("grossprofit_margin"))
                    else None,
                    "debt_to_assets": float(latest["debt_to_assets"])
                    if pd.notna(latest.get("debt_to_assets"))
                    else None,
                    "ocfps": float(latest["ocfps"]) if pd.notna(latest.get("ocfps")) else None,
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "fin_ann_date",
                    "fin_end_date",
                    "roe_dt",
                    "grossprofit_margin",
                    "debt_to_assets",
                    "ocfps",
                ]
            )
        return pd.DataFrame(rows)


def get_provider(custom_universe: list[UniverseItem] | None = None) -> BaseProvider:
    settings = get_settings()
    provider_name = settings.data_provider.lower()
    if provider_name == "akshare":
        return AkshareProvider(custom_universe=custom_universe)
    if provider_name == "tushare":
        return TushareProvider().with_custom_universe(custom_universe)
    return MockAShareProvider(custom_universe=custom_universe)
