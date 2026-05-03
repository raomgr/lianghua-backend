from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from app.config import get_settings
from app.services.data_provider import BaseProvider


class MarketRepository:
    def __init__(self) -> None:
        settings = get_settings()
        self.default_provider = settings.data_provider
        self.db_path = settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS universe (
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(symbol, provider)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_bars (
                    symbol TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    amount REAL NOT NULL DEFAULT 0,
                    turnover_rate REAL NOT NULL DEFAULT 0,
                    provider TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(symbol, trade_date, provider)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    run_at TEXT NOT NULL,
                    symbols_synced INTEGER NOT NULL,
                    bars_written INTEGER NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "daily_bars", "amount", "REAL NOT NULL DEFAULT 0")
            self._ensure_column(conn, "daily_bars", "turnover_rate", "REAL NOT NULL DEFAULT 0")
            self._migrate_universe_primary_key(conn)
            self._migrate_daily_bars_primary_key(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    run_at TEXT NOT NULL,
                    train_rows INTEGER NOT NULL,
                    validation_rows INTEGER NOT NULL,
                    validation_ic REAL NOT NULL,
                    validation_directional_accuracy REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    coefficients_json TEXT NOT NULL,
                    feature_stats_json TEXT NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    predicted_return_5d REAL NOT NULL,
                    score REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(model_run_id) REFERENCES model_runs(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS custom_universe (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_accounts (
                    account_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    initial_cash REAL NOT NULL,
                    cash REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_positions (
                    account_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    last_price REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(account_id, symbol)
                )
                """
            )
            self._ensure_column(conn, "paper_positions", "sellable_quantity", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "paper_positions", "buy_locked_quantity", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "paper_positions", "buy_locked_at", "TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    name TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    notional REAL NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "paper_orders", "filled_quantity", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "paper_orders", "remaining_quantity", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "paper_orders", "fill_ratio", "REAL NOT NULL DEFAULT 1")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    snapshot_at TEXT NOT NULL,
                    cash REAL NOT NULL,
                    market_value REAL NOT NULL,
                    equity REAL NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_rebalance_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    top_n INTEGER NOT NULL,
                    capital_fraction REAL NOT NULL,
                    max_position_weight REAL NOT NULL,
                    min_cash_buffer_ratio REAL NOT NULL,
                    max_turnover_ratio REAL NOT NULL,
                    orders_created INTEGER NOT NULL,
                    turnover_ratio REAL NOT NULL,
                    note TEXT NOT NULL,
                    target_symbols_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_rebalance_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    preview_json TEXT NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_execution_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_daily_settings (
                    account_id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    run_time TEXT NOT NULL DEFAULT '15:10',
                    auto_sync INTEGER NOT NULL DEFAULT 1,
                    auto_train INTEGER NOT NULL DEFAULT 1,
                    auto_rebalance INTEGER NOT NULL DEFAULT 1,
                    top_n INTEGER NOT NULL DEFAULT 3,
                    capital_fraction REAL NOT NULL DEFAULT 0.95,
                    max_position_weight REAL NOT NULL DEFAULT 0.35,
                    min_cash_buffer_ratio REAL NOT NULL DEFAULT 0.05,
                    max_turnover_ratio REAL NOT NULL DEFAULT 1.0,
                    stop_loss_pct REAL NOT NULL DEFAULT 0.1,
                    take_profit_pct REAL NOT NULL DEFAULT 0.2,
                    fill_ratio REAL NOT NULL DEFAULT 1.0,
                    max_drawdown_limit REAL NOT NULL DEFAULT 0.18,
                    max_equity_change_limit REAL NOT NULL DEFAULT 0.04,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_daily_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    note TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_reviews (
                    model_run_id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    note TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(model_run_id) REFERENCES model_runs(id)
                )
                """
            )

    def _table_primary_key_columns(self, conn: sqlite3.Connection, table_name: str) -> list[str]:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall() if int(row[5]) > 0]

    def _migrate_universe_primary_key(self, conn: sqlite3.Connection) -> None:
        if self._table_primary_key_columns(conn, "universe") == ["symbol", "provider"]:
            return

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS universe_v2 (
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                provider TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(symbol, provider)
            )
            """
        )
        conn.execute("DELETE FROM universe_v2")
        conn.execute(
            """
            INSERT INTO universe_v2(symbol, name, provider, updated_at)
            SELECT symbol, name, provider, updated_at
            FROM universe
            """
        )
        conn.execute("DROP TABLE universe")
        conn.execute("ALTER TABLE universe_v2 RENAME TO universe")

    def _migrate_daily_bars_primary_key(self, conn: sqlite3.Connection) -> None:
        if self._table_primary_key_columns(conn, "daily_bars") == ["symbol", "trade_date", "provider"]:
            return

        columns = {row[1] for row in conn.execute("PRAGMA table_info(daily_bars)").fetchall()}
        amount_expr = "amount" if "amount" in columns else "0"
        turnover_expr = "turnover_rate" if "turnover_rate" in columns else "0"

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_bars_v2 (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                amount REAL NOT NULL DEFAULT 0,
                turnover_rate REAL NOT NULL DEFAULT 0,
                provider TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(symbol, trade_date, provider)
            )
            """
        )
        conn.execute("DELETE FROM daily_bars_v2")
        conn.execute(
            f"""
            INSERT INTO daily_bars_v2(
                symbol, trade_date, open, high, low, close, volume, amount, turnover_rate, provider, updated_at
            )
            SELECT
                symbol,
                trade_date,
                open,
                high,
                low,
                close,
                volume,
                {amount_expr},
                {turnover_expr},
                provider,
                updated_at
            FROM daily_bars
            """
        )
        conn.execute("DROP TABLE daily_bars")
        conn.execute("ALTER TABLE daily_bars_v2 RENAME TO daily_bars")

    def _ensure_column(self, conn: sqlite3.Connection, table_name: str, column_name: str, ddl: str) -> None:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}")

    def record_sync_run(
        self,
        provider_name: str,
        symbols_synced: int,
        bars_written: int,
        note: str,
        run_at: str | None = None,
    ) -> str:
        timestamp = run_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sync_runs(provider, run_at, symbols_synced, bars_written, note)
                VALUES(?, ?, ?, ?, ?)
                """,
                (provider_name, timestamp, symbols_synced, bars_written, note),
            )
        return timestamp

    def sync_provider_data(self, provider: BaseProvider, provider_name: str) -> dict:
        universe = provider.get_universe()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bars_written = 0

        with self.connect() as conn:
            for item in universe:
                conn.execute(
                    """
                    INSERT INTO universe(symbol, name, provider, updated_at)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(symbol, provider) DO UPDATE SET
                        name=excluded.name,
                        updated_at=excluded.updated_at
                    """,
                    (item.symbol, item.name, provider_name, now),
                )

                bars = provider.get_daily_bars(item.symbol, limit=180).copy()
                bars["trade_date"] = bars["trade_date"].astype(str)
                for _, row in bars.iterrows():
                    conn.execute(
                        """
                        INSERT INTO daily_bars(
                            symbol, trade_date, open, high, low, close, volume, amount, turnover_rate, provider, updated_at
                        )
                        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(symbol, trade_date, provider) DO UPDATE SET
                            open=excluded.open,
                            high=excluded.high,
                            low=excluded.low,
                            close=excluded.close,
                            volume=excluded.volume,
                            amount=excluded.amount,
                            turnover_rate=excluded.turnover_rate,
                            updated_at=excluded.updated_at
                        """,
                        (
                            item.symbol,
                            row["trade_date"],
                            float(row["open"]),
                            float(row["high"]),
                            float(row["low"]),
                            float(row["close"]),
                            float(row["volume"]),
                            float(row.get("amount", 0.0)),
                            float(row.get("turnover_rate", 0.0)),
                            provider_name,
                            now,
                        ),
                    )
                bars_written += len(bars)

            conn.execute(
                """
                INSERT INTO sync_runs(provider, run_at, symbols_synced, bars_written, note)
                VALUES(?, ?, ?, ?, ?)
                """,
                (provider_name, now, len(universe), bars_written, "sync complete"),
            )

        return {
            "symbols_synced": len(universe),
            "bars_written": bars_written,
            "run_at": now,
        }

    def has_data(self, provider: str | None = None) -> bool:
        active_provider = provider or self.default_provider
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM daily_bars WHERE provider = ?",
                (active_provider,),
            ).fetchone()
        return bool(row and row[0] > 0)

    def load_symbol_history(self, symbol: str, limit: int = 180, provider: str | None = None) -> pd.DataFrame:
        active_provider = provider or self.default_provider
        query = """
        SELECT trade_date, open, high, low, close, volume, amount, turnover_rate
        FROM daily_bars
        WHERE symbol = ? AND provider = ?
        ORDER BY trade_date DESC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(symbol, active_provider, limit))

        if frame.empty:
            return frame

        frame = frame.sort_values("trade_date").reset_index(drop=True)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        return frame

    def load_universe(self, provider: str | None = None) -> pd.DataFrame:
        active_provider = provider or self.default_provider
        with self.connect() as conn:
            return pd.read_sql_query(
                "SELECT symbol, name FROM universe WHERE provider = ? ORDER BY symbol",
                conn,
                params=(active_provider,),
            )

    def load_all_histories(self, limit: int = 180, provider: str | None = None) -> dict[str, pd.DataFrame]:
        active_provider = provider or self.default_provider
        universe = self.load_universe(provider=active_provider)
        histories: dict[str, pd.DataFrame] = {}
        for _, row in universe.iterrows():
            history = self.load_symbol_history(row["symbol"], limit=limit, provider=active_provider)
            if not history.empty:
                histories[row["symbol"]] = history
        return histories

    def load_latest_sync(self, provider: str | None = None) -> dict:
        active_provider = provider or self.default_provider
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT provider, run_at, symbols_synced, bars_written, note
                FROM sync_runs
                WHERE provider = ?
                ORDER BY id DESC
                LIMIT 1
                """
            , (active_provider,)).fetchone()

        if not row:
            return {
                "provider": "unknown",
                "run_at": "never",
                "symbols_synced": 0,
                "bars_written": 0,
                "note": "No sync has been run yet.",
            }

        return {
            "provider": row[0],
            "run_at": row[1],
            "symbols_synced": int(row[2]),
            "bars_written": int(row[3]),
            "note": row[4],
        }

    def load_bar_count(self, provider: str | None = None) -> int:
        active_provider = provider or self.default_provider
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM daily_bars WHERE provider = ?",
                (active_provider,),
            ).fetchone()
        return int(row[0]) if row else 0

    def record_model_run(
        self,
        *,
        model_name: str,
        provider: str,
        run_at: str,
        train_rows: int,
        validation_rows: int,
        validation_ic: float,
        validation_directional_accuracy: float,
        metrics: dict,
        coefficients: dict,
        feature_stats: dict,
        note: str,
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO model_runs(
                    model_name,
                    provider,
                    run_at,
                    train_rows,
                    validation_rows,
                    validation_ic,
                    validation_directional_accuracy,
                    metrics_json,
                    coefficients_json,
                    feature_stats_json,
                    note
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    provider,
                    run_at,
                    train_rows,
                    validation_rows,
                    validation_ic,
                    validation_directional_accuracy,
                    json.dumps(metrics, ensure_ascii=True),
                    json.dumps(coefficients, ensure_ascii=True),
                    json.dumps(feature_stats, ensure_ascii=True),
                    note,
                ),
            )
            return int(cursor.lastrowid)

    def replace_predictions(self, model_run_id: int, predictions: list[dict], created_at: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM model_predictions WHERE model_run_id = ?", (model_run_id,))
            for row in predictions:
                conn.execute(
                    """
                    INSERT INTO model_predictions(
                        model_run_id,
                        symbol,
                        name,
                        trade_date,
                        predicted_return_5d,
                        score,
                        rank,
                        created_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_run_id,
                        row["symbol"],
                        row["name"],
                        str(row["trade_date"]),
                        row["predicted_return_5d"],
                        row["score"],
                        row["rank"],
                        created_at,
                    ),
                )

    def load_latest_model_run(self, provider: str | None = None, model_name: str | None = None) -> dict:
        active_provider = provider or self.default_provider
        params: list[object] = [active_provider]
        model_name_clause = ""
        if model_name:
            model_name_clause = "AND model_name = ?"
            params.append(model_name)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT
                    id,
                    model_name,
                    provider,
                    run_at,
                    train_rows,
                    validation_rows,
                    validation_ic,
                    validation_directional_accuracy,
                    metrics_json,
                    coefficients_json,
                    feature_stats_json,
                    note
                FROM model_runs
                WHERE provider = ?
                {model_name_clause}
                ORDER BY id DESC
                LIMIT 1
                """
            , params).fetchone()

        if not row:
            return {}

        return {
            "id": int(row[0]),
            "model_name": row[1],
            "provider": row[2],
            "run_at": row[3],
            "train_rows": int(row[4]),
            "validation_rows": int(row[5]),
            "validation_ic": float(row[6]),
            "validation_directional_accuracy": float(row[7]),
            "metrics": json.loads(row[8]),
            "coefficients": json.loads(row[9]),
            "feature_stats": json.loads(row[10]),
            "note": row[11],
        }

    def load_latest_predictions(
        self,
        limit: int = 10,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> list[dict]:
        active_provider = provider or self.default_provider
        model_name_clause = ""
        params: list[object] = [active_provider]
        if model_name:
            model_name_clause = "AND model_name = ?"
            params.append(model_name)
        params.append(limit)
        query = """
        SELECT
            p.symbol,
            p.name,
            p.trade_date,
            p.predicted_return_5d,
            p.score,
            p.rank,
            mr.model_name,
            mr.run_at
        FROM model_predictions p
        JOIN model_runs mr ON mr.id = p.model_run_id
        WHERE p.model_run_id = (
            SELECT id FROM model_runs WHERE provider = ? {model_name_clause} ORDER BY id DESC LIMIT 1
        )
        ORDER BY p.rank ASC
        LIMIT ?
        """.format(model_name_clause=model_name_clause)
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=params)

        if frame.empty:
            return []

        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        fallback_date = None
        if frame["trade_date"].isna().any():
            fallback_date = datetime.now().date()
            frame["trade_date"] = frame["trade_date"].where(frame["trade_date"].notna(), fallback_date)
        frame["predicted_return_5d"] = frame["predicted_return_5d"].astype(float)
        frame["score"] = frame["score"].astype(float)
        frame["rank"] = frame["rank"].astype(int)
        return frame.to_dict(orient="records")

    def load_recent_model_runs(self, limit: int = 5, provider: str | None = None) -> list[dict]:
        active_provider = provider or self.default_provider
        query = """
        SELECT
            model_name,
            run_at,
            validation_ic,
            validation_directional_accuracy,
            train_rows,
            validation_rows,
            note
        FROM model_runs
        WHERE provider = ?
        ORDER BY id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(active_provider, limit))

        if frame.empty:
            return []

        return frame.to_dict(orient="records")

    def save_signal_review(self, model_run_id: int, status: str, note: str = "") -> dict:
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        normalized_status = str(status or "pending").strip().lower()
        if normalized_status not in {"pending", "executed", "ignored"}:
            normalized_status = "pending"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO signal_reviews(model_run_id, status, note, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(model_run_id) DO UPDATE SET
                    status = excluded.status,
                    note = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (int(model_run_id), normalized_status, str(note or "").strip(), updated_at),
            )
        return {
            "model_run_id": int(model_run_id),
            "status": normalized_status,
            "note": str(note or "").strip(),
            "updated_at": updated_at,
        }

    def load_signal_review(self, model_run_id: int) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT model_run_id, status, note, updated_at
                FROM signal_reviews
                WHERE model_run_id = ?
                LIMIT 1
                """,
                (int(model_run_id),),
            ).fetchone()
        if not row:
            return None
        return {
            "model_run_id": int(row[0]),
            "status": str(row[1]),
            "note": str(row[2]),
            "updated_at": str(row[3]),
        }

    def load_signal_reviews(self, model_run_ids: list[int]) -> dict[int, dict]:
        if not model_run_ids:
            return {}
        placeholders = ",".join("?" for _ in model_run_ids)
        query = f"""
        SELECT model_run_id, status, note, updated_at
        FROM signal_reviews
        WHERE model_run_id IN ({placeholders})
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=[int(item) for item in model_run_ids])
        if frame.empty:
            return {}
        reviews: dict[int, dict] = {}
        for _, row in frame.iterrows():
            reviews[int(row["model_run_id"])] = {
                "model_run_id": int(row["model_run_id"]),
                "status": str(row["status"]),
                "note": str(row["note"]),
                "updated_at": str(row["updated_at"]),
            }
        return reviews

    def load_recent_signal_batches(
        self,
        limit: int = 12,
        top_n: int = 5,
        provider: str | None = None,
    ) -> list[dict]:
        active_provider = provider or self.default_provider
        query = """
        SELECT
            id,
            model_name,
            provider,
            run_at
        FROM model_runs
        WHERE provider = ?
        ORDER BY id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            runs = pd.read_sql_query(query, conn, params=(active_provider, limit))
        if runs.empty:
            return []

        run_ids = [int(item) for item in runs["id"].tolist()]
        placeholders = ",".join("?" for _ in run_ids)
        prediction_query = f"""
        SELECT
            model_run_id,
            symbol,
            name,
            trade_date,
            predicted_return_5d,
            score,
            rank
        FROM model_predictions
        WHERE model_run_id IN ({placeholders})
        ORDER BY model_run_id DESC, rank ASC
        """
        with self.connect() as conn:
            predictions = pd.read_sql_query(prediction_query, conn, params=run_ids)

        prediction_groups: dict[int, list[dict]] = {}
        if not predictions.empty:
            predictions["predicted_return_5d"] = predictions["predicted_return_5d"].astype(float)
            predictions["score"] = predictions["score"].astype(float)
            predictions["rank"] = predictions["rank"].astype(int)
            for run_id, group in predictions.groupby("model_run_id"):
                prediction_groups[int(run_id)] = group.sort_values("rank").head(top_n).to_dict(orient="records")

        reviews = self.load_signal_reviews(run_ids)
        rows: list[dict] = []
        for _, run in runs.iterrows():
            model_run_id = int(run["id"])
            batch_predictions = prediction_groups.get(model_run_id, [])
            review = reviews.get(
                model_run_id,
                {
                    "model_run_id": model_run_id,
                    "status": "pending",
                    "note": "",
                    "updated_at": "",
                },
            )
            avg_return = (
                float(sum(float(item["predicted_return_5d"]) for item in batch_predictions) / len(batch_predictions))
                if batch_predictions
                else 0.0
            )
            best_return = float(batch_predictions[0]["predicted_return_5d"]) if batch_predictions else 0.0
            rows.append(
                {
                    "model_run_id": model_run_id,
                    "model_name": str(run["model_name"]),
                    "provider": str(run["provider"]),
                    "generated_at": str(run["run_at"]),
                    "signal_trade_date": str(batch_predictions[0]["trade_date"]) if batch_predictions else "",
                    "top_symbols": [str(item["symbol"]) for item in batch_predictions],
                    "top_names": [str(item["name"]) for item in batch_predictions],
                    "avg_predicted_return_5d": avg_return,
                    "best_predicted_return_5d": best_return,
                    "review_status": str(review["status"]),
                    "review_note": str(review["note"]),
                    "review_updated_at": str(review["updated_at"]),
                }
            )
        return rows

    def load_latest_model_comparison(self, provider: str | None = None, limit: int = 6) -> list[dict]:
        active_provider = provider or self.default_provider
        query = """
        SELECT
            mr.id,
            mr.model_name,
            mr.provider,
            mr.run_at,
            mr.validation_ic,
            mr.validation_directional_accuracy,
            mr.metrics_json,
            mr.note
        FROM model_runs mr
        JOIN (
            SELECT model_name, MAX(id) AS latest_id
            FROM model_runs
            WHERE provider = ?
            GROUP BY model_name
        ) latest
          ON latest.latest_id = mr.id
        ORDER BY mr.id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(active_provider, limit))
        if frame.empty:
            return []

        latest_run = self.load_latest_model_run(provider=active_provider)
        champion_id = latest_run.get("id")
        items: list[dict] = []
        for _, row in frame.iterrows():
            metrics = json.loads(row["metrics_json"])
            items.append(
                {
                    "model_name": row["model_name"],
                    "provider": row["provider"],
                    "run_at": row["run_at"],
                    "validation_ic": float(row["validation_ic"]),
                    "validation_directional_accuracy": float(row["validation_directional_accuracy"]),
                    "walk_forward_mean_ic": float(metrics.get("walk_forward_mean_ic", 0.0)),
                    "walk_forward_positive_ic_ratio": float(metrics.get("walk_forward_positive_ic_ratio", 0.0)),
                    "walk_forward_mean_long_short_return": float(metrics.get("walk_forward_mean_long_short_return", 0.0)),
                    "note": row["note"],
                    "is_champion": int(row["id"]) == champion_id,
                }
            )
        return items

    def lookup_symbol_names(self, symbols: list[str]) -> dict[str, str]:
        if not symbols:
            return {}

        placeholders = ",".join("?" for _ in symbols)
        query = f"""
        SELECT u.symbol, u.name
        FROM universe u
        JOIN (
            SELECT symbol, MAX(updated_at) AS latest_updated_at
            FROM universe
            WHERE symbol IN ({placeholders})
            GROUP BY symbol
        ) latest
          ON latest.symbol = u.symbol
         AND latest.latest_updated_at = u.updated_at
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=symbols)
        if frame.empty:
            return {}
        return {str(row["symbol"]): str(row["name"]) for _, row in frame.iterrows()}

    def save_custom_universe(self, items: list[dict]) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            conn.execute("DELETE FROM custom_universe")
            for position, item in enumerate(items):
                conn.execute(
                    """
                    INSERT INTO custom_universe(symbol, name, position, updated_at)
                    VALUES(?, ?, ?, ?)
                    """,
                    (
                        str(item["symbol"]).zfill(6),
                        str(item["name"]),
                        position,
                        now,
                    ),
                )

    def load_custom_universe(self) -> list[dict]:
        query = """
        SELECT symbol, name, position, updated_at
        FROM custom_universe
        ORDER BY position ASC, symbol ASC
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn)
        if frame.empty:
            return []
        return frame.to_dict(orient="records")

    def search_universe_candidates(self, query_text: str, limit: int = 12) -> list[dict]:
        normalized = query_text.strip()
        if not normalized:
            return []

        like_pattern = f"%{normalized}%"
        query = """
        SELECT symbol, name
        FROM (
            SELECT
                symbol,
                name,
                MAX(updated_at) AS latest_updated_at
            FROM universe
            WHERE symbol LIKE ? OR name LIKE ?
            GROUP BY symbol, name
        )
        ORDER BY symbol ASC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(like_pattern, like_pattern, limit))
        if frame.empty:
            return []
        return frame.to_dict(orient="records")

    def ensure_paper_account(self, account_id: str = "default", name: str = "模拟账户", initial_cash: float = 1_000_000.0) -> dict:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_accounts(account_id, name, initial_cash, cash, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO NOTHING
                """,
                (account_id, name, float(initial_cash), float(initial_cash), now, now),
            )
            row = conn.execute(
                """
                SELECT account_id, name, initial_cash, cash, created_at, updated_at
                FROM paper_accounts
                WHERE account_id = ?
                """,
                (account_id,),
            ).fetchone()
        return {
            "account_id": row[0],
            "name": row[1],
            "initial_cash": float(row[2]),
            "cash": float(row[3]),
            "created_at": row[4],
            "updated_at": row[5],
        }

    def load_paper_account(self, account_id: str = "default") -> dict:
        account = self.ensure_paper_account(account_id=account_id)
        return account

    def update_paper_account_cash(self, cash: float, account_id: str = "default") -> dict:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE paper_accounts
                SET cash = ?, updated_at = ?
                WHERE account_id = ?
                """,
                (float(cash), now, account_id),
            )
        return self.load_paper_account(account_id=account_id)

    def load_paper_positions(self, account_id: str = "default") -> list[dict]:
        query = """
        SELECT
            account_id, symbol, name, quantity, avg_cost, last_price, updated_at,
            sellable_quantity, buy_locked_quantity, buy_locked_at
        FROM paper_positions
        WHERE account_id = ?
        ORDER BY symbol ASC
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(account_id,))
        if frame.empty:
            return []
        frame["quantity"] = frame["quantity"].astype(int)
        frame["sellable_quantity"] = frame["sellable_quantity"].astype(int)
        frame["buy_locked_quantity"] = frame["buy_locked_quantity"].astype(int)
        return frame.to_dict(orient="records")

    def upsert_paper_position(
        self,
        *,
        account_id: str,
        symbol: str,
        name: str,
        quantity: int,
        avg_cost: float,
        last_price: float,
        sellable_quantity: int | None = None,
        buy_locked_quantity: int | None = None,
        buy_locked_at: str | None = None,
    ) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        next_sellable_quantity = int(quantity if sellable_quantity is None else sellable_quantity)
        next_buy_locked_quantity = int(0 if buy_locked_quantity is None else buy_locked_quantity)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_positions(
                    account_id, symbol, name, quantity, avg_cost, last_price, updated_at,
                    sellable_quantity, buy_locked_quantity, buy_locked_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, symbol) DO UPDATE SET
                    name = excluded.name,
                    quantity = excluded.quantity,
                    avg_cost = excluded.avg_cost,
                    last_price = excluded.last_price,
                    updated_at = excluded.updated_at,
                    sellable_quantity = excluded.sellable_quantity,
                    buy_locked_quantity = excluded.buy_locked_quantity,
                    buy_locked_at = excluded.buy_locked_at
                """,
                (
                    account_id,
                    symbol,
                    name,
                    int(quantity),
                    float(avg_cost),
                    float(last_price),
                    now,
                    next_sellable_quantity,
                    next_buy_locked_quantity,
                    buy_locked_at,
                ),
            )

    def delete_paper_position(self, account_id: str, symbol: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM paper_positions WHERE account_id = ? AND symbol = ?",
                (account_id, symbol),
            )

    def refresh_paper_position_prices(self, price_map: dict[str, float], account_id: str = "default") -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            for symbol, price in price_map.items():
                conn.execute(
                    """
                    UPDATE paper_positions
                    SET last_price = ?, updated_at = ?
                    WHERE account_id = ? AND symbol = ?
                    """,
                    (float(price), now, account_id, symbol),
                )

    def unlock_paper_t1_positions(self, account_id: str = "default", today: str | None = None) -> int:
        trade_date = today or datetime.now().strftime("%Y-%m-%d")
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, quantity, sellable_quantity, buy_locked_quantity, buy_locked_at
                FROM paper_positions
                WHERE account_id = ? AND buy_locked_quantity > 0
                """,
                (account_id,),
            ).fetchall()
            updated = 0
            for symbol, quantity, sellable_quantity, buy_locked_quantity, buy_locked_at in rows:
                lock_date = str(buy_locked_at or "")[:10]
                if lock_date and lock_date < trade_date:
                    next_sellable = min(int(quantity), int(sellable_quantity) + int(buy_locked_quantity))
                    conn.execute(
                        """
                        UPDATE paper_positions
                        SET sellable_quantity = ?, buy_locked_quantity = 0, buy_locked_at = NULL
                        WHERE account_id = ? AND symbol = ?
                        """,
                        (next_sellable, account_id, symbol),
                    )
                    updated += 1
        return updated

    def record_paper_order(
        self,
        *,
        account_id: str,
        symbol: str,
        name: str,
        side: str,
        quantity: int,
        filled_quantity: int | None = None,
        remaining_quantity: int | None = None,
        fill_ratio: float | None = None,
        price: float,
        notional: float,
        status: str = "filled",
        source: str = "paper-rebalance",
        note: str = "",
    ) -> int:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        next_filled_quantity = int(quantity if filled_quantity is None else filled_quantity)
        next_remaining_quantity = int(max(int(quantity) - next_filled_quantity, 0) if remaining_quantity is None else remaining_quantity)
        next_fill_ratio = float((next_filled_quantity / max(int(quantity), 1)) if fill_ratio is None else fill_ratio)
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO paper_orders(
                    account_id, created_at, symbol, name, side, quantity, filled_quantity, remaining_quantity,
                    fill_ratio, price, notional, status, source, note
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    created_at,
                    symbol,
                    name,
                    side,
                    int(quantity),
                    next_filled_quantity,
                    next_remaining_quantity,
                    next_fill_ratio,
                    float(price),
                    float(notional),
                    status,
                    source,
                    note,
                ),
            )
            return int(cursor.lastrowid)

    def load_paper_orders(self, account_id: str = "default", limit: int = 30) -> list[dict]:
        query = """
        SELECT
            id, account_id, created_at, symbol, name, side, quantity, filled_quantity,
            remaining_quantity, fill_ratio, price, notional, status, source, note
        FROM paper_orders
        WHERE account_id = ?
        ORDER BY id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(account_id, limit))
        if frame.empty:
            return []
        frame["id"] = frame["id"].astype(int)
        frame["quantity"] = frame["quantity"].astype(int)
        frame["filled_quantity"] = frame["filled_quantity"].astype(int)
        frame["remaining_quantity"] = frame["remaining_quantity"].astype(int)
        return frame.to_dict(orient="records")

    def load_paper_order(self, order_id: int, account_id: str = "default") -> dict | None:
        query = """
        SELECT
            id, account_id, created_at, symbol, name, side, quantity, filled_quantity,
            remaining_quantity, fill_ratio, price, notional, status, source, note
        FROM paper_orders
        WHERE account_id = ? AND id = ?
        LIMIT 1
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(account_id, int(order_id)))
        if frame.empty:
            return None
        row = frame.iloc[0].to_dict()
        row["id"] = int(row["id"])
        row["quantity"] = int(row["quantity"])
        row["filled_quantity"] = int(row["filled_quantity"])
        row["remaining_quantity"] = int(row["remaining_quantity"])
        return row

    def update_paper_order_status(
        self,
        order_id: int,
        *,
        account_id: str = "default",
        status: str,
        note: str | None = None,
    ) -> dict | None:
        existing = self.load_paper_order(order_id, account_id=account_id)
        if not existing:
            return None
        next_note = existing.get("note", "") if note is None else note
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE paper_orders
                SET status = ?, note = ?
                WHERE account_id = ? AND id = ?
                """,
                (status, next_note, account_id, int(order_id)),
            )
        return self.load_paper_order(order_id, account_id=account_id)

    def reset_paper_account(self, account_id: str = "default", initial_cash: float | None = None) -> dict:
        account = self.ensure_paper_account(account_id=account_id, initial_cash=initial_cash or 1_000_000.0)
        next_initial_cash = float(initial_cash) if initial_cash is not None else float(account["initial_cash"])
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            conn.execute("DELETE FROM paper_positions WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM paper_orders WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM paper_equity_snapshots WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM paper_rebalance_runs WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM paper_rebalance_plans WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM paper_risk_events WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM paper_execution_reports WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM paper_daily_runs WHERE account_id = ?", (account_id,))
            conn.execute(
                """
                UPDATE paper_accounts
                SET initial_cash = ?, cash = ?, updated_at = ?
                WHERE account_id = ?
                """,
                (next_initial_cash, next_initial_cash, now, account_id),
            )
        return self.load_paper_account(account_id=account_id)

    def load_latest_prices(self, symbols: list[str], provider: str | None = None) -> dict[str, float]:
        if not symbols:
            return {}
        active_provider = provider or self.default_provider
        placeholders = ",".join("?" for _ in symbols)
        query = f"""
        SELECT d.symbol, d.close
        FROM daily_bars d
        JOIN (
            SELECT symbol, MAX(trade_date) AS latest_trade_date
            FROM daily_bars
            WHERE provider = ? AND symbol IN ({placeholders})
            GROUP BY symbol
        ) latest
          ON latest.symbol = d.symbol
         AND latest.latest_trade_date = d.trade_date
        WHERE d.provider = ?
        """
        params: list[object] = [active_provider, *symbols, active_provider]
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=params)
        if frame.empty:
            return {}
        return {str(row["symbol"]): float(row["close"]) for _, row in frame.iterrows()}

    def record_paper_equity_snapshot(
        self,
        *,
        account_id: str,
        cash: float,
        market_value: float,
        equity: float,
        note: str = "",
    ) -> int:
        snapshot_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO paper_equity_snapshots(account_id, snapshot_at, cash, market_value, equity, note)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (account_id, snapshot_at, float(cash), float(market_value), float(equity), note),
            )
            return int(cursor.lastrowid)

    def load_paper_equity_curve(self, account_id: str = "default", limit: int = 120) -> list[dict]:
        query = """
        SELECT id, account_id, snapshot_at, cash, market_value, equity, note
        FROM paper_equity_snapshots
        WHERE account_id = ?
        ORDER BY id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(account_id, limit))
        if frame.empty:
            return []
        frame = frame.sort_values("id").reset_index(drop=True)
        frame["id"] = frame["id"].astype(int)
        return frame.to_dict(orient="records")

    def record_paper_rebalance_run(
        self,
        *,
        account_id: str,
        top_n: int,
        capital_fraction: float,
        max_position_weight: float,
        min_cash_buffer_ratio: float,
        max_turnover_ratio: float,
        orders_created: int,
        turnover_ratio: float,
        target_symbols: list[str],
        note: str = "",
    ) -> int:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO paper_rebalance_runs(
                    account_id, created_at, top_n, capital_fraction, max_position_weight,
                    min_cash_buffer_ratio, max_turnover_ratio, orders_created, turnover_ratio, note, target_symbols_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    created_at,
                    int(top_n),
                    float(capital_fraction),
                    float(max_position_weight),
                    float(min_cash_buffer_ratio),
                    float(max_turnover_ratio),
                    int(orders_created),
                    float(turnover_ratio),
                    note,
                    json.dumps(target_symbols, ensure_ascii=True),
                ),
            )
            return int(cursor.lastrowid)

    def load_paper_rebalance_runs(self, account_id: str = "default", limit: int = 20) -> list[dict]:
        query = """
        SELECT
            id, account_id, created_at, top_n, capital_fraction, max_position_weight,
            min_cash_buffer_ratio, max_turnover_ratio, orders_created, turnover_ratio, note, target_symbols_json
        FROM paper_rebalance_runs
        WHERE account_id = ?
        ORDER BY id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(account_id, limit))
        if frame.empty:
            return []
        frame["id"] = frame["id"].astype(int)
        frame["top_n"] = frame["top_n"].astype(int)
        frame["orders_created"] = frame["orders_created"].astype(int)
        frame["target_symbols"] = frame["target_symbols_json"].apply(lambda text: json.loads(text or "[]"))
        frame = frame.drop(columns=["target_symbols_json"])
        return frame.to_dict(orient="records")

    def create_paper_rebalance_plan(
        self,
        *,
        account_id: str,
        config: dict,
        preview: dict,
        note: str = "",
    ) -> dict:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE paper_rebalance_plans
                SET status = 'superseded'
                WHERE account_id = ? AND status = 'pending'
                """,
                (account_id,),
            )
            cursor = conn.execute(
                """
                INSERT INTO paper_rebalance_plans(account_id, created_at, status, config_json, preview_json, note)
                VALUES(?, ?, 'pending', ?, ?, ?)
                """,
                (
                    account_id,
                    created_at,
                    json.dumps(config, ensure_ascii=True),
                    json.dumps(preview, ensure_ascii=True),
                    note,
                ),
            )
            plan_id = int(cursor.lastrowid)
        return self.load_paper_rebalance_plan(plan_id, account_id=account_id) or {}

    def load_paper_rebalance_plan(self, plan_id: int, account_id: str = "default") -> dict | None:
        query = """
        SELECT id, account_id, created_at, status, config_json, preview_json, note
        FROM paper_rebalance_plans
        WHERE account_id = ? AND id = ?
        LIMIT 1
        """
        with self.connect() as conn:
            row = conn.execute(query, (account_id, int(plan_id))).fetchone()
        if not row:
            return None
        return {
            "id": int(row[0]),
            "account_id": row[1],
            "created_at": row[2],
            "status": row[3],
            "config": json.loads(row[4] or "{}"),
            "preview": json.loads(row[5] or "{}"),
            "note": row[6],
        }

    def load_latest_pending_paper_plan(self, account_id: str = "default") -> dict | None:
        query = """
        SELECT id
        FROM paper_rebalance_plans
        WHERE account_id = ? AND status = 'pending'
        ORDER BY id DESC
        LIMIT 1
        """
        with self.connect() as conn:
            row = conn.execute(query, (account_id,)).fetchone()
        if not row:
            return None
        return self.load_paper_rebalance_plan(int(row[0]), account_id=account_id)

    def update_paper_rebalance_plan_status(
        self,
        plan_id: int,
        *,
        status: str,
        account_id: str = "default",
        note: str | None = None,
    ) -> dict | None:
        existing = self.load_paper_rebalance_plan(plan_id, account_id=account_id)
        if not existing:
            return None
        next_note = note if note is not None else existing.get("note", "")
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE paper_rebalance_plans
                SET status = ?, note = ?
                WHERE account_id = ? AND id = ?
                """,
                (status, next_note, account_id, int(plan_id)),
            )
        return self.load_paper_rebalance_plan(plan_id, account_id=account_id)

    def record_paper_risk_event(
        self,
        *,
        account_id: str,
        event_type: str,
        severity: str,
        title: str,
        details: dict,
        note: str = "",
    ) -> int:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO paper_risk_events(account_id, created_at, event_type, severity, title, details_json, note)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    created_at,
                    event_type,
                    severity,
                    title,
                    json.dumps(details, ensure_ascii=True),
                    note,
                ),
            )
            return int(cursor.lastrowid)

    def load_paper_risk_events(self, account_id: str = "default", limit: int = 20) -> list[dict]:
        query = """
        SELECT id, account_id, created_at, event_type, severity, title, details_json, note
        FROM paper_risk_events
        WHERE account_id = ?
        ORDER BY id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(account_id, limit))
        if frame.empty:
            return []
        frame["id"] = frame["id"].astype(int)
        frame["details"] = frame["details_json"].apply(lambda text: json.loads(text or "{}"))
        frame = frame.drop(columns=["details_json"])
        return frame.to_dict(orient="records")

    def record_paper_execution_report(
        self,
        *,
        account_id: str,
        report_type: str,
        title: str,
        summary: dict,
        note: str = "",
    ) -> int:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO paper_execution_reports(account_id, created_at, report_type, title, summary_json, note)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    created_at,
                    report_type,
                    title,
                    json.dumps(summary, ensure_ascii=True),
                    note,
                ),
            )
            return int(cursor.lastrowid)

    def load_paper_execution_reports(self, account_id: str = "default", limit: int = 20) -> list[dict]:
        query = """
        SELECT id, account_id, created_at, report_type, title, summary_json, note
        FROM paper_execution_reports
        WHERE account_id = ?
        ORDER BY id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(account_id, limit))
        if frame.empty:
            return []
        frame["id"] = frame["id"].astype(int)
        frame["summary"] = frame["summary_json"].apply(lambda text: json.loads(text or "{}"))
        frame = frame.drop(columns=["summary_json"])
        return frame.to_dict(orient="records")

    def ensure_paper_daily_settings(self, account_id: str = "default") -> dict:
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_daily_settings(
                    account_id, enabled, run_time, auto_sync, auto_train, auto_rebalance,
                    top_n, capital_fraction, max_position_weight, min_cash_buffer_ratio,
                    max_turnover_ratio, stop_loss_pct, take_profit_pct, fill_ratio,
                    max_drawdown_limit, max_equity_change_limit, updated_at
                )
                VALUES(?, 0, '15:10', 1, 1, 1, 3, 0.95, 0.35, 0.05, 1.0, 0.1, 0.2, 1.0, 0.18, 0.04, ?)
                ON CONFLICT(account_id) DO NOTHING
                """,
                (account_id, updated_at),
            )
            row = conn.execute(
                """
                SELECT
                    account_id, enabled, run_time, auto_sync, auto_train, auto_rebalance,
                    top_n, capital_fraction, max_position_weight, min_cash_buffer_ratio,
                    max_turnover_ratio, stop_loss_pct, take_profit_pct, fill_ratio,
                    max_drawdown_limit, max_equity_change_limit, updated_at
                FROM paper_daily_settings
                WHERE account_id = ?
                LIMIT 1
                """,
                (account_id,),
            ).fetchone()
        return {
            "account_id": row[0],
            "enabled": bool(row[1]),
            "run_time": row[2],
            "auto_sync": bool(row[3]),
            "auto_train": bool(row[4]),
            "auto_rebalance": bool(row[5]),
            "top_n": int(row[6]),
            "capital_fraction": float(row[7]),
            "max_position_weight": float(row[8]),
            "min_cash_buffer_ratio": float(row[9]),
            "max_turnover_ratio": float(row[10]),
            "stop_loss_pct": float(row[11]),
            "take_profit_pct": float(row[12]),
            "fill_ratio": float(row[13]),
            "max_drawdown_limit": float(row[14]),
            "max_equity_change_limit": float(row[15]),
            "updated_at": row[16],
        }

    def load_paper_daily_settings(self, account_id: str = "default") -> dict:
        return self.ensure_paper_daily_settings(account_id=account_id)

    def update_paper_daily_settings(self, account_id: str = "default", **values) -> dict:
        current = self.ensure_paper_daily_settings(account_id=account_id)
        next_values = {**current, **values}
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE paper_daily_settings
                SET enabled = ?, run_time = ?, auto_sync = ?, auto_train = ?, auto_rebalance = ?,
                    top_n = ?, capital_fraction = ?, max_position_weight = ?, min_cash_buffer_ratio = ?,
                    max_turnover_ratio = ?, stop_loss_pct = ?, take_profit_pct = ?, fill_ratio = ?,
                    max_drawdown_limit = ?, max_equity_change_limit = ?, updated_at = ?
                WHERE account_id = ?
                """,
                (
                    1 if next_values["enabled"] else 0,
                    str(next_values["run_time"]),
                    1 if next_values["auto_sync"] else 0,
                    1 if next_values["auto_train"] else 0,
                    1 if next_values["auto_rebalance"] else 0,
                    int(next_values["top_n"]),
                    float(next_values["capital_fraction"]),
                    float(next_values["max_position_weight"]),
                    float(next_values["min_cash_buffer_ratio"]),
                    float(next_values["max_turnover_ratio"]),
                    float(next_values["stop_loss_pct"]),
                    float(next_values["take_profit_pct"]),
                    float(next_values["fill_ratio"]),
                    float(next_values["max_drawdown_limit"]),
                    float(next_values["max_equity_change_limit"]),
                    updated_at,
                    account_id,
                ),
            )
        return self.load_paper_daily_settings(account_id=account_id)

    def record_paper_daily_run(
        self,
        *,
        account_id: str,
        run_date: str,
        status: str,
        steps: list[dict],
        note: str = "",
    ) -> int:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO paper_daily_runs(account_id, run_date, created_at, status, steps_json, note)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    run_date,
                    created_at,
                    status,
                    json.dumps(steps, ensure_ascii=True),
                    note,
                ),
            )
            return int(cursor.lastrowid)

    def load_paper_daily_runs(self, account_id: str = "default", limit: int = 12) -> list[dict]:
        query = """
        SELECT id, account_id, run_date, created_at, status, steps_json, note
        FROM paper_daily_runs
        WHERE account_id = ?
        ORDER BY id DESC
        LIMIT ?
        """
        with self.connect() as conn:
            frame = pd.read_sql_query(query, conn, params=(account_id, limit))
        if frame.empty:
            return []
        frame["id"] = frame["id"].astype(int)
        frame["steps"] = frame["steps_json"].apply(lambda text: json.loads(text or "[]"))
        frame = frame.drop(columns=["steps_json"])
        return frame.to_dict(orient="records")

    def load_latest_paper_daily_run(self, account_id: str = "default") -> dict | None:
        runs = self.load_paper_daily_runs(account_id=account_id, limit=1)
        return runs[0] if runs else None
