from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timedelta

from app.services.paper_trading import PaperTradingService
from app.services.storage import MarketRepository


class PaperDailyScheduler:
    def __init__(self, paper: PaperTradingService, poll_seconds: int = 30) -> None:
        self.paper = paper
        self.repo = MarketRepository()
        self.poll_seconds = max(int(poll_seconds), 10)
        self.max_retry_attempts = 2
        self.retry_delay_seconds = 300
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._pending_retry_at: datetime | None = None
        self._retry_attempt = 0
        self._retry_from_step = "sync"
        self._status = {
            "loop_running": False,
            "enabled": False,
            "last_checked_at": "",
            "last_triggered_at": "",
            "last_completed_at": "",
            "last_outcome": "",
            "retry_from_step": "",
            "next_run_at": "",
            "next_retry_at": "",
            "retry_attempt": 0,
            "max_retry_attempts": self.max_retry_attempts,
            "note": "等待后端启动调度循环。",
        }

    def get_status(self) -> dict:
        return dict(self._status)

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_loop(), name="paper-daily-scheduler")

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self._status["loop_running"] = False
        self._status["note"] = "调度循环已停止。"
        self._status["next_retry_at"] = ""
        self._status["retry_attempt"] = 0
        self._status["retry_from_step"] = ""

    def _schedule_retry(self, now: datetime, reason: str, retry_from_step: str) -> None:
        self._retry_attempt += 1
        self._pending_retry_at = now + timedelta(seconds=self.retry_delay_seconds)
        self._retry_from_step = retry_from_step
        self._status["retry_attempt"] = self._retry_attempt
        self._status["retry_from_step"] = retry_from_step
        self._status["next_retry_at"] = self._pending_retry_at.strftime("%Y-%m-%d %H:%M:%S")
        self._status["note"] = f"{reason} 将在稍后自动重试。"
        self.repo.record_paper_risk_event(
            account_id="default",
            event_type="scheduler_retry_scheduled",
            severity="warning",
            title="自动日更已安排重试",
            details={
                "retry_attempt": self._retry_attempt,
                "max_retry_attempts": self.max_retry_attempts,
                "next_retry_at": self._status["next_retry_at"],
                "retry_from_step": retry_from_step,
                "reason": reason,
            },
            note="Scheduler queued a retry.",
        )

    def _clear_retry(self) -> None:
        self._pending_retry_at = None
        self._retry_attempt = 0
        self._retry_from_step = "sync"
        self._status["next_retry_at"] = ""
        self._status["retry_attempt"] = 0
        self._status["retry_from_step"] = ""

    def _retry_due(self, now: datetime) -> bool:
        return self._pending_retry_at is not None and now >= self._pending_retry_at

    def _resolve_retry_from_step(self, latest_run: dict | None) -> str:
        if not latest_run:
            return "sync"
        for step in latest_run.get("steps", []):
            if str(step.get("status")) == "failed":
                return str(step.get("step") or "sync")
        return "sync"

    async def _execute_cycle(self, *, now: datetime, trigger_source: str, retry_attempt: int = 0) -> None:
        active_start_step = self._retry_from_step if "retry" in trigger_source else "sync"
        self._status["last_triggered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.repo.record_paper_risk_event(
            account_id="default",
            event_type="scheduler_triggered",
            severity="info",
            title="自动日更已触发",
            details={
                "trigger_source": trigger_source,
                "retry_attempt": retry_attempt,
                "retry_from_step": active_start_step,
            },
            note="Scheduler started a daily cycle.",
        )
        snapshot = await asyncio.to_thread(
            self.paper.run_daily_cycle,
            trigger_source=trigger_source,
            retry_attempt=retry_attempt,
            start_from_step=active_start_step,
        )
        latest_run = (snapshot.get("daily_runs") or [None])[0]
        run_status = str((latest_run or {}).get("status", "success"))
        self._status["last_completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._status["last_outcome"] = run_status

        if run_status == "failed":
            retry_from_step = self._resolve_retry_from_step(latest_run)
            if self._retry_attempt < self.max_retry_attempts:
                self._schedule_retry(now, "最近一次自动日更失败。", retry_from_step)
            else:
                self._status["note"] = "自动日更失败且已达到最大重试次数。"
                self.repo.record_paper_risk_event(
                    account_id="default",
                    event_type="scheduler_retry_exhausted",
                    severity="error",
                    title="自动日更达到最大重试次数",
                    details={
                        "retry_attempt": self._retry_attempt,
                        "max_retry_attempts": self.max_retry_attempts,
                        "retry_from_step": retry_from_step,
                    },
                    note="Scheduler exhausted retries.",
                )
                self._clear_retry()
        else:
            self._clear_retry()
            self._status["note"] = (
                "自动日更执行完成。"
                if run_status == "success"
                else "自动日更完成，但带有部分告警。"
            )
            self.repo.record_paper_risk_event(
                account_id="default",
                event_type="scheduler_completed",
                severity="info" if run_status == "success" else "warning",
                title="自动日更执行完成",
                details={
                    "status": run_status,
                    "trigger_source": trigger_source,
                    "retry_attempt": retry_attempt,
                    "retry_from_step": active_start_step if "retry" in trigger_source else "",
                },
                note="Scheduler completed a daily cycle.",
            )

    def _resolve_next_run_at(self, now: datetime, run_time: str, latest_run: dict | None, enabled: bool) -> str:
        if not enabled:
            return ""
        try:
            hour, minute = [int(part) for part in str(run_time or "15:10").split(":", 1)]
        except ValueError:
            hour, minute = 15, 10
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        latest_date = str(latest_run.get("run_date", "")) if latest_run else ""
        today = now.strftime("%Y-%m-%d")
        if latest_date == today and candidate <= now:
            candidate = candidate + timedelta(days=1)
        elif candidate <= now and latest_date != today:
            candidate = now
        return candidate.strftime("%Y-%m-%d %H:%M:%S")

    def _should_run_now(self, now: datetime, run_time: str, latest_run: dict | None, enabled: bool) -> bool:
        if not enabled:
            return False
        try:
            hour, minute = [int(part) for part in str(run_time or "15:10").split(":", 1)]
        except ValueError:
            hour, minute = 15, 10
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        latest_date = str(latest_run.get("run_date", "")) if latest_run else ""
        if latest_date == now.strftime("%Y-%m-%d"):
            return False
        return now >= target

    async def _run_loop(self) -> None:
        while True:
            now = datetime.now()
            settings = self.repo.load_paper_daily_settings()
            latest_run = self.repo.load_latest_paper_daily_run()
            self._status["loop_running"] = True
            self._status["enabled"] = bool(settings.get("enabled", False))
            self._status["last_checked_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
            self._status["next_run_at"] = self._resolve_next_run_at(
                now,
                str(settings.get("run_time", "15:10")),
                latest_run,
                bool(settings.get("enabled", False)),
            )
            if self._pending_retry_at is not None:
                self._status["note"] = "自动日更上次失败，等待重试中。"
            else:
                self._status["note"] = (
                    "应用内定时任务运行中，需保持后端进程持续启动。"
                    if settings.get("enabled", False)
                    else "已停用自动日更，可手动执行。"
                )
            should_run = self._should_run_now(
                now,
                str(settings.get("run_time", "15:10")),
                latest_run,
                bool(settings.get("enabled", False)),
            )
            should_retry = self._retry_due(now)

            if should_run or should_retry:
                async with self._lock:
                    try:
                        await self._execute_cycle(
                            now=now,
                            trigger_source="scheduler-retry" if should_retry else "scheduler",
                            retry_attempt=self._retry_attempt,
                        )
                    except Exception as exc:  # pragma: no cover - runtime safety guard
                        self._status["last_completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self._status["last_outcome"] = "failed"
                        self.repo.record_paper_risk_event(
                            account_id="default",
                            event_type="scheduler_failed",
                            severity="error",
                            title="应用内定时任务执行失败",
                            details={
                                "error": str(exc),
                                "retry_attempt": self._retry_attempt,
                                "retry_from_step": self._retry_from_step,
                            },
                            note="Unhandled scheduler failure.",
                        )
                        if self._retry_attempt < self.max_retry_attempts:
                            self._schedule_retry(now, f"自动任务异常：{exc}", self._retry_from_step)
                        else:
                            self._status["note"] = f"最近一次自动执行失败：{exc}"
                            self._clear_retry()

            await asyncio.sleep(self.poll_seconds)
