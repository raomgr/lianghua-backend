from __future__ import annotations

from datetime import datetime
from threading import Lock, Thread
from uuid import uuid4

from app.services.market_service import MarketService
from app.services.storage import MarketRepository
from app.services.training import train_local_model


class TrainJobService:
    def __init__(self, market: MarketService) -> None:
        self.market = market
        self.repo = MarketRepository()
        self._lock = Lock()
        self._active_job_id: str | None = None

    def start(self) -> dict:
        with self._lock:
            if self._active_job_id:
                active_job = self.repo.load_train_job(self._active_job_id)
                if active_job and active_job.get("status") in {"queued", "running"}:
                    return active_job

            job_id = uuid4().hex
            job = self.repo.create_train_job(job_id, "queued", "训练任务已创建，等待执行。")
            self._active_job_id = job_id

            worker = Thread(target=self._run_job, args=(job_id,), name=f"train-job-{job_id[:8]}", daemon=True)
            worker.start()
            return job

    def get(self, job_id: str) -> dict:
        return self.repo.load_train_job(job_id)

    def _run_job(self, job_id: str) -> None:
        started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.repo.update_train_job(
            job_id,
            status="running",
            message="正在加载训练数据并执行本地模型训练。",
            started_at=started_at,
            error="",
            result={},
        )
        try:
            histories, names = self.market.get_training_inputs(limit=220)
            result = train_local_model(
                histories,
                names,
                provider_name=self.market.active_data_provider,
                configured_provider=self.market.settings.data_provider,
            )
            finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.repo.update_train_job(
                job_id,
                status="succeeded",
                message=result.get("message", "训练完成。"),
                finished_at=finished_at,
                result=result,
            )
        except Exception as exc:
            finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.repo.update_train_job(
                job_id,
                status="failed",
                message="训练失败。",
                finished_at=finished_at,
                error=str(exc),
                result={},
            )
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None
