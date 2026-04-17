"""
Observability integration (LangSmith + local fallback).

This module provides a safe optional wrapper around LangSmith so the app can
run fully local without credentials, while enabling cloud observability when
configured.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, List
from uuid import uuid4

from utils.config import settings
from utils.logger import logger


class ObservabilityClient:
    """Optional LangSmith client wrapper with graceful no-op behavior."""

    def __init__(self) -> None:
        self.enabled = settings.LANGSMITH_ENABLED
        self.project = settings.LANGSMITH_PROJECT
        self.client = None
        self._auth_disabled = False

        if not self.enabled:
            logger.info("Observability: LangSmith disabled")
            return

        if not settings.LANGSMITH_API_KEY:
            logger.warning("Observability: LangSmith enabled but API key missing")
            return

        try:
            from langsmith import Client

            kwargs: Dict[str, Any] = {"api_key": settings.LANGSMITH_API_KEY}
            if settings.LANGSMITH_ENDPOINT:
                kwargs["api_url"] = settings.LANGSMITH_ENDPOINT

            self.client = Client(**kwargs)
            logger.info(
                f"Observability: LangSmith initialized (project={self.project})"
            )
        except Exception as e:
            logger.warning(f"Observability: LangSmith init failed, fallback to local logs: {e}")
            self.client = None

    def _handle_runtime_error(self, error: Exception, stage: str) -> None:
        message = str(error)
        is_auth_error = "403" in message or "401" in message or "Forbidden" in message or "Unauthorized" in message
        if is_auth_error and self.client is not None:
            self.client = None
            self._auth_disabled = True
            logger.warning(
                f"Observability: disabling LangSmith after auth failure during {stage}; falling back to local logs"
            )

    @property
    def is_active(self) -> bool:
        return self.client is not None

    def start_run(
        self,
        name: str,
        inputs: Dict[str, Any],
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Start an observability run and return run ID (or None if disabled)."""
        run_id = str(uuid4())

        if not self.is_active:
            logger.info(
                "Observability(local): start",
                run_name=name,
                run_id=run_id,
                metadata=metadata or {},
            )
            return run_id

        try:
            self.client.create_run(
                id=run_id,
                name=name,
                run_type="chain",
                project_name=self.project,
                inputs=inputs,
                tags=tags or [],
                extra={"metadata": metadata or {}},
            )
            return run_id
        except Exception as e:
            logger.warning(f"Observability: start_run failed ({name}): {e}")
            self._handle_runtime_error(e, "start_run")
            return run_id

    def end_run(
        self,
        run_id: Optional[str],
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Close an observability run with outputs or error."""
        if not run_id:
            return

        if not self.is_active:
            if error:
                logger.error("Observability(local): end_error", run_id=run_id, error=error)
            else:
                logger.info("Observability(local): end", run_id=run_id)
            return

        try:
            update_kwargs: Dict[str, Any] = {
                "run_id": run_id,
                "outputs": outputs or {},
                "end_time": datetime.utcnow(),
            }
            if error:
                update_kwargs["error"] = error
            self.client.update_run(**update_kwargs)
        except Exception as e:
            logger.warning(f"Observability: end_run failed ({run_id}): {e}")
            self._handle_runtime_error(e, "end_run")


_observer: Optional[ObservabilityClient] = None


def get_observer() -> ObservabilityClient:
    global _observer
    if _observer is None:
        _observer = ObservabilityClient()
    return _observer
