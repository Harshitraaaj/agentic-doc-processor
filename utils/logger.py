"""
Structured logging configuration
"""
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import structlog
from pythonjsonlogger import jsonlogger

from utils.config import settings


class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every record — ensures real-time output under uvicorn."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


_LOGGING_CONFIGURED = False  # Module-level guard — prevents duplicate handler registration


def setup_logging() -> structlog.BoundLogger:
    """Configure structured logging"""
    global _LOGGING_CONFIGURED

    # Ensure logs directory exists
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    log_file = settings.LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"

    root_logger = logging.getLogger()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Only add handlers once — prevents duplicate log entries when both
    # FastAPI (api/main.py) and Streamlit (streamlit_app.py) both import this module.
    if not _LOGGING_CONFIGURED:
        _fmt = "%(message)s"

        stdout_handler = _FlushingStreamHandler(sys.stdout)
        stdout_handler.setLevel(log_level)
        stdout_handler.setFormatter(logging.Formatter(_fmt))
        root_logger.addHandler(stdout_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(_fmt))
        root_logger.addHandler(file_handler)

        _LOGGING_CONFIGURED = True

    # Always ConsoleRenderer — human-readable in uvicorn terminal.
    # cache_logger_on_first_use=False is required: True causes silent logs after uvicorn --reload
    # because structlog caches the old processor chain and new handlers are never reached.
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    return structlog.get_logger()


# Global logger instance
logger = setup_logging()
