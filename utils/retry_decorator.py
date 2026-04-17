"""
Retry decorator with exponential backoff using tenacity
"""
import functools
import asyncio 
import logging
from typing import Callable, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
from botocore.exceptions import ClientError, EndpointConnectionError

from utils.config import settings
from utils.logger import logger


def with_retry(
    max_attempts: int = None,
    min_wait: int = None,
    max_wait: int = None,
    multiplier: int = None
) -> Callable:
    """
    Decorator to add retry logic with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        multiplier: Exponential backoff multiplier
    
    Returns:
        Decorated function with retry logic
    """
    max_attempts = max_attempts or settings.MAX_RETRIES
    min_wait = min_wait or settings.RETRY_MIN_WAIT
    max_wait = max_wait or settings.RETRY_MAX_WAIT
    multiplier = multiplier or settings.RETRY_MULTIPLIER
    
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=multiplier,
                min=min_wait,
                max=max_wait
            ),
            retry=retry_if_exception_type((
                ClientError,
                EndpointConnectionError,
                TimeoutError,
                ConnectionError,
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO),
            reraise=True
        )
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Import asyncio for coroutine check
