"""
AuditHub - Reusable Decorators
===============================

Utility decorators for common cross-cutting concerns:
timing, logging, retries, and input validation.

Usage::

    from src.utils.decorators import timer, log_execution, retry, validate_input

    @timer
    def my_function():
        ...

    @log_execution
    def my_function():
        ...

    @retry(max_attempts=3, delay=1.0)
    def flaky_function():
        ...

    @validate_input(a=lambda x: x > 0, b=lambda x: isinstance(x, str))
    def typed_function(a: int, b: str):
        ...
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from src.utils.exceptions import AuditHubException
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# timer
# ============================================================================


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that logs the execution time of a function.

    Logs the duration at the INFO level after the function completes.

    Parameters
    ----------
    func : Callable
        The function to time.

    Returns
    -------
    Callable
        Wrapped function that logs execution time.

    Examples
    --------
    >>> @timer
    ... def process_data(n: int) -> int:
    ...     return sum(range(n))
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.info(
                "TIMER | %s | %.4fs",
                func.__qualname__,
                elapsed,
            )

    return wrapper


# ============================================================================
# log_execution
# ============================================================================


def log_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that logs function entry, exit, and exceptions.

    Logs at DEBUG level on entry, INFO on success, and ERROR on
    exception.

    Parameters
    ----------
    func : Callable
        The function to log.

    Returns
    -------
    Callable
        Wrapped function with execution logging.

    Examples
    --------
    >>> @log_execution
    ... def load_data(path: str) -> dict:
    ...     return {"data": path}
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__qualname__
        logger.debug("ENTER | %s", func_name)

        try:
            result = func(*args, **kwargs)
            logger.info("EXIT  | %s | OK", func_name)
            return result
        except Exception as exc:
            logger.error(
                "EXIT  | %s | ERROR: %s: %s",
                func_name,
                type(exc).__name__,
                exc,
            )
            raise

    return wrapper


# ============================================================================
# retry
# ============================================================================


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[
        Type[Exception], Tuple[Type[Exception], ...]
    ] = Exception,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that retries a function on failure.

    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts (default: ``3``).
    delay : float
        Initial delay between retries in seconds (default: ``1.0``).
    backoff : float
        Multiplier applied to delay after each retry (default: ``2.0``).
    exceptions : type or tuple of types
        Exception types to catch and retry on (default: ``Exception``).

    Returns
    -------
    Callable
        Decorated function with retry logic.

    Raises
    ------
    AuditHubException
        If the function fails after all retry attempts.

    Examples
    --------
    >>> @retry(max_attempts=3, delay=0.5)
    ... def fetch_data(url: str) -> dict:
    ...     return {"status": "ok"}
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_attempts:
                        logger.warning(
                            "RETRY | %s | Attempt %d/%d failed: %s. "
                            "Retrying in %.2fs...",
                            func.__qualname__,
                            attempt,
                            max_attempts,
                            exc,
                            current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "RETRY | %s | All %d attempts failed.",
                            func.__qualname__,
                            max_attempts,
                        )

            raise AuditHubException(
                f"Function '{func.__qualname__}' failed after "
                f"{max_attempts} attempts",
                details={
                    "max_attempts": max_attempts,
                    "last_error": str(last_exception),
                },
                cause=last_exception,
            )

        return wrapper

    return decorator


# ============================================================================
# validate_input
# ============================================================================


def validate_input(**validators: Callable[[Any], bool]) -> Callable:
    """Decorator that validates function arguments against predicates.

    Each keyword argument maps a parameter name to a callable predicate
    that receives the argument value and returns ``True`` if valid.

    Parameters
    ----------
    **validators : Callable
        Mapping of parameter name -> predicate function.

    Returns
    -------
    Callable
        Decorated function with input validation.

    Raises
    ------
    AuditHubException
        If any argument fails its validation predicate.

    Examples
    --------
    >>> @validate_input(a=lambda x: x > 0, b=lambda x: isinstance(x, str))
    ... def process(a: int, b: str) -> str:
    ...     return f"{a}: {b}"

    >>> process(5, "hello")  # OK
    >>> process(-1, "bad")   # raises AuditHubException
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build a mapping of parameter name -> value
            sig = _get_signature(func)
            bound = _bind_args(sig, func, *args, **kwargs)
            arg_map: Dict[str, Any] = dict(bound.arguments)

            # Validate each specified parameter
            for param_name, predicate in validators.items():
                if param_name not in arg_map:
                    logger.warning(
                        "validate_input: '%s' not found in args of %s",
                        param_name,
                        func.__qualname__,
                    )
                    continue

                value = arg_map[param_name]
                if not predicate(value):
                    raise AuditHubException(
                        f"Input validation failed for '{param_name}' "
                        f"in {func.__qualname__}: "
                        f"value={value!r} does not satisfy predicate",
                        details={
                            "function": func.__qualname__,
                            "parameter": param_name,
                            "value": repr(value),
                        },
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _get_signature(func: Callable) -> Any:
    """Get function signature, handling bound methods."""
    import inspect

    if hasattr(func, "__signature__"):
        return func.__signature__
    return inspect.signature(func)


def _bind_args(
    sig: Any,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Bind positional and keyword arguments to a function signature."""
    import inspect

    bound = sig.bind(*args, **kwargs)
    # Apply defaults for parameters not provided
    for name, param in sig.parameters.items():
        if name not in bound.arguments and param.default is not inspect.Parameter.empty:
            bound.arguments[name] = param.default
    return bound


__all__ = [
    "timer",
    "log_execution",
    "retry",
    "validate_input",
]
