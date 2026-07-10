"""
Tests for the Decorators module.
"""

import time

import pytest

from src.utils.decorators import log_execution, retry, timer, validate_input
from src.utils.exceptions import AuditHubException


class TestTimer:
    """Test suite for the timer decorator."""

    def test_timer_logs_duration(self, caplog):
        """timer should log the function name and duration."""
        caplog.set_level("INFO")

        @timer
        def my_func():
            return 42

        result = my_func()
        assert result == 42

        # Check that the timer log message was emitted
        assert any("TIMER" in msg and "my_func" in msg for msg in caplog.messages)

    def test_timer_with_args(self, caplog):
        """timer should work with functions that have arguments."""
        caplog.set_level("INFO")

        @timer
        def add(a: int, b: int) -> int:
            return a + b

        result = add(3, 4)
        assert result == 7

        assert any("TIMER" in msg and "add" in msg for msg in caplog.messages)

    def test_timer_preserves_signature(self):
        """timer should preserve the wrapped function's signature."""

        @timer
        def process(data: str, /, limit: int = 10) -> str:
            return data[:limit]

        assert process.__wrapped__(  # type: ignore[attr-defined]
            "hello world", limit=5
        ) == "hello"


class TestLogExecution:
    """Test suite for the log_execution decorator."""

    def test_log_execution_success(self, caplog):
        """log_execution should log entry and exit on success."""
        caplog.set_level("DEBUG")

        @log_execution
        def my_func():
            return "done"

        result = my_func()
        assert result == "done"

        messages = " ".join(caplog.messages)
        assert "ENTER" in messages
        assert "EXIT" in messages
        assert "OK" in messages

    def test_log_execution_error(self, caplog):
        """log_execution should log entry and error on exception."""
        caplog.set_level("DEBUG")

        @log_execution
        def failing_func():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            failing_func()

        messages = " ".join(caplog.messages)
        assert "ENTER" in messages
        assert "EXIT" in messages
        assert "ERROR" in messages
        assert "ValueError" in messages


class TestRetry:
    """Test suite for the retry decorator."""

    def test_retry_success_first_attempt(self):
        """retry should succeed on first attempt."""

        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        def my_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = my_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """retry should succeed after some failures."""

        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "finally"

        result = flaky_func()
        assert result == "finally"
        assert call_count == 3

    def test_retry_all_failures(self):
        """retry should raise AuditHubException after all attempts fail."""

        call_count = 0

        @retry(max_attempts=3, delay=0.1)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        with pytest.raises(AuditHubException) as exc:
            always_fails()

        assert call_count == 3
        assert "failed after 3 attempts" in str(exc.value)

    def test_retry_custom_exceptions(self):
        """retry should only catch specified exceptions."""

        @retry(max_attempts=2, delay=0.1, exceptions=ValueError)
        def raises_type_error():
            raise TypeError("not caught")

        with pytest.raises(TypeError):
            raises_type_error()

    def test_retry_backoff(self):
        """retry should increase delay with backoff."""

        call_count = 0
        start = time.perf_counter()

        @retry(max_attempts=3, delay=0.2, backoff=2.0)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(AuditHubException):
            always_fails()

        elapsed = time.perf_counter() - start
        # With delay 0.2, 0.4, total should be at least 0.6s
        assert elapsed >= 0.5
        assert call_count == 3


class TestValidateInput:
    """Test suite for the validate_input decorator."""

    def test_validate_success(self):
        """validate_input should pass valid arguments."""

        @validate_input(a=lambda x: x > 0, b=lambda x: isinstance(x, str))
        def process(a: int, b: str) -> str:
            return f"{a}: {b}"

        result = process(5, "hello")
        assert result == "5: hello"

    def test_validate_failure(self):
        """validate_input should raise on invalid arguments."""

        @validate_input(a=lambda x: x > 0)
        def process(a: int) -> int:
            return a

        with pytest.raises(AuditHubException) as exc:
            process(-1)
        assert "Input validation failed" in str(exc.value)

    def test_validate_multiple_params(self):
        """validate_input should validate multiple parameters."""

        @validate_input(
            name=lambda x: len(x) > 0,
            age=lambda x: 0 <= x <= 150,
        )
        def create_user(name: str, age: int) -> dict:
            return {"name": name, "age": age}

        result = create_user("Alice", 30)
        assert result["name"] == "Alice"

        with pytest.raises(AuditHubException):
            create_user("", 30)

        with pytest.raises(AuditHubException):
            create_user("Bob", -1)

    def test_validate_with_defaults(self):
        """validate_input should handle default argument values."""

        @validate_input(value=lambda x: x >= 0)
        def set_value(value: int, name: str = "default") -> str:
            return f"{name}={value}"

        result = set_value(42)
        assert result == "default=42"

        with pytest.raises(AuditHubException):
            set_value(-1)

    def test_validate_missing_param(self):
        """validate_input should warn on unknown params (not raise)."""

        @validate_input(nonexistent=lambda x: False)
        def my_func(x: int) -> int:
            return x

        # Should not raise, just warn
        result = my_func(42)
        assert result == 42
