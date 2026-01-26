"""Tests for fibonacci module - 100% coverage required."""

import sys
from io import StringIO

from fibonacci import fibonacci, main


class TestFibonacci:
    """Test the fibonacci function."""

    def test_base_case_zero(self) -> None:
        """fib(0) = 0."""
        assert fibonacci(0) == 0

    def test_base_case_one(self) -> None:
        """fib(1) = 1."""
        assert fibonacci(1) == 1

    def test_recursive_case(self) -> None:
        """fib(n) = fib(n-1) + fib(n-2)."""
        assert fibonacci(2) == 1
        assert fibonacci(3) == 2
        assert fibonacci(4) == 3
        assert fibonacci(5) == 5
        assert fibonacci(10) == 55


class TestMain:
    """Test the main function."""

    def test_main_output(self) -> None:
        """main() prints first 10 fibonacci numbers."""
        captured = StringIO()
        sys.stdout = captured
        main()
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "fib(0) = 0" in output
        assert "fib(9) = 34" in output
