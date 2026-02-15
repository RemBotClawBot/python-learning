#!/usr/bin/env python3
"""
Advanced Testing Examples
=========================

This module demonstrates advanced Python testing techniques including:
- Pytest fixtures and parameterization
- Mocking and patching
- Async testing
- Property-based testing with Hypothesis
- Integration and end-to-end tests
- Test coverage reporting
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import asyncio
from typing import List, Dict, Any, Optional
import random
import statistics
from datetime import datetime, timedelta
import json
import tempfile
import os

# Try to import Hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.strategies import text, integers, floats, lists, dictionaries
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("Note: Hypothesis not installed. Run: pip install hypothesis")


# ---------- Example Functions to Test ----------

class Calculator:
    """A simple calculator class for demonstration."""
    
    def __init__(self, name: str = "Calculator"):
        self.name = name
        self.history: List[str] = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """Get calculation history."""
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear calculation history."""
        self.history.clear()


class DataProcessor:
    """Process data with external dependencies."""
    
    def __init__(self, api_client=None):
        self.api_client = api_client
    
    def process_user_data(self, user_id: int) -> Dict[str, Any]:
        """Process user data with API calls."""
        if not self.api_client:
            raise ValueError("API client required")
        
        # Fetch user data
        user_data = self.api_client.fetch_user(user_id)
        
        # Process data
        processed = {
            "id": user_id,
            "name": user_data.get("name", "Unknown"),
            "email": user_data.get("email", ""),
            "age": user_data.get("age", 0),
            "is_adult": user_data.get("age", 0) >= 18,
            "processed_at": datetime.now().isoformat()
        }
        
        # Save processed data
        self.api_client.save_processed_data(user_id, processed)
        
        return processed


class AsyncProcessor:
    """Async data processor."""
    
    async def fetch_data(self, url: str) -> Dict[str, Any]:
        """Simulate async data fetching."""
        await asyncio.sleep(0.1)  # Simulate network delay
        return {"url": url, "data": f"content_from_{url}", "timestamp": datetime.now().isoformat()}
    
    async def process_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process multiple URLs concurrently."""
        tasks = [self.fetch_data(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results


class StatisticsCalculator:
    """Calculate statistics with validation."""
    
    @staticmethod
    def calculate_mean(numbers: List[float]) -> float:
        """Calculate mean of numbers."""
        if not numbers:
            raise ValueError("Cannot calculate mean of empty list")
        return sum(numbers) / len(numbers)
    
    @staticmethod
    def calculate_median(numbers: List[float]) -> float:
        """Calculate median of numbers."""
        if not numbers:
            raise ValueError("Cannot calculate median of empty list")
        
        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)
        
        if n % 2 == 1:
            return sorted_numbers[n // 2]
        else:
            mid1 = sorted_numbers[n // 2 - 1]
            mid2 = sorted_numbers[n // 2]
            return (mid1 + mid2) / 2
    
    @staticmethod
    def calculate_std_dev(numbers: List[float]) -> float:
        """Calculate standard deviation."""
        if len(numbers) < 2:
            raise ValueError("Need at least 2 numbers for standard deviation")
        
        mean = StatisticsCalculator.calculate_mean(numbers)
        variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        return variance ** 0.5


# ---------- Unittest Examples ----------

class TestCalculator(unittest.TestCase):
    """Test Calculator class using unittest."""
    
    def setUp(self):
        """Set up fresh calculator for each test."""
        self.calc = Calculator("TestCalc")
    
    def tearDown(self):
        """Clean up after each test."""
        self.calc.clear_history()
    
    def test_add(self):
        """Test addition."""
        result = self.calc.add(5, 3)
        self.assertEqual(result, 8)
        self.assertEqual(len(self.calc.history), 1)
        self.assertIn("5 + 3 = 8", self.calc.history)
    
    def test_subtract(self):
        """Test subtraction."""
        result = self.calc.subtract(10, 4)
        self.assertEqual(result, 6)
    
    def test_multiply(self):
        """Test multiplication."""
        result = self.calc.multiply(7, 6)
        self.assertEqual(result, 42)
    
    def test_divide(self):
        """Test division."""
        result = self.calc.divide(15, 3)
        self.assertEqual(result, 5)
    
    def test_divide_by_zero(self):
        """Test division by zero raises error."""
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)
    
    def test_history(self):
        """Test calculation history."""
        self.calc.add(1, 2)
        self.calc.subtract(5, 3)
        self.calc.multiply(4, 5)
        
        history = self.calc.get_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0], "1 + 2 = 3")
        self.assertEqual(history[1], "5 - 3 = 2")
        self.assertEqual(history[2], "4 * 5 = 20")
    
    def test_clear_history(self):
        """Test clearing history."""
        self.calc.add(1, 1)
        self.assertEqual(len(self.calc.history), 1)
        
        self.calc.clear_history()
        self.assertEqual(len(self.calc.history), 0)


# ---------- Pytest Examples ----------

def test_calculator_add():
    """Test Calculator.add with pytest."""
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0
    assert calc.add(0, 0) == 0


@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
    (3.5, 2.5, 6.0),
])
def test_add_parameterized(a, b, expected):
    """Parameterized test for addition."""
    calc = Calculator()
    result = calc.add(a, b)
    assert result == expected


@pytest.fixture
def calculator():
    """Fixture providing a Calculator instance."""
    return Calculator("FixtureCalc")


@pytest.fixture
def calculator_with_history():
    """Fixture with pre-populated history."""
    calc = Calculator()
    calc.add(10, 20)
    calc.subtract(50, 30)
    return calc


def test_with_fixture(calculator):
    """Test using fixture."""
    assert calculator.name == "FixtureCalc"
    calculator.add(5, 3)
    assert len(calculator.history) == 1


def test_history_fixture(calculator_with_history):
    """Test with history fixture."""
    assert len(calculator_with_history.history) == 2
    assert "10 + 20 = 30" in calculator_with_history.history


@pytest.mark.parametrize("operation,a,b,expected", [
    ("add", 5, 3, 8),
    ("subtract", 10, 4, 6),
    ("multiply", 7, 6, 42),
    ("divide", 15, 3, 5),
])
def test_all_operations(calculator, operation, a, b, expected):
    """Test all operations with parameterization."""
    if operation == "add":
        result = calculator.add(a, b)
    elif operation == "subtract":
        result = calculator.subtract(a, b)
    elif operation == "multiply":
        result = calculator.multiply(a, b)
    elif operation == "divide":
        result = calculator.divide(a, b)
    else:
        pytest.fail(f"Unknown operation: {operation}")
    
    assert result == expected


# ---------- Mocking Examples ----------

def test_data_processor_with_mock():
    """Test DataProcessor with mocked API client."""
    # Create mock API client
    mock_client = Mock()
    
    # Configure mock responses
    mock_client.fetch_user.return_value = {
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30
    }
    
    # Create processor with mocked client
    processor = DataProcessor(mock_client)
    
    # Process data
    result = processor.process_user_data(123)
    
    # Verify mock was called correctly
    mock_client.fetch_user.assert_called_once_with(123)
    mock_client.save_processed_data.assert_called_once()
    
    # Check result
    assert result["id"] == 123
    assert result["name"] == "John Doe"
    assert result["is_adult"] == True


def test_data_processor_error():
    """Test DataProcessor error handling."""
    processor = DataProcessor(None)
    
    with pytest.raises(ValueError, match="API client required"):
        processor.process_user_data(123)


def test_patching():
    """Test using patch to mock random module."""
    # Test with predictable "random" value
    with patch('random.randint', return_value=42):
        result = random.randint(1, 100)
        assert result == 42
    
    # Verify normal behavior restored after patch
    result = random.randint(1, 100)
    assert 1 <= result <= 100


# ---------- Async Testing Examples ----------

@pytest.mark.asyncio
async def test_async_processor():
    """Test AsyncProcessor."""
    processor = AsyncProcessor()
    
    # Test single fetch
    result = await processor.fetch_data("https://example.com")
    assert "url" in result
    assert result["url"] == "https://example.com"
    assert "data" in result
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_process_multiple():
    """Test concurrent processing."""
    processor = AsyncProcessor()
    urls = [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3"
    ]
    
    results = await processor.process_multiple(urls)
    
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result["url"] == urls[i]
        assert "data" in result
        assert "timestamp" in result


# ---------- Hypothesis Property-Based Testing ----------

if HYPOTHESIS_AVAILABLE:
    @given(st.lists(st.floates(allow_nan=False, allow_infinity=False), min_size=1))
    def test_mean_property(numbers):
        """Property-based test for mean calculation."""
        mean = StatisticsCalculator.calculate_mean(numbers)
        
        # Property: Mean should be between min and max
        assert min(numbers) <= mean <= max(numbers)
        
        # Property: Sum equals mean * count
        total = sum(numbers)
        count = len(numbers)
        # Use relative tolerance for floating point comparison
        assert abs(total - mean * count) < 1e-10 * abs(total)
    
    @given(st.lists(st.floates(allow_nan=False, allow_infinity=False), min_size=1))
    def test_median_property(numbers):
        """Property-based test for median calculation."""
        median = StatisticsCalculator.calculate_median(numbers)
        sorted_numbers = sorted(numbers)
        
        # Property: Median is in the sorted list
        assert median in sorted_numbers
        
        # Property: For odd length, median is middle element
        if len(numbers) % 2 == 1:
            assert median == sorted_numbers[len(numbers) // 2]
    
    @given(st.lists(st.floates(allow_nan=False, allow_infinity=False), min_size=2))
    def test_std_dev_property(numbers):
        """Property-based test for standard deviation."""
        std_dev = StatisticsCalculator.calculate_std_dev(numbers)
        
        # Property: Standard deviation is non-negative
        assert std_dev >= 0
        
        # Property: If all numbers are equal, std dev is 0
        if all(x == numbers[0] for x in numbers):
            assert abs(std_dev) < 1e-10


# ---------- Integration Test Examples ----------

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_calculator_integration(self):
        """Integration test for calculator workflow."""
        calc = Calculator("IntegrationTest")
        
        # Perform series of operations
        calc.add(10, 20)
        calc.subtract(50, 30)
        calc.multiply(5, 4)
        calc.divide(100, 25)
        
        # Verify history
        history = calc.get_history()
        assert len(history) == 4
        assert "10 + 20 = 30" in history
        assert "50 - 30 = 20" in history
        assert "5 * 4 = 20" in history
        assert "100 / 25 = 4" in history
        
        # Clear and verify
        calc.clear_history()
        assert len(calc.get_history()) == 0
    
    def test_file_processing_integration(self):
        """Integration test with file I/O."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            test_data = {"numbers": [1, 2, 3, 4, 5]}
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            # Read and process file
            with open(temp_file, 'r') as f:
                data = json.load(f)
            
            # Process data
            mean = StatisticsCalculator.calculate_mean(data["numbers"])
            median = StatisticsCalculator.calculate_median(data["numbers"])
            
            assert mean == 3.0
            assert median == 3.0
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_async_integration(self):
        """Integration test with async operations."""
        processor = AsyncProcessor()
        
        # Fetch multiple URLs
        urls = ["https://api.example.com/data1", "https://api.example.com/data2"]
        results = await processor.process_multiple(urls)
        
        assert len(results) == 2
        for result in results:
            assert "url" in result
            assert "data" in result
            assert "timestamp" in result


# ---------- Test Configuration and Reporting ----------

def test_configuration():
    """Test with custom configuration."""
    # Environment variable testing
    os.environ["TEST_MODE"] = "true"
    assert os.getenv("TEST_MODE") == "true"
    
    # Clean up
    del os.environ["TEST_MODE"]


@pytest.mark.skip(reason="Example of a skipped test")
def test_skipped():
    """This test is skipped."""
    assert False  # Won't run


@pytest.mark.xfail(reason="Expected to fail")
def test_expected_failure():
    """Test that's expected to fail."""
    assert 1 == 2  # This will fail, but that's expected


def test_timing(monkeypatch):
    """Test timing behavior with monkeypatch."""
    fake_time = 1000.0
    
    # Mock time.time()
    monkeypatch.setattr('time.time', lambda: fake_time)
    import time
    
    assert time.time() == fake_time


# ---------- Test Runner Examples ----------

def run_unit_tests():
    """Run unittest tests."""
    print("\n" + "="*60)
    print("Running unittest Tests")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCalculator)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_pytest_tests():
    """Run pytest tests."""
    print("\n" + "="*60)
    print("Running pytest Tests")
    print("="*60)
    
    import sys
    import pytest
    
    # Run specific test module
    return pytest.main([__file__, "-v"])


# ---------- Main Execution ----------

def main():
    """Run all test examples."""
    print("Advanced Testing Examples")
    print("="*60)
    
    # Run unittest examples
    print("\n1. Running unittest examples...")
    unittest_success = run_unit_tests()
    print(f"Unittest successful: {unittest_success}")
    
    # Run pytest examples
    print("\n\n2. Running pytest examples...")
    pytest.main([__file__, "-v", "--tb=short"])
    
    print("\n" + "="*60)
    print("Test Examples Complete!")
    print("="*60)
    
    # Create requirements file for testing
    requirements = """# Testing dependencies
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
hypothesis>=6.0.0
"""
    
    with open("requirements-testing.txt", "w") as f:
        f.write(requirements)
    print("\nCreated requirements-testing.txt file")
    
    # Create pytest configuration
    pytest_ini = """; pytest.ini - Configuration file for pytest
[pytest]
testpaths = .
python_files = test_*.py *_test.py
python_classes = Test* *Test
python_functions = test_*

; Add options
addopts = 
    -v
    --tb=short
    --strict-markers
    
; Markers
markers =
    slow: marks tests as slow (deselect with '-m \"not slow\"')
    integration: integration tests
    e2e: end-to-end tests
    
; Asyncio support
asyncio_mode = auto
"""
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_ini)
    print("Created pytest.ini configuration file")
    
    # Create .coveragerc for coverage reporting
    coverage_rc = """# .coveragerc - Coverage.py configuration
[run]
source = .
omit = 
    */tests/*
    */__pycache__/*
    */site-packages/*
    */dist-packages/*
    */venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    
show_missing = true
skip_covered = true
"""
    
    with open(".coveragerc", "w") as f:
        f.write(coverage_rc)
    print("Created .coveragerc for coverage reporting")


if __name__ == "__main__":
    main()