# Testing Procedures

This document outlines the testing procedures for the bat project.

## Testing Framework

We use **pytest** as the primary testing framework for this project.

## Test Structure

### Test File Naming
- Test files should be named `test_<module_name>.py`
- Place test files in the project root directory
- Each module should have a corresponding test file

### Test Function Naming
- Test functions should follow the pattern: `test_<functionality>_<expected_outcome>`
- Use descriptive names that explain what is being tested
- Example: `test_constant_velocity_position_calculation()`

## Test Organization

### Arrange-Act-Assert (AAA) Pattern
All tests should follow the AAA pattern:

```python
def test_example():
    # Arrange: Set up test data and conditions
    initial_position = np.array([0.0, 0.0])
    velocity = np.array([5.0, 0.0])
    time = 2.0
    
    # Act: Execute the function being tested
    result = position_constant_velocity(initial_position, velocity, time)
    
    # Assert: Verify the expected outcome
    expected = np.array([10.0, 0.0])
    np.testing.assert_array_almost_equal(result, expected)
```

## Running Tests

### On MacBook (Local Testing)
```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests
python3 -m pytest

# Run specific test file
python3 -m pytest test_kinematics_formulas.py

# Run with verbose output
python3 -m pytest -v

# Run with coverage
python3 -m pytest --cov=. --cov-report=html
```

### On vast.ai Instance (Remote Testing)
```bash
# From MacBook, run tests remotely
python3 main_macbook.py --command "cd /root/bat && python3 -m pytest"

# Or SSH directly and run
ssh -i ~/.ssh/vastai_key -p 40542 root@[instance-ip]
cd /root/bat
python3 -m pytest
```

## Test Categories

### Unit Tests
- Test individual functions and methods
- Should be fast and isolated
- Mock external dependencies when necessary

### Integration Tests
- Test interactions between modules
- Test end-to-end workflows
- Verify data flow between components

## Test Coverage Goals

- Aim for >80% code coverage for critical functionality
- Focus on:
  - All mathematical formulas
  - Data generation functions
  - File I/O operations
  - Error handling

## Common Test Patterns

### Testing Numerical Functions
Use `np.testing.assert_array_almost_equal()` for floating-point comparisons:
```python
import numpy as np
np.testing.assert_array_almost_equal(actual, expected, decimal=6)
```

### Testing File Operations
Use temporary directories and verify file creation:
```python
import tempfile
import os

def test_file_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test file creation
        filepath = os.path.join(tmpdir, 'test.csv')
        # ... create file ...
        assert os.path.exists(filepath)
```

### Testing Error Cases
Use `pytest.raises()` to test exceptions:
```python
import pytest

def test_invalid_input():
    with pytest.raises(ValueError):
        function_with_invalid_input()
```

## Continuous Integration

- Run tests before committing code
- All tests must pass before merging
- Keep tests updated with code changes

## Test Data

- Use small, predictable test data
- Avoid large datasets unless necessary
- Use fixtures for reusable test data

## Best Practices

1. **One assertion per test** (when possible) - makes failures clear
2. **Test edge cases** - zero, negative, boundary values
3. **Test error conditions** - invalid inputs, missing files
4. **Keep tests independent** - no dependencies between tests
5. **Use descriptive assertions** - clear failure messages
6. **Test both positive and negative cases**
7. **Document complex test logic** - explain why the test exists

