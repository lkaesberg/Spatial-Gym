# Spatial-Gym Test Suite

This directory contains comprehensive tests for the Spatial-Gym environment.

## Test Structure

### `test_environment.py`
Basic environment tests covering:
- Environment initialization with various configurations
- Gymnasium API compliance (reset, step, action_space, observation_space)
- Rendering modes (human, llm)
- Episode termination and truncation
- Robustness and edge cases

### `test_random_agent.py`
Random agent tests including:
- Single and multiple episode runs
- Statistics collection (success rate, episode length, rewards)
- Different observation formats
- Stress tests with extended runs
- Action distribution validation

### `test_predefined_paths.py`
Predefined action sequence tests:
- **Valid paths**: Simple forward, circular, zigzag, back-and-forth patterns
- **Invalid paths**: Boundary violations, contradictory movements, excessive actions
- **Specific puzzles**: Reproducibility tests with seeds
- Path validation and reward accumulation

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_environment.py -v
pytest tests/test_random_agent.py -v
pytest tests/test_predefined_paths.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=Spatial_Gym --cov-report=html
```

### Run specific test class
```bash
pytest tests/test_environment.py::TestEnvironmentInitialization -v
```

### Run specific test
```bash
pytest tests/test_random_agent.py::TestRandomAgent::test_random_agent_statistics -v
```

## Test Requirements

Install test dependencies:
```bash
pip install -e ".[test]"
```

Or manually:
```bash
pip install pytest pytest-cov
```

## CI/CD Integration

Tests are automatically run on:
- Every push to `main` and `develop` branches
- All pull requests
- Multiple OS (Ubuntu, macOS) and Python versions (3.9, 3.10, 3.11)

See `.github/workflows/tests.yml` for the full CI configuration.

## Test Categories

### Unit Tests
- Environment initialization
- API compliance
- Observation/action spaces

### Integration Tests
- Multi-episode runs
- Different configurations
- Random agent behavior

### Validation Tests
- Predefined action sequences
- Path validation
- Reward structure

## Adding New Tests

1. Create a new test file: `tests/test_<feature>.py`
2. Follow the existing structure with test classes
3. Use descriptive test names: `test_<what_is_being_tested>`
4. Add docstrings explaining what each test validates
5. Ensure tests are independent and don't rely on execution order

## Debugging Failed Tests

### Verbose output
```bash
pytest tests/ -v -s
```

### Show print statements
```bash
pytest tests/ -v -s --capture=no
```

### Stop on first failure
```bash
pytest tests/ -x
```

### Run only failed tests
```bash
pytest tests/ --lf
```

## Test Coverage

Current test coverage includes:
- ✅ Environment initialization
- ✅ Gymnasium API compliance
- ✅ Action/observation spaces
- ✅ Rendering modes
- ✅ Episode lifecycle
- ✅ Random agent behavior
- ✅ Predefined paths (valid and invalid)
- ✅ Multi-episode stability
- ✅ Configuration variations

## Continuous Improvement

Tests are continuously improved to:
- Increase code coverage
- Validate edge cases
- Ensure cross-platform compatibility
- Test new features
- Prevent regressions
