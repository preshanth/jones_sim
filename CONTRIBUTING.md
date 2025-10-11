# Contributing to jones_sim

## Local Development Setup

### Installation

```bash
# Clone and install in editable mode with dev dependencies
pip install -e .[dev]
```

## Running Tests Locally

### Quick CI Check (Recommended)

Run the full CI pipeline locally before pushing:

```bash
./run_ci_locally.sh
```

This script runs:
- Code formatting checks (black)
- Linting and import sorting (ruff)
- Core test suite with coverage
- Package build verification

**Coverage report**: After running, open `htmlcov/index.html` in your browser.

### Running Specific Test Subsets

```bash
# Fast tests only (excludes slow MCMC and end-to-end tests)
pytest tests/ -n auto -m "not slow and not end_to_end and not requires_casa"

# Include slow tests
RUN_MEDIUM_TESTS=1 ./run_ci_locally.sh

# All tests including CASA integration (requires casatools)
pytest tests/ -n auto -v

# Single test file
pytest tests/test_effects.py -v

# Single test function
pytest tests/test_effects.py::test_parallactic_angle -v

# With coverage and parallelization
pytest tests/ -n auto --cov=jones_sim --cov-report=html
```

### Parallel Testing

Tests run in parallel by default using `pytest-xdist` for 2-4x speedup:

```bash
# Run on all CPU cores (default with -n auto)
pytest tests/ -n auto

# Run on specific number of workers
pytest tests/ -n 4

# Disable parallelization (sequential)
pytest tests/ -n 0
# or simply omit the -n flag
pytest tests/
```

**Note**: Individual slow tests (marked `@pytest.mark.slow`) won't benefit from parallelization, but having many fast tests run in parallel significantly speeds up the overall suite.

## Pre-commit Hooks (Optional)

Pre-commit hooks automatically check code quality before each commit:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Skip hooks for a specific commit (use sparingly)
git commit --no-verify -m "message"
```

The hooks will automatically:
- Format code with black
- Lint and sort imports with ruff (replaces flake8 + isort)
- Check for trailing whitespace, large files, etc.

## Code Quality Standards

### Formatting

We use **black** for code formatting (88 character line length):

```bash
# Check formatting
black --check jones_sim/ tests/

# Auto-format
black jones_sim/ tests/
```

### Linting and Import Sorting

We use **ruff** for linting and import organization (replaces flake8 + isort + more):

```bash
# Check all issues
ruff check jones_sim/ tests/

# Auto-fix issues
ruff check --fix jones_sim/ tests/

# Check only import sorting
ruff check --select I jones_sim/ tests/

# Format code (alternative to black)
ruff format jones_sim/ tests/
```

Ruff is 10-100x faster than the traditional Python tooling stack and catches more issues.

## Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.slow` - Long-running tests (MCMC sampling, >10s)
- `@pytest.mark.end_to_end` - Full pipeline integration tests
- `@pytest.mark.requires_casa` - Tests requiring CASA tools installation

Run specific markers:

```bash
pytest tests/ -m "slow"           # Only slow tests
pytest tests/ -m "not slow"       # Skip slow tests
pytest tests/ -m "requires_casa"  # Only CASA tests
```

## CI/CD Pipeline

GitHub Actions runs on every push:

1. **Linting** (Python 3.11): black, ruff
2. **Tests** (Python 3.9, 3.10, 3.11, 3.12):
   - Core tests with coverage
   - Medium tests (Python 3.11 only)
3. **Package Build**: Verify installability
4. **Integration Tests** (main branch only): Slow tests

**Before pushing**, run `./run_ci_locally.sh` to catch issues early.

## Making Changes

### Workflow

1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes
3. Run local CI: `./run_ci_locally.sh`
4. Commit changes: `git commit -m "descriptive message"`
5. Push and create PR: `git push origin feature-name`

### Commit Messages

Follow conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code restructuring
- `style:` Formatting changes
- `chore:` Maintenance tasks

Example: `feat: add RotationMeasure effect class`

## Common Issues

### Import Errors

If you see import errors after pulling changes:

```bash
pip install -e .[dev]  # Reinstall with updated dependencies
```

### Coverage Not Generating

```bash
# Make sure pytest-cov is installed
pip install pytest-cov

# Run with explicit coverage flags
pytest --cov=jones_sim --cov-report=html
```

### Pre-commit Hook Failures

```bash
# Update hooks to latest versions
pre-commit autoupdate

# Clear cache if having issues
pre-commit clean
```

## Getting Help

- Check existing tests for examples
- Review CLAUDE.md for project context and architecture
- Open an issue for bugs or feature requests
