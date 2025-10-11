#!/bin/bash
# Local CI/CD test script - mimics GitHub Actions workflow
# Run this before pushing to catch issues early

set -e  # Exit on first error

PYTHON_VERSION=${PYTHON_VERSION:-$(python --version | cut -d' ' -f2)}
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Running Local CI/CD Tests =====${NC}"
echo "Python version: $PYTHON_VERSION"
echo ""

# Function to print section headers
section() {
    echo ""
    echo -e "${YELLOW}===== $1 =====${NC}"
}

# Function to print success
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    error "Not in project root directory (no pyproject.toml found)"
    exit 1
fi

# 1. Install dependencies
section "Installing dependencies"
pip install -e .[dev] -q
success "Dependencies installed"

# 2. Linting checks
section "Running linting checks"

# Black formatting check
echo "Checking code formatting with black..."
if black --check --diff jones_sim/ tests/; then
    success "Black formatting check passed"
else
    error "Black formatting check failed"
    echo "Run: black jones_sim/ tests/"
    exit 1
fi

# Ruff linting (replaces flake8 + isort + more)
echo ""
echo "Linting with ruff..."
if ruff check jones_sim/ tests/; then
    success "Ruff linting passed"
else
    error "Ruff found issues"
    echo "Run: ruff check --fix jones_sim/ tests/"
    exit 1
fi

success "All linting checks passed"

# 3. Core tests with coverage
section "Running core tests with coverage"
pytest tests/ \
    -n auto \
    -m "not slow and not end_to_end and not requires_casa" \
    --cov=jones_sim \
    --cov-report=term-missing \
    --cov-report=html \
    --maxfail=5 \
    -v

success "Core tests passed"

# 4. Optional: Run medium tests
if [ "$RUN_MEDIUM_TESTS" = "1" ]; then
    section "Running medium tests (including MCMC)"
    pytest tests/ \
        -n auto \
        -m "not end_to_end and not requires_casa" \
        --maxfail=3 \
        -v
    success "Medium tests passed"
fi

# 5. Package build check
section "Testing package build"
pip install build twine -q
python -m build
twine check dist/*
success "Package build check passed"

# 6. Test installation
section "Testing package installation"
pip uninstall jones_sim -y -q
pip install . -q
python -c "import jones_sim; print(f'jones_sim {jones_sim.__version__} installed successfully')"
pip install -e .[dev] -q  # Reinstall in editable mode
success "Package installation test passed"

# Summary
echo ""
echo -e "${GREEN}===== All CI checks passed! =====${NC}"
echo ""
echo "Coverage report available at: htmlcov/index.html"
echo ""
echo "Optional: Set RUN_MEDIUM_TESTS=1 to include slow tests"
echo "Example: RUN_MEDIUM_TESTS=1 ./run_ci_locally.sh"
