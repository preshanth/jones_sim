#!/bin/bash
# Run calibration validation tests
#
# Usage:
#   ./run_validation_tests.sh           # Run all tests
#   ./run_validation_tests.sh fast      # Run only fast tests
#   ./run_validation_tests.sh slow      # Run only integration tests
#   ./run_validation_tests.sh coverage  # Run with coverage report

set -e

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEST_DIR/.."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}JONES_SIM CALIBRATION VALIDATION TESTS${NC}"
echo -e "${GREEN}======================================================================${NC}"

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Install with: pip install pytest pytest-timeout pytest-benchmark"
    exit 1
fi

MODE="${1:-all}"

case "$MODE" in
    fast)
        echo -e "${YELLOW}Running fast tests only...${NC}"
        pytest -m "fast" tests/test_calibration_validation.py tests/test_config.py
        ;;
    slow)
        echo -e "${YELLOW}Running integration tests (slow)...${NC}"
        pytest -m "slow or integration" tests/test_calibration_validation.py
        ;;
    integration)
        echo -e "${YELLOW}Running integration tests...${NC}"
        pytest -m "integration" tests/test_calibration_validation.py
        ;;
    benchmark)
        echo -e "${YELLOW}Running benchmark tests...${NC}"
        pytest -m "benchmark" tests/test_calibration_validation.py
        ;;
    coverage)
        echo -e "${YELLOW}Running tests with coverage...${NC}"
        pytest --cov=jones_sim --cov-report=html --cov-report=term \
            tests/test_calibration_validation.py tests/test_config.py
        echo -e "${GREEN}Coverage report: htmlcov/index.html${NC}"
        ;;
    all)
        echo -e "${YELLOW}Running all tests...${NC}"
        pytest tests/test_calibration_validation.py tests/test_config.py
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 {fast|slow|integration|benchmark|coverage|all}"
        exit 1
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo -e "${GREEN}======================================================================${NC}"
else
    echo -e "${RED}======================================================================${NC}"
    echo -e "${RED}✗ TESTS FAILED${NC}"
    echo -e "${RED}======================================================================${NC}"
fi

exit $EXIT_CODE
