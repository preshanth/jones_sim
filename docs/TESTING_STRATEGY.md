# Testing Strategy & Coverage Analysis

## Current Status (as of 2025-10-10)

### Overall Coverage: 45%
But this number is **misleading** - the core simulation code has excellent coverage.

### Coverage Breakdown

| Module | Coverage | Lines | Status |
|--------|----------|-------|--------|
| **Core Simulation** | | | |
| `simulator.py` | 100% | 35 | ✅ Excellent |
| `effects.py` | 93% | 108 | ✅ Excellent |
| `source_models.py` | 97% | 100 | ✅ Excellent |
| `symbolic.py` | 97% | 67 | ✅ Excellent |
| `visibility_generator.py` | 97% | 106 | ✅ Excellent |
| **MCMC/Sampling** | | | |
| `mc_sampler.py` | 58% | 95 | ⚠️ Moderate |
| `unified_sampler.py` | 54% | 178 | ⚠️ Moderate |
| `bandpass_sampler.py` | 0% | 95 | ❌ Untested |
| **Visualization** | | | |
| `plotting.py` | 0% | 105 | ❌ Untested |
| `unified_plotter.py` | 9% | 208 | ❌ Barely tested |
| **External Dependencies** | | | |
| `casa_interface.py` | 11% | 243 | ⚠️ Optional dependency |

**Core coverage: 97%** (simulation, effects, sources, symbolic)
**Full coverage: 45%** (includes MCMC, plotting, CASA)

## What CI Actually Runs

### On Every Push/PR (Python 3.9-3.12)
```bash
pytest -m "not slow and not end_to_end and not requires_casa"
```
- ✅ Core simulation tests
- ✅ Fast unit tests
- ❌ Excludes MCMC sampling
- ❌ Excludes plotting
- ❌ Excludes CASA integration

### On Push to Main (Python 3.11 only)
```bash
pytest -m "slow and not end_to_end and not requires_casa"
```
- ✅ MCMC sampling tests
- ❌ Still excludes end-to-end
- ❌ Still excludes CASA

### Never Runs in CI
- `@pytest.mark.end_to_end` tests
- `@pytest.mark.requires_casa` tests

**Result**: CI generates the 45% coverage number, which reflects fast tests only.

## Critical Analysis: What Actually Needs Testing?

### Philosophy
**Don't test for coverage numbers. Test for:**
1. Correctness of critical algorithms
2. Regression prevention
3. API contract validation
4. Error handling

### Module-by-Module Assessment

#### 1. `bandpass_sampler.py` (0% coverage)
**What it does**: PyMC MCMC sampling for bandpass parameters

**Should we test?**
- ❌ Don't test: MCMC convergence (PyMC's responsibility)
- ❌ Don't test: Statistical accuracy (requires domain expertise)
- ✅ Do test: Interface doesn't crash with toy data
- ✅ Do test: Parameter extraction methods

**Verdict**: Add 1-2 smoke tests. **Target: 20-30%**

#### 2. `plotting.py` (0% coverage)
**What it does**: Bokeh visualization functions

**Should we test?**
- ❌ Don't test: Visual appearance (requires human inspection)
- ❌ Don't test: Layout details (brittle, low value)
- ✅ Do test: Returns figure objects
- ✅ Do test: Handles edge cases (empty data, NaN)

**Verdict**: Add 3-5 smoke tests. **Target: 30-40%**

#### 3. `unified_plotter.py` (9% coverage)
**What it does**: Dashboard generation from MCMC results

**Should we test?**
- ❌ Don't test: HTML rendering (Bokeh's job)
- ❌ Don't test: Visual layout
- ✅ Do test: Summary statistics extraction
- ✅ Do test: Error handling for missing data
- ✅ Do test: File generation completes

**Verdict**: Add 5-10 integration tests. **Target: 40-50%**

#### 4. `mc_sampler.py` (58% coverage)
**What it does**: Gain MCMC sampling

**Should we test more?**
- ✅ Current 58% is good for interface validation
- ❌ Don't add: More convergence tests
- ⚠️ Could add: Edge case handling (zero antennas, etc.)

**Verdict**: Current level acceptable. **Target: 60-70%**

#### 5. `unified_sampler.py` (54% coverage)
**What it does**: Unified PyMC model for all Jones effects

**Should we test more?**
- ✅ Current 54% covers main workflow
- ❌ Untested parts (lines 238-357) are mostly reconstruction code
- ⚠️ Could add: Test `reconstruct_jones_matrices()` with sample data

**Verdict**: Current level acceptable. **Target: 65-75%**

#### 6. `casa_interface.py` (11% coverage)
**What it does**: Interface to CASA measurement sets

**Should we test more?**
- ❌ **No** - Requires CASA tools (external dependency)
- ✅ Correctly marked with `@pytest.mark.requires_casa`
- ❌ Don't mock CASA (tests the mock, not real behavior)
- ⚠️ Could add: Import tests without CASA installed

**Verdict**: 11% is fine for optional dependency. **Target: 15-25%**

## Three Options for Moving Forward

### Option A: Accept Current Coverage ✅ RECOMMENDED
**Effort**: None
**Coverage**: Stays at 45%

**Rationale**:
- Core code (97%) is what matters most
- MCMC/plotting hard to test meaningfully
- Coverage number doesn't reflect code quality

**Action**:
- Keep README explanation about core vs full coverage
- Don't add tests just for numbers
- Focus testing effort on new features

### Option B: Add Minimal Smoke Tests
**Effort**: 2-3 hours, ~100-150 lines of test code
**Coverage**: 45% → 60-65%

**What to add**:
```python
# test_plotting_smoke.py
def test_gains_plot_generates():
    """Verify plots generate without errors."""
    plotter = JonesPlotter()
    fig_amp, fig_phase = plotter.plot_gains_vs_time(...)
    assert fig_amp is not None
    assert fig_phase is not None

# test_mcmc_smoke.py
def test_bandpass_sampler_runs():
    """Verify sampler completes with toy data."""
    sampler = BandpassMCSampler(n_antennas=2, n_channels=8)
    sampler.build_bandpass_model()
    trace = sampler.sample(draws=10, tune=5)  # Minimal sampling
    assert trace is not None
```

**Benefits**:
- Catch regressions in plotting/MCMC code
- Minimal maintenance burden
- Still don't test "correctness", just "doesn't crash"

**Drawbacks**:
- Tests add little value beyond import testing
- Could give false confidence

### Option C: Full Coverage Push ❌ NOT RECOMMENDED
**Effort**: 1-2 weeks
**Coverage**: 45% → 80%+

**Would require**:
- Testing visual output (brittle)
- Testing MCMC convergence (domain expertise)
- Mocking CASA (tests mock, not real code)
- High maintenance burden

**Why not**:
- Testing for coverage sake
- Low value tests
- Better to spend time on validation work

## Recommendation

**Accept Option A**: Current testing is appropriate.

### Why This is OK

1. **Core simulation code (97%) is excellent**
   - This is the algorithmic heart
   - Well-tested, high confidence

2. **MCMC code (0-58%) is specialist tooling**
   - Built on PyMC (already tested)
   - Best validated through scientific use, not unit tests
   - Smoke tests add little value

3. **Plotting code (0-9%) is visual**
   - Hard to test programmatically
   - Best validated by human inspection
   - Regressions caught by visual QA

4. **CASA code (11%) is optional**
   - External dependency
   - Correctly excluded from CI
   - Users with CASA will discover issues

### Better Use of Testing Effort

Instead of chasing coverage, focus on:
1. **Validation studies** - Compare outputs to known results
2. **Integration tests** - Real-world workflows
3. **Documentation** - Examples and tutorials
4. **Scientific validation** - Does it match theory?

## CI Configuration

### Current Setup (Correct)
```yaml
# Fast tests: Every commit, all Python versions
pytest -m "not slow and not end_to_end and not requires_casa"

# Medium tests: Python 3.11 only
pytest -m "not end_to_end and not requires_casa"

# Slow tests: Main branch only
pytest -m "slow and not end_to_end and not requires_casa"
```

**Why this is good**:
- Fast feedback on core code
- Don't waste CI time on slow MCMC tests
- CASA tests run locally only (developer has CASA)

### Don't Change To
```yaml
# Bad: Run everything (slow, overkill)
pytest tests/
```

## Conclusion

**Current 45% coverage is fine.** The number is misleading because:
- Core code: 97% (excellent)
- Specialist code: 0-58% (hard to test, less critical)

**Action**: Accept current coverage, focus effort on scientific validation and documentation.

## Future Considerations

### When to Add Tests
- ✅ New core simulation features → Aim for 90%+ coverage
- ✅ Bug fixes → Add regression test
- ⚠️ New MCMC samplers → Minimal smoke test only
- ⚠️ New visualizations → Manual visual QA
- ❌ Coverage number improvement → Not a goal

### When Coverage Will Drop
- Adding more MCMC code
- Adding more visualization
- Adding more external dependencies

**This is OK.** Total coverage percentage is not a quality metric for this codebase.
