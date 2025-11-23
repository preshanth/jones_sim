# Calibration Testing Extension - TODO List

## Current Status
✅ K (Delay) validation implemented - `validate_delay_recovery.py`
✅ G (Gains) partial validation implemented
✅ CalibrationSolver supports K, G, B effects
✅ Basic comparison framework in place

## TODO: Extend Testing to All Caltable Types

### Phase 1: Core Caltable Types (High Priority)
These are the most commonly used calibration types in real observations.

#### 1. **Bandpass (B) Validation**
**Branch**: `test/bandpass-validation`
**Parallelizable**: Yes (independent)
**Estimated complexity**: Medium

- [ ] Create `scripts/validate_bandpass_recovery.py`
  - [ ] Generate ground truth bandpass (frequency-dependent gains)
    - [ ] Trapezoidal bandpass shape (edge rolloff)
    - [ ] Ripple in passband
    - [ ] Per-antenna, per-polarization
  - [ ] Corrupt MS with bandpass effect
  - [ ] Run CASA `bandpass()` to get B-table
  - [ ] Run our `CalibrationSolver` with B effect
  - [ ] Compare truth vs CASA vs ours
  - [ ] Generate diagnostic plots
- [ ] Test cases:
  - [ ] 64 channels (moderate bandwidth)
  - [ ] 128 channels (wideband)
  - [ ] With and without noise
  - [ ] Different edge rolloff parameters
  - [ ] Combined with K (delay + bandpass)

#### 2. **Leakage (D) Validation**
**Branch**: `test/leakage-validation`
**Parallelizable**: Yes (independent)
**Estimated complexity**: Medium

- [ ] Create `scripts/validate_leakage_recovery.py`
  - [ ] Generate ground truth D-terms (leakage)
    - [ ] Complex leakage per antenna: d_xy, d_yx
    - [ ] Typical values: |d| ~ 0.01-0.1
  - [ ] Corrupt MS with leakage effect
  - [ ] Run CASA `polcal()` to get D-table
  - [ ] Run our `CalibrationSolver` with D effect
  - [ ] Compare truth vs CASA vs ours
  - [ ] Generate diagnostic plots (Stokes Q, U, V)
- [ ] Test cases:
  - [ ] Full polarization MS (4 correlations)
  - [ ] Different leakage magnitudes
  - [ ] With and without noise
  - [ ] Combined with K+G (delay + gain + leakage)

#### 3. **Combined Effects Validation**
**Branch**: `test/combined-effects`
**Parallelizable**: Depends on 1 & 2
**Estimated complexity**: High

- [ ] Create `scripts/validate_combined_effects.py`
  - [ ] Test K+G (delay + gain)
  - [ ] Test K+B (delay + bandpass)
  - [ ] Test K+G+B (delay + gain + bandpass)
  - [ ] Test K+G+D (delay + gain + leakage) - full polarization
  - [ ] Test K+G+B+D (all effects combined)
- [ ] Sequential solving workflow:
  - [ ] K → G → B → D (standard pipeline)
  - [ ] Compare with CASA sequential calibration
  - [ ] Validate error propagation between stages
- [ ] Generate comparison metrics:
  - [ ] RMS errors per effect
  - [ ] Cross-effect correlations
  - [ ] Convergence diagnostics

### Phase 2: Advanced Calibration Types (Medium Priority)

#### 4. **Time-Varying Gains (G with solint)**
**Branch**: `test/time-varying-gains`
**Parallelizable**: Yes
**Estimated complexity**: Medium

- [ ] Extend `validate_delay_recovery.py` for time variation
  - [ ] Generate time-varying ground truth gains
    - [ ] Smooth variations (atmospheric)
    - [ ] Step changes (pointing errors)
  - [ ] Test different solint modes:
    - [ ] `solint='int'` - per integration
    - [ ] `solint='60s'` - time averaging
    - [ ] `solint='inf'` - single solution
  - [ ] Compare solver performance vs solint
- [ ] Test cases:
  - [ ] 30 min observation, 2s integrations
  - [ ] Different time variation timescales
  - [ ] Noise impact on time resolution

#### 5. **Opacity/Tropospheric (T) Validation**
**Branch**: `test/opacity-validation`
**Parallelizable**: Yes
**Estimated complexity**: Low-Medium

- [ ] Create `scripts/validate_opacity_recovery.py`
  - [ ] Generate ground truth opacity (elevation-dependent)
  - [ ] Corrupt MS with opacity effect
  - [ ] Run CASA `gencal(caltype='amp')` for T-table
  - [ ] Compare with atmospheric model
- [ ] Test cases:
  - [ ] Different elevation ranges
  - [ ] Different opacity values (tau)

### Phase 3: Testing Infrastructure (High Priority)
These enable better testing and can be done in parallel.

#### 6. **Automated Test Suite**
**Branch**: `test/automated-suite`
**Parallelizable**: Yes
**Estimated complexity**: Medium

- [ ] Create `tests/test_calibration_validation.py`
  - [ ] Pytest-based integration tests
  - [ ] Parametrized tests for different configs
  - [ ] Automated comparison checks (assert RMS < threshold)
  - [ ] CI/CD integration (run on PR)
- [ ] Test fixtures:
  - [ ] Standard simulated MS (cached)
  - [ ] Ground truth data generators
  - [ ] Comparison utilities
- [ ] Performance benchmarks:
  - [ ] Timing tests (CPU vs GPU)
  - [ ] Memory usage monitoring
  - [ ] Convergence speed metrics

#### 7. **Diagnostic Plotting Enhancement**
**Branch**: `test/enhanced-plotting`
**Parallelizable**: Yes
**Estimated complexity**: Low

- [ ] Enhance `jones_sim/plotting.py`
  - [ ] Bandpass plots (amp/phase vs frequency)
  - [ ] Leakage plots (D-term amplitude/phase)
  - [ ] Time-series plots (gains vs time)
  - [ ] Residual plots (model vs data)
  - [ ] Convergence plots (MCMC traces)
- [ ] Create comparison dashboard:
  - [ ] Side-by-side: Truth | CASA | Ours
  - [ ] Difference plots
  - [ ] Error distributions

#### 8. **JSON Configuration for Tests**
**Branch**: `test/json-configs`
**Parallelizable**: Yes
**Estimated complexity**: Low

- [ ] Create test configuration files:
  - [ ] `configs/test_k_only.json`
  - [ ] `configs/test_g_only.json`
  - [ ] `configs/test_b_only.json`
  - [ ] `configs/test_d_only.json`
  - [ ] `configs/test_k_g_b_d.json`
- [ ] Update validation scripts to use JSON configs
- [ ] Document config schema for testing

### Phase 4: Advanced Features (Lower Priority)

#### 9. **Cross-Validation Framework**
**Branch**: `test/cross-validation`
**Parallelizable**: After Phase 1-3
**Estimated complexity**: High

- [ ] Implement k-fold cross-validation:
  - [ ] Split data by time
  - [ ] Split data by frequency
  - [ ] Split data by baseline
- [ ] Generate cross-validation metrics:
  - [ ] Predictive accuracy
  - [ ] Generalization error
  - [ ] Overfitting detection

#### 10. **Real Data Testing**
**Branch**: `test/real-data`
**Parallelizable**: After all synthetic tests pass
**Estimated complexity**: High

- [ ] Identify test datasets:
  - [ ] VLA calibrator observations
  - [ ] ALMA calibrator observations
  - [ ] Public archive data
- [ ] Create real data validation pipeline:
  - [ ] Load real MS
  - [ ] Run CASA standard calibration
  - [ ] Run our solver
  - [ ] Compare quality metrics (not ground truth)
- [ ] Metrics for real data (no ground truth):
  - [ ] Self-consistency (closure phases)
  - [ ] Residual RMS after calibration
  - [ ] Image quality metrics

## Parallelization Strategy (Git Worktrees)

### Setup Git Worktrees
```bash
# Create worktrees for parallel work
git worktree add ../jones_sim_bandpass test/bandpass-validation
git worktree add ../jones_sim_leakage test/leakage-validation
git worktree add ../jones_sim_combined test/combined-effects
git worktree add ../jones_sim_plotting test/enhanced-plotting
git worktree add ../jones_sim_automated test/automated-suite
```

### Work Distribution
- **Claude Instance 1**: Bandpass validation (#1)
- **Claude Instance 2**: Leakage validation (#2)
- **Claude Instance 3**: Plotting enhancements (#7)
- **Claude Instance 4**: Automated test suite (#6)
- **Claude Instance 5**: JSON configs (#8)

### Integration Points
- All branches merge back to `cleanup`
- After merge: run full test suite
- Document results in `docs/VALIDATION_RESULTS.md`

## Success Criteria

### For Each Validation Script
- [ ] Passes with RMS error < 0.01 ns (K)
- [ ] Passes with RMS error < 0.01 (G amplitude)
- [ ] Passes with RMS error < 0.1° (G phase)
- [ ] Passes with RMS error < 0.02 (B)
- [ ] Passes with RMS error < 0.001 (D leakage)
- [ ] Comparison with CASA: ±10% or better
- [ ] All plots generated without errors
- [ ] Documented in validation report

### For Test Suite
- [ ] All pytest tests pass
- [ ] Code coverage > 80% for calibration modules
- [ ] CI/CD pipeline runs successfully
- [ ] Performance benchmarks meet targets

## Timeline Estimate
- **Phase 1** (Core types): 2-3 days (parallelized)
- **Phase 2** (Advanced): 1-2 days (parallelized)
- **Phase 3** (Infrastructure): 1-2 days (parallelized)
- **Phase 4** (Advanced features): 3-5 days
- **Total**: ~1-2 weeks with 5 parallel instances

## Notes
- Always test with and without noise
- Always test with multiple random seeds
- Always compare against CASA baseline
- Always generate diagnostic plots
- Always document failures and edge cases
