# Jones Matrix Parameter Distributions

## Free Parameters in Jones Matrix Equations

From our Jones factorization `J = P·E·X·G·B·T·R·C`, here are the free parameters for error estimation and uncertainty propagation:

### Electronic Gains G (most critical for error budget)
- `A_xx(ant, freq, time)`, `φ_xx(ant, freq, time)` - XX amplitude/phase
- `A_yy(ant, freq, time)`, `φ_yy(ant, freq, time)` - YY amplitude/phase
- **Dependencies**: (antenna, frequency, time)
- **Typical uncertainties**: ~1% amplitude, ~few degrees phase
- **Correlations**: XX/YY often correlated, temporal correlation ~minutes to hours

### Instrumental Leakage X (antenna-specific, stable)
- `D_hv(ant)`, `D_vh(ant)` - complex leakage terms
- `θ(ant)` - misalignment angle
- **Dependencies**: (antenna) - stable over observation
- **Typical uncertainties**: ~0.1% for leakage, ~0.1° for misalignment
- **Correlations**: D_hv and D_vh often anticorrelated

### Bandpass/Delays B (frequency-dependent)
- `τ_xx(ant, freq)`, `τ_yy(ant, freq)` - delays per polarization
- **Dependencies**: (antenna, frequency)
- **Typical uncertainties**: ~picoseconds to nanoseconds
- **Correlations**: Smooth frequency structure, XX/YY correlated

### TEC Effects T (time-varying)
- `t_xx(ant, time)`, `t_yy(ant, time)` - differential TEC
- `θ_xx(ant)`, `θ_yy(ant)` - constant phase offsets
- **Dependencies**: (antenna, time)
- **Typical uncertainties**: ~few % of TEC value
- **Correlations**: Spatially and temporally correlated across antennas

### R/L Delay Difference R
- `Δτ(ant)` - R-L delay difference
- **Dependencies**: (antenna) - stable
- **Typical uncertainties**: ~100 picoseconds
- **Correlations**: Independent per antenna

### Cross-hand Phase C
- `φ(ant)` - cross-hand phase offset
- **Dependencies**: (antenna) - stable
- **Typical uncertainties**: ~few degrees
- **Correlations**: Independent per antenna

## Statistical Sampling Strategy

### Parameter Distribution Requirements
1. **Hold distributions** (not just point values) - Gaussian, uniform, etc.
2. **Sample correlated parameters** - gains often correlated across polarizations
3. **Interpolate** sparse time/frequency sampling
4. **Propagate uncertainties** through Jones matrices to visibilities

### Error Estimation Workflow
1. Define parameter distributions for each effect
2. Sample parameter realizations from distributions
3. Compute Jones matrices for each realization
4. Apply to visibilities and compute statistics
5. Analyze error propagation and correlations

### Implementation Strategy
- `ParameterDistribution` class for each parameter type
- Support for different distribution types (normal, log-normal, uniform)
- Correlation matrices between related parameters
- Time/frequency interpolation with uncertainty propagation
- Monte Carlo sampling for error estimation