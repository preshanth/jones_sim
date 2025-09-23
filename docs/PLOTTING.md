# Plotting and Visualization Guide

## Overview

The `jones_sim` package provides comprehensive interactive visualization capabilities for Monte Carlo results. All plots are generated using Bokeh for interactive exploration, with proper uncertainty quantification and statistical analysis.

## Dashboard Architecture

### Main Dashboard Components

The unified dashboard organizes visualizations into logical sections:

1. **Summary Panel**: Statistical overview of all effects
2. **Time Domain Plots**: Effects varying with observation time
3. **Frequency Domain Plots**: Effects varying across spectral channels
4. **Jones Matrix Analysis**: Individual matrix element behavior
5. **Export Options**: Data extraction and summary reports

### Accessing Visualizations

**Command Line Interface:**
```bash
jones-sim --config observation.json --plot --plot-output results.html
```

**Python API:**
```python
from jones_sim.unified_plotter import JonesPlotter

plotter = JonesPlotter(sampler)
plotter.create_comprehensive_dashboard("dashboard.html")
```

## Plot Types and Interpretation

### Time Domain Visualizations

#### Electronic Gains Evolution

**Plot Features:**
- Amplitude and phase evolution over observation time
- Semi-transparent Monte Carlo sample traces
- 90% confidence intervals (shaded regions)
- Median curves (thick lines)
- Separate XX and YY polarization tracking

**Physical Interpretation:**
- Smooth thermal variations on hour timescales
- Random walk phase behavior
- Correlated amplitude changes between polarizations
- Systematic drifts vs. random fluctuations

**Configuration Parameters:**
```python
amp_fig, phase_fig = plotter.plot_gains_vs_time(
    antenna_id=0,        # Which antenna to analyze
    n_traces=30          # Number of sample traces to display
)
```

**Analysis Guidelines:**
- Look for systematic trends in amplitude evolution
- Phase should show gradual drift, not sudden jumps
- XX/YY correlations indicate common-mode effects
- Large uncertainties suggest poorly constrained parameters

#### Parallactic Angle Rotation

**Plot Features:**
- Deterministic celestial rotation pattern
- Linear evolution at 15 degrees per hour
- No uncertainty (determined by observation geometry)

**Physical Interpretation:**
- Earth rotation effects on linear feeds
- Predictable from source position and observation time
- Critical for polarization calibration

### Frequency Domain Visualizations

#### Bandpass Response

**Plot Features:**
- Amplitude and phase response across frequency channels
- Jagged structure from instrumental effects
- Cable delay signatures (linear phase slopes)
- Per-channel amplitude variations

**Physical Interpretation:**
- Linear phase slopes indicate cable length differences
- Amplitude variations from filter responses
- Standing wave patterns from impedance mismatches
- Antenna-dependent characteristics

**Configuration Parameters:**
```python
bp_amp_fig, bp_phase_fig = plotter.plot_bandpass_vs_frequency(
    antenna_id=0,        # Which antenna to analyze
    n_traces=30          # Number of sample traces to display
)
```

**Analysis Guidelines:**
- Phase slopes reveal cable delay differences
- Amplitude structure indicates filter characteristics
- Smooth vs. jagged patterns distinguish effect types
- Compare antennas to identify systematic vs. random variations

### Static Effect Analysis

#### Instrumental Leakage

**Visualization:**
- Bar charts or scatter plots showing leakage magnitudes
- Complex plane representation of leakage terms
- Antenna-by-antenna comparison

**Physical Interpretation:**
- Cross-polarization coupling strength
- Feed alignment quality
- Receiver isolation performance

#### Cross-hand Phase Offsets

**Visualization:**
- Phase distribution histograms
- Antenna-dependent offset values
- Uncertainty quantification

## Interactive Features

### Bokeh Interactivity

All plots support standard Bokeh interactions:

**Pan and Zoom:**
- Mouse drag to pan
- Scroll wheel to zoom
- Box zoom tool for precise regions

**Data Inspection:**
- Hover tooltips show exact values
- Click to highlight specific traces
- Selection tools for data subset analysis

**Plot Configuration:**
- Toggle legend items to hide/show traces
- Reset view to original zoom level
- Save plot as PNG image

### Custom Analysis Tools

**Effect Isolation:**
```python
# Plot individual effects
gains_only = plotter.plot_gains_vs_time(antenna_id=0)
bandpass_only = plotter.plot_bandpass_vs_frequency(antenna_id=0)

# Compare multiple antennas
for ant_id in range(4):
    fig = plotter.plot_gains_vs_time(ant_id)
```

**Parameter Sensitivity:**
```python
# Extract specific parameter distributions
summaries = plotter.create_effect_summary()
thermal_amplitude = summaries['gains']['thermal_amplitude_mean']
```

## Jones Matrix Element Analysis

### Complex Scatter Plots

**Visualization:**
- Real vs. imaginary plane scatter plots
- Color-coded by matrix element (J11, J12, J21, J22)
- Distribution shapes reveal parameter correlations

**Configuration:**
```python
jones_fig = plotter.plot_individual_jones_matrices(
    antenna_id=0,        # Which antenna
    time_idx=0,          # Which time sample
    freq_idx=0,          # Which frequency channel
    n_samples=50         # Number of Monte Carlo samples
)
```

**Interpretation Guidelines:**
- Diagonal elements (J11, J22) should be near (1,0)
- Off-diagonal elements (J12, J21) indicate cross-coupling
- Scatter size reflects parameter uncertainty
- Clustering patterns show correlations

### Matrix Properties

**Unitarity Checks:**
- Determinant distributions
- Eigenvalue analysis
- Condition number assessment

**Physical Constraints:**
- Amplitude ranges
- Phase unwrapping
- Polarization conservation

## Statistical Analysis Features

### Uncertainty Quantification

**Confidence Intervals:**
- 90% intervals (default shading)
- Percentile-based error bars
- Sample distribution overlays

**Convergence Diagnostics:**
- R-hat statistics display
- Effective sample size indicators
- Trace plot quality assessment

### Effect Ranking

**Relative Importance:**
- Variance contributions by effect
- Signal-to-noise ratios
- Calibration impact assessment

**Summary Statistics:**
```json
{
  "gains": {
    "type": "time_varying",
    "base_amplitude_xx_mean": 1.023,
    "base_amplitude_xx_std": 0.045,
    "thermal_timescale": 3600.0,
    "n_antennas_affected": 8
  }
}
```

## Export and Reporting

### Dashboard Output

**HTML Dashboard:**
- Self-contained interactive plots
- Embedded summary statistics
- Tabbed organization by effect type

**JSON Summary:**
- Machine-readable effect parameters
- Uncertainty estimates
- Configuration metadata

### Plot Extraction

**Individual Figures:**
```python
# Save specific plots
from bokeh.io import save, output_file

output_file("gains_evolution.html")
save(gains_figure)
```

**Data Export:**
```python
# Extract underlying data
times, gains_xx, gains_yy = sampler.extract_gain_samples(antenna_id=0)
np.save("gains_samples.npy", gains_xx)
```

## Customization Options

### Appearance Settings

**Color Schemes:**
- Effect-specific color coding
- Polarization-based coloring (blue/red for XX/YY)
- Configurable transparency levels

**Plot Dimensions:**
```python
plotter = JonesPlotter(sampler)
plotter.width = 800   # Plot width in pixels
plotter.height = 600  # Plot height in pixels
```

### Analysis Parameters

**Sample Selection:**
```python
# Vary number of traces displayed
plotter.plot_gains_vs_time(n_traces=100)  # More traces, lower individual transparency

# Focus on specific time/frequency ranges
plotter.plot_bandpass_vs_frequency(antenna_id=0)  # Full range
```

**Statistical Options:**
- Confidence interval levels (5-95%, 10-90%, etc.)
- Outlier detection and highlighting
- Robust statistics vs. standard estimates

## Performance Considerations

### Large Dataset Handling

**Memory Management:**
- Automatic trace thinning for display
- Progressive loading for large posterior samples
- Efficient Bokeh data source management

**Render Performance:**
- Optimize number of displayed traces
- Use data decimation for high-resolution plots
- Leverage Bokeh's built-in performance features

### Batch Processing

**Multiple Antenna Analysis:**
```python
# Generate plots for all antennas
for ant_id in range(sampler.n_antennas):
    amp_fig, phase_fig = plotter.plot_gains_vs_time(ant_id)
    # Save individual plots or combine into grid
```

**Automated Reporting:**
```python
# Generate standardized reports
def create_observation_report(sampler, output_dir):
    plotter = JonesPlotter(sampler)

    # Main dashboard
    plotter.create_comprehensive_dashboard(f"{output_dir}/dashboard.html")

    # Individual antenna analysis
    for ant_id in range(sampler.n_antennas):
        amp_fig, phase_fig = plotter.plot_gains_vs_time(ant_id)
        # Save antenna-specific analysis
```

## Integration with Analysis Pipelines

### Calibration Assessment

**Error Budget Visualization:**
- Compare effect magnitudes
- Identify dominant error sources
- Assess calibration requirements

**Strategy Evaluation:**
- Model different calibration approaches
- Visualize residual errors
- Optimize observation strategies

### Scientific Impact Analysis

**Observable Sensitivity:**
- Propagate errors to science products
- Visualize systematic vs. statistical uncertainties
- Guide observation planning decisions

This visualization framework provides comprehensive tools for understanding instrumental effects and their uncertainties in radio interferometric observations.