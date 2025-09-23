# Jones Simulation API Documentation

## Overview

The `jones_sim` package provides Monte Carlo sampling and analysis of Jones matrix parameters for radio interferometry. The package implements Bayesian parameter estimation for instrumental effects across antenna, time, and frequency dimensions.

## Core Classes

### JonesMCSampler

Main class for unified Monte Carlo sampling of Jones matrix parameters.

```python
from jones_sim.unified_sampler import JonesMCSampler

sampler = JonesMCSampler(config)
```

#### Constructor

```python
JonesMCSampler(config: Union[Dict, str, Path])
```

**Parameters:**
- `config`: Configuration dictionary, JSON string, or path to JSON configuration file

**Attributes:**
- `n_antennas`: Number of antennas in simulation
- `n_times`: Number of time samples
- `n_freqs`: Number of frequency channels
- `times`: Time grid array (seconds)
- `frequencies`: Frequency grid array (Hz)
- `model`: PyMC model object (after building)
- `trace`: ArviZ InferenceData object (after sampling)

#### Methods

##### `build_unified_model() -> pm.Model`

Constructs unified PyMC model for all enabled effects.

**Returns:** PyMC model object

**Effects Supported:**
- `gains`: Time-varying electronic gains with thermal drift
- `bandpass`: Frequency-varying response with cable delays
- `leakage`: Static instrumental polarization leakage
- `parallactic`: Deterministic parallactic angle rotation

##### `sample(draws=1000, tune=500, chains=2, target_accept=0.9) -> az.InferenceData`

Executes MCMC sampling using NUTS algorithm.

**Parameters:**
- `draws`: Posterior samples per chain
- `tune`: Tuning samples for step size adaptation
- `chains`: Number of parallel MCMC chains
- `target_accept`: Target acceptance rate (0.8-0.95)

**Returns:** ArviZ InferenceData with posterior samples

##### `compute_jones_matrices(sample_idx=None) -> np.ndarray`

Computes Jones matrices from sampled parameters.

**Parameters:**
- `sample_idx`: Specific sample index (None for all samples)

**Returns:**
- All samples: `(n_samples, n_antennas, n_times, n_freqs, 2, 2)` complex array
- Single sample: `(n_antennas, n_times, n_freqs, 2, 2)` complex array

##### `from_json_file(config_path) -> JonesMCSampler`

Class method to create sampler from JSON configuration file.

**Parameters:**
- `config_path`: Path to JSON configuration

**Returns:** Configured sampler instance

### JonesPlotter

Interactive visualization and analysis for Monte Carlo results.

```python
from jones_sim.unified_plotter import JonesPlotter

plotter = JonesPlotter(sampler, logger)
```

#### Constructor

```python
JonesPlotter(sampler: JonesMCSampler, logger: Optional[logging.Logger] = None)
```

**Parameters:**
- `sampler`: Sampler instance with completed trace
- `logger`: Optional logger for progress tracking

#### Methods

##### `create_effect_summary() -> Dict[str, Dict[str, Any]]`

Generates statistical summaries for each enabled effect.

**Returns:** Dictionary with effect names as keys, summary statistics as values

**Summary Fields:**
- `type`: Effect classification (time_varying, frequency_varying, static)
- `*_mean`, `*_std`: Parameter estimates and uncertainties
- `n_antennas_affected`: Number of antennas with this effect
- Effect-specific statistics (timescales, frequency ranges, etc.)

##### `plot_gains_vs_time(antenna_id=0, n_traces=30) -> Tuple[figure, figure]`

Creates time-domain plots for gain evolution.

**Parameters:**
- `antenna_id`: Antenna index to plot
- `n_traces`: Number of Monte Carlo traces to display

**Returns:** Tuple of (amplitude_figure, phase_figure)

**Plot Features:**
- Semi-transparent sample traces
- 90% confidence intervals
- Median evolution curves
- Interactive hover tooltips

##### `plot_bandpass_vs_frequency(antenna_id=0, n_traces=30) -> Tuple[figure, figure]`

Creates frequency-domain plots for bandpass response.

**Parameters:**
- `antenna_id`: Antenna index to plot
- `n_traces`: Number of Monte Carlo traces to display

**Returns:** Tuple of (amplitude_figure, phase_figure)

##### `plot_individual_jones_matrices(antenna_id=0, time_idx=0, freq_idx=0, n_samples=50) -> figure`

Creates complex scatter plot of Jones matrix elements.

**Parameters:**
- `antenna_id`: Antenna index
- `time_idx`: Time index
- `freq_idx`: Frequency index
- `n_samples`: Number of samples to plot

**Returns:** Bokeh figure with complex plane scatter plot

##### `create_comprehensive_dashboard(output_file="jones_dashboard.html") -> Dict`

Generates complete interactive dashboard with all analyses.

**Parameters:**
- `output_file`: HTML output filename

**Returns:** Dictionary of effect summaries

**Dashboard Components:**
- Effect summary panel
- Tabbed interface for individual effects
- Time-domain plots (gains, parallactic angle)
- Frequency-domain plots (bandpass)
- Jones matrix element visualizations

## Configuration Schema

### Grid Configuration

```json
{
  "grid": {
    "n_antennas": 4,
    "n_times": 30,
    "n_frequencies": 64,
    "time": {
      "start": 0.0,
      "end": 7200.0,
      "units": "seconds"
    },
    "frequency": {
      "start": 1.3e9,
      "end": 1.5e9,
      "units": "Hz"
    }
  }
}
```

### Effect Configurations

#### Electronic Gains

```json
{
  "gains": {
    "base_amplitude": 1.0,
    "amplitude_std": 0.02,
    "thermal_amplitude": 0.01,
    "thermal_timescale": 3600.0,
    "phase_drift_std": 1e-5
  }
}
```

**Parameters:**
- `base_amplitude`: Mean gain amplitude
- `amplitude_std`: Standard deviation of base amplitudes
- `thermal_amplitude`: Amplitude of thermal variations
- `thermal_timescale`: Thermal cycle period (seconds)
- `phase_drift_std`: Phase drift rate standard deviation (rad/sec)

#### Bandpass Response

```json
{
  "bandpass": {
    "delay_std": 1e-9,
    "jagged_amplitude": 0.05
  }
}
```

**Parameters:**
- `delay_std`: Cable delay standard deviation (seconds)
- `jagged_amplitude`: Amplitude of per-channel variations

#### Instrumental Leakage

```json
{
  "leakage": {
    "amplitude": 0.001
  }
}
```

**Parameters:**
- `amplitude`: Standard deviation of complex leakage terms

#### Parallactic Angle

```json
{
  "parallactic": {
    "rate_deg_per_hour": 15.0
  }
}
```

**Parameters:**
- `rate_deg_per_hour`: Parallactic angle rotation rate (degrees/hour)

### Sampling Configuration

```json
{
  "sampling": {
    "draws": 1000,
    "tune": 500,
    "chains": 2,
    "target_accept": 0.9
  }
}
```

## Command Line Interface

### Basic Usage

```bash
# Create default configuration
jones-sim --create-default-config config.json

# Run sampling
jones-sim --config config.json --output samples.nc

# Include interactive plotting
jones-sim --config config.json --output samples.nc --plot --plot-output dashboard.html
```

### Full Options

```bash
jones-sim [options]

Options:
  --config PATH              JSON configuration file
  --output PATH              Output file for samples (default: jones_samples.nc)
  --create-default-config PATH  Create default config at specified path
  --plot                     Generate interactive plots dashboard
  --plot-output PATH         Output file for dashboard (default: jones_dashboard.html)
  --log-level LEVEL          Logging verbosity (DEBUG/INFO/WARNING/ERROR)
```

## Error Handling

### Common Issues

**PyMC Sampling Failures:**
- Reduce `target_accept` to 0.8-0.85
- Increase `tune` samples
- Check for numerical instabilities in parameter ranges

**Memory Limitations:**
- Reduce grid dimensions (`n_times`, `n_frequencies`)
- Use fewer MCMC chains
- Process antennas separately

**Plotting Failures:**
- Jones matrix extraction may fail for complex parameter structures
- Individual effect plots should always work
- Check log output for specific error messages

### Logging

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via CLI:
```bash
jones-sim --config config.json --log-level DEBUG
```

## Performance Considerations

### GPU Acceleration

PyMC automatically detects and uses GPU acceleration when available. Install with:
```bash
pip install pymc[gpu]
```

### Memory Usage

Memory scales as `O(n_samples × n_antennas × n_times × n_frequencies)`. For large problems:

1. Use streaming analysis
2. Process antenna subsets
3. Reduce temporal/frequency resolution
4. Use thinned posterior samples

### Computational Scaling

MCMC convergence time depends on:
- Parameter dimensionality (number of effects)
- Correlation structure
- Prior specification
- Grid resolution

Typical scaling: 2-4 antennas, 20-50 time points, 32-128 frequency channels.