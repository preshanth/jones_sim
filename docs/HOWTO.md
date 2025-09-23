# Jones Matrix Forward Modeling: Setup and Usage Guide

## Introduction

This guide demonstrates how to set up and use the `jones_sim` package for Monte Carlo forward modeling of Jones matrix effects in radio interferometry. The package implements Bayesian parameter estimation for instrumental effects, providing uncertainty quantification and statistical analysis of Jones matrix parameters.

## Installation and Setup

### Package Installation

```bash
# Install from source
cd jones_sim
pip install -e .

# Verify installation
jones-sim --help
```

### Dependencies

The package requires:
- PyMC (Bayesian modeling)
- PyTensor (computational backend)
- ArviZ (posterior analysis)
- Bokeh (interactive visualization)
- NumPy, SciPy (numerical computing)

## Basic Workflow

### Step 1: Create Configuration

Generate a default configuration file:

```bash
jones-sim --create-default-config my_observation.json
```

This creates a JSON file with reasonable defaults for a 4-antenna, 2-hour observation covering 200 MHz bandwidth.

### Step 2: Customize Parameters

Edit the configuration file to match your observation parameters:

```json
{
  "grid": {
    "n_antennas": 8,
    "n_times": 24,
    "n_frequencies": 128,
    "time": {
      "start": 0.0,
      "end": 14400.0,
      "units": "seconds"
    },
    "frequency": {
      "start": 1.28e9,
      "end": 1.48e9,
      "units": "Hz"
    }
  }
}
```

### Step 3: Configure Effects

Enable and parameterize the instrumental effects relevant to your analysis:

```json
{
  "effects": {
    "gains": {
      "base_amplitude": 1.0,
      "amplitude_std": 0.03,
      "thermal_amplitude": 0.02,
      "thermal_timescale": 3600.0,
      "phase_drift_std": 2e-5
    },
    "bandpass": {
      "delay_std": 2e-9,
      "jagged_amplitude": 0.08
    },
    "leakage": {
      "amplitude": 0.002
    }
  }
}
```

### Step 4: Run Analysis

Execute the complete analysis pipeline:

```bash
jones-sim --config my_observation.json \
          --output results.nc \
          --plot \
          --plot-output dashboard.html \
          --log-level INFO
```

This command:
- Builds the Bayesian model
- Runs MCMC sampling
- Generates posterior analysis
- Creates interactive visualization dashboard
- Saves all results to files

## Configuration Guide

### Grid Parameters

The observation grid defines the parameter space dimensions:

**Temporal Grid:**
- `n_times`: Number of time samples (typically 20-50)
- `start`, `end`: Observation time range in seconds
- Time resolution should capture thermal variations (minutes to hours)

**Frequency Grid:**
- `n_frequencies`: Number of channels (typically 32-256)
- `start`, `end`: Frequency range in Hz
- Channel resolution should capture bandpass structure

**Array Configuration:**
- `n_antennas`: Number of antennas (2-64)
- Each antenna gets independent parameters for most effects

### Effect Configuration

#### Electronic Gains

Time-varying complex gains with thermal drift:

```json
{
  "gains": {
    "base_amplitude": 1.0,           // Nominal gain amplitude
    "amplitude_std": 0.02,           // 2% amplitude variation between antennas
    "thermal_amplitude": 0.01,       // 1% thermal amplitude modulation
    "thermal_timescale": 3600.0,     // 1-hour thermal cycle
    "phase_drift_std": 1e-5          // Phase drift rate (rad/sec)
  }
}
```

**Physical Interpretation:**
- `base_amplitude`: Reflects amplifier settings and cable losses
- `thermal_amplitude`: Temperature-dependent gain variations
- `thermal_timescale`: Environmental thermal cycles
- `phase_drift_std`: Electronic phase stability

#### Bandpass Response

Frequency-dependent amplitude and phase response:

```json
{
  "bandpass": {
    "delay_std": 1e-9,               // Cable delay standard deviation (ns)
    "jagged_amplitude": 0.05         // Per-channel amplitude variations
  }
}
```

**Physical Interpretation:**
- `delay_std`: Cable length differences between antennas
- `jagged_amplitude`: Frequency-dependent filter responses, standing waves

#### Instrumental Leakage

Cross-polarization coupling (static per antenna):

```json
{
  "leakage": {
    "amplitude": 0.001               // Leakage term standard deviation
  }
}
```

**Physical Interpretation:**
- `amplitude`: Feed misalignment, imperfect polarizers, receiver crosstalk

#### Parallactic Angle

Deterministic rotation due to celestial mechanics:

```json
{
  "parallactic": {
    "rate_deg_per_hour": 15.0        // Earth rotation rate
  }
}
```

### Sampling Configuration

MCMC sampling parameters affect convergence and computation time:

```json
{
  "sampling": {
    "draws": 1000,                   // Posterior samples per chain
    "tune": 500,                     // Tuning samples for step size
    "chains": 2,                     // Number of parallel chains
    "target_accept": 0.9             // Target acceptance rate
  }
}
```

**Tuning Guidelines:**
- `draws`: More samples improve statistics but increase computation time
- `tune`: Increase if convergence diagnostics show problems
- `chains`: Use 2-4 chains for convergence validation
- `target_accept`: 0.8-0.95 range; higher values for complex posteriors

## Analysis Workflows

### Basic Error Budget Analysis

1. Configure realistic parameter ranges for your instrument
2. Run sampling with moderate resolution (20 times, 64 channels)
3. Examine effect summaries to identify dominant error sources
4. Iterate with refined parameters

### Parameter Uncertainty Analysis

1. Configure effects with realistic parameter ranges
2. Sample joint parameter distributions using MCMC
3. Analyze correlations between different effects
4. Quantify uncertainty propagation through Jones matrices

### Effect Characterization

1. Model individual effects independently
2. Study time/frequency dependencies
3. Assess relative importance of different effects
4. Generate statistical summaries for each effect type

## Visualization and Results

### Dashboard Components

The interactive dashboard includes:

**Summary Panel:**
- Effect-by-effect statistical summaries
- Parameter estimates with uncertainties
- Physical interpretations and scaling

**Time Domain Plots:**
- Gain amplitude and phase evolution
- Thermal drift patterns
- Parallactic angle rotation
- Monte Carlo uncertainty envelopes

**Frequency Domain Plots:**
- Bandpass amplitude and phase response
- Cable delay signatures
- Per-channel variations
- Correlated frequency structure

**Jones Matrix Analysis:**
- Complex scatter plots of matrix elements
- Individual antenna characteristics
- Time/frequency point analysis

### Interpreting Results

**Gain Evolution:**
- Smooth thermal variations on hour timescales
- Random walk phase behavior
- Correlated XX/YY polarizations

**Bandpass Structure:**
- Linear phase slopes from cable delays
- Jagged amplitude variations from filter responses
- Antenna-to-antenna differences

**Leakage Patterns:**
- Small complex-valued cross-coupling terms
- Static values per antenna
- Magnitude typically 0.1-1% level

## Example Configurations

### Small Array Configuration

```json
{
  "grid": {
    "n_antennas": 8,
    "n_times": 24,
    "n_frequencies": 64,
    "time": {"start": 0.0, "end": 7200.0},
    "frequency": {"start": 1.4e9, "end": 1.6e9}
  },
  "effects": {
    "gains": {
      "base_amplitude": 1.0,
      "amplitude_std": 0.02,
      "thermal_amplitude": 0.01,
      "thermal_timescale": 3600.0,
      "phase_drift_std": 2e-5
    },
    "bandpass": {
      "delay_std": 1e-9,
      "jagged_amplitude": 0.05
    },
    "leakage": {
      "amplitude": 0.001
    }
  }
}
```

### Large Array Configuration

```json
{
  "grid": {
    "n_antennas": 32,
    "n_times": 48,
    "n_frequencies": 128,
    "time": {"start": 0.0, "end": 14400.0},
    "frequency": {"start": 1.0e9, "end": 2.0e9}
  },
  "effects": {
    "gains": {
      "base_amplitude": 1.0,
      "amplitude_std": 0.03,
      "thermal_amplitude": 0.02,
      "thermal_timescale": 1800.0,
      "phase_drift_std": 3e-5
    },
    "bandpass": {
      "delay_std": 2e-9,
      "jagged_amplitude": 0.08
    }
  }
}
```

## Performance Optimization

### Memory Management

For large parameter spaces:
- Process antenna subsets separately
- Reduce temporal/frequency resolution
- Use fewer Monte Carlo samples
- Enable GPU acceleration if available

### Convergence Diagnostics

Monitor MCMC convergence:
- R-hat statistics should be close to 1.0
- Effective sample size should be > 400
- Trace plots should show good mixing
- Increase tuning samples if needed

### Computational Scaling

Typical performance scaling:
- 4 antennas, 24 times, 64 frequencies: ~10 minutes
- 8 antennas, 36 times, 128 frequencies: ~45 minutes
- 16 antennas, 48 times, 256 frequencies: ~3 hours

GPU acceleration can provide 5-10x speedup for large problems.

## Troubleshooting

### Common Issues

**Sampling Failures:**
- Reduce target acceptance rate to 0.8-0.85
- Check parameter ranges for physical reasonableness
- Increase tuning samples
- Simplify model by removing effects

**Memory Errors:**
- Reduce grid dimensions
- Use fewer chains
- Process data in chunks

**Plotting Failures:**
- Jones matrix extraction may fail for complex models
- Individual effect plots should always work
- Check log files for specific error messages

### Getting Help

For implementation questions:
1. Check the API documentation
2. Examine test files for usage examples
3. Enable debug logging for detailed diagnostics
4. Validate configuration against schema

This framework provides tools for Jones matrix parameter analysis and uncertainty quantification in radio interferometry applications.