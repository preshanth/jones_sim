# Jones Simulator - Code Architecture

## Overview

Jones simulator is a Python package for simulating and calibrating radio interferometric observations using Jones matrix formalism. It provides forward modeling (corruption) and inverse modeling (calibration) capabilities.

## Directory Structure

```
jones_sim/
├── jones_sim/              # Main package
│   ├── effects.py          # Jones matrix effect implementations
│   ├── simulator.py        # Core simulator and corruption engine
│   ├── config.py           # JSON configuration system
│   ├── jax_config.py       # JAX/GPU configuration
│   ├── casa_interface.py   # CASA MS/caltable I/O
│   ├── calibration_solver.py  # NumPyro/JAX Bayesian solver
│   ├── antsol.py           # Fast antenna-based gain solver
│   ├── dbs_solver.py       # Delay Bayesian sampler
│   ├── plotting.py         # Basic plotting utilities
│   ├── plotting_enhanced.py # Enhanced validation plots
│   ├── source_models.py    # Source Stokes models
│   └── visibility_generator.py # Visibility synthesis
├── scripts/                # Validation and workflow scripts
│   ├── validate_delay_recovery.py
│   └── validate_bandpass_recovery.py
├── tests/                  # Pytest test suite
│   └── test_*.py
├── configs/                # JSON configuration files
│   └── test_*.json
└── docs/                   # Documentation
```

## Data Flow

### Forward Path (Simulation/Corruption)

```
Source Model → Visibility Generator → Jones Simulator → Corrupted Visibilities
     ↓                 ↓                    ↓                      ↓
  Stokes I,Q,U,V   Ideal Vis [2×2]    Apply Effects         Write to MS
```

**Steps:**
1. **Source Model** (`source_models.py`) - Define Stokes parameters
2. **Visibility Generator** (`visibility_generator.py`) - Synthesize ideal visibilities
3. **Jones Simulator** (`simulator.py`) - Apply corruption effects
4. **MS Writer** (`casa_interface.py`) - Write corrupted DATA column

### Inverse Path (Calibration)

```
MS DATA → CalibrationSolver → MCMC/MAP → Solutions → Comparison
   ↓            ↓                ↓           ↓           ↓
 Read      Build Model      Sample      Extract     Plot/Validate
```

**Steps:**
1. **MS Reader** (`casa_interface.py`) - Load DATA, MODEL_DATA
2. **Solver Setup** (`calibration_solver.py`) - Configure effects, priors
3. **MCMC/MAP** (NumPyro/JAX) - Sample posterior or optimize
4. **Solution Extraction** - Get calibration parameters
5. **Validation** (`plotting_enhanced.py`) - Compare with CASA/truth

## Core Components

### 1. Effects (`effects.py`)

Jones matrix effects implement the corruption model.

**Base Class Pattern:**
```python
class BaseEffect:
    def compute_jones(self, freq, time, ant) -> np.ndarray[2, 2]
        """Return 2×2 Jones matrix for given freq, time, antenna"""
```

**Available Effects:**
- `ParallacticAngle` - Parallactic rotation (time-dependent)
- `ElectronicGains` - Complex antenna gains (G)
- `InstrumentalLeakage` - Feed leakage (D-terms)
- `BandpassDelay` - Frequency-dependent delays (K) and amplitude (B)
- `RLDelayDifference` - Cross-hand delay
- `CrosshandPhase` - XY phase offset
- `RotationMeasure` - Faraday rotation

**Usage:**
```python
from jones_sim import BandpassDelay, ElectronicGains, JonesSimulator

# Create effects
delay_effect = BandpassDelay(tau_xx=delays, tau_yy=delays, ref_freq=1.5e9)
gain_effect = ElectronicGains(g_xx=gains[:, 0], g_yy=gains[:, 1])

# Add to simulator
sim = JonesSimulator()
sim.add_effect("delays", delay_effect)
sim.add_effect("gains", gain_effect)

# Corrupt visibilities
corrupted = sim.corrupt_visibilities(
    ideal_vis, frequencies, times, ant1, ant2
)
```

### 2. Simulator (`simulator.py`)

Coordinates application of multiple Jones effects in proper order.

**Key Methods:**
- `add_effect(name, effect)` - Register effect instance
- `compute_jones_matrix(freq, time, ant)` - Compute combined Jones matrix
- `corrupt_visibilities(...)` - Apply corruption (CPU or GPU)
- `predict_visibilities(...)` - Forward model for solving (JAX)

**Effect Chain Order:**
```python
effect_order = [
    "parallactic",     # P - Parallactic angle
    "leakage",         # D - Instrumental leakage
    "gains",           # G - Electronic gains
    "bandpass",        # B - Bandpass (delays + freq gains)
    "rotation_measure", # Ω - Faraday rotation
    "rl_delay",        # K_RL - Cross-hand delay
    "crosshand_phase", # X - XY phase
]
```

**Visibility Equation:**
```
V_corrupted = J₁ V_ideal J₂^H
```

Where:
- `J₁ = P₁ D₁ G₁ B₁ Ω₁ K₁ X₁` (antenna 1 Jones matrix)
- `J₂^H` = Hermitian transpose of antenna 2 Jones matrix

### 3. Configuration System (`config.py`)

JSON-based configuration for reproducible simulations.

**Key Classes:**
- `JonesConfig` - Load and parse JSON configs
- `DistributionSampler` - Sample from parameter distributions

**Config Structure:**
```json
{
  "jones_chain": {
    "order": ["gain", "bandpass"],
    "enabled_effects": ["gain"]
  },
  "effects": {
    "gain": {
      "type": "complex_electronic_gain",
      "amplitude": {"x_pol": {"distribution": "log_normal", ...}}
    }
  },
  "noise": {"enabled": true, ...},
  "processing": {"use_gpu": false, ...}
}
```

**Usage:**
```python
from jones_sim import JonesConfig

config = JonesConfig("configs/test_k_delays.json")
sim = config.create_simulator(n_antennas=27)
noise_params = config.get_noise_config()
```

### 4. Calibration Solver (`calibration_solver.py`)

Bayesian calibration using NumPyro/JAX.

**Key Features:**
- Sequential solving (K → G → B → D)
- MCMC (NUTS) or MAP optimization
- Flexible priors (Gaussian, uniform, etc.)
- CASA solution initialization
- Solution interval support (solint)

**Workflow:**
```python
from jones_sim.calibration_solver import CalibrationSolver

# Initialize with JAX config
solver = CalibrationSolver(
    "my.ms",
    max_cpu_fraction=0.5,  # Use 50% CPU cores if no GPU
    gpu_device=0            # Use GPU 0 if available
)

# Load data
solver.load_data(spw="0", solint="inf")

# Add effects
solver.add_effect("K", solint="inf", prior_bound_ns=1.0)
solver.add_effect("G", solint="inf", calmode="ap", prior_std=0.3)

# Load CASA solutions for priors
solver.load_casa_solutions(K="delays.K", G="gains.G")

# Build model
solver.build_model()

# Solve
solver.sample(draws=1000, tune=1000, chains=2)
# OR
solver.optimize(num_steps=1000)  # Faster MAP

# Extract solutions
k_solution = solver.get_solution("K")
g_solution = solver.get_solution("G")

# Print summary
solver.print_summary()
```

**Effect Types:**
- `K` - Delays (seconds per antenna)
- `G` - Complex gains (amplitude + phase)
- `B` - Bandpass (frequency-dependent gains)
- `D` - Leakage (D-terms)

### 5. CASA Interface (`casa_interface.py`)

I/O for CASA Measurement Sets and calibration tables.

**Key Classes:**
- `MeasurementSetHandler` - Read/write MS data
- `CalibrationTableHandler` - Read CASA caltables

**Usage:**
```python
from jones_sim.casa_interface import MeasurementSetHandler

# Read MS
ms = MeasurementSetHandler("my.ms")
summary = ms.get_observation_summary()
data = ms.read_visibilities(field=0, spw="0")

# Access data
vis = data["data"]           # (n_pol, n_chan, n_row)
model = data["model_data"]
frequencies = data["frequencies"]
antenna1 = data["antenna1"]
antenna2 = data["antenna2"]

ms.close()
```

### 6. Enhanced Plotting (`plotting_enhanced.py`)

Validation and diagnostic plots.

**Key Functions:**
- `plot_bandpass_comparison()` - Amp/phase vs frequency
- `plot_three_way_comparison()` - Truth vs CASA vs Ours
- `plot_leakage_dterms()` - D-term visualizations
- `plot_time_series_gains()` - Time-varying gains
- `plot_error_histogram()` - Error distributions
- `create_validation_dashboard()` - Comprehensive dashboard

**Usage:**
```python
from jones_sim.plotting_enhanced import plot_three_way_comparison

truth_delays = ...
casa_delays = ...
recovered_delays = ...

p1, p2 = plot_three_way_comparison(
    truth_delays,
    casa_delays,
    recovered_delays,
    x_label="Antenna",
    y_label="Delay (ns)",
    title="Delay Recovery",
    output_file_path="delays_comparison.html"
)
```

## GPU/CPU Management

### JAX Configuration (`jax_config.py`)

Handles GPU detection and CPU thread limiting for NumPyro/JAX.

**Problem Solved:**
JAX/NumPyro MCMC with `num_chains=N` can spawn N×CPU_cores threads, causing oversubscription.

**Solution:**
```python
from jones_sim import configure_jax

# Auto-configure: GPU if available, else 50% CPU cores
configure_jax(max_cpu_fraction=0.5, gpu_device=0)

# Then use solver
solver = CalibrationSolver("my.ms")  # Uses pre-configured JAX
```

**How it works:**
1. Check for GPUs via `jax.devices("gpu")`
2. If GPU found → `jax.config.update("jax_default_device", gpu[device_id])`
3. If no GPU → Set `XLA_FLAGS=--xla_force_host_platform_device_count=N`
   - Limits JAX to N virtual devices (e.g., 50% of cores)
   - Prevents oversubscription during MCMC

### CuPy (Legacy Path)

CuPy is used in:
- `corrupt_visibilities(use_gpu=True)` - Fast corruption on GPU
- `antsol.py` - GPU-accelerated gain solving

**Note:** CuPy path is maintained by collaborator Arpan. JAX is preferred for new code.

## Validation Scripts

### Purpose
End-to-end validation that compares our solver against CASA and ground truth.

### Script Structure
All validation scripts follow this pattern:

```python
def generate_ground_truth(...):
    """Generate known corruption parameters"""
    return truth_values

def corrupt_ms_with_effects(...):
    """Apply corruption to MS DATA column"""
    ...

def run_casa_solver(...):
    """Run CASA calibration (gaincal, bandpass, etc.)"""
    return casa_solutions

def run_our_solver(...):
    """Run CalibrationSolver"""
    return solver

def compare_results(...):
    """Compare truth vs CASA vs ours"""
    return metrics

def main():
    """Orchestrate validation and return exit code"""
    # 1. Simulate or load MS
    # 2. Generate ground truth
    # 3. Corrupt MS
    # 4. Run CASA
    # 5. Run our solver
    # 6. Compare and validate
    # 7. Return 0 (pass) or 1 (fail)
```

### Available Scripts
- `validate_delay_recovery.py` - K (delay) validation
- `validate_bandpass_recovery.py` - B (bandpass) validation

### Running Validations
```bash
# No noise, MAP optimization (fast)
python scripts/validate_bandpass_recovery.py --no_noise --map

# With noise, MCMC sampling (realistic)
python scripts/validate_bandpass_recovery.py --draws 1000 --tune 1000

# Check exit code
echo $?  # 0 = pass, 1 = fail
```

## Testing Framework

### Test Organization

**Markers:**
- `@pytest.mark.fast` - Unit tests (< 1 second)
- `@pytest.mark.slow` - Integration tests (creates MS files)
- `@pytest.mark.integration` - Requires CASA/MS
- `@pytest.mark.benchmark` - Performance tests

**Test Files:**
- `test_config.py` - Config system tests
- `test_calibration_validation.py` - Integration tests for validation scripts
- `test_effects.py` - Jones matrix effect tests
- `test_simulator.py` - Simulator tests
- ... (other component tests)

### Running Tests
```bash
# Fast unit tests only
pytest -m fast

# All tests
pytest

# Specific test file
pytest tests/test_calibration_validation.py -v

# With coverage
pytest --cov=jones_sim --cov-report=html

# Using test runner
./tests/run_validation_tests.sh all
./tests/run_validation_tests.sh fast
./tests/run_validation_tests.sh coverage
```

## Common Workflows

### 1. Simulate Corrupted MS
```python
from jones_sim import JonesConfig, JonesSimulator
from jones_sim.casa_interface import MeasurementSetHandler

# Create MS (using CASA simobserve or similar)
# ...

# Load config
config = JonesConfig("configs/test_k_g_combined.json")

# Get MS info
ms = MeasurementSetHandler("sim.ms")
summary = ms.get_observation_summary()
n_antennas = summary["n_antennas"]

# Create simulator with config
sim = config.create_simulator(n_antennas=n_antennas)

# Read MODEL_DATA
data = ms.read_visibilities(field=0, spw="0")
ideal_vis = data["model_data"]

# Corrupt
corrupted_vis = sim.corrupt_visibilities(
    ideal_vis,
    data["frequencies"],
    data["time"],
    data["antenna1"],
    data["antenna2"]
)

# Write back to DATA
# (use CASA table tool to write)
```

### 2. Calibrate and Compare
```python
from jones_sim.calibration_solver import CalibrationSolver

# Run CASA calibration first
# gaincal(vis="sim.ms", caltable="sim.K", ...)

# Run our solver
solver = CalibrationSolver("sim.ms")
solver.load_data(spw="0", solint="inf")
solver.add_effect("K", solint="inf")
solver.load_casa_solutions(K="sim.K")
solver.build_model()
solver.sample(draws=1000, tune=1000)

# Compare
k_solution = solver.get_solution("K")
# Compare k_solution with CASA and ground truth
```

### 3. Create Custom Effect
```python
import numpy as np

class MyCustomEffect:
    """Custom Jones matrix effect."""

    def __init__(self, my_param):
        self.my_param = my_param

    def compute_jones(self, freq, time, ant_id):
        """Compute 2×2 Jones matrix.

        Args:
            freq: Frequency in Hz
            time: Time in seconds
            ant_id: Antenna index

        Returns:
            jones: 2×2 complex array
        """
        # Your implementation here
        jones = np.eye(2, dtype=complex)
        # ... apply your effect ...
        return jones

# Use it
from jones_sim import JonesSimulator

effect = MyCustomEffect(my_param=1.0)
sim = JonesSimulator()
sim.add_effect("my_effect", effect)
```

## Performance Considerations

### CPU vs GPU

**CPU Path:**
- Default for `corrupt_visibilities(use_gpu=False)`
- Loop over visibilities
- Good for small datasets (< 100k vis)

**GPU Path (CuPy):**
- `corrupt_visibilities(use_gpu=True)`
- Vectorized on GPU
- 10-100× faster for large datasets
- Requires CuPy installation

**JAX Path:**
- Used in `predict_visibilities(use_jax=True)`
- For calibration/solving (needs autodiff)
- GPU or CPU with XLA compilation

### Memory Management

**Large MS files:**
- Use chunked processing in `corrupt_delay.py`
- Default chunk size: 100,000 rows
- Adjust via `chunk_size` parameter

**MCMC Memory:**
- Trace size ∝ draws × chains × n_parameters
- Use MAP for large problems
- Or reduce draws/chains

## Extending the Code

### Adding New Effect Type

1. **Create effect class in `effects.py`:**
```python
class MyNewEffect:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def compute_jones(self, freq, time, ant_id):
        # Implementation
        return jones_matrix
```

2. **Add to config system in `config.py`:**
```python
def _create_effect(self, effect_name, effect_config, n_antennas):
    # ... existing code ...
    elif effect_type == "my_new_effect":
        param1 = self._sample_distribution(effect_config.get("param1"))
        param2 = self._sample_distribution(effect_config.get("param2"))
        return MyNewEffect(param1=param1, param2=param2)
```

3. **Add to solver in `calibration_solver.py`:**
```python
# Define priors and model for new effect
```

4. **Add validation script:**
```python
# scripts/validate_myneweffect_recovery.py
```

5. **Add tests:**
```python
# tests/test_myneweffect.py
```

### Adding New Solver

Create in `jones_sim/my_new_solver.py`:
```python
class MyNewSolver:
    def __init__(self, ms_path):
        self.ms_path = ms_path

    def solve(self, ...):
        # Implementation
        return solutions
```

Register in `__init__.py` and add tests.

## Debugging

### Common Issues

**1. JAX GPU not found:**
```python
# Check GPU availability
import jax
print(jax.devices())  # Should show GPU devices

# If no GPU, JAX will auto-fallback to CPU
```

**2. CASA table errors:**
```python
# Ensure CASA tables are closed
ms_handler.close()

# Or use context manager (if implemented)
```

**3. Memory errors in MCMC:**
```python
# Reduce draws or chains
solver.sample(draws=100, tune=100, chains=1)

# Or use MAP
solver.optimize(num_steps=1000)
```

**4. Slow convergence:**
```python
# Increase tuning steps
solver.sample(draws=500, tune=2000, chains=2)

# Check priors are reasonable
# Check data quality (SNR, flagging)
```

### Logging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Development Workflow

### 1. Setup
```bash
git clone https://github.com/preshanth/jones_sim
cd jones_sim
pip install -e ".[dev]"  # Editable install with dev dependencies
```

### 2. Make Changes
```bash
# Create feature branch
git checkout -b feature/my-new-feature

# Make changes, add tests
# Run tests
pytest

# Run validation
python scripts/validate_bandpass_recovery.py --no_noise --map
```

### 3. Submit PR
```bash
git add .
git commit -m "Add my new feature"
git push origin feature/my-new-feature
# Create PR on GitHub
```

## API Stability

**Stable APIs** (won't change):
- `JonesSimulator.add_effect()`
- `JonesSimulator.corrupt_visibilities()`
- Effect `compute_jones()` signature
- `JonesConfig` interface

**Experimental APIs** (may change):
- `CalibrationSolver` internals
- GPU/JAX configuration details
- Specific prior distributions

## Dependencies

**Core:**
- numpy, scipy
- casatools, casatasks (CASA I/O)

**Calibration:**
- jax, jaxlib (autodiff, GPU)
- numpyro (Bayesian inference)

**Plotting:**
- bokeh (interactive plots)

**Optional:**
- cupy (GPU corruption)
- pytest, pytest-benchmark (testing)

See `pyproject.toml` for complete list.
