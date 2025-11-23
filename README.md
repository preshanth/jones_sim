# jones_sim

[![CI](https://github.com/preshanth/jones_sim/workflows/CI/badge.svg)](https://github.com/preshanth/jones_sim/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Forward modeling of radio interferometric visibilities using Jones matrix formalism.

## Installation

### Basic Installation

Core functionality (symbolic math, numerical simulation):

```bash
pip install jones_sim
```

### Installation with Optional Features

```bash
# With interactive plotting (Bokeh)
pip install jones_sim[plotting]

# With Monte Carlo sampling (PyMC)
pip install jones_sim[mcmc]

# With Simulation-Based Inference (SBI)
pip install jones_sim[sbi]

# With CASA integration
pip install jones_sim[casa]

# Everything except CASA
pip install jones_sim[all]

# Development installation (includes all features + testing tools)
pip install -e .[dev]
```

## Features

- **Symbolic Jones matrices** - Generate Jones matrix equations using SymPy
- **Numerical simulation** - Compute Jones effects (gains, bandpass, leakage, parallactic angle, etc.)
- **Visibility corruption** - Apply Jones matrices to corrupt ideal visibilities
- **Source models** - Unpolarized, linear, circular, and rotation measure sources
- **Interactive plots** (optional) - Bokeh visualizations
- **MCMC sampling** (optional) - PyMC/NumPyro-based parameter estimation
- **Simulation-Based Inference** (optional) - Neural posterior estimation for fast calibration with uncertainties
- **CASA interface** (optional) - Read/write measurement sets and calibration tables

## Quick Start

```python
import numpy as np
from jones_sim.effects import ElectronicGains, ParallacticAngle
from jones_sim.simulator import JonesSimulator

# Create simulator
sim = JonesSimulator()

# Add effects
sim.add_effect('parallactic', ParallacticAngle(latitude=34.0))
sim.add_effect('gains', ElectronicGains(n_antennas=10))

# Compute Jones matrix
J = sim.compute_jones_matrix(freq=1.4e9, time=0.0, antenna_id=0)
```

## SBI Quick Start (NEW!)

Solve for bandpass calibration with **uncertainty quantification**:

```python
from jones_sim.sbi_solver import BandpassSBISimulator, SBICalibrationSolver
from jones_sim.solvable_effects import BandpassEffect

# Setup simulator
simulator = BandpassSBISimulator(
    effect=BandpassEffect(),
    visibility_model=your_vis_model,
    n_antennas=4,
    n_channels=16,
)

# Train neural network (once)
solver = SBICalibrationSolver(simulator, n_rounds=2)
solver.train(n_simulations=10000)

# Infer with uncertainties (fast!)
samples, summary = solver.infer(observed_visibilities)
print(f"Bandpass gain = {summary['mean'][0]:.3f} ± {summary['std'][0]:.3f}")
# Output: Bandpass gain = 1.210 ± 0.052
```

See [docs/SBI_GUIDE.md](docs/SBI_GUIDE.md) for complete documentation.

## Documentation

- [docs/SBI_GUIDE.md](docs/SBI_GUIDE.md) - Simulation-Based Inference guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide and local testing
- [docs/](docs/) - API documentation, testing strategy, and algorithm details

## Testing

Run the full test suite locally:

```bash
./run_ci_locally.sh
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more testing options.