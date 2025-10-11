# jones_sim

[![CI](https://github.com/preshanth/jones_sim/workflows/CI/badge.svg)](https://github.com/preshanth/jones_sim/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Core Coverage](https://img.shields.io/badge/coverage%20(core)-97%25-brightgreen)](https://github.com/preshanth/jones_sim/actions)
<!-- [![Full Coverage](https://img.shields.io/badge/coverage%20(full)-45%25-orange)](https://github.com/preshanth/jones_sim/actions) -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Forward modeling of radio interferometric visibilities using Jones matrix formalism.

> **Coverage Note**: Core coverage (97%) measures simulation/effects modules tested on every commit. Full coverage (45%) includes MCMC/plotting code tested in slow integration tests.

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

# With CASA integration
pip install jones_sim[casa]

# Everything except CASA
pip install jones_sim[all]

# Development installation (includes all features + testing tools)
pip install -e .[dev]
```

## Features

### Core Capabilities
- **Symbolic equation generation** - SymPy-based Jones matrix algebra
- **Numerical simulation** - Forward modeling of Jones effects (gains, bandpass, leakage, etc.)
- **Visibility corruption** - Apply Jones matrices to ideal visibilities
- **Source models** - Multiple polarization scenarios (unpolarized, linear, circular, rotation measure)

### Optional Features
- **Interactive visualization** (requires `[plotting]`) - Bokeh-based dashboards for time/frequency effects
- **Monte Carlo analysis** (requires `[mcmc]`) - Bayesian parameter estimation with PyMC
- **CASA integration** (requires `[casa]`) - Read/write measurement sets and calibration tables

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

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide and local testing
- [CLAUDE.md](CLAUDE.md) - Project architecture and implementation details

## Testing

Run the full test suite locally:

```bash
./run_ci_locally.sh
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more testing options.