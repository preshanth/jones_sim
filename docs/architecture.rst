Architecture
============

System Design and Code Paths
-----------------------------

For detailed information about the Jones Simulator architecture, including:

- Complete code architecture and data flow
- Core components with usage examples
- Forward path (corruption) and inverse path (calibration)
- GPU/CPU management
- Validation script structure
- Common workflows
- Performance considerations
- How to extend the code

Please see `ARCHITECTURE.md <../ARCHITECTURE.md>`_ in the docs directory.

Quick Overview
--------------

The Jones Simulator follows a modular architecture:

Core Components
~~~~~~~~~~~~~~~

1. **Simulator** (``jones_sim/simulator.py``)
   - Main Jones matrix corruption engine
   - Chains multiple effects together
   - Applies J1 * V * J2^H corruption

2. **Calibration Solver** (``jones_sim/calibration_solver.py``)
   - Bayesian calibration using JAX/NumPyro
   - Supports MAP and MCMC methods
   - Handles K, G, B, D effect types

3. **Configuration** (``jones_sim/config.py``)
   - JSON-based parameter system
   - Distribution sampling for stochastic effects
   - Validation thresholds

4. **MS Interface** (``jones_sim/ms_interface.py``)
   - CASA Measurement Set I/O
   - Converts between MS and internal formats

5. **CASA Interface** (``jones_sim/casa_interface.py``)
   - Runs CASA calibration tasks
   - Imports/exports caltables
   - Comparison utilities

Data Flow
~~~~~~~~~

Forward Path (Simulation)::

    Config → Simulator → Effects → Corrupted Visibilities → MS

Inverse Path (Calibration)::

    MS → CalibrationSolver → PyMC Model → JAX Optimization → Solutions → Caltable

Validation::

    Ground Truth → Corrupt MS → [CASA, Our Solver] → Compare → Pass/Fail

See the full `ARCHITECTURE.md <../ARCHITECTURE.md>`_ document for complete details.
