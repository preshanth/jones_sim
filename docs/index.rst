Jones Simulator Documentation
==============================

A Bayesian calibration framework for radio interferometry using Jones matrix formalism.

Overview
--------

Jones Simulator provides tools for:

- Simulating radio interferometry observations with realistic instrumental effects
- Bayesian calibration using JAX/NumPyro/BlackJAX
- Comparison with CASA calibration solutions
- Validation and testing of calibration algorithms

Quick Start
-----------

Installation::

    pip install -e .

Basic usage::

    from jones_sim import JonesConfig, CalibrationSolver

    # Load configuration
    config = JonesConfig("configs/test_k_delays.json")

    # Create simulator
    sim = config.create_simulator(n_antennas=27)

    # Run calibration
    solver = CalibrationSolver("observation.ms")
    solver.solve_k(method="map")

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   architecture
   workflows
   configuration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/effects
   api/calibration
   api/utilities

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   contributing_docs

Core Components
---------------

Main Modules
~~~~~~~~~~~~

:mod:`jones_sim.simulator`
    Core Jones matrix simulator and effect chain

:mod:`jones_sim.calibration_solver`
    Bayesian calibration solver using JAX/NumPyro

:mod:`jones_sim.config`
    JSON-based configuration system

:mod:`jones_sim.ms_interface`
    Measurement Set I/O interface

Effect Types
~~~~~~~~~~~~

- **K effects**: Antenna-based delays
- **G effects**: Time-varying complex gains
- **B effects**: Bandpass (frequency-dependent gains)
- **D effects**: Polarization leakage
- **P effects**: Parallactic angle rotation

Calibration Methods
~~~~~~~~~~~~~~~~~~~

- **MAP**: Maximum A Posteriori (fast optimization)
- **MCMC**: Full posterior sampling (NumPyro/BlackJAX)
- **CASA**: Interface to CASA calibration for comparison

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
