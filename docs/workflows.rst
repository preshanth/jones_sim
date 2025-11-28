Common Workflows
================

Simulation Workflow
-------------------

Basic simulation with delays and gains::

    from jones_sim import JonesConfig

    # Load config
    config = JonesConfig("configs/test_k_g_combined.json")

    # Create simulator
    sim = config.create_simulator(n_antennas=27)

    # Simulate visibilities
    vis_corrupted = sim.corrupt_visibilities(vis_true, times, freqs, antennas1, antennas2)

Calibration Workflow
--------------------

MAP calibration (fast)::

    from jones_sim import CalibrationSolver

    # Initialize solver
    solver = CalibrationSolver("observation.ms")

    # Solve for delays (K)
    solver.solve_k(method="map")

    # Solve for gains (G)
    solver.solve_g(method="map")

    # Export to CASA caltable
    solver.export_to_caltable("solution.cal", caltype="K")

MCMC calibration (full posterior)::

    # Run MCMC sampling
    solver.solve_k(
        method="mcmc",
        num_warmup=500,
        num_samples=1000
    )

    # Access posterior samples
    samples = solver.get_samples()

Validation Workflow
-------------------

End-to-end validation with CASA comparison::

    # Run validation script
    python scripts/validate_delay_recovery.py --msname test.ms --no_noise --map

    # Check results
    # Exit code 0 = PASSED
    # Exit code 1 = FAILED

The validation workflow:

1. Generates ground truth effects
2. Corrupts MS with known effects
3. Runs CASA solver
4. Runs our Bayesian solver
5. Compares Truth vs CASA vs Ours
6. Validates against thresholds
7. Generates diagnostic plots

CASA Comparison Workflow
-------------------------

Compare with CASA calibration::

    from jones_sim import CalibrationSolver
    from jones_sim.ms_calibration import compare_to_casa_caltable

    # Run our solver
    solver = CalibrationSolver("observation.ms")
    solver.solve_k(method="map")

    # Compare with CASA K table
    comparison = compare_to_casa_caltable(
        solver,
        "casa_k_solution.cal",
        caltype="K"
    )

    # Check residuals
    print(f"RMS difference: {comparison['rms']}")

Custom Effect Workflow
----------------------

Add a new Jones effect::

    from jones_sim.effects.base import JonesEffect
    import jax.numpy as jnp

    class CustomEffect(JonesEffect):
        def __init__(self, params):
            super().__init__(params)

        def compute_jones(self, times, freqs, ant1, ant2):
            # Return shape: (n_vis, n_freq, 2, 2) for Jones matrices
            # or (n_antennas, n_time, n_freq, 2, 2) for antenna-based
            pass

    # Register in config system
    # Add to jones_sim/config.py effect factory

Plotting Workflow
-----------------

Generate validation dashboard::

    from jones_sim.plotting_enhanced import create_validation_dashboard

    # Create comprehensive dashboard
    create_validation_dashboard(
        "validation_results.pkl",
        output_dir="plots/"
    )

Custom comparison plot::

    from jones_sim.plotting_enhanced import plot_three_way_comparison

    plot_three_way_comparison(
        truth_values,
        casa_values,
        our_values,
        title="Delay Comparison",
        output_file_path="delay_comparison.html"
    )
