#!/usr/bin/env python3
"""End-to-end validation of Faraday rotation / rotation measure (F) recovery.

Ionospheric Faraday rotation causes frequency-dependent polarization angle rotation
proportional to λ². The rotation measure (RM) quantifies this effect.

Rotation angle: χ = RM × λ² (radians)
where λ = c/ν is wavelength

This script:
1. Creates simulated full-polarization MS (4 correlations)
2. Generates ground truth rotation measures (RM) per antenna
3. Corrupts DATA with Faraday rotation (frequency-dependent)
4. Runs RM synthesis or solver to recover RM
5. Compares: truth vs recovered RM
6. Returns exit code 0 if validation passes

Note: CASA doesn't have built-in RM calibration. This validates our solver's
ability to recover ionospheric RM from multi-frequency polarization data.

Usage:
    python validate_faraday_recovery.py [options]

Options:
    --msname NAME       MS name (default: sim_faraday_test.ms)
    --skip_sim          Skip simulation if MS exists
    --rm_range R        RM range in rad/m² (default: 2.0)
    --n_channels N      Number of channels (default: 64, need many for RM)
    --seed N            Random seed (default: 60)
    --no_noise          Skip thermal noise
    --map               Use MAP instead of MCMC
"""

import argparse
import os
import sys

import numpy as np
from casatools import table

from jones_sim import JonesSimulator
from jones_sim.calibration_solver import CalibrationSolver
from jones_sim.casa_interface import MeasurementSetHandler
from jones_sim.effects import RotationMeasure
from jones_sim.plotting_enhanced import plot_three_way_comparison
from scripts.simulate_3c286 import simulate_3c286


def generate_ground_truth_rm(
    n_antennas: int,
    rm_range: float = 2.0,
    seed: int = 60,
):
    """Generate ground truth rotation measures.

    Ionospheric RM varies spatially and temporally. Different antennas can see
    different RM due to ionospheric gradients.

    Typical values:
    - Daytime, mid-latitudes: 0-5 rad/m²
    - Nighttime: < 1 rad/m²
    - High solar activity: up to 10+ rad/m²

    Args:
        n_antennas: Number of antennas
        rm_range: RM range (±) in rad/m²
        seed: Random seed

    Returns:
        rm: Rotation measures [n_ant] in rad/m²
    """
    np.random.seed(seed)

    # Generate RMs with some spatial structure
    # (could model ionospheric gradients more realistically)
    rm = np.random.uniform(-rm_range, rm_range, n_antennas)

    # Reference antenna has RM=0 (or mean RM)
    rm[0] = 0.0

    return rm


def corrupt_ms_with_faraday(ms_path, rm_values, add_noise=True, sefd=420.0):
    """Corrupt MS DATA with Faraday rotation using JonesSimulator.

    Args:
        ms_path: Path to MS
        rm_values: Rotation measures [n_ant] in rad/m²
        add_noise: Whether to add thermal noise
        sefd: System equivalent flux density (Jy)
    """
    ms_handler = MeasurementSetHandler(ms_path)

    # Read data
    tb = table()
    tb.open(ms_path, nomodify=False)

    model_data = tb.getcol("MODEL_DATA")  # Clean data
    antenna1 = tb.getcol("ANTENNA1")
    antenna2 = tb.getcol("ANTENNA2")

    n_corr, n_chan, n_row = model_data.shape

    # Check full polarization
    if n_corr != 4:
        raise ValueError(
            f"MS must have 4 polarizations for Faraday rotation, got {n_corr}"
        )

    # Get frequency info
    summary = ms_handler.get_observation_summary()
    spw_info = summary["frequency_info"][0]
    freqs = spw_info["chan_freqs"]
    chan_width = spw_info.get(
        "chan_width", freqs[1] - freqs[0] if len(freqs) > 1 else 1e6
    )

    # Create simulator with Faraday rotation
    sim = JonesSimulator()
    faraday_effect = RotationMeasure(rotation_measure=rm_values)
    sim.add_effect("faraday", faraday_effect)

    # Reshape for simulator: (n_vis, 4)
    ideal_vis = model_data.transpose(2, 1, 0).reshape(-1, n_corr)  # (n_row*n_chan, 4)
    frequencies = np.tile(freqs, n_row)
    times = np.zeros(n_row * n_chan)
    ant1_arr = np.repeat(antenna1, n_chan)
    ant2_arr = np.repeat(antenna2, n_chan)

    # Corrupt using simulator
    print("Corrupting visibilities with Faraday rotation...", end=" ", flush=True)
    corrupted_vis = sim.corrupt_visibilities(
        ideal_vis, frequencies, times, ant1_arr, ant2_arr, use_gpu=False
    )
    print("Done")

    # Reshape back to (n_corr, n_chan, n_row)
    corrupted_data = corrupted_vis.reshape(n_row, n_chan, n_corr).transpose(2, 1, 0)

    # Add thermal noise if requested
    if add_noise:
        # Get integration time
        times_col = tb.getcol("TIME")
        unique_times = np.unique(times_col)
        if len(unique_times) > 1:
            int_time = np.median(np.diff(unique_times))
        else:
            int_time = 2.0

        # Radiometer equation
        sigma = sefd / np.sqrt(2 * chan_width * int_time)

        # Add complex Gaussian noise
        sigma_complex = sigma / np.sqrt(2)
        noise = np.random.normal(
            0, sigma_complex, corrupted_data.shape
        ) + 1j * np.random.normal(0, sigma_complex, corrupted_data.shape)
        corrupted_data = corrupted_data + noise
        print(f"  Added thermal noise (sigma={sigma:.4f} Jy)")

    # Write back to MS
    tb.putcol("DATA", corrupted_data)
    tb.close()

    print(f"✓ Applied Faraday rotation (noise={add_noise})")


def run_solver(ms_path, rm_initial, use_map=True, draws=500, tune=500):
    """Run CalibrationSolver to recover RM.

    Args:
        ms_path: Path to MS
        rm_initial: Initial RM guess [n_ant]
        use_map: Use MAP instead of MCMC
        draws: Number of MCMC samples
        tune: Number of tuning samples

    Returns:
        CalibrationSolver instance
    """
    print(f"Running CalibrationSolver (method={'MAP' if use_map else 'MCMC'})...")
    solver = CalibrationSolver(ms_path)
    solver.load_data()

    # Configure F effect
    solver.configure_effect(
        effect_name="F",
        effect_type="faraday",
        casa_caltable=None,  # No CASA RM calibration
    )

    # Solve
    if use_map:
        solver.solve(method="map", max_iter=1000, tol=1e-6)
    else:
        solver.solve(method="mcmc", num_warmup=tune, num_samples=draws)

    print("✓ Solver complete")
    return solver


def compare_rm_results(
    truth_rm,
    solver,
    n_antennas,
    output_prefix="faraday_validation",
    rm_threshold=0.1,
):
    """Compare ground truth vs recovered RM.

    Args:
        truth_rm: Ground truth RM [n_ant] in rad/m²
        solver: CalibrationSolver instance
        n_antennas: Number of antennas
        output_prefix: Prefix for output plots
        rm_threshold: RM RMS threshold (rad/m²)

    Returns:
        0 if passed, 1 if failed
    """
    # Get solver solutions
    our_rm = solver.get_solutions()["F"]["rm"]  # [n_ant]

    # Calculate residuals (excluding reference antenna)
    rm_res = our_rm[1:] - truth_rm[1:]
    rms_rm = np.sqrt(np.mean(rm_res**2))

    print("\n" + "=" * 60)
    print("FARADAY ROTATION (RM) VALIDATION RESULTS")
    print("=" * 60)
    print(f"Truth RM range: {np.min(truth_rm):.3f} to {np.max(truth_rm):.3f} rad/m²")
    print(f"Recovered RM range: {np.min(our_rm):.3f} to {np.max(our_rm):.3f} rad/m²")
    print(f"\nRM RMS residual: {rms_rm:.4f} rad/m²")
    print(f"Threshold: {rm_threshold:.2f} rad/m²")

    # Print antenna-by-antenna comparison
    print("\nPer-antenna RM (rad/m²):")
    print(f"{'Ant':>4} {'Truth':>8} {'Recovered':>10} {'Residual':>10}")
    print("-" * 36)
    for ant in range(min(10, n_antennas)):  # Show first 10
        res = our_rm[ant] - truth_rm[ant] if ant > 0 else 0.0
        print(f"{ant:4d} {truth_rm[ant]:8.3f} {our_rm[ant]:10.3f} {res:10.4f}")
    if n_antennas > 10:
        print(f"... ({n_antennas - 10} more antennas)")

    # Validation
    passed = rms_rm < rm_threshold

    if passed:
        print(f"\n✓ PASS: RM RMS {rms_rm:.4f} < {rm_threshold:.2f} rad/m²")
    else:
        print(f"\n✗ FAIL: RM RMS {rms_rm:.4f} >= {rm_threshold:.2f} rad/m²")

    # Generate plots
    print("\nGenerating validation plots...")
    ant_indices = np.arange(n_antennas)

    plot_three_way_comparison(
        truth_rm,
        truth_rm,  # No CASA RM calibration
        our_rm,
        x_values=ant_indices,
        title="Rotation Measure vs Antenna",
        xlabel="Antenna",
        ylabel="RM (rad/m²)",
        output_file_path=f"{output_prefix}_rm_comparison.html",
    )

    print("=" * 60)
    if passed:
        print("✓ FARADAY (RM) VALIDATION PASSED")
        return 0
    else:
        print("✗ FARADAY (RM) VALIDATION FAILED")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate Faraday rotation / RM recovery"
    )
    parser.add_argument("--msname", default="sim_faraday_test.ms", help="MS name")
    parser.add_argument("--skip_sim", action="store_true", help="Skip simulation")
    parser.add_argument("--rm_range", type=float, default=2.0, help="RM range (rad/m²)")
    parser.add_argument("--n_channels", type=int, default=64, help="Number of channels")
    parser.add_argument("--seed", type=int, default=60, help="Random seed")
    parser.add_argument("--no_noise", action="store_true", help="Skip thermal noise")
    parser.add_argument("--map", action="store_true", help="Use MAP instead of MCMC")
    args = parser.parse_args()

    # Create simulated MS if needed (full polarization, many channels for RM)
    if not args.skip_sim or not os.path.exists(args.msname):
        print(
            f"Creating simulated 3C286 MS with full polarization ({args.n_channels} channels for RM)..."
        )
        simulate_3c286(
            msname=args.msname,
            n_channels=args.n_channels,
            obs_time_min=5.0,
            int_time_sec=2.0,
        )
    else:
        print(f"Using existing MS: {args.msname}")

    # Load MS
    ms_handler = MeasurementSetHandler(args.msname)
    n_antennas = len(ms_handler.get_antenna_names())
    data_dict = ms_handler.read_data()
    n_channels_actual = len(data_dict["frequencies"])

    if n_channels_actual < 16:
        print(f"⚠ Warning: Only {n_channels_actual} channels - RM recovery may be poor")
        print("  Recommend >= 32 channels for reliable RM measurement")

    # Generate ground truth RM
    truth_rm = generate_ground_truth_rm(
        n_antennas=n_antennas,
        rm_range=args.rm_range,
        seed=args.seed,
    )

    print("Generated rotation measures:")
    print(f"  RM range: ±{args.rm_range:.2f} rad/m²")
    print(f"  RM RMS: {np.sqrt(np.mean(truth_rm**2)):.3f} rad/m²")
    print(f"  Channels: {n_channels_actual} (λ² leverage for RM)")

    # Corrupt MS with Faraday rotation
    corrupt_ms_with_faraday(
        args.msname,
        truth_rm,
        add_noise=not args.no_noise,
    )

    # Run solver
    rm_initial = np.zeros(n_antennas)  # Start from zero RM
    solver = run_solver(args.msname, rm_initial, use_map=args.map)

    # Compare and validate
    exit_code = compare_rm_results(
        truth_rm,
        solver,
        n_antennas,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
