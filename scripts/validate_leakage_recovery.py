#!/usr/bin/env python3
"""End-to-end validation of polarization leakage (D-term) recovery pipeline.

This script:
1. Creates simulated full-polarization MS (4 correlations: XX, XY, YX, YY)
2. Generates known ground truth D-terms (leakage)
3. Corrupts DATA with leakage effects
4. Runs CASA polcal() to get D-table
5. Runs our CalibrationSolver with D effect
6. Compares: truth vs CASA vs ours
7. Generates validation plots
8. Returns exit code 0 if validation passes

Usage:
    python validate_leakage_recovery.py [options]

Options:
    --msname NAME       MS name (default: sim_leakage_test.ms)
    --skip_sim          Skip simulation if MS exists
    --leakage_level L   Leakage level (default: 0.05)
    --seed N            Random seed (default: 45)
    --no_noise          Skip thermal noise (for exact recovery test)
    --map               Use MAP instead of MCMC
"""

import argparse
import os
import sys

import numpy as np

from casatasks import polcal
from casatools import table

from jones_sim import JonesSimulator, JonesConfig
from jones_sim.effects import InstrumentalLeakage
from jones_sim.calibration_solver import CalibrationSolver
from jones_sim.casa_interface import MeasurementSetHandler
from jones_sim.plotting_enhanced import plot_three_way_comparison
from scripts.simulate_3c286 import simulate_3c286


def generate_ground_truth_dterms(
    n_antennas: int,
    leakage_level: float = 0.05,
    seed: int = 45,
):
    """Generate known ground truth D-terms (polarization leakage).

    D-terms represent instrumental polarization leakage between feeds:
    - d_xy: Leakage from Y to X
    - d_yx: Leakage from X to Y

    Typical values: |d| ~ 0.01 to 0.1

    Args:
        n_antennas: Number of antennas
        leakage_level: RMS leakage level (typical: 0.01-0.1)
        seed: Random seed

    Returns:
        d_terms: Complex D-terms [n_ant, 2] where [:, 0] = d_xy, [:, 1] = d_yx
    """
    np.random.seed(seed)

    # D-terms are complex with magnitude ~ leakage_level
    # Generate as Gaussian in real and imaginary parts
    d_terms = np.zeros((n_antennas, 2), dtype=complex)

    for ant in range(n_antennas):
        for pol in range(2):  # d_xy, d_yx
            real_part = np.random.normal(0, leakage_level)
            imag_part = np.random.normal(0, leakage_level)
            d_terms[ant, pol] = real_part + 1j * imag_part

    # Reference antenna typically has zero leakage
    d_terms[0, :] = 0.0 + 0j

    return d_terms


def corrupt_ms_with_leakage(ms_path, d_terms, add_noise=True, sefd=420.0):
    """Corrupt MS DATA column with polarization leakage effects using JonesSimulator.

    Args:
        ms_path: Path to MS
        d_terms: Complex D-terms [n_ant, 2] where [:, 0] = d_hv, [:, 1] = d_vh
        add_noise: Whether to add thermal noise
        sefd: System equivalent flux density (Jy) for noise
    """
    ms_handler = MeasurementSetHandler(ms_path)

    # Read data
    tb = table()
    tb.open(ms_path, nomodify=False)

    model_data = tb.getcol("MODEL_DATA")  # Clean data
    antenna1 = tb.getcol("ANTENNA1")
    antenna2 = tb.getcol("ANTENNA2")

    n_corr, n_chan, n_row = model_data.shape

    # Check that MS has full polarization (4 correlations)
    if n_corr != 4:
        raise ValueError(
            f"MS must have 4 polarizations for leakage calibration, got {n_corr}"
        )

    # Get frequency info
    summary = ms_handler.get_observation_summary()
    spw_info = summary["frequency_info"][0]
    freqs = spw_info["chan_freqs"]
    chan_width = spw_info.get("chan_width", freqs[1] - freqs[0] if len(freqs) > 1 else 1e6)

    # Create simulator with leakage
    sim = JonesSimulator()
    leakage_effect = InstrumentalLeakage(
        d_hv=d_terms[:, 0],  # H->V leakage (d_xy)
        d_vh=d_terms[:, 1],  # V->H leakage (d_yx)
        theta=0.0,  # No misalignment
    )
    sim.add_effect("leakage", leakage_effect)

    # Reshape for simulator: (n_vis, 4)
    ideal_vis = model_data.transpose(2, 1, 0).reshape(-1, n_corr)  # (n_row*n_chan, 4)
    frequencies = np.tile(freqs, n_row)
    times = np.zeros(n_row * n_chan)
    ant1_arr = np.repeat(antenna1, n_chan)
    ant2_arr = np.repeat(antenna2, n_chan)

    # Corrupt using simulator
    print("Corrupting visibilities with leakage...", end=" ", flush=True)
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
        noise = np.random.normal(0, sigma_complex, corrupted_data.shape) + \
                1j * np.random.normal(0, sigma_complex, corrupted_data.shape)
        corrupted_data = corrupted_data + noise
        print(f"  Added thermal noise (sigma={sigma:.4f} Jy)")

    # Write back to MS
    tb.putcol("DATA", corrupted_data)
    tb.close()

    print(f"✓ Corrupted MS with leakage (noise={add_noise})")


def run_casa_polcal(ms_path, dtable, refant="0"):
    """Run CASA polcal() to calibrate D-terms.

    Args:
        ms_path: Path to MS
        dtable: Output caltable path
        refant: Reference antenna

    Returns:
        Success boolean
    """
    try:
        print(f"Running CASA polcal()...")
        polcal(
            vis=ms_path,
            caltable=dtable,
            refant=refant,
            poltype="D",  # D-term calibration
        )
        print(f"✓ CASA polcal complete: {dtable}")
        return True
    except Exception as e:
        print(f"✗ CASA polcal failed: {e}")
        return False


def run_solver(ms_path, dtable, use_map=True, draws=500, tune=500):
    """Run our CalibrationSolver with D effect.

    Args:
        ms_path: Path to MS
        dtable: CASA caltable for comparison
        use_map: Use MAP instead of MCMC
        draws: Number of MCMC samples
        tune: Number of tuning samples

    Returns:
        CalibrationSolver instance
    """
    print(f"Running CalibrationSolver (method={'MAP' if use_map else 'MCMC'})...")
    solver = CalibrationSolver(ms_path)
    solver.load_data()

    # Configure D effect
    solver.configure_effect(
        effect_name="D",
        effect_type="leakage",
        casa_caltable=dtable,
    )

    # Solve
    if use_map:
        solver.solve(method="map", max_iter=1000, tol=1e-6)
    else:
        solver.solve(method="mcmc", num_warmup=tune, num_samples=draws)

    print(f"✓ Solver complete")
    return solver


def compare_dterm_results(
    truth_dterms,
    casa_dtable,
    solver,
    n_antennas,
    output_prefix="leakage_validation",
    magnitude_threshold=0.01,
    phase_threshold_deg=5.0,
):
    """Compare ground truth vs CASA vs our solver.

    Args:
        truth_dterms: Ground truth D-terms [n_ant, 2]
        casa_dtable: Path to CASA D-table
        solver: Our CalibrationSolver
        n_antennas: Number of antennas
        output_prefix: Prefix for output plots
        magnitude_threshold: Magnitude RMS threshold
        phase_threshold_deg: Phase RMS threshold (degrees)

    Returns:
        0 if passed, 1 if failed
    """
    # Read CASA solutions
    tb = table()
    tb.open(casa_dtable)
    casa_dterms_raw = tb.getcol("CPARAM")  # [n_pol, n_chan, n_rows]
    casa_ant = tb.getcol("ANTENNA1")
    tb.close()

    # Reshape CASA D-terms to [n_ant, 2]
    casa_dterms = np.zeros((n_antennas, 2), dtype=complex)
    for ant in range(n_antennas):
        mask = casa_ant == ant
        if np.any(mask):
            # CASA stores as [pol, chan, ant]
            casa_dterms[ant, 0] = casa_dterms_raw[0, 0, mask][0]  # d_xy
            casa_dterms[ant, 1] = casa_dterms_raw[1, 0, mask][0]  # d_yx

    # Get our solver solutions
    our_dterms = solver.get_solutions()["D"]["dterms"]  # [n_ant, 2]

    # Calculate magnitudes and phases
    truth_mag = np.abs(truth_dterms)
    casa_mag = np.abs(casa_dterms)
    our_mag = np.abs(our_dterms)

    truth_phase = np.angle(truth_dterms, deg=True)
    casa_phase = np.angle(casa_dterms, deg=True)
    our_phase = np.angle(our_dterms, deg=True)

    # Calculate residuals (excluding reference antenna)
    casa_mag_res = casa_mag[1:] - truth_mag[1:]
    our_mag_res = our_mag[1:] - truth_mag[1:]

    # Phase residuals
    casa_phase_res = (casa_phase[1:] - truth_phase[1:] + 180) % 360 - 180
    our_phase_res = (our_phase[1:] - truth_phase[1:] + 180) % 360 - 180

    # RMS metrics
    casa_mag_rms = np.sqrt(np.mean(casa_mag_res**2))
    our_mag_rms = np.sqrt(np.mean(our_mag_res**2))
    casa_phase_rms = np.sqrt(np.mean(casa_phase_res**2))
    our_phase_rms = np.sqrt(np.mean(our_phase_res**2))

    # Print comparison
    print("\n" + "="*60)
    print("LEAKAGE (D-TERM) RECOVERY VALIDATION RESULTS")
    print("="*60)
    print(f"Magnitude RMS (Truth vs CASA): {casa_mag_rms:.5f}")
    print(f"Magnitude RMS (Truth vs Ours): {our_mag_rms:.5f}")
    print(f"Phase RMS (Truth vs CASA):     {casa_phase_rms:.2f}°")
    print(f"Phase RMS (Truth vs Ours):     {our_phase_rms:.2f}°")
    print(f"\nThresholds: mag={magnitude_threshold}, phase={phase_threshold_deg}°")

    # Validation checks
    passed = True
    if our_mag_rms > magnitude_threshold:
        print(f"✗ FAIL: Magnitude RMS {our_mag_rms:.5f} > {magnitude_threshold}")
        passed = False
    else:
        print(f"✓ PASS: Magnitude RMS {our_mag_rms:.5f} < {magnitude_threshold}")

    if our_phase_rms > phase_threshold_deg:
        print(f"✗ FAIL: Phase RMS {our_phase_rms:.2f}° > {phase_threshold_deg}°")
        passed = False
    else:
        print(f"✓ PASS: Phase RMS {our_phase_rms:.2f}° < {phase_threshold_deg}°")

    # Generate plots
    print(f"\nGenerating validation plots...")

    # Plot D-term magnitudes vs antenna
    ant_indices = np.arange(n_antennas)

    # d_xy magnitude
    plot_three_way_comparison(
        truth_mag[:, 0],
        casa_mag[:, 0],
        our_mag[:, 0],
        x_values=ant_indices,
        title="D-term Magnitude (d_xy) vs Antenna",
        xlabel="Antenna",
        ylabel="|d_xy|",
        output_file_path=f"{output_prefix}_dxy_magnitude.html",
    )

    # d_yx magnitude
    plot_three_way_comparison(
        truth_mag[:, 1],
        casa_mag[:, 1],
        our_mag[:, 1],
        x_values=ant_indices,
        title="D-term Magnitude (d_yx) vs Antenna",
        xlabel="Antenna",
        ylabel="|d_yx|",
        output_file_path=f"{output_prefix}_dyx_magnitude.html",
    )

    # d_xy phase
    plot_three_way_comparison(
        truth_phase[:, 0],
        casa_phase[:, 0],
        our_phase[:, 0],
        x_values=ant_indices,
        title="D-term Phase (d_xy) vs Antenna",
        xlabel="Antenna",
        ylabel="Phase (degrees)",
        output_file_path=f"{output_prefix}_dxy_phase.html",
    )

    print("="*60)
    if passed:
        print("✓ LEAKAGE (D-TERM) VALIDATION PASSED")
        return 0
    else:
        print("✗ LEAKAGE (D-TERM) VALIDATION FAILED")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Validate polarization leakage recovery")
    parser.add_argument("--msname", default="sim_leakage_test.ms", help="MS name")
    parser.add_argument("--skip_sim", action="store_true", help="Skip simulation if MS exists")
    parser.add_argument("--leakage_level", type=float, default=0.05, help="Leakage level")
    parser.add_argument("--n_channels", type=int, default=64, help="Number of channels")
    parser.add_argument("--seed", type=int, default=45, help="Random seed")
    parser.add_argument("--no_noise", action="store_true", help="Skip thermal noise")
    parser.add_argument("--map", action="store_true", help="Use MAP instead of MCMC")
    args = parser.parse_args()

    # Create simulated MS if needed (full polarization required)
    if not args.skip_sim or not os.path.exists(args.msname):
        print("Creating simulated 3C286 MS with full polarization...")
        simulate_3c286(
            msname=args.msname,
            n_channels=args.n_channels,
            obs_time_min=5.0,
            int_time_sec=2.0,
        )
    else:
        print(f"Using existing MS: {args.msname}")

    # Load MS to get metadata
    ms_handler = MeasurementSetHandler(args.msname)
    n_antennas = len(ms_handler.get_antenna_names())

    # Generate ground truth D-terms
    truth_dterms = generate_ground_truth_dterms(
        n_antennas=n_antennas,
        leakage_level=args.leakage_level,
        seed=args.seed,
    )

    print(f"Generated D-terms: |d| ~ {args.leakage_level:.3f}")
    print(f"  d_xy range: {np.min(np.abs(truth_dterms[:, 0])):.4f} - {np.max(np.abs(truth_dterms[:, 0])):.4f}")
    print(f"  d_yx range: {np.min(np.abs(truth_dterms[:, 1])):.4f} - {np.max(np.abs(truth_dterms[:, 1])):.4f}")

    # Corrupt MS with leakage
    corrupt_ms_with_leakage(
        args.msname,
        truth_dterms,
        add_noise=not args.no_noise,
    )

    # Run CASA polcal
    dtable = args.msname.replace(".ms", "_casa.dcal")
    if os.path.exists(dtable):
        os.system(f"rm -rf {dtable}")

    if not run_casa_polcal(args.msname, dtable):
        return 1

    # Run our solver
    solver = run_solver(args.msname, dtable, use_map=args.map)

    # Compare and validate
    exit_code = compare_dterm_results(
        truth_dterms,
        dtable,
        solver,
        n_antennas,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
