#!/usr/bin/env python3
"""End-to-end validation of time-varying gain (G) recovery pipeline.

This script:
1. Creates simulated 3C286 MS (or uses existing)
2. Generates known ground truth time-varying gains
3. Corrupts DATA with gain effects
4. Runs CASA gaincal() to get G-table
5. Runs our CalibrationSolver with G effect
6. Compares: truth vs CASA vs ours
7. Generates validation plots
8. Returns exit code 0 if validation passes

Usage:
    python validate_gain_recovery.py [options]

Options:
    --msname NAME       MS name (default: sim_gain_test.ms)
    --skip_sim          Skip simulation if MS exists
    --n_times N         Number of time samples (default: 60)
    --seed N            Random seed (default: 43)
    --no_noise          Skip thermal noise (for exact recovery test)
    --map               Use MAP instead of MCMC
"""

import argparse
import os
import sys

import numpy as np

from casatasks import gaincal
from casatools import table

from jones_sim import JonesSimulator, JonesConfig
from jones_sim.effects import ElectronicGains
from jones_sim.calibration_solver import CalibrationSolver
from jones_sim.casa_interface import MeasurementSetHandler
from jones_sim.plotting_enhanced import plot_three_way_comparison
from scripts.simulate_3c286 import simulate_3c286


def generate_ground_truth_gains(
    n_antennas: int,
    n_times: int,
    timescale_minutes: float = 10.0,
    amp_mean: float = 1.0,
    amp_std: float = 0.05,
    phase_std_deg: float = 5.0,
    seed: int = 43,
):
    """Generate known ground truth time-varying gains.

    Creates smooth time-varying complex gains with:
    - Amplitude variations (log-normal around mean)
    - Phase variations (Gaussian random walk)
    - Smooth variations on timescale

    Args:
        n_antennas: Number of antennas
        n_times: Number of time samples
        timescale_minutes: Characteristic timescale for variations
        amp_mean: Mean amplitude (1.0 = unity gain)
        amp_std: Standard deviation for amplitude variations
        phase_std_deg: Standard deviation for phase variations (degrees)
        seed: Random seed

    Returns:
        gains: Complex gains [n_ant, n_time, 2 pols]
        times_rel: Relative time array (minutes)
    """
    np.random.seed(seed)

    # Time array
    times_rel = np.linspace(0, 30, n_times)  # 30 minutes total
    dt = times_rel[1] - times_rel[0]

    # Create gains
    gains = np.zeros((n_antennas, n_times, 2), dtype=complex)

    for ant in range(n_antennas):
        for pol in range(2):
            # Amplitude - smooth variations with Gaussian process-like behavior
            amp_innovations = np.random.normal(0, amp_std, n_times)
            # Apply exponential smoothing (low-pass filter)
            alpha = dt / timescale_minutes
            amp_smooth = np.zeros(n_times)
            amp_smooth[0] = amp_mean
            for t in range(1, n_times):
                amp_smooth[t] = (1 - alpha) * amp_smooth[t-1] + alpha * (amp_mean + amp_innovations[t])

            # Ensure positive amplitudes
            amp = np.maximum(amp_smooth, 0.1)

            # Phase - random walk with drift
            phase_innovations = np.random.normal(0, phase_std_deg, n_times)
            # Apply smoothing
            phase_smooth = np.zeros(n_times)
            for t in range(1, n_times):
                phase_smooth[t] = phase_smooth[t-1] + alpha * phase_innovations[t]

            phase_rad = np.deg2rad(phase_smooth)

            # Complex gains
            gains[ant, :, pol] = amp * np.exp(1j * phase_rad)

    # Reference antenna = unity gain
    gains[0, :, :] = 1.0 + 0j

    return gains, times_rel


def corrupt_ms_with_gains(ms_path, gains, add_noise=True, sefd=420.0):
    """Corrupt MS DATA column with time-varying gain effects using JonesSimulator.

    Args:
        ms_path: Path to MS
        gains: Complex gains [n_ant, n_time, 2]
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
    times_col = tb.getcol("TIME")

    n_corr, n_chan, n_row = model_data.shape

    # Get frequency info
    summary = ms_handler.get_observation_summary()
    spw_info = summary["frequency_info"][0]
    freqs = spw_info["chan_freqs"]
    chan_width = spw_info.get("chan_width", freqs[1] - freqs[0] if len(freqs) > 1 else 1e6)

    # Map times to indices
    unique_times = np.unique(times_col)
    time_to_idx = {t: i for i, t in enumerate(unique_times)}

    # Create time-varying gain callables
    def g_xx_func(freq, time, antenna_id):
        t_idx = min(time_to_idx.get(time, 0), gains.shape[1] - 1)
        return gains[antenna_id, t_idx, 0]

    def g_yy_func(freq, time, antenna_id):
        t_idx = min(time_to_idx.get(time, 0), gains.shape[1] - 1)
        return gains[antenna_id, t_idx, 1]

    # Create simulator with time-varying gains
    sim = JonesSimulator()
    gain_effect = ElectronicGains(g_xx=g_xx_func, g_yy=g_yy_func)
    sim.add_effect("gains", gain_effect)

    # Reshape for simulator: (n_vis, 4)
    ideal_vis = model_data.transpose(2, 1, 0).reshape(-1, n_corr)  # (n_row*n_chan, 4)
    frequencies = np.tile(freqs, n_row)
    times = np.repeat(times_col, n_chan)
    ant1_arr = np.repeat(antenna1, n_chan)
    ant2_arr = np.repeat(antenna2, n_chan)

    # Corrupt using simulator
    print("Corrupting visibilities with time-varying gains...", end=" ", flush=True)
    corrupted_vis = sim.corrupt_visibilities(
        ideal_vis, frequencies, times, ant1_arr, ant2_arr, use_gpu=False
    )
    print("Done")

    # Reshape back to (n_corr, n_chan, n_row)
    corrupted_data = corrupted_vis.reshape(n_row, n_chan, n_corr).transpose(2, 1, 0)

    # Add thermal noise if requested
    if add_noise:
        # Calculate integration time
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

    # Write corrupted data
    tb.putcol("DATA", corrupted_data)
    tb.close()

    print(f"✓ Corrupted MS with time-varying gain effects (noise={add_noise})")


def run_casa_gaincal(ms_path, gtable, refant="0", solint="int"):
    """Run CASA gaincal() to calibrate gains.

    Args:
        ms_path: Path to MS
        gtable: Output caltable path
        refant: Reference antenna
        solint: Solution interval

    Returns:
        Success boolean
    """
    try:
        print(f"Running CASA gaincal (solint={solint})...")
        gaincal(
            vis=ms_path,
            caltable=gtable,
            refant=refant,
            solint=solint,
            calmode="ap",  # Amplitude and phase
            gaintype="G",
        )
        print(f"✓ CASA gaincal complete: {gtable}")
        return True
    except Exception as e:
        print(f"✗ CASA gaincal failed: {e}")
        return False


def run_solver(ms_path, gtable, use_map=True, draws=500, tune=500):
    """Run our CalibrationSolver with G effect.

    Args:
        ms_path: Path to MS
        gtable: CASA caltable for comparison
        use_map: Use MAP instead of MCMC
        draws: Number of MCMC samples
        tune: Number of tuning samples

    Returns:
        CalibrationSolver instance
    """
    print(f"Running CalibrationSolver (method={'MAP' if use_map else 'MCMC'})...")
    solver = CalibrationSolver(ms_path)
    solver.load_data()

    # Configure G effect
    solver.configure_effect(
        effect_name="G",
        effect_type="time_variable_gain",
        solint="int",  # Per integration
        calmode="ap",
        casa_caltable=gtable,
    )

    # Solve
    if use_map:
        solver.solve(method="map", max_iter=1000, tol=1e-6)
    else:
        solver.solve(method="mcmc", num_warmup=tune, num_samples=draws)

    print(f"✓ Solver complete")
    return solver


def compare_gain_results(
    truth_gains,
    casa_gtable,
    solver,
    times_rel,
    n_antennas,
    output_prefix="gain_validation",
    amp_threshold=0.03,
    phase_threshold_deg=3.0,
):
    """Compare ground truth vs CASA vs our solver.

    Args:
        truth_gains: Ground truth gains [n_ant, n_time, 2]
        casa_gtable: Path to CASA G-table
        solver: Our CalibrationSolver
        times_rel: Time array (minutes)
        n_antennas: Number of antennas
        output_prefix: Prefix for output plots
        amp_threshold: Amplitude RMS threshold
        phase_threshold_deg: Phase RMS threshold (degrees)

    Returns:
        0 if passed, 1 if failed
    """
    # Read CASA solutions
    tb = table()
    tb.open(casa_gtable)
    casa_gains_raw = tb.getcol("CPARAM")  # [n_pol, n_chan, n_rows]
    casa_ant = tb.getcol("ANTENNA1")
    casa_time = tb.getcol("TIME")
    tb.close()

    # Reshape CASA gains to [n_ant, n_time, 2]
    n_times = len(np.unique(casa_time))
    casa_gains = np.zeros((n_antennas, n_times, 2), dtype=complex)

    for ant in range(n_antennas):
        mask = casa_ant == ant
        gains_ant = casa_gains_raw[:, 0, mask]  # [2, n_time]
        if gains_ant.shape[1] > 0:
            casa_gains[ant, :, :] = gains_ant.T

    # Get our solver solutions
    our_gains = solver.get_solutions()["G"]["gains"]  # [n_ant, n_time, 2]

    # Compare amplitudes and phases
    truth_amp = np.abs(truth_gains)
    casa_amp = np.abs(casa_gains)
    our_amp = np.abs(our_gains)

    truth_phase = np.angle(truth_gains, deg=True)
    casa_phase = np.angle(casa_gains, deg=True)
    our_phase = np.angle(our_gains, deg=True)

    # Calculate residuals (excluding reference antenna)
    casa_amp_res = casa_amp[1:] - truth_amp[1:]
    our_amp_res = our_amp[1:] - truth_amp[1:]

    # Phase residuals (unwrap phase differences)
    casa_phase_res = (casa_phase[1:] - truth_phase[1:] + 180) % 360 - 180
    our_phase_res = (our_phase[1:] - truth_phase[1:] + 180) % 360 - 180

    # RMS metrics
    casa_amp_rms = np.sqrt(np.mean(casa_amp_res**2))
    our_amp_rms = np.sqrt(np.mean(our_amp_res**2))
    casa_phase_rms = np.sqrt(np.mean(casa_phase_res**2))
    our_phase_rms = np.sqrt(np.mean(our_phase_res**2))

    # Print comparison
    print("\n" + "="*60)
    print("GAIN RECOVERY VALIDATION RESULTS")
    print("="*60)
    print(f"Amplitude RMS (Truth vs CASA): {casa_amp_rms:.4f}")
    print(f"Amplitude RMS (Truth vs Ours): {our_amp_rms:.4f}")
    print(f"Phase RMS (Truth vs CASA):     {casa_phase_rms:.2f}°")
    print(f"Phase RMS (Truth vs Ours):     {our_phase_rms:.2f}°")
    print(f"\nThresholds: amp={amp_threshold}, phase={phase_threshold_deg}°")

    # Validation checks
    passed = True
    if our_amp_rms > amp_threshold:
        print(f"✗ FAIL: Amplitude RMS {our_amp_rms:.4f} > {amp_threshold}")
        passed = False
    else:
        print(f"✓ PASS: Amplitude RMS {our_amp_rms:.4f} < {amp_threshold}")

    if our_phase_rms > phase_threshold_deg:
        print(f"✗ FAIL: Phase RMS {our_phase_rms:.2f}° > {phase_threshold_deg}°")
        passed = False
    else:
        print(f"✓ PASS: Phase RMS {our_phase_rms:.2f}° < {phase_threshold_deg}°")

    # Generate plots
    print(f"\nGenerating validation plots...")

    # Plot for a few representative antennas
    for ant in [1, 5, 10]:
        if ant >= n_antennas:
            continue

        # Amplitude comparison
        plot_three_way_comparison(
            truth_amp[ant, :, 0],
            casa_amp[ant, :, 0],
            our_amp[ant, :, 0],
            x_values=times_rel,
            title=f"Gain Amplitude vs Time (Ant {ant}, Pol X)",
            xlabel="Time (minutes)",
            ylabel="Amplitude",
            output_file_path=f"{output_prefix}_amp_ant{ant}_polX.html",
        )

        # Phase comparison
        plot_three_way_comparison(
            truth_phase[ant, :, 0],
            casa_phase[ant, :, 0],
            our_phase[ant, :, 0],
            x_values=times_rel,
            title=f"Gain Phase vs Time (Ant {ant}, Pol X)",
            xlabel="Time (minutes)",
            ylabel="Phase (degrees)",
            output_file_path=f"{output_prefix}_phase_ant{ant}_polX.html",
        )

    print("="*60)
    if passed:
        print("✓ GAIN VALIDATION PASSED")
        return 0
    else:
        print("✗ GAIN VALIDATION FAILED")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Validate gain recovery pipeline")
    parser.add_argument("--msname", default="sim_gain_test.ms", help="MS name")
    parser.add_argument("--skip_sim", action="store_true", help="Skip simulation if MS exists")
    parser.add_argument("--n_times", type=int, default=60, help="Number of time samples")
    parser.add_argument("--n_channels", type=int, default=64, help="Number of channels")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--no_noise", action="store_true", help="Skip thermal noise")
    parser.add_argument("--map", action="store_true", help="Use MAP instead of MCMC")
    args = parser.parse_args()

    # Create simulated MS if needed
    if not args.skip_sim or not os.path.exists(args.msname):
        print("Creating simulated 3C286 MS...")
        # Calculate obs time from n_times and 2s integration
        obs_time_min = (args.n_times * 2.0) / 60.0
        simulate_3c286(
            msname=args.msname,
            n_channels=args.n_channels,
            obs_time_min=obs_time_min,
            int_time_sec=2.0,
        )
    else:
        print(f"Using existing MS: {args.msname}")

    # Load MS to get metadata
    ms_handler = MeasurementSetHandler(args.msname)
    n_antennas = len(ms_handler.get_antenna_names())

    # Generate ground truth gains
    truth_gains, times_rel = generate_ground_truth_gains(
        n_antennas=n_antennas,
        n_times=args.n_times,
        seed=args.seed,
    )

    # Corrupt MS with gains
    corrupt_ms_with_gains(
        args.msname,
        truth_gains,
        add_noise=not args.no_noise,
    )

    # Run CASA gaincal
    gtable = args.msname.replace(".ms", "_casa.gcal")
    if os.path.exists(gtable):
        os.system(f"rm -rf {gtable}")

    if not run_casa_gaincal(args.msname, gtable):
        return 1

    # Run our solver
    solver = run_solver(args.msname, gtable, use_map=args.map)

    # Compare and validate
    exit_code = compare_gain_results(
        truth_gains,
        gtable,
        solver,
        times_rel,
        n_antennas,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
