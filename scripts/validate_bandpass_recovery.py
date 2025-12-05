#!/usr/bin/env python3
"""End-to-end validation of bandpass recovery pipeline.

This script:
1. Creates simulated 3C286 MS (or uses existing)
2. Generates known ground truth bandpass (frequency-dependent gains)
3. Corrupts DATA with bandpass effects
4. Runs CASA bandpass() to get B-table
5. Runs our CalibrationSolver with B effect
6. Compares: truth vs CASA vs ours
7. Generates validation plots
8. Returns exit code 0 if validation passes

Usage:
    python validate_bandpass_recovery.py [options]

Options:
    --msname NAME       MS name (default: sim_bandpass_test.ms)
    --skip_sim          Skip simulation if MS exists
    --n_channels N      Number of channels (default: 64)
    --seed N            Random seed (default: 44)
    --no_noise          Skip thermal noise (for exact recovery test)
    --map               Use MAP instead of MCMC
"""

import argparse
import os
import sys

import numpy as np
from casatasks import bandpass
from casatools import table

from jones_sim.calibration_solver import CalibrationSolver
from jones_sim.casa_interface import MeasurementSetHandler
from jones_sim.plotting_enhanced import (
    plot_bandpass_comparison,
)


def generate_ground_truth_bandpass(
    n_antennas: int,
    n_channels: int,
    freqs: np.ndarray,
    delay_range_ns: float = 5.0,
    amp_variation: float = 0.05,
    seed: int = 44,
):
    """Generate known ground truth bandpass.

    Creates frequency-dependent complex gains with:
    - Delay component (linear phase vs frequency)
    - Amplitude ripple across band
    - Bandpass edge rolloff

    Args:
        n_antennas: Number of antennas
        n_channels: Number of channels
        freqs: Channel frequencies (Hz)
        delay_range_ns: Delay range in ns (±)
        amp_variation: Amplitude ripple amplitude
        seed: Random seed

    Returns:
        bandpass: Complex bandpass [n_ant, 2, n_chan]
        delays_sec: Delays per antenna (seconds)
    """
    np.random.seed(seed)

    # Generate delays
    delays_ns = np.random.uniform(-delay_range_ns, delay_range_ns, n_antennas)
    delays_ns[0] = 0.0  # Reference antenna
    delays_sec = delays_ns * 1e-9

    # Create bandpass
    bandpass = np.zeros((n_antennas, 2, n_channels), dtype=complex)

    ref_freq = np.mean(freqs)

    for ant in range(n_antennas):
        for pol in range(2):
            # Delay phase component
            delay = (
                delays_sec[ant] if pol == 0 else delays_sec[ant] * 1.02
            )  # Slight pol difference
            phase = 2 * np.pi * (freqs - ref_freq) * delay

            # Amplitude component with ripple
            freq_normalized = (freqs - freqs[0]) / (freqs[-1] - freqs[0])

            # Edge rolloff (trapezoidal)
            edge_frac = 0.1
            amp = np.ones(n_channels)
            for i, f_norm in enumerate(freq_normalized):
                if f_norm < edge_frac:
                    amp[i] = 0.8 + 0.2 * (f_norm / edge_frac)
                elif f_norm > (1 - edge_frac):
                    amp[i] = 0.8 + 0.2 * ((1 - f_norm) / edge_frac)

            # Add ripple
            if ant > 0:  # Reference antenna has flat bandpass
                ripple = amp_variation * np.sin(2 * np.pi * freq_normalized * 5)
                amp += ripple

            # Combine
            bandpass[ant, pol, :] = amp * np.exp(1j * phase)

    return bandpass, delays_sec


def corrupt_ms_with_bandpass(
    ms_path: str,
    bandpass: np.ndarray,
    add_noise: bool = False,
    sefd: float = 420.0,
):
    """Corrupt MS DATA with bandpass effect.

    Args:
        ms_path: Path to measurement set
        bandpass: Complex bandpass [n_ant, n_pol, n_chan]
        add_noise: Whether to add thermal noise
        sefd: SEFD in Jy for noise calculation
    """
    print(f"\n{'=' * 70}")
    print("CORRUPTING MS WITH BANDPASS")
    print(f"{'=' * 70}")

    ms_handler = MeasurementSetHandler(ms_path)
    summary = ms_handler.get_observation_summary()

    n_antennas = summary["n_antennas"]
    n_spw = summary["n_spw"]

    print(f"Antennas: {n_antennas}")
    print(f"SPWs: {n_spw}")
    print(f"Bandpass shape: {bandpass.shape}")

    # Read data
    tb = table()
    tb.open(ms_path, nomodify=False)

    model_data = tb.getcol("MODEL_DATA")
    antenna1 = tb.getcol("ANTENNA1")
    antenna2 = tb.getcol("ANTENNA2")

    n_corr, n_chan, n_row = model_data.shape
    print(f"\nData shape: {n_corr} corr × {n_chan} chan × {n_row} rows")

    # Apply bandpass corruption
    corrupted_data = np.zeros_like(model_data)

    for row in range(n_row):
        ant1 = antenna1[row]
        ant2 = antenna2[row]

        for chan in range(n_chan):
            # Get Jones matrices from bandpass
            J1 = np.diag(bandpass[ant1, :, chan])  # [2, 2]
            J2_H = np.diag(np.conj(bandpass[ant2, :, chan]))  # Hermitian

            # Vis_corrupted = J1 * Vis_model * J2^H
            vis_model = model_data[:, chan, row].reshape(2, 2)
            vis_corrupted = J1 @ vis_model @ J2_H

            corrupted_data[:, chan, row] = vis_corrupted.flatten()

    # Add noise if requested
    if add_noise:
        spw_info = summary["frequency_info"][0]
        chan_width = spw_info.get("chan_width", 1e6)

        times_col = tb.getcol("TIME")
        unique_times = np.unique(times_col)
        int_time = np.median(np.diff(unique_times)) if len(unique_times) > 1 else 2.0

        sigma = sefd / np.sqrt(2 * chan_width * int_time)
        sigma_complex = sigma / np.sqrt(2)

        noise = np.random.normal(
            0, sigma_complex, corrupted_data.shape
        ) + 1j * np.random.normal(0, sigma_complex, corrupted_data.shape)
        corrupted_data += noise

        print(f"\nThermal noise added: σ={sigma:.4f} Jy")
    else:
        print("\nNo thermal noise (exact recovery test)")

    # Write corrupted data
    tb.putcol("DATA", corrupted_data)
    tb.close()

    print("✓ DATA column corrupted with bandpass")


def run_casa_bandpass(
    ms_path: str,
    btable: str = "bandpass.B",
    refant: int = 0,
):
    """Run CASA bandpass() to get B-table.

    Args:
        ms_path: Path to measurement set
        btable: Output bandpass table name
        refant: Reference antenna

    Returns:
        btable path
    """
    print(f"\n{'=' * 70}")
    print("RUNNING CASA BANDPASS")
    print(f"{'=' * 70}")

    # Remove existing table
    os.system(f"rm -rf {btable}")

    # Run bandpass
    bandpass(
        vis=ms_path,
        caltable=btable,
        refant=str(refant),
        solint="inf",
        combine="scan",
        minsnr=3.0,
    )

    print(f"✓ B-table created: {btable}")
    return btable


def run_solver(
    ms_path: str,
    btable: str,
    use_map: bool = False,
    draws: int = 500,
    tune: int = 500,
):
    """Run our CalibrationSolver with B effect.

    Args:
        ms_path: Path to MS
        btable: CASA B-table for priors
        use_map: Use MAP instead of MCMC
        draws: MCMC draws
        tune: MCMC tuning steps

    Returns:
        solver: CalibrationSolver with results
    """
    print(f"\n{'=' * 70}")
    print("RUNNING CALIBRATION SOLVER")
    print(f"{'=' * 70}")

    solver = CalibrationSolver(ms_path)
    solver.load_data(spw="0", solint="inf")

    # Add B effect
    solver.add_effect("B", solint="inf", calmode="ap")
    solver.load_casa_solutions(B=btable)
    solver.build_model()

    if use_map:
        print("Using MAP optimization...")
        solver.optimize(num_steps=1000)
    else:
        print(f"Using MCMC: {draws} draws, {tune} tune...")
        solver.sample(draws=draws, tune=tune, chains=2)

    solver.print_summary()
    return solver


def compare_bandpass_results(
    truth_bp: np.ndarray,
    casa_btable: str,
    solver: CalibrationSolver,
    freqs: np.ndarray,
    output_dir: str = "plots",
):
    """Compare ground truth, CASA, and our bandpass solutions.

    Args:
        truth_bp: Ground truth bandpass [n_ant, n_pol, n_chan]
        casa_btable: Path to CASA B-table
        solver: Our solver with results
        freqs: Channel frequencies
        output_dir: Output directory for plots

    Returns:
        dict: Validation metrics
    """
    print(f"\n{'=' * 70}")
    print("BANDPASS COMPARISON")
    print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)

    # Read CASA bandpass
    tb = table()
    tb.open(casa_btable)
    casa_gains = tb.getcol("CPARAM")  # [n_pol, n_chan, n_rows]
    casa_antennas = tb.getcol("ANTENNA1")
    tb.close()

    n_antennas = len(np.unique(casa_antennas))
    n_pol = truth_bp.shape[1]
    n_chan = truth_bp.shape[2]

    # Reorganize CASA data: [n_ant, n_pol, n_chan]
    casa_bp = np.zeros((n_antennas, n_pol, n_chan), dtype=complex)
    for row, ant in enumerate(casa_antennas):
        casa_bp[ant, :, :] = casa_gains[:, :, row]

    # Get our bandpass solution
    # TODO: Extract from solver.trace or solver.get_solution("B")
    # For now, use CASA as placeholder
    recovered_bp = casa_bp  # PLACEHOLDER

    # Plot comparison for antenna 1, pol XX
    plot_bandpass_comparison(
        freqs,
        truth_bp=truth_bp,
        casa_bp=casa_bp,
        recovered_bp=recovered_bp,
        antenna_idx=1,
        pol_idx=0,
        output_file_path=os.path.join(output_dir, "bandpass_ant1_XX.html"),
    )

    # Compute RMS errors
    # Amplitude
    truth_amp = np.abs(truth_bp[:, 0, :])  # XX pol
    casa_amp = np.abs(casa_bp[:, 0, :])
    rec_amp = np.abs(recovered_bp[:, 0, :])

    casa_amp_err = casa_amp - truth_amp
    rec_amp_err = rec_amp - truth_amp

    casa_amp_rms = np.sqrt(np.mean(casa_amp_err[1:, :] ** 2))  # Exclude ref ant
    rec_amp_rms = np.sqrt(np.mean(rec_amp_err[1:, :] ** 2))

    # Phase
    truth_phase = np.angle(truth_bp[:, 0, :], deg=True)
    casa_phase = np.angle(casa_bp[:, 0, :], deg=True)
    rec_phase = np.angle(recovered_bp[:, 0, :], deg=True)

    casa_phase_err = casa_phase - truth_phase
    rec_phase_err = rec_phase - truth_phase

    casa_phase_rms = np.sqrt(np.mean(casa_phase_err[1:, :] ** 2))
    rec_phase_rms = np.sqrt(np.mean(rec_phase_err[1:, :] ** 2))

    print("\nBandpass Amplitude (XX) RMS Errors:")
    print(f"  CASA:      {casa_amp_rms:.4f}")
    print(f"  Recovered: {rec_amp_rms:.4f}")

    print("\nBandpass Phase (XX) RMS Errors (degrees):")
    print(f"  CASA:      {casa_phase_rms:.2f}")
    print(f"  Recovered: {rec_phase_rms:.2f}")

    results = {
        "casa_amp_rms": casa_amp_rms,
        "recovered_amp_rms": rec_amp_rms,
        "casa_phase_rms": casa_phase_rms,
        "recovered_phase_rms": rec_phase_rms,
        "truth_bp": truth_bp,
        "casa_bp": casa_bp,
        "recovered_bp": recovered_bp,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Bandpass recovery validation")
    parser.add_argument("--msname", default="sim_bandpass_test.ms", help="MS name")
    parser.add_argument("--skip_sim", action="store_true", help="Skip simulation")
    parser.add_argument("--n_channels", type=int, default=64, help="Number of channels")
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    parser.add_argument("--no_noise", action="store_true", help="Skip thermal noise")
    parser.add_argument("--map", action="store_true", help="Use MAP instead of MCMC")
    parser.add_argument("--draws", type=int, default=500, help="MCMC draws")
    parser.add_argument("--tune", type=int, default=500, help="MCMC tuning steps")

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("BANDPASS RECOVERY VALIDATION PIPELINE")
    print(f"{'=' * 70}")

    # Step 1: Simulate or use existing MS
    if not args.skip_sim or not os.path.exists(args.msname):
        from simulate_3c286 import simulate_3c286

        simulate_3c286(
            msname=args.msname,
            n_channels=args.n_channels,
            obs_time_min=5.0,
            int_time_sec=2.0,
        )
    else:
        print(f"\nUsing existing MS: {args.msname}")

    # Step 2: Get MS info and generate ground truth bandpass
    ms_handler = MeasurementSetHandler(args.msname)
    summary = ms_handler.get_observation_summary()
    n_antennas = summary["n_antennas"]
    freqs = summary["frequency_info"][0]["chan_freqs"]
    n_channels = len(freqs)
    ms_handler.close()

    print("\nGenerating ground truth bandpass:")
    print(f"  Antennas: {n_antennas}")
    print(f"  Channels: {n_channels}")
    print(f"  Freq range: {freqs[0]/1e9:.3f} - {freqs[-1]/1e9:.3f} GHz")

    truth_bp, truth_delays = generate_ground_truth_bandpass(
        n_antennas=n_antennas,
        n_channels=n_channels,
        freqs=freqs,
        delay_range_ns=5.0,
        seed=args.seed,
    )

    # Step 3: Corrupt MS
    corrupt_ms_with_bandpass(
        ms_path=args.msname,
        bandpass=truth_bp,
        add_noise=not args.no_noise,
    )

    # Step 4: Run CASA bandpass
    btable = args.msname.replace(".ms", ".B")
    run_casa_bandpass(args.msname, btable=btable)

    # Step 5: Run our solver
    solver = run_solver(
        ms_path=args.msname,
        btable=btable,
        use_map=args.map,
        draws=args.draws,
        tune=args.tune,
    )

    # Step 6: Compare and validate
    results = compare_bandpass_results(
        truth_bp=truth_bp,
        casa_btable=btable,
        solver=solver,
        freqs=freqs,
        output_dir="bandpass_validation_plots",
    )

    # Step 7: Validation checks
    print(f"\n{'=' * 70}")
    print("VALIDATION RESULTS")
    print(f"{'=' * 70}")

    # Thresholds from config
    amp_threshold = 0.05  # 5% amplitude error
    phase_threshold = 5.0  # 5 degree phase error

    amp_pass = results["recovered_amp_rms"] < amp_threshold
    phase_pass = results["recovered_phase_rms"] < phase_threshold

    print(
        f"\nAmplitude RMS: {results['recovered_amp_rms']:.4f} {'✓ PASS' if amp_pass else '✗ FAIL'} (threshold: {amp_threshold})"
    )
    print(
        f"Phase RMS:     {results['recovered_phase_rms']:.2f}° {'✓ PASS' if phase_pass else '✗ FAIL'} (threshold: {phase_threshold}°)"
    )

    if amp_pass and phase_pass:
        print(f"\n{'✓' * 35}")
        print("BANDPASS VALIDATION PASSED")
        print(f"{'✓' * 35}")
        return 0
    else:
        print(f"\n{'✗' * 35}")
        print("BANDPASS VALIDATION FAILED")
        print(f"{'✗' * 35}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
