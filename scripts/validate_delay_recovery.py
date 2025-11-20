#!/usr/bin/env python3
"""End-to-end validation of delay recovery pipeline.

This script:
1. Creates simulated 3C286 MS (or uses existing)
2. Generates known ground truth delays
3. Corrupts DATA with those delays
4. Runs CASA gaincal to get K-table
5. Runs our BayesianDelaySampler
6. Compares: truth vs CASA vs ours
7. Generates comparison plots

Usage:
    python validate_delay_recovery.py [options]

Options:
    --msname NAME       MS name (default: sim_3c286.ms)
    --skip_sim          Skip simulation if MS exists
    --n_channels N      Number of channels (default: 64)
    --delay_range NS    Delay range in ns (default: 10)
    --seed N            Random seed (default: 42)
    --no_noise          Skip thermal noise (for exact recovery test)
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory for development
sys.path.insert(0, '/home/pjaganna/Software/jones_sim')

from casatasks import gaincal
from casatools import table

from jones_sim import BandpassDelay, JonesSimulator
from jones_sim.casa_interface import MeasurementSetHandler
from jones_sim.dbs_solver import BayesianDelaySampler


def generate_ground_truth_delays(n_antennas: int, delay_range_ns: float, seed: int = 42):
    """Generate known ground truth delays.

    Args:
        n_antennas: Number of antennas
        delay_range_ns: Delay range in nanoseconds (±)
        seed: Random seed for reproducibility

    Returns:
        delays_ns: Array of delays in nanoseconds
        delays_sec: Array of delays in seconds
    """
    np.random.seed(seed)

    # Generate random delays
    delays_ns = np.random.uniform(-delay_range_ns, delay_range_ns, n_antennas)

    # Fix antenna 0 as reference
    delays_ns[0] = 0.0

    delays_sec = delays_ns * 1e-9

    return delays_ns, delays_sec


def corrupt_ms_with_delays(
    ms_path: str,
    delays_sec: np.ndarray,
    add_noise: bool = False,
    tsys: float = 30.0,
    aperture_eff: float = 0.45,
    antenna_diameter: float = 25.0,
):
    """Corrupt MS DATA column with known delays.

    Args:
        ms_path: Path to measurement set
        delays_sec: Delay per antenna in seconds
        add_noise: Whether to add thermal noise
        tsys: System temperature in K
        aperture_eff: Aperture efficiency
        antenna_diameter: Antenna diameter in meters
    """
    print(f"\n{'=' * 70}")
    print("CORRUPTING MS WITH DELAYS")
    print(f"{'=' * 70}")

    # Create delay effect
    delay_effect = BandpassDelay(
        tau_xx=delays_sec,
        tau_yy=delays_sec,
        ref_freq=0.0,
    )

    # Create simulator
    sim = JonesSimulator()
    sim.add_effect("delays", delay_effect)

    # Open MS
    ms_handler = MeasurementSetHandler(ms_path)
    summary = ms_handler.get_observation_summary()

    n_antennas = summary["n_antennas"]
    n_spw = summary["n_spw"]

    print(f"Antennas: {n_antennas}")
    print(f"SPWs: {n_spw}")
    print(f"Delays applied (ns):")
    for ant in range(min(n_antennas, 10)):
        print(f"  Ant {ant}: {delays_sec[ant] * 1e9:.3f}")
    if n_antennas > 10:
        print(f"  ... ({n_antennas - 10} more)")

    # Read data
    tb = table()
    tb.open(ms_path, nomodify=False)

    model_data = tb.getcol('MODEL_DATA')  # Clean data
    antenna1 = tb.getcol('ANTENNA1')
    antenna2 = tb.getcol('ANTENNA2')

    n_corr, n_chan, n_row = model_data.shape
    print(f"\nData shape: {n_corr} corr × {n_chan} chan × {n_row} rows")

    # Get frequencies
    spw_info = summary["frequency_info"][0]
    freqs = spw_info["chan_freqs"]
    chan_width = spw_info.get("chan_width", freqs[1] - freqs[0] if len(freqs) > 1 else 1e6)

    # Reshape for corruption
    # Convert (n_corr, n_chan, n_row) to list of (n_vis, 4)
    ideal_list = []
    freq_list = []
    ant1_list = []
    ant2_list = []

    for row in range(n_row):
        for chan in range(n_chan):
            ideal_list.append(model_data[:, chan, row])
            freq_list.append(freqs[chan])
            ant1_list.append(antenna1[row])
            ant2_list.append(antenna2[row])

    ideal_vis = np.array(ideal_list, dtype=complex)
    frequencies = np.array(freq_list)
    times = np.zeros(len(frequencies))  # Not used for delays
    ant1_arr = np.array(ant1_list, dtype=int)
    ant2_arr = np.array(ant2_list, dtype=int)

    n_vis = len(ideal_vis)
    print(f"Total visibilities: {n_vis:,}")

    # Prepare noise parameters if needed
    noise_params = None
    if add_noise:
        # Calculate integration time from MS
        times_col = tb.getcol('TIME')
        unique_times = np.unique(times_col)
        if len(unique_times) > 1:
            int_time = np.median(np.diff(unique_times))
        else:
            int_time = 2.0  # Default

        bandwidth = np.full(n_vis, chan_width)
        int_time_arr = np.full(n_vis, int_time)

        noise_params = {
            'tsys': tsys,
            'aperture_eff': aperture_eff,
            'antenna_diameter': antenna_diameter,
            'bandwidth': bandwidth,
            'int_time': int_time_arr,
        }
        print(f"\nAdding thermal noise:")
        print(f"  Tsys: {tsys} K")
        print(f"  Aperture eff: {aperture_eff}")
        print(f"  Diameter: {antenna_diameter} m")
        print(f"  Int time: {int_time} s")
    else:
        print("\nNo thermal noise (exact recovery test)")

    # Corrupt
    print("\nCorrupting visibilities...", end=" ", flush=True)
    corrupted_vis = sim.corrupt_visibilities(
        ideal_vis,
        frequencies,
        times,
        ant1_arr,
        ant2_arr,
        use_gpu=False,  # Use CPU for consistency
        noise_params=noise_params,
    )
    print("Done")

    # Reshape back to (n_corr, n_chan, n_row)
    corrupted_data = np.zeros_like(model_data)
    vis_idx = 0
    for row in range(n_row):
        for chan in range(n_chan):
            corrupted_data[:, chan, row] = corrupted_vis[vis_idx]
            vis_idx += 1

    # Write corrupted data
    tb.putcol('DATA', corrupted_data)
    tb.close()

    print("✓ DATA column corrupted with delays")


def run_casa_gaincal(ms_path: str, caltable: str = 'delays.K', refant: int = 0):
    """Run CASA gaincal to get K-table.

    Args:
        ms_path: Path to measurement set
        caltable: Output calibration table name
        refant: Reference antenna index

    Returns:
        Path to calibration table
    """
    print(f"\n{'=' * 70}")
    print("RUNNING CASA GAINCAL")
    print(f"{'=' * 70}")

    # Remove existing table
    os.system(f'rm -rf {caltable}')

    # Run gaincal for delays
    gaincal(
        vis=ms_path,
        caltable=caltable,
        gaintype='K',  # Delay
        refant=str(refant),
        solint='inf',  # One solution for entire observation
        combine='scan',
        minsnr=3.0,
    )

    print(f"✓ K-table created: {caltable}")

    # Read and print CASA delays
    tb = table()
    tb.open(caltable)
    fparam = tb.getcol('FPARAM')
    antennas = tb.getcol('ANTENNA1')
    flags = tb.getcol('FLAG')
    tb.close()

    # Extract delays per antenna
    n_antennas = int(np.max(antennas)) + 1
    casa_delays_ns = np.zeros(n_antennas)

    if fparam.ndim == 3:
        n_pol, n_chan, n_rows = fparam.shape
        for row in range(n_rows):
            ant = antennas[row]
            # Average over pols and chans
            count = 0
            for pol in range(n_pol):
                for chan in range(n_chan):
                    if not flags[pol, chan, row]:
                        casa_delays_ns[ant] += fparam[pol, chan, row]
                        count += 1
            if count > 0:
                casa_delays_ns[ant] /= count
    elif fparam.ndim == 2:
        n_pol, n_rows = fparam.shape
        for row in range(n_rows):
            ant = antennas[row]
            count = 0
            for pol in range(n_pol):
                if not flags[pol, row]:
                    casa_delays_ns[ant] += fparam[pol, row]
                    count += 1
            if count > 0:
                casa_delays_ns[ant] /= count

    print(f"\nCASA delays (ns):")
    for ant in range(min(n_antennas, 10)):
        print(f"  Ant {ant}: {casa_delays_ns[ant]:.3f}")
    if n_antennas > 10:
        print(f"  ... ({n_antennas - 10} more)")

    return caltable, casa_delays_ns


def run_bayesian_solver(
    ms_path: str,
    caltable: str,
    output_trace: str = 'delay_trace.nc',
    draws: int = 1000,
    tune: int = 1000,
):
    """Run our Bayesian delay solver.

    Args:
        ms_path: Path to measurement set
        caltable: Path to CASA K-table (for priors)
        output_trace: Output trace filename
        draws: MCMC draws per chain
        tune: MCMC tuning steps

    Returns:
        Sampler object with results
    """
    print(f"\n{'=' * 70}")
    print("RUNNING BAYESIAN DELAY SOLVER")
    print(f"{'=' * 70}")

    sampler = BayesianDelaySampler(ms_path, caltable)
    sampler.load_data(spw=0, field=0)
    sampler.read_casa_delays()
    sampler.estimate_thermal_noise_from_time_scatter()
    sampler.build_model(prior_bound_ns=1.0, use_numpyro=True)
    sampler.sample(draws=draws, tune=tune, chains=2, target_accept=0.9)
    sampler.print_summary()
    sampler.save_trace(output_trace)

    return sampler


def compare_results(
    truth_delays_ns: np.ndarray,
    casa_delays_ns: np.ndarray,
    sampler: BayesianDelaySampler,
):
    """Compare ground truth, CASA, and our solver results.

    Args:
        truth_delays_ns: Ground truth delays in ns
        casa_delays_ns: CASA delays in ns
        sampler: Our sampler with results
    """
    print(f"\n{'=' * 70}")
    print("COMPARISON: TRUTH vs CASA vs OURS")
    print(f"{'=' * 70}")

    # Extract our posterior means
    delays_free_post = sampler.trace.posterior["delays_free"].values
    n_samples = delays_free_post.shape[0] * delays_free_post.shape[1]
    delays_free_flat = delays_free_post.reshape(n_samples, -1)

    n_antennas = sampler.n_antennas
    our_delays_ns = np.zeros(n_antennas)
    our_delays_std = np.zeros(n_antennas)

    our_delays_ns[0] = sampler.casa_delays[0] * 1e9  # Fixed
    for ant in range(1, n_antennas):
        our_delays_ns[ant] = np.mean(delays_free_flat[:, ant - 1]) * 1e9
        our_delays_std[ant] = np.std(delays_free_flat[:, ant - 1]) * 1e9

    # Print comparison table
    print(f"\n{'Ant':<5} {'Truth':<12} {'CASA':<12} {'Ours':<12} {'CASA-Truth':<12} {'Ours-Truth':<12}")
    print("-" * 70)

    casa_errors = []
    our_errors = []

    for ant in range(n_antennas):
        truth = truth_delays_ns[ant]
        casa = casa_delays_ns[ant]
        ours = our_delays_ns[ant]

        casa_diff = casa - truth
        our_diff = ours - truth

        if ant > 0:  # Skip reference antenna
            casa_errors.append(casa_diff)
            our_errors.append(our_diff)

        print(f"{ant:<5} {truth:>11.3f} {casa:>11.3f} {ours:>11.3f} {casa_diff:>11.3f} {our_diff:>11.3f}")

    # Summary statistics
    casa_errors = np.array(casa_errors)
    our_errors = np.array(our_errors)

    print(f"\n{'=' * 70}")
    print("ERROR STATISTICS (excluding reference antenna)")
    print(f"{'=' * 70}")

    print(f"\nCASA errors (ns):")
    print(f"  Mean: {np.mean(casa_errors):.4f}")
    print(f"  Std:  {np.std(casa_errors):.4f}")
    print(f"  RMS:  {np.sqrt(np.mean(casa_errors**2)):.4f}")
    print(f"  Max:  {np.max(np.abs(casa_errors)):.4f}")

    print(f"\nOur errors (ns):")
    print(f"  Mean: {np.mean(our_errors):.4f}")
    print(f"  Std:  {np.std(our_errors):.4f}")
    print(f"  RMS:  {np.sqrt(np.mean(our_errors**2)):.4f}")
    print(f"  Max:  {np.max(np.abs(our_errors)):.4f}")

    # Comparison
    casa_rms = np.sqrt(np.mean(casa_errors**2))
    our_rms = np.sqrt(np.mean(our_errors**2))

    print(f"\nComparison:")
    if our_rms < casa_rms:
        improvement = (1 - our_rms / casa_rms) * 100
        print(f"  ✓ Our solver is {improvement:.1f}% better than CASA (RMS)")
    else:
        degradation = (our_rms / casa_rms - 1) * 100
        print(f"  ✗ Our solver is {degradation:.1f}% worse than CASA (RMS)")

    return {
        'truth': truth_delays_ns,
        'casa': casa_delays_ns,
        'ours': our_delays_ns,
        'ours_std': our_delays_std,
        'casa_errors': casa_errors,
        'our_errors': our_errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end delay recovery validation"
    )
    parser.add_argument(
        '--msname',
        default='sim_3c286.ms',
        help='MS name (default: sim_3c286.ms)'
    )
    parser.add_argument(
        '--skip_sim',
        action='store_true',
        help='Skip simulation if MS exists'
    )
    parser.add_argument(
        '--n_channels',
        type=int,
        default=64,
        help='Number of channels (default: 64)'
    )
    parser.add_argument(
        '--delay_range',
        type=float,
        default=10.0,
        help='Delay range in ns (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--no_noise',
        action='store_true',
        help='Skip thermal noise for exact recovery test'
    )
    parser.add_argument(
        '--draws',
        type=int,
        default=1000,
        help='MCMC draws per chain (default: 1000)'
    )
    parser.add_argument(
        '--tune',
        type=int,
        default=1000,
        help='MCMC tuning steps (default: 1000)'
    )

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("DELAY RECOVERY VALIDATION PIPELINE")
    print(f"{'=' * 70}")

    # Step 1: Simulate (or use existing MS)
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

    # Step 2: Get antenna count and generate ground truth
    ms_handler = MeasurementSetHandler(args.msname)
    summary = ms_handler.get_observation_summary()
    n_antennas = summary["n_antennas"]
    ms_handler.close()

    truth_delays_ns, truth_delays_sec = generate_ground_truth_delays(
        n_antennas=n_antennas,
        delay_range_ns=args.delay_range,
        seed=args.seed,
    )

    print(f"\nGround truth delays generated:")
    print(f"  Range: ±{args.delay_range} ns")
    print(f"  Seed: {args.seed}")

    # Step 3: Corrupt MS with delays
    corrupt_ms_with_delays(
        ms_path=args.msname,
        delays_sec=truth_delays_sec,
        add_noise=not args.no_noise,
    )

    # Step 4: Run CASA gaincal
    caltable = args.msname.replace('.ms', '.K')
    caltable, casa_delays_ns = run_casa_gaincal(
        ms_path=args.msname,
        caltable=caltable,
        refant=0,
    )

    # Step 5: Run our solver
    trace_file = args.msname.replace('.ms', '_trace.nc')
    sampler = run_bayesian_solver(
        ms_path=args.msname,
        caltable=caltable,
        output_trace=trace_file,
        draws=args.draws,
        tune=args.tune,
    )

    # Step 6: Compare results
    results = compare_results(
        truth_delays_ns=truth_delays_ns,
        casa_delays_ns=casa_delays_ns,
        sampler=sampler,
    )

    print(f"\n{'=' * 70}")
    print("✓ VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"MS: {args.msname}")
    print(f"K-table: {caltable}")
    print(f"Trace: {trace_file}")
    print(f"\nNext steps:")
    print(f"  1. Run chain_plotter.py {trace_file} for diagnostic plots")
    print(f"  2. Add noise (remove --no_noise) to test degradation")
    print(f"  3. Increase channels for wideband test")

    return results


if __name__ == '__main__':
    results = main()
