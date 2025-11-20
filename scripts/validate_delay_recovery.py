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

import argparse
import os
import sys

import numpy as np

# Add parent directory for development
sys.path.insert(0, "/home/pjaganna/Software/jones_sim")

from casatasks import gaincal
from casatools import table

from jones_sim import BandpassDelay, ElectronicGains, JonesSimulator
from jones_sim.calibration_solver import CalibrationSolver
from jones_sim.casa_interface import MeasurementSetHandler


def generate_ground_truth_delays(
    n_antennas: int, delay_range_ns: float, seed: int = 42
):
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


def generate_ground_truth_gains(
    n_antennas: int, amp_std: float = 0.1, phase_std: float = 0.1, seed: int = 43
):
    """Generate known ground truth gains.

    Args:
        n_antennas: Number of antennas
        amp_std: Standard deviation of amplitude variations around 1.0
        phase_std: Standard deviation of phase in radians
        seed: Random seed for reproducibility

    Returns:
        gains: Complex array (n_antennas, 2) for two polarizations
    """
    np.random.seed(seed)

    # Generate amplitude (log-normal around 1.0)
    amp = np.exp(np.random.normal(0, amp_std, (n_antennas, 2)))

    # Generate phase
    phase = np.random.normal(0, phase_std, (n_antennas, 2))

    # Combine
    gains = amp * np.exp(1j * phase)

    # Fix antenna 0 as reference
    gains[0, :] = 1.0 + 0j

    return gains


def corrupt_ms_with_effects(
    ms_path: str,
    delays_sec: np.ndarray = None,
    gains: np.ndarray = None,
    add_noise: bool = False,
    sefd: float = 420.0,
):
    """Corrupt MS DATA column with delays and/or gains.

    Args:
        ms_path: Path to measurement set
        delays_sec: Delay per antenna in seconds (optional)
        gains: Complex gains (n_antennas, 2) (optional)
        add_noise: Whether to add thermal noise
        sefd: System Equivalent Flux Density in Jy (VLA L-band ~420 Jy)
    """
    print(f"\n{'=' * 70}")
    print("CORRUPTING MS WITH EFFECTS")
    print(f"{'=' * 70}")

    # Open MS
    ms_handler = MeasurementSetHandler(ms_path)
    summary = ms_handler.get_observation_summary()

    n_antennas = summary["n_antennas"]
    n_spw = summary["n_spw"]

    print(f"Antennas: {n_antennas}")
    print(f"SPWs: {n_spw}")

    if delays_sec is not None:
        print("Delays applied (ns):")
        for ant in range(min(n_antennas, 5)):
            print(f"  Ant {ant}: {delays_sec[ant] * 1e9:.3f}")
        if n_antennas > 5:
            print(f"  ... ({n_antennas - 5} more)")

    if gains is not None:
        print("Gains applied (amp, phase in deg):")
        for ant in range(min(n_antennas, 5)):
            amp = np.abs(gains[ant, 0])
            phase = np.angle(gains[ant, 0]) * 180 / np.pi
            print(f"  Ant {ant}: {amp:.3f}, {phase:.1f}°")
        if n_antennas > 5:
            print(f"  ... ({n_antennas - 5} more)")

    # Read data
    tb = table()
    tb.open(ms_path, nomodify=False)

    model_data = tb.getcol("MODEL_DATA")  # Clean data
    antenna1 = tb.getcol("ANTENNA1")
    antenna2 = tb.getcol("ANTENNA2")

    n_corr, n_chan, n_row = model_data.shape
    print(f"\nData shape: {n_corr} corr × {n_chan} chan × {n_row} rows")

    # Get frequencies
    spw_info = summary["frequency_info"][0]
    freqs = spw_info["chan_freqs"]
    chan_width = spw_info.get(
        "chan_width", freqs[1] - freqs[0] if len(freqs) > 1 else 1e6
    )

    n_vis = n_chan * n_row
    print(f"Total visibilities: {n_vis:,}")

    # Create simulator and add effects
    sim = JonesSimulator()

    if delays_sec is not None:
        delay_effect = BandpassDelay(
            tau_xx=delays_sec,
            tau_yy=delays_sec,
            ref_freq=0.0,
        )
        sim.add_effect("delays", delay_effect)
        print("\nDelay effect added")

    if gains is not None:
        gain_effect = ElectronicGains(
            g_xx=gains[:, 0],
            g_yy=gains[:, 1],
        )
        sim.add_effect("gains", gain_effect)
        print("Gain effect added")

    # Reshape for simulator: (n_vis, 4)
    ideal_vis = model_data.transpose(2, 1, 0).reshape(-1, n_corr)  # (n_row*n_chan, 4)
    frequencies = np.tile(freqs, n_row)
    times = np.zeros(n_vis)
    ant1_arr = np.repeat(antenna1, n_chan)
    ant2_arr = np.repeat(antenna2, n_chan)

    # Corrupt using simulator
    print("Corrupting visibilities...", end=" ", flush=True)
    corrupted_vis = sim.corrupt_visibilities(
        ideal_vis, frequencies, times, ant1_arr, ant2_arr, use_gpu=False
    )
    print("Done")

    # Reshape back to (n_corr, n_chan, n_row)
    corrupted_data = corrupted_vis.reshape(n_row, n_chan, n_corr).transpose(2, 1, 0)

    # Add thermal noise if requested
    if add_noise:
        # Calculate integration time from MS
        times_col = tb.getcol("TIME")
        unique_times = np.unique(times_col)
        if len(unique_times) > 1:
            int_time = np.median(np.diff(unique_times))
        else:
            int_time = 2.0  # Default

        # Radiometer equation: sigma = SEFD / sqrt(bandwidth * int_time)
        # For visibility: sigma_vis = SEFD / sqrt(2 * bandwidth * int_time)
        # (factor of 2 for two antennas)
        sigma = sefd / np.sqrt(2 * chan_width * int_time)

        print("\nAdding thermal noise:")
        print(f"  SEFD: {sefd} Jy")
        print(f"  Channel width: {chan_width/1e6:.2f} MHz")
        print(f"  Int time: {int_time} s")
        print(f"  Expected sigma: {sigma:.4f} Jy per visibility")

        # Add complex Gaussian noise
        sigma_complex = sigma / np.sqrt(2)
        noise = np.random.normal(
            0, sigma_complex, corrupted_data.shape
        ) + 1j * np.random.normal(0, sigma_complex, corrupted_data.shape)
        corrupted_data = corrupted_data + noise
        print("  Noise added")
    else:
        print("\nNo thermal noise (exact recovery test)")

    # Write corrupted data
    tb.putcol("DATA", corrupted_data)
    tb.close()

    print("✓ DATA column corrupted with delays")


def run_casa_gaincal(
    ms_path: str, caltable: str = "delays.K", refant: int = 0, with_gains: bool = False
):
    """Run CASA gaincal to get K-table (and optionally G-table).

    Args:
        ms_path: Path to measurement set
        caltable: Output calibration table name
        refant: Reference antenna index
        with_gains: Also solve for G (gains)

    Returns:
        caltable, casa_delays_ns, (casa_gains if with_gains else None)
    """
    print(f"\n{'=' * 70}")
    print("RUNNING CASA GAINCAL")
    print(f"{'=' * 70}")

    # Remove existing table
    os.system(f"rm -rf {caltable}")

    # Run gaincal for delays
    gaincal(
        vis=ms_path,
        caltable=caltable,
        gaintype="K",  # Delay
        refant=str(refant),
        solint="inf",  # One solution for entire observation
        combine="scan",
        minsnr=3.0,
    )

    print(f"✓ K-table created: {caltable}")

    # Optionally solve for gains
    gtable = None
    if with_gains:
        gtable = caltable.replace(".K", ".G")
        os.system(f"rm -rf {gtable}")
        gaincal(
            vis=ms_path,
            caltable=gtable,
            gaintype="G",
            refant=str(refant),
            solint="inf",
            combine="scan",
            calmode="ap",
            gaintable=[caltable],  # Apply K first
        )
        print(f"✓ G-table created: {gtable}")

    # Read and print CASA delays
    tb = table()
    tb.open(caltable)
    fparam = tb.getcol("FPARAM")
    antennas = tb.getcol("ANTENNA1")
    flags = tb.getcol("FLAG")
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

    print("\nCASA delays (ns):")
    for ant in range(min(n_antennas, 10)):
        print(f"  Ant {ant}: {casa_delays_ns[ant]:.3f}")
    if n_antennas > 10:
        print(f"  ... ({n_antennas - 10} more)")

    # Read CASA gains if requested
    casa_gains = None
    if with_gains and gtable:
        tb.open(gtable)
        cparam = tb.getcol("CPARAM")  # Complex gains
        g_antennas = tb.getcol("ANTENNA1")
        tb.close()

        casa_gains = np.ones((n_antennas, 2), dtype=complex)
        for row in range(len(g_antennas)):
            ant = g_antennas[row]
            casa_gains[ant, 0] = cparam[0, 0, row]  # XX
            casa_gains[ant, 1] = cparam[1, 0, row]  # YY

        print("\nCASA gains (amp):")
        for ant in range(min(n_antennas, 5)):
            print(
                f"  Ant {ant}: XX={np.abs(casa_gains[ant,0]):.3f}, YY={np.abs(casa_gains[ant,1]):.3f}"
            )

    return caltable, casa_delays_ns, casa_gains


def run_bayesian_solver(
    ms_path: str,
    caltable: str,
    output_trace: str = "delay_trace.nc",
    draws: int = 1000,
    tune: int = 1000,
    use_map: bool = False,
    spw: str = "0",
    with_gains: bool = False,
):
    """Run our Bayesian calibration solver.

    Args:
        ms_path: Path to measurement set
        caltable: Path to CASA K-table (for priors)
        output_trace: Output trace filename
        draws: MCMC draws per chain
        tune: MCMC tuning steps
        use_map: Use MAP optimization instead of MCMC
        spw: SPW selection (e.g., '0', '0:32')
        with_gains: Also solve for G (gains)

    Returns:
        Solver object with results (last solver if sequential)
    """
    print(f"\n{'=' * 70}")
    print("RUNNING CALIBRATION SOLVER")
    print(f"{'=' * 70}")

    # Step 1: Solve K (delays)
    print("\n--- Solving K (delays) ---")
    solver_k = CalibrationSolver(ms_path)
    solver_k.load_data(spw=spw, solint="inf")
    solver_k.add_effect("K", solint="inf", prior_bound_ns=1.0)
    solver_k.load_casa_solutions(K=caltable)
    solver_k.build_model()

    if use_map:
        print("Using MAP optimization for K...")
        solver_k.optimize(num_steps=1000)
    else:
        print(f"Using MCMC sampling for K: {draws} draws, {tune} tune...")
        solver_k.sample(draws=draws, tune=tune, chains=2, target_accept=0.8)

    solver_k.print_summary()
    k_solution = solver_k.get_solution("K")

    if not with_gains:
        solver_k.save_trace(output_trace)
        # Save results for separate plotting
        from jones_sim.plotting import save_calibration_results

        save_calibration_results(
            output_trace.replace(".nc", "_plot_data.npz"),
            delays_casa=solver_k.effects["K"]["casa_values"],
            delays_recovered=k_solution,
            freqs=solver_k.frequencies,
        )
        return solver_k

    # Step 2: Solve G (gains) with K corrected
    print("\n--- Solving G (gains) with K applied ---")
    solver_g = CalibrationSolver(ms_path)
    solver_g.load_data(spw=spw, solint="inf")
    solver_g.apply_corrections(K=k_solution)  # Correct data with K solution

    gtable = caltable.replace(".K", ".G")
    solver_g.add_effect("G", solint="inf", calmode="ap", prior_std=0.3)
    solver_g.load_casa_solutions(G=gtable)
    solver_g.build_model()

    if use_map:
        print("Using MAP optimization for G...")
        solver_g.optimize(num_steps=1000)
    else:
        print(f"Using MCMC sampling for G: {draws} draws, {tune} tune...")
        solver_g.sample(draws=draws, tune=tune, chains=2, target_accept=0.8)

    solver_g.print_summary()
    solver_g.save_trace(output_trace.replace(".nc", "_G.nc"))

    # Save results for separate plotting
    from jones_sim.plotting import save_calibration_results

    g_solution = solver_g.get_solution("G")
    save_calibration_results(
        output_trace.replace(".nc", "_plot_data.npz"),
        delays_casa=solver_k.effects["K"]["casa_values"],
        delays_recovered=k_solution,
        gains_casa=solver_g.effects["G"]["casa_values"],
        gains_recovered=g_solution,
        freqs=solver_g.frequencies,
    )

    # Store K results in G solver for comparison
    solver_g.k_trace = solver_k.trace
    solver_g.effects["K"] = solver_k.effects["K"]

    return solver_g


def compare_results(
    truth_delays_ns: np.ndarray,
    casa_delays_ns: np.ndarray,
    solver: CalibrationSolver,
    truth_gains: np.ndarray = None,
    casa_gains: np.ndarray = None,
):
    """Compare ground truth, CASA, and our solver results.

    Args:
        truth_delays_ns: Ground truth delays in ns
        casa_delays_ns: CASA delays in ns
        solver: Our solver with results
        truth_gains: Ground truth gains (n_antennas, 2)
        casa_gains: CASA gains (n_antennas, 2)
    """
    print(f"\n{'=' * 70}")
    print("COMPARISON: TRUTH vs CASA vs OURS")
    print(f"{'=' * 70}")

    # Extract our posterior means
    # Handle both single solver and sequential solver cases
    if hasattr(solver, "k_trace"):
        # Sequential solve - K results stored separately
        delays_free_samples = solver.k_trace["delays_free"]
    else:
        delays_free_samples = solver.trace["delays_free"]

    # n_samples = delays_free_samples.shape[0]

    n_antennas = solver.n_antennas
    our_delays_ns = np.zeros(n_antennas)
    our_delays_std = np.zeros(n_antennas)

    casa_delays = solver.effects["K"]["casa_values"]
    our_delays_ns[0] = casa_delays[0] * 1e9  # Fixed
    for ant in range(1, n_antennas):
        our_delays_ns[ant] = np.mean(delays_free_samples[:, ant - 1]) * 1e9
        our_delays_std[ant] = np.std(delays_free_samples[:, ant - 1]) * 1e9

    # Print comparison table
    print(
        f"\n{'Ant':<5} {'Truth':<12} {'CASA':<12} {'Ours':<12} {'CASA-Truth':<12} {'Ours-Truth':<12}"
    )
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

        print(
            f"{ant:<5} {truth:>11.3f} {casa:>11.3f} {ours:>11.3f} {casa_diff:>11.3f} {our_diff:>11.3f}"
        )

    # Summary statistics
    casa_errors = np.array(casa_errors)
    our_errors = np.array(our_errors)

    print(f"\n{'=' * 70}")
    print("ERROR STATISTICS (excluding reference antenna)")
    print(f"{'=' * 70}")

    print("\nCASA errors (ns):")
    print(f"  Mean: {np.mean(casa_errors):.4f}")
    print(f"  Std:  {np.std(casa_errors):.4f}")
    print(f"  RMS:  {np.sqrt(np.mean(casa_errors**2)):.4f}")
    print(f"  Max:  {np.max(np.abs(casa_errors)):.4f}")

    print("\nOur errors (ns):")
    print(f"  Mean: {np.mean(our_errors):.4f}")
    print(f"  Std:  {np.std(our_errors):.4f}")
    print(f"  RMS:  {np.sqrt(np.mean(our_errors**2)):.4f}")
    print(f"  Max:  {np.max(np.abs(our_errors)):.4f}")

    # Comparison
    casa_rms = np.sqrt(np.mean(casa_errors**2))
    our_rms = np.sqrt(np.mean(our_errors**2))

    print("\nComparison:")
    if our_rms < casa_rms:
        improvement = (1 - our_rms / casa_rms) * 100
        print(f"  ✓ Our solver is {improvement:.1f}% better than CASA (RMS)")
    else:
        degradation = (our_rms / casa_rms - 1) * 100
        print(f"  ✗ Our solver is {degradation:.1f}% worse than CASA (RMS)")

    results = {
        "truth": truth_delays_ns,
        "casa": casa_delays_ns,
        "ours": our_delays_ns,
        "ours_std": our_delays_std,
        "casa_errors": casa_errors,
        "our_errors": our_errors,
    }

    # Compare gains if available
    if truth_gains is not None and casa_gains is not None and "G" in solver.effects:
        print(f"\n{'=' * 70}")
        print("GAIN COMPARISON")
        print(f"{'=' * 70}")

        # Extract our gains from trace
        n_antennas = solver.n_antennas
        our_gains = np.ones((n_antennas, 2), dtype=complex)

        # Get amplitude and phase samples
        if "gain_amp" in solver.trace and "gain_phase" in solver.trace:
            amp_samples = solver.trace["gain_amp"]  # (n_samples, n_ant-1, 2)
            phase_samples = solver.trace["gain_phase"]
            for ant in range(1, n_antennas):
                amp = np.mean(amp_samples[:, ant - 1, :], axis=0)
                phase = np.mean(phase_samples[:, ant - 1, :], axis=0)
                our_gains[ant, :] = amp * np.exp(1j * phase)
        else:
            print(
                f"  WARNING: gain_amp/gain_phase not found in trace. Keys: {list(solver.trace.keys())}"
            )

        print(
            f"\n{'Ant':<5} {'Truth XX':<12} {'CASA XX':<12} {'Ours XX':<12} {'Diff':<12}"
        )
        print("-" * 60)
        for ant in range(n_antennas):
            t_amp = np.abs(truth_gains[ant, 0])
            c_amp = np.abs(casa_gains[ant, 0])
            o_amp = np.abs(our_gains[ant, 0])
            diff = o_amp - t_amp
            print(
                f"{ant:<5} {t_amp:>11.3f} {c_amp:>11.3f} {o_amp:>11.3f} {diff:>11.3f}"
            )

        results["truth_gains"] = truth_gains
        results["casa_gains"] = casa_gains
        results["our_gains"] = our_gains

    return results


def main():
    parser = argparse.ArgumentParser(description="End-to-end delay recovery validation")
    parser.add_argument(
        "--msname", default="sim_3c286.ms", help="MS name (default: sim_3c286.ms)"
    )
    parser.add_argument(
        "--skip_sim", action="store_true", help="Skip simulation if MS exists"
    )
    parser.add_argument(
        "--n_channels", type=int, default=64, help="Number of channels (default: 64)"
    )
    parser.add_argument(
        "--delay_range",
        type=float,
        default=10.0,
        help="Delay range in ns (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no_noise",
        action="store_true",
        help="Skip thermal noise for exact recovery test",
    )
    parser.add_argument(
        "--draws", type=int, default=1000, help="MCMC draws per chain (default: 1000)"
    )
    parser.add_argument(
        "--tune", type=int, default=1000, help="MCMC tuning steps (default: 1000)"
    )
    parser.add_argument(
        "--map", action="store_true", help="Use MAP optimization instead of MCMC"
    )
    parser.add_argument(
        "--spw",
        default="0",
        help='SPW selection (e.g., "0", "0:32" for single channel)',
    )
    parser.add_argument(
        "--sefd",
        type=float,
        default=420.0,
        help="SEFD in Jy (default: 420 for VLA L-band)",
    )
    parser.add_argument(
        "--with_gains",
        action="store_true",
        help="Also corrupt with and solve for gains",
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

    print("\nGround truth delays generated:")
    print(f"  Range: ±{args.delay_range} ns")
    print(f"  Seed: {args.seed}")

    # Step 3: Generate ground truth gains (if requested)
    if args.with_gains:
        truth_gains = generate_ground_truth_gains(
            n_antennas=n_antennas,
            amp_std=0.3,  # Larger range: ~0.5 to 2.0
            phase_std=0.3,
            seed=args.seed + 1,
        )
        print("\nGround truth gains generated:")
        print("  Amp std: 0.3, Phase std: 0.3 rad")
    else:
        truth_gains = None

    # Step 4: Corrupt MS with effects
    corrupt_ms_with_effects(
        ms_path=args.msname,
        delays_sec=truth_delays_sec,
        gains=truth_gains,
        add_noise=not args.no_noise,
        sefd=args.sefd,
    )

    # Step 4: Run CASA gaincal
    caltable = args.msname.replace(".ms", ".K")
    caltable, casa_delays_ns, casa_gains = run_casa_gaincal(
        ms_path=args.msname,
        caltable=caltable,
        refant=0,
        with_gains=args.with_gains,
    )

    # Step 5: Run our solver
    trace_file = args.msname.replace(".ms", "_trace.nc")
    sampler = run_bayesian_solver(
        ms_path=args.msname,
        caltable=caltable,
        output_trace=trace_file,
        draws=args.draws,
        tune=args.tune,
        use_map=args.map,
        spw=args.spw,
        with_gains=args.with_gains,
    )

    # Step 6: Compare results
    results = compare_results(
        truth_delays_ns=truth_delays_ns,
        casa_delays_ns=casa_delays_ns,
        solver=sampler,
        truth_gains=truth_gains,
        casa_gains=casa_gains,
    )

    print(f"\n{'=' * 70}")
    print("✓ VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"MS: {args.msname}")
    print(f"K-table: {caltable}")
    print(f"Trace: {trace_file}")
    print("\nNext steps:")
    print(f"  1. Run chain_plotter.py {trace_file} for diagnostic plots")
    print("  2. Add noise (remove --no_noise) to test degradation")
    print("  3. Increase channels for wideband test")

    return results


if __name__ == "__main__":
    results = main()
