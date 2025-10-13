#!/usr/bin/env python
"""Run gaincal comparison: AntSol vs CASA gaincal.

Example usage matching your gaincal command:
    python run_gaincal_comparison.py \\
        --ms 3c391_ctm_mosaic_10s_spw0.ms \\
        --field '0,1,9' \\
        --spw '0:27~36' \\
        --refant ea21 \\
        --calmode p \\
        --output antsol_solutions.npz \\
        --casa-table 3c391_ctm_mosaic_10s_spw0.G0all
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from jones_sim import MSCalibrator
    from jones_sim.casa_interface import CalibrationTableHandler
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure jones_sim is installed and casatools is available")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run AntSol gaincal and compare to CASA"
    )
    parser.add_argument("--ms", required=True, help="Measurement set path")
    parser.add_argument("--field", default="", help="Field selection (e.g., '0,1,9')")
    parser.add_argument("--spw", default="", help="SPW selection (e.g., '0:27~36')")
    parser.add_argument("--refant", default=0, help="Reference antenna (name or index)")
    parser.add_argument(
        "--calmode", default="p", choices=["p", "a", "ap"], help="Calibration mode"
    )
    parser.add_argument(
        "--solint", default="int", help="Solution interval (currently only 'int')"
    )
    parser.add_argument(
        "--minsnr", type=float, default=0.0, help="Minimum SNR threshold"
    )
    parser.add_argument(
        "--output", default="antsol_solutions.npz", help="Output file for solutions"
    )
    parser.add_argument(
        "--casa-table", help="CASA gaincal table for comparison (optional)"
    )
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU acceleration (requires CuPy)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("AntSol Gaincal")
    print("=" * 70)

    # Initialize calibrator
    print(f"\nOpening MS: {args.ms}")
    calibrator = MSCalibrator(args.ms, use_gpu=args.gpu)

    # Run gaincal
    print("\nRunning AntSol solver...")
    try:
        results = calibrator.gaincal(
            caltable=args.output,
            field=args.field,
            spw=args.spw,
            refant=args.refant,
            calmode=args.calmode,
            solint=args.solint,
            minsnr=args.minsnr,
        )
    finally:
        calibrator.close()

    # Print summary
    print("\n" + "=" * 70)
    print("Solution Summary")
    print("=" * 70)
    print(f"Solution intervals: {results['n_solutions']}")
    print(f"Antennas: {results['n_antennas']}")
    print(
        f"Flagged solutions: {results['n_flagged']}/{results['n_solutions'] * results['n_antennas'] * 2}"
    )

    # Check convergence
    n_converged = sum(
        1 for info in results["solutions"]["convergence"] if info["converged"]
    )
    print(f"Converged: {n_converged}/{results['n_solutions']} intervals")

    # Compare to CASA if provided
    if args.casa_table:
        print(f"\n{'='*70}")
        print("Comparison to CASA gaincal")
        print("=" * 70)
        print(f"Reading CASA table: {args.casa_table}")

        try:
            cal_handler = CalibrationTableHandler(args.casa_table)
            casa_sols = cal_handler.read_gain_solutions()

            print(
                f"CASA solutions: {len(casa_sols['unique_times'])} times, "
                f"{len(casa_sols['unique_antennas'])} antennas"
            )

            # Basic comparison
            compare_solutions(results["solutions"], casa_sols, args.calmode)

            if args.plot:
                plot_comparison(
                    results["solutions"],
                    casa_sols,
                    args.refant,
                    args.calmode,
                    args.output,
                )

        except Exception as e:
            print(f"Error comparing to CASA table: {e}")
            import traceback

            traceback.print_exc()

    print(f"\nSolutions saved to: {args.output}.npz")
    print("Done!")


def compare_solutions(our_sols, casa_sols, calmode):
    """Compare our solutions to CASA solutions.

    Args:
        our_sols: Our solutions dictionary
        casa_sols: CASA solutions from CalibrationTableHandler
        calmode: Calibration mode ('p', 'a', 'ap')
    """
    # Extract shapes
    n_times_ours = our_sols["gains_xx"].shape[0]
    n_ant_ours = our_sols["gains_xx"].shape[1]

    print(f"\nOur solutions shape: ({n_times_ours} times, {n_ant_ours} ants)")
    print(f"CASA gains shape: {casa_sols['gains'].shape}")

    # CASA gains are typically [npol, nchan, nrow]
    # We need to reshape/index to match our time-antenna structure
    # This requires understanding CASA's antenna and time indexing

    print("\nNote: Direct comparison requires matching CASA table row indexing")
    print("      to our time-antenna structure. This is non-trivial.")

    # For now, just show statistics
    our_phase_std_xx = np.std(
        np.angle(our_sols["gains_xx"][~our_sols["flags"][:, :, 0]])
    )
    our_phase_std_yy = np.std(
        np.angle(our_sols["gains_yy"][~our_sols["flags"][:, :, 1]])
    )

    casa_phase_std = np.std(np.angle(casa_sols["gains"][~casa_sols["flags"]]))

    print("\nPhase scatter:")
    print(f"  AntSol XX: {np.degrees(our_phase_std_xx):.2f} deg")
    print(f"  AntSol YY: {np.degrees(our_phase_std_yy):.2f} deg")
    print(f"  CASA:      {np.degrees(casa_phase_std):.2f} deg")


def plot_comparison(our_sols, casa_sols, refant, calmode, output_base):
    """Generate comparison plots.

    Args:
        our_sols: Our solutions
        casa_sols: CASA solutions
        refant: Reference antenna
        calmode: Calibration mode
        output_base: Base name for output files
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot our solutions
    times_rel = (our_sols["times"] - our_sols["times"][0]) / 60.0  # Minutes

    # XX phases
    ax = axes[0, 0]
    for ant in range(our_sols["gains_xx"].shape[1]):
        if ant == refant:
            continue
        phases = np.angle(our_sols["gains_xx"][:, ant])
        flags = our_sols["flags"][:, ant, 0]
        phases_masked = np.ma.masked_where(flags, phases)
        ax.plot(
            times_rel, np.degrees(phases_masked), "o-", label=f"Ant {ant}", alpha=0.7
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("XX Phase (deg)")
    ax.set_title("AntSol XX Phases")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    # YY phases
    ax = axes[0, 1]
    for ant in range(our_sols["gains_yy"].shape[1]):
        if ant == refant:
            continue
        phases = np.angle(our_sols["gains_yy"][:, ant])
        flags = our_sols["flags"][:, ant, 1]
        phases_masked = np.ma.masked_where(flags, phases)
        ax.plot(
            times_rel, np.degrees(phases_masked), "o-", label=f"Ant {ant}", alpha=0.7
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("YY Phase (deg)")
    ax.set_title("AntSol YY Phases")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    # Residuals per iteration (first interval as example)
    ax = axes[1, 0]
    if our_sols.get("convergence"):
        residuals = our_sols["convergence"][0]["residual_history"]
        ax.semilogy(residuals, "b-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMS Residual")
        ax.set_title("Convergence (first interval)")
        ax.grid(True, alpha=0.3)

    # Convergence statistics
    ax = axes[1, 1]
    if our_sols.get("convergence"):
        iterations = [info["iterations"] for info in our_sols["convergence"]]
        converged = [info["converged"] for info in our_sols["convergence"]]

        ax.hist(iterations, bins=20, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Iterations to converge")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Convergence histogram ({sum(converged)}/{len(converged)} converged)"
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_base.replace(".npz", "_plots.png")
    plt.savefig(plot_file, dpi=150)
    print(f"\nPlots saved to: {plot_file}")
    plt.close()


if __name__ == "__main__":
    main()
