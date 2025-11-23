#!/usr/bin/env python3
"""Demonstration of SBI-based bandpass calibration.

This script shows how to:
1. Set up an SBI simulator for bandpass calibration
2. Train a neural density estimator
3. Perform inference to recover bandpass with credible intervals
4. Visualize the results

Example output:
    Channel 0, Antenna 1, XX pol:
        True:      1.21 + 0.15j
        Inferred:  1.21 ± 0.05 (amplitude)
                   0.14 ± 0.03 (phase)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    HAS_SBI = True
except ImportError:
    HAS_SBI = False
    print("ERROR: SBI dependencies not installed.")
    print("Install with: pip install -e '.[sbi]'")
    exit(1)

from jones_sim.sbi_solver import BandpassSBISimulator, SBICalibrationSolver
from jones_sim.solvable_effects import BandpassEffect


def create_point_source_model(flux: float = 1.0):
    """Create a simple point source visibility model.

    Args:
        flux: Source flux in Jy

    Returns:
        Visibility model function
    """
    def vis_model(ant1, ant2, freqs, n_antennas):
        """Point source at phase center: constant visibility."""
        n_baselines = len(ant1)
        n_channels = len(freqs)
        n_pol = 4

        # Point source: all visibilities = flux
        vis = np.ones((n_baselines, n_channels, n_pol), dtype=complex) * flux

        return vis

    return vis_model


def main():
    parser = argparse.ArgumentParser(
        description="SBI Bandpass Calibration Demonstration"
    )
    parser.add_argument(
        "--n-antennas", type=int, default=4, help="Number of antennas"
    )
    parser.add_argument(
        "--n-channels", type=int, default=8, help="Number of frequency channels"
    )
    parser.add_argument(
        "--n-train", type=int, default=5000, help="Number of training simulations"
    )
    parser.add_argument(
        "--n-rounds", type=int, default=2, help="Number of SBI training rounds"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000, help="Number of posterior samples"
    )
    parser.add_argument(
        "--output-dir", type=str, default="sbi_output", help="Output directory"
    )
    parser.add_argument(
        "--load-posterior", type=str, help="Load pre-trained posterior from file"
    )
    parser.add_argument(
        "--save-posterior", type=str, help="Save trained posterior to file"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SBI Bandpass Calibration Demonstration")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Setup simulator
    print(f"\n[1/5] Setting up simulator")
    print(f"  Antennas:  {args.n_antennas}")
    print(f"  Channels:  {args.n_channels}")

    vis_model = create_point_source_model(flux=1.0)
    freqs = np.linspace(1.0e9, 2.0e9, args.n_channels)

    effect = BandpassEffect()
    simulator = BandpassSBISimulator(
        effect=effect,
        visibility_model=vis_model,
        n_antennas=args.n_antennas,
        n_channels=args.n_channels,
        freqs=freqs,
        noise_std=0.02,  # 2% noise
    )

    print(f"  Parameter dimensionality: {simulator.get_param_dim()}")
    print(f"  Observation dimensionality: {simulator.get_obs_dim()}")

    # Setup solver
    print(f"\n[2/5] Setting up SBI solver")
    solver = SBICalibrationSolver(
        simulator=simulator,
        n_rounds=args.n_rounds,
        density_estimator="maf",  # Masked autoregressive flow
        device="cpu",
    )

    # Train or load
    if args.load_posterior:
        print(f"\n[3/5] Loading pre-trained posterior from {args.load_posterior}")
        solver.load(args.load_posterior)
    else:
        print(f"\n[3/5] Training neural density estimator")
        print(f"  Simulations per round: {args.n_train}")
        print(f"  Number of rounds: {args.n_rounds}")
        print(f"  This may take a few minutes...\n")

        solver.train(
            n_simulations=args.n_train,
            training_batch_size=50,
            show_progress_bars=True,
        )

        if args.save_posterior:
            print(f"\nSaving posterior to {args.save_posterior}")
            solver.save(args.save_posterior)

    # Generate test observation with known bandpass
    print(f"\n[4/5] Generating test observation")

    # Create a known bandpass with specific features
    np.random.seed(42)
    true_params = simulator.get_prior().sample((1,)).numpy()[0]
    true_bandpass = simulator.params_to_bandpass(true_params)

    # Simulate observations
    observed_vis = simulator.simulate(true_params)

    print(f"  Generated {len(observed_vis)} visibility measurements")

    # Perform inference
    print(f"\n[5/5] Performing inference")
    print(f"  Sampling {args.n_samples} from posterior...")

    samples, summary = solver.infer(observed_vis, num_samples=args.n_samples)

    # Convert samples to bandpass arrays
    bandpass_samples = np.array([
        simulator.params_to_bandpass(samples[i])
        for i in range(min(1000, len(samples)))  # Use subset for speed
    ])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Print results for a few antennas/channels
    print("\nBandpass Amplitude Recovery (select channels):")
    print("-" * 60)

    for ant in range(1, min(3, args.n_antennas)):  # Skip ref antenna
        for chan in [0, args.n_channels // 2, args.n_channels - 1]:
            for pol in range(2):
                pol_name = ["XX", "YY"][pol]

                true_val = true_bandpass[ant, chan, pol]
                true_amp = np.abs(true_val)
                true_phase = np.angle(true_val)

                # Get samples for this element
                samples_amp = np.abs(bandpass_samples[:, ant, chan, pol])
                samples_phase = np.angle(bandpass_samples[:, ant, chan, pol])

                mean_amp = np.mean(samples_amp)
                std_amp = np.std(samples_amp)
                mean_phase = np.mean(samples_phase)
                std_phase = np.std(samples_phase)

                print(f"\nAnt {ant}, Chan {chan}, {pol_name}:")
                print(f"  Amplitude: {mean_amp:.3f} ± {std_amp:.3f} (true: {true_amp:.3f})")
                print(f"  Phase:     {mean_phase:.3f} ± {std_phase:.3f} (true: {true_phase:.3f})")

                # Check if recovered within 2-sigma
                within_bounds_amp = abs(true_amp - mean_amp) < 2 * std_amp
                within_bounds_phase = abs(true_phase - mean_phase) < 2 * std_phase

                status = "✓" if (within_bounds_amp and within_bounds_phase) else "✗"
                print(f"  Status: {status}")

    # Create visualization
    print(f"\n\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("SBI Bandpass Calibration Results", fontsize=14, fontweight="bold")

    # Plot 1: Bandpass amplitude vs channel for one antenna
    ax = axes[0, 0]
    ant_idx = 1
    pol_idx = 0

    true_amps = np.abs(true_bandpass[ant_idx, :, pol_idx])
    mean_amps = np.mean(np.abs(bandpass_samples[:, ant_idx, :, pol_idx]), axis=0)
    std_amps = np.std(np.abs(bandpass_samples[:, ant_idx, :, pol_idx]), axis=0)

    channels = np.arange(args.n_channels)
    ax.plot(channels, true_amps, 'k-', linewidth=2, label="True")
    ax.plot(channels, mean_amps, 'b-', linewidth=2, label="SBI Mean")
    ax.fill_between(
        channels,
        mean_amps - 2 * std_amps,
        mean_amps + 2 * std_amps,
        alpha=0.3,
        label="95% CI"
    )
    ax.set_xlabel("Channel")
    ax.set_ylabel("Bandpass Amplitude")
    ax.set_title(f"Amplitude Recovery (Ant {ant_idx}, XX pol)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Bandpass phase vs channel
    ax = axes[0, 1]
    true_phases = np.angle(true_bandpass[ant_idx, :, pol_idx])
    mean_phases = np.mean(np.angle(bandpass_samples[:, ant_idx, :, pol_idx]), axis=0)
    std_phases = np.std(np.angle(bandpass_samples[:, ant_idx, :, pol_idx]), axis=0)

    ax.plot(channels, true_phases, 'k-', linewidth=2, label="True")
    ax.plot(channels, mean_phases, 'r-', linewidth=2, label="SBI Mean")
    ax.fill_between(
        channels,
        mean_phases - 2 * std_phases,
        mean_phases + 2 * std_phases,
        alpha=0.3,
        label="95% CI"
    )
    ax.set_xlabel("Channel")
    ax.set_ylabel("Bandpass Phase (rad)")
    ax.set_title(f"Phase Recovery (Ant {ant_idx}, XX pol)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Posterior samples for one channel
    ax = axes[1, 0]
    chan_idx = args.n_channels // 2
    samples_real = np.real(bandpass_samples[:, ant_idx, chan_idx, pol_idx])
    samples_imag = np.imag(bandpass_samples[:, ant_idx, chan_idx, pol_idx])
    true_val = true_bandpass[ant_idx, chan_idx, pol_idx]

    ax.scatter(samples_real, samples_imag, alpha=0.1, s=1, color='blue', label='Posterior samples')
    ax.scatter([np.real(true_val)], [np.imag(true_val)], s=100, color='red',
               marker='*', label='True value', zorder=10)
    ax.scatter([np.mean(samples_real)], [np.mean(samples_imag)], s=100, color='green',
               marker='o', label='Posterior mean', zorder=10)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(f"Posterior Distribution (Ant {ant_idx}, Chan {chan_idx}, XX)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 4: Uncertainty (std) across channels
    ax = axes[1, 1]
    std_amps_all = np.std(np.abs(bandpass_samples[:, ant_idx, :, pol_idx]), axis=0)
    ax.plot(channels, std_amps_all, 'g-', linewidth=2, marker='o')
    ax.set_xlabel("Channel")
    ax.set_ylabel("Amplitude Uncertainty (std)")
    ax.set_title(f"Uncertainty Estimate (Ant {ant_idx}, XX pol)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / "sbi_bandpass_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"\nKey takeaway:")
    print(f"  SBI provides FULL POSTERIOR DISTRIBUTIONS")
    print(f"  Not just point estimates, but credible intervals!")
    print(f"  Example: Bandpass gain = {mean_amps[0]:.3f} ± {2*std_amps[0]:.3f} (95% CI)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
