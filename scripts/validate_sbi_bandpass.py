#!/usr/bin/env python3
"""Validation script for SBI bandpass calibration.

Compares SBI vs MCMC for bandpass recovery on simulated data.

Workflow:
1. Generate/use simulated 3C286 MS
2. Generate ground truth bandpass
3. Corrupt DATA with bandpass
4. Run MCMC (CalibrationSolver) for bandpass
5. Run SBI for bandpass
6. Compare: truth vs MCMC vs SBI

Usage:
    python validate_sbi_bandpass.py [options]
"""

import argparse
import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jones_sim.sbi_bandpass_solver import BandpassSBISimulator, SBIBandpassSolver

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def generate_ground_truth_bandpass(
    n_antennas: int,
    n_channels: int,
    amp_std: float = 0.2,
    phase_std: float = 0.3,
    seed: int = 42,
):
    """Generate realistic ground truth bandpass.

    Args:
        n_antennas: Number of antennas
        n_channels: Number of channels
        amp_std: Std of amplitude variations (log-space)
        phase_std: Std of phase variations (radians)
        seed: Random seed

    Returns:
        bandpass: Complex array (n_antennas, n_channels, 2)
    """
    np.random.seed(seed)

    # Generate per-channel variations
    # Future: Add smoothness/ripples here
    log_amp = np.random.normal(0, amp_std, (n_antennas, n_channels, 2))
    phase = np.random.normal(0, phase_std, (n_antennas, n_channels, 2))

    # Construct complex bandpass
    bandpass = np.exp(log_amp) * np.exp(1j * phase)

    # Fix reference antenna
    bandpass[0, :, :] = 1.0 + 0j

    logger.info(f"Generated ground truth bandpass:")
    logger.info(f"  Shape: {bandpass.shape}")
    logger.info(f"  Amp range: {np.min(np.abs(bandpass)):.3f} - {np.max(np.abs(bandpass)):.3f}")
    logger.info(f"  Phase range: {np.min(np.angle(bandpass)):.3f} - {np.max(np.angle(bandpass)):.3f}")

    return bandpass


def test_simulator_basic():
    """Basic test of BandpassSBISimulator."""
    logger.info("\n" + "="*70)
    logger.info("BASIC SIMULATOR TEST")
    logger.info("="*70)

    # Small test case
    sim = BandpassSBISimulator(
        n_antennas=4,
        n_channels=8,
        vla_band='L',
    )

    logger.info(f"Param dim: {sim.get_param_dim()}")
    logger.info(f"Obs dim: {sim.get_obs_dim()}")

    # Sample from prior
    prior = sim.get_prior()
    params = prior.sample((1,)).numpy()[0]

    logger.info(f"Sampled params shape: {params.shape}")

    # Simulate
    obs = sim.simulate(params)

    logger.info(f"Observation shape: {obs.shape}")
    logger.info(f"Observation stats: mean={np.mean(obs):.3f}, std={np.std(obs):.3f}")

    # Test round-trip
    bp = sim.unflatten_bandpass(params)
    logger.info(f"Bandpass shape: {bp.shape}")
    logger.info(f"Ref ant bandpass: {bp[0, 0, :]}")

    params2 = sim.flatten_bandpass(bp)

    # Test round-trip with phase wrapping awareness
    # The parameter vector is: [log_amp_xx, log_amp_yy, phase_xx, phase_yy]
    # For amplitudes, direct comparison is fine
    # For phases, need to account for ±π equivalence
    n_free = sim.n_antennas - 1
    n_params_per_pol = n_free * sim.n_channels

    # Split params into amplitude and phase sections
    params_amp = params[:2*n_params_per_pol]  # log_amp_xx and log_amp_yy
    params_phase = params[2*n_params_per_pol:]  # phase_xx and phase_yy

    params2_amp = params2[:2*n_params_per_pol]
    params2_phase = params2[2*n_params_per_pol:]

    # Check amplitudes (float32 precision from PyTorch prior)
    # For float32: machine epsilon ~1.2e-7, use 1e-6 for exp/log round-trip
    amp_match = np.allclose(params_amp, params2_amp, rtol=1e-6, atol=1e-7)

    # Check phases (account for 2π wrapping)
    # Two phases are equivalent if diff ≈ 0 or diff ≈ ±2π
    phase_diff = np.abs(params_phase - params2_phase)
    phase_diff_wrapped = np.minimum(phase_diff, 2*np.pi - phase_diff)
    # Use float32-appropriate tolerance
    phase_match = np.allclose(phase_diff_wrapped, 0, rtol=1e-6, atol=1e-7)

    if not (amp_match and phase_match):
        logger.error("Round-trip failed!")

        if not amp_match:
            amp_diff = np.abs(params_amp - params2_amp)
            max_amp_idx = np.argmax(amp_diff)
            logger.error(f"\n  Amplitude mismatch:")
            logger.error(f"    Max diff: {amp_diff[max_amp_idx]:.15e} at index {max_amp_idx}")
            logger.error(f"    params_amp[{max_amp_idx}] = {params_amp[max_amp_idx]:.15e}")
            logger.error(f"    params2_amp[{max_amp_idx}] = {params2_amp[max_amp_idx]:.15e}")

        if not phase_match:
            max_phase_idx = np.argmax(phase_diff_wrapped)
            logger.error(f"\n  Phase mismatch:")
            logger.error(f"    Max diff (wrapped): {phase_diff_wrapped[max_phase_idx]:.15e} at index {max_phase_idx}")
            logger.error(f"    params_phase[{max_phase_idx}] = {params_phase[max_phase_idx]:.15e}")
            logger.error(f"    params2_phase[{max_phase_idx}] = {params2_phase[max_phase_idx]:.15e}")
            logger.error(f"    Raw diff: {phase_diff[max_phase_idx]:.15e}")

        raise AssertionError("Round-trip failed!")

    logger.info("✓ Basic simulator test passed")


def test_sbi_training_minimal():
    """Minimal SBI training test with very few simulations."""
    logger.info("\n" + "="*70)
    logger.info("MINIMAL SBI TRAINING TEST")
    logger.info("="*70)

    # Very small for speed
    sim = BandpassSBISimulator(
        n_antennas=4,
        n_channels=8,
    )

    solver = SBIBandpassSolver(
        simulator=sim,
        n_rounds=1,
        density_estimator="mdn",  # Fastest
    )

    logger.info("Training with minimal simulations (100)...")
    solver.train(n_simulations=100, show_progress_bars=False)

    logger.info("✓ Minimal training completed")

    # Test inference
    logger.info("Testing inference...")
    true_params = sim.get_prior().sample((1,)).numpy()[0]
    obs = sim.simulate(true_params)

    samples, summary = solver.infer(obs, num_samples=100)

    logger.info(f"Posterior samples shape: {samples.shape}")
    logger.info(f"Posterior mean std: {np.std(summary['mean']):.3f}")

    logger.info("✓ Minimal inference test passed")


def plot_bandpass_comparison(
    frequencies: np.ndarray,
    true_bandpass: np.ndarray,
    sbi_bandpass: np.ndarray,
    sbi_std: np.ndarray,
    casa_bandpass: np.ndarray = None,
    antennas_to_plot: list = [1, 10, 20],
    output_path: str = "bandpass_comparison.png",
):
    """Plot bandpass comparison: Truth vs SBI (vs CASA if available).

    Args:
        frequencies: Frequency array in Hz
        true_bandpass: Ground truth bandpass (n_antennas, n_channels, 2)
        sbi_bandpass: SBI recovered bandpass (n_antennas, n_channels, 2)
        sbi_std: SBI standard deviation in log-space (n_antennas, n_channels, 2)
        casa_bandpass: CASA bandpass if available (n_antennas, n_channels, 2)
        antennas_to_plot: List of antenna indices to plot
        output_path: Where to save the figure
    """
    n_channels = len(frequencies)
    freq_ghz = frequencies / 1e9

    # Create figure with subplots for each antenna
    n_ants = len(antennas_to_plot)
    fig, axes = plt.subplots(n_ants, 4, figsize=(16, 4*n_ants))
    if n_ants == 1:
        axes = axes.reshape(1, -1)

    for idx, ant in enumerate(antennas_to_plot):
        for pol in [0, 1]:
            pol_name = ["XX", "YY"][pol]

            # Amplitude plot
            ax_amp = axes[idx, pol*2]

            # Truth
            true_amp = np.abs(true_bandpass[ant, :, pol])
            ax_amp.plot(freq_ghz, true_amp, 'k-', linewidth=2, label='Truth', alpha=0.7)

            # SBI with uncertainty
            sbi_amp = np.abs(sbi_bandpass[ant, :, pol])
            # Convert log-space std to amplitude uncertainty
            sbi_amp_err = sbi_amp * np.abs(sbi_std[ant, :, pol])
            ax_amp.plot(freq_ghz, sbi_amp, 'b-', linewidth=2, label='SBI')
            ax_amp.fill_between(
                freq_ghz,
                sbi_amp - sbi_amp_err,
                sbi_amp + sbi_amp_err,
                alpha=0.3,
                color='blue',
                label='SBI ±1σ'
            )

            # CASA if available
            if casa_bandpass is not None:
                casa_amp = np.abs(casa_bandpass[ant, :, pol])
                ax_amp.plot(freq_ghz, casa_amp, 'r--', linewidth=1.5, label='CASA', alpha=0.7)

            ax_amp.set_xlabel('Frequency (GHz)')
            ax_amp.set_ylabel('Amplitude')
            ax_amp.set_title(f'Ant {ant} {pol_name} - Amplitude')
            ax_amp.legend(loc='best', fontsize=8)
            ax_amp.grid(True, alpha=0.3)

            # Phase plot
            ax_phase = axes[idx, pol*2 + 1]

            # Truth
            true_phase = np.angle(true_bandpass[ant, :, pol])
            ax_phase.plot(freq_ghz, true_phase, 'k-', linewidth=2, label='Truth', alpha=0.7)

            # SBI
            sbi_phase = np.angle(sbi_bandpass[ant, :, pol])
            ax_phase.plot(freq_ghz, sbi_phase, 'b-', linewidth=2, label='SBI')

            # CASA if available
            if casa_bandpass is not None:
                casa_phase = np.angle(casa_bandpass[ant, :, pol])
                ax_phase.plot(freq_ghz, casa_phase, 'r--', linewidth=1.5, label='CASA', alpha=0.7)

            ax_phase.set_xlabel('Frequency (GHz)')
            ax_phase.set_ylabel('Phase (rad)')
            ax_phase.set_title(f'Ant {ant} {pol_name} - Phase')
            ax_phase.legend(loc='best', fontsize=8)
            ax_phase.grid(True, alpha=0.3)
            ax_phase.set_ylim(-np.pi, np.pi)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved bandpass comparison plot to {output_path}")
    plt.close()


def test_sbi_full_vla(resume: bool = False):
    """Full VLA-scale test (27 antennas, 64 channels).

    Args:
        resume: If True, resume from checkpoint if available
    """
    logger.info("\n" + "="*70)
    logger.info("FULL VLA-SCALE TEST")
    logger.info("="*70)

    # Checkpoint paths
    checkpoint_dir = "sbi_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    solver_path = os.path.join(checkpoint_dir, "sbi_solver_full.pkl")
    data_path = os.path.join(checkpoint_dir, "sbi_data_full.npz")

    # VLA D-config scale
    n_antennas = 27
    n_channels = 64

    sim = BandpassSBISimulator(
        n_antennas=n_antennas,
        n_channels=n_channels,
        vla_band='L',
    )

    logger.info(f"Full scale simulator:")
    logger.info(f"  Parameter dim: {sim.get_param_dim():,}")
    logger.info(f"  Observation dim: {sim.get_obs_dim():,}")

    # Check if we can resume
    can_resume = resume and os.path.exists(solver_path) and os.path.exists(data_path)

    if can_resume:
        logger.info("\n" + "="*70)
        logger.info("RESUMING FROM CHECKPOINT")
        logger.info("="*70)

        # Load saved data
        logger.info(f"Loading data from {data_path}...")
        data = np.load(data_path)
        true_bandpass = data['true_bandpass']
        obs = data['obs']

        logger.info(f"Loading trained solver from {solver_path}...")
        solver = SBIBandpassSolver(
            simulator=sim,
            n_rounds=2,
            density_estimator="maf",
        )
        solver.load(solver_path)

        logger.info("✓ Resumed from checkpoint successfully")

    else:
        if resume:
            logger.info("\n⚠ Checkpoint not found, training from scratch")

        logger.info("\n" + "="*70)
        logger.info("TRAINING FROM SCRATCH")
        logger.info("="*70)

        # Generate ground truth
        true_bandpass = generate_ground_truth_bandpass(
            n_antennas=n_antennas,
            n_channels=n_channels,
            amp_std=0.2,
            phase_std=0.3,
        )

        # Flatten to SBI format
        true_params = sim.flatten_bandpass(true_bandpass)

        # Simulate observations
        logger.info("Simulating observations...")
        obs = sim.simulate(true_params)

        logger.info(f"Observation shape: {obs.shape}")
        logger.info(f"Observation range: {np.min(obs):.3f} - {np.max(obs):.3f}")

        # Train SBI (with moderate number of simulations)
        solver = SBIBandpassSolver(
            simulator=sim,
            n_rounds=2,
            density_estimator="maf",
        )

        logger.info("\nTraining SBI (this will take a few minutes)...")
        logger.info("  Using 5000 simulations for speed (use 10k+ for production)")
        solver.train(n_simulations=5000, show_progress_bars=True)

        # Save checkpoint
        logger.info("\n" + "="*70)
        logger.info("SAVING CHECKPOINT")
        logger.info("="*70)

        solver.save(solver_path)
        np.savez(
            data_path,
            true_bandpass=true_bandpass,
            obs=obs,
        )

        logger.info(f"✓ Saved solver to {solver_path}")
        logger.info(f"✓ Saved data to {data_path}")

    # Infer
    logger.info("\nPerforming inference...")
    samples, summary = solver.infer(obs, num_samples=1000)

    # Convert summary back to bandpass for comparison
    mean_bandpass = sim.unflatten_bandpass(summary['mean'])
    std_bandpass_log = sim.unflatten_bandpass(summary['std'])

    # Compare to truth
    logger.info("\n" + "="*70)
    logger.info("RESULTS COMPARISON")
    logger.info("="*70)

    # Select a few channels and antennas for detailed comparison
    test_ants = [1, 10, 20]
    test_chans = [0, n_channels//2, n_channels-1]

    for ant in test_ants:
        for chan in test_chans:
            for pol in [0, 1]:
                pol_name = ["XX", "YY"][pol]

                true_val = true_bandpass[ant, chan, pol]
                mean_val = mean_bandpass[ant, chan, pol]

                true_amp = np.abs(true_val)
                true_phase = np.angle(true_val)

                mean_amp = np.abs(mean_val)
                mean_phase = np.angle(mean_val)

                # Rough uncertainty from std
                std_amp = np.abs(std_bandpass_log[ant, chan, pol])

                amp_error = abs(mean_amp - true_amp)
                phase_error = abs(mean_phase - true_phase)

                logger.info(f"\nAnt {ant}, Chan {chan}, {pol_name}:")
                logger.info(f"  Amp: true={true_amp:.3f}, SBI={mean_amp:.3f} ± {std_amp:.3f}")
                logger.info(f"  Phase: true={true_phase:.3f}, SBI={mean_phase:.3f}")
                logger.info(f"  Errors: amp={amp_error:.3f}, phase={phase_error:.3f}")

                status = "✓" if (amp_error < 2*std_amp) else "✗"
                logger.info(f"  Status: {status}")

    # Generate comparison plots
    logger.info("\n" + "="*70)
    logger.info("GENERATING PLOTS")
    logger.info("="*70)

    plot_bandpass_comparison(
        frequencies=sim.frequencies,
        true_bandpass=true_bandpass,
        sbi_bandpass=mean_bandpass,
        sbi_std=std_bandpass_log,
        casa_bandpass=None,  # TODO: Add CASA comparison
        antennas_to_plot=test_ants,
        output_path="sbi_bandpass_comparison.png",
    )

    logger.info("\n✓ Full VLA-scale test completed")


def main():
    parser = argparse.ArgumentParser(description="SBI Bandpass Validation")
    parser.add_argument(
        "--test",
        choices=["basic", "minimal", "full", "all"],
        default="basic",
        help="Which test to run"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available (for 'full' test)"
    )

    args = parser.parse_args()

    if args.test in ["basic", "all"]:
        test_simulator_basic()

    if args.test in ["minimal", "all"]:
        test_sbi_training_minimal()

    if args.test in ["full", "all"]:
        test_sbi_full_vla(resume=args.resume)

    logger.info("\n" + "="*70)
    logger.info("ALL TESTS COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    main()
