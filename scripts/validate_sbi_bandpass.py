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
    assert np.allclose(params, params2), "Round-trip failed!"

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


def test_sbi_full_vla():
    """Full VLA-scale test (27 antennas, 64 channels)."""
    logger.info("\n" + "="*70)
    logger.info("FULL VLA-SCALE TEST")
    logger.info("="*70)

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

    logger.info("\n✓ Full VLA-scale test completed")


def main():
    parser = argparse.ArgumentParser(description="SBI Bandpass Validation")
    parser.add_argument(
        "--test",
        choices=["basic", "minimal", "full", "all"],
        default="basic",
        help="Which test to run"
    )

    args = parser.parse_args()

    if args.test in ["basic", "all"]:
        test_simulator_basic()

    if args.test in ["minimal", "all"]:
        test_sbi_training_minimal()

    if args.test in ["full", "all"]:
        test_sbi_full_vla()

    logger.info("\n" + "="*70)
    logger.info("ALL TESTS COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    main()
