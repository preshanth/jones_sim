"""Tests for SBI calibration solver."""

import numpy as np
import pytest

# Test imports work
try:
    import torch
    import sbi
    HAS_SBI = True
except ImportError:
    HAS_SBI = False

if HAS_SBI:
    from jones_sim.sbi_solver import (
        BandpassSBISimulator,
        GainSBISimulator,
        SBICalibrationSolver,
    )
    from jones_sim.solvable_effects import BandpassEffect, GainEffect


@pytest.mark.skipif(not HAS_SBI, reason="SBI dependencies not installed")
class TestBandpassSBISimulator:
    """Test bandpass SBI simulator."""

    def test_initialization(self):
        """Test simulator can be initialized."""
        n_antennas = 4
        n_channels = 8

        # Simple visibility model: constant 1+0j
        def simple_vis_model(ant1, ant2, freqs, n_ant):
            n_baselines = len(ant1)
            n_chan = len(freqs)
            return np.ones((n_baselines, n_chan, 4), dtype=complex)

        effect = BandpassEffect()
        sim = BandpassSBISimulator(
            effect=effect,
            visibility_model=simple_vis_model,
            n_antennas=n_antennas,
            n_channels=n_channels,
            noise_std=0.01,
        )

        assert sim.n_antennas == n_antennas
        assert sim.n_channels == n_channels
        assert sim.get_param_dim() == (n_antennas - 1) * n_channels * 2 * 2

    def test_simulation(self):
        """Test that simulation produces correct output shape."""
        n_antennas = 4
        n_channels = 8

        def simple_vis_model(ant1, ant2, freqs, n_ant):
            n_baselines = len(ant1)
            n_chan = len(freqs)
            return np.ones((n_baselines, n_chan, 4), dtype=complex)

        effect = BandpassEffect()
        sim = BandpassSBISimulator(
            effect=effect,
            visibility_model=simple_vis_model,
            n_antennas=n_antennas,
            n_channels=n_channels,
        )

        # Sample from prior
        prior = sim.get_prior()
        params = prior.sample((1,)).numpy()[0]

        # Simulate
        obs = sim.simulate(params)

        # Check output shape
        expected_dim = sim.get_obs_dim()
        assert len(obs) == expected_dim
        assert np.all(np.isfinite(obs))

    def test_params_to_bandpass(self):
        """Test parameter conversion to bandpass array."""
        n_antennas = 4
        n_channels = 8

        def simple_vis_model(ant1, ant2, freqs, n_ant):
            n_baselines = len(ant1)
            n_chan = len(freqs)
            return np.ones((n_baselines, n_chan, 4), dtype=complex)

        effect = BandpassEffect()
        sim = BandpassSBISimulator(
            effect=effect,
            visibility_model=simple_vis_model,
            n_antennas=n_antennas,
            n_channels=n_channels,
        )

        # Sample params
        prior = sim.get_prior()
        params = prior.sample((1,)).numpy()[0]

        # Convert to bandpass
        bp = sim.params_to_bandpass(params)

        # Check shape
        assert bp.shape == (n_antennas, n_channels, 2)
        assert bp.dtype == complex

        # Check reference antenna is all 1s
        assert np.allclose(bp[0], 1.0)


@pytest.mark.skipif(not HAS_SBI, reason="SBI dependencies not installed")
class TestGainSBISimulator:
    """Test gain SBI simulator."""

    def test_initialization(self):
        """Test gain simulator initialization."""
        n_antennas = 4

        def simple_vis_model(ant1, ant2, freqs, n_ant):
            n_baselines = len(ant1)
            n_chan = len(freqs)
            return np.ones((n_baselines, n_chan, 4), dtype=complex)

        effect = GainEffect()
        sim = GainSBISimulator(
            effect=effect,
            visibility_model=simple_vis_model,
            n_antennas=n_antennas,
            n_channels=1,
        )

        assert sim.get_param_dim() == (n_antennas - 1) * 2 * 2

    def test_params_to_gains(self):
        """Test parameter conversion to gains."""
        n_antennas = 4

        def simple_vis_model(ant1, ant2, freqs, n_ant):
            n_baselines = len(ant1)
            n_chan = len(freqs)
            return np.ones((n_baselines, n_chan, 4), dtype=complex)

        effect = GainEffect()
        sim = GainSBISimulator(
            effect=effect,
            visibility_model=simple_vis_model,
            n_antennas=n_antennas,
        )

        prior = sim.get_prior()
        params = prior.sample((1,)).numpy()[0]

        gains = sim.params_to_gains(params)

        assert gains.shape == (n_antennas, 2)
        assert gains.dtype == complex
        assert np.allclose(gains[0], 1.0)  # Reference antenna


@pytest.mark.skipif(not HAS_SBI, reason="SBI dependencies not installed")
@pytest.mark.slow
class TestSBICalibrationSolver:
    """Test SBI calibration solver."""

    def test_training_simple(self):
        """Test training on simple gain problem."""
        n_antennas = 3  # Keep small for speed

        def simple_vis_model(ant1, ant2, freqs, n_ant):
            n_baselines = len(ant1)
            n_chan = len(freqs)
            return np.ones((n_baselines, n_chan, 4), dtype=complex)

        effect = GainEffect()
        simulator = GainSBISimulator(
            effect=effect,
            visibility_model=simple_vis_model,
            n_antennas=n_antennas,
            noise_std=0.05,
        )

        solver = SBICalibrationSolver(
            simulator=simulator,
            n_rounds=1,
            density_estimator="mdn",  # Fastest for testing
        )

        # Train with very few simulations for speed
        solver.train(n_simulations=100, show_progress_bars=False)

        assert solver.posterior is not None

    def test_inference_simple(self):
        """Test inference produces reasonable output."""
        n_antennas = 3

        def simple_vis_model(ant1, ant2, freqs, n_ant):
            n_baselines = len(ant1)
            n_chan = len(freqs)
            return np.ones((n_baselines, n_chan, 4), dtype=complex)

        effect = GainEffect()
        simulator = GainSBISimulator(
            effect=effect,
            visibility_model=simple_vis_model,
            n_antennas=n_antennas,
            noise_std=0.01,
        )

        solver = SBICalibrationSolver(
            simulator=simulator,
            n_rounds=1,
            density_estimator="mdn",
        )

        # Train
        solver.train(n_simulations=100, show_progress_bars=False)

        # Create fake observation
        true_params = simulator.get_prior().sample((1,)).numpy()[0]
        obs = simulator.simulate(true_params)

        # Infer
        samples, summary = solver.infer(obs, num_samples=100)

        # Check output format
        assert samples.shape[0] == 100
        assert samples.shape[1] == simulator.get_param_dim()

        assert "mean" in summary
        assert "std" in summary
        assert "credible_interval_68" in summary
        assert "credible_interval_95" in summary

        # Check shapes
        assert summary["mean"].shape == (simulator.get_param_dim(),)
        assert summary["std"].shape == (simulator.get_param_dim(),)
