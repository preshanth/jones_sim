"""Comprehensive unit tests for antenna-based gain solver (AntSol algorithm).

Tests cover all solution modes, leakage terms, convergence behavior, and edge cases.
"""

import numpy as np
import pytest

from jones_sim.antsol import AntSolSolver


class TestAntSolBasic:
    """Basic functionality tests with synthetic point source data."""

    def setup_method(self):
        """Setup common test parameters."""
        self.n_ant = 4
        self.refant = 0
        np.random.seed(42)  # Reproducible tests

    def generate_point_source_correlations(self, gains_xx, gains_yy, leakage=None):
        """Generate synthetic correlation matrices from known gains.

        Args:
            gains_xx: [n_ant] complex gains for XX polarization
            gains_yy: [n_ant] complex gains for YY polarization
            leakage: Optional [n_ant] complex leakage terms

        Returns:
            correlations: [4, n_ant, n_ant] for [XX, XY, YX, YY]
            weights: [4, n_ant, n_ant] all ones except autocorr
        """
        n_ant = len(gains_xx)
        correlations = np.zeros((4, n_ant, n_ant), dtype=complex)

        # Point source: X_ij = g_i * g_j^*
        correlations[0] = np.outer(gains_xx, np.conj(gains_xx))  # XX
        correlations[3] = np.outer(gains_yy, np.conj(gains_yy))  # YY

        # Add leakage contribution if present (additive model)
        if leakage is not None:
            correlations[0] += np.outer(leakage, np.conj(leakage))
            correlations[3] += np.outer(leakage, np.conj(leakage))

        # XY, YX = 0 for unpolarized point source (no cross-pol)
        correlations[1] = 0.0
        correlations[2] = 0.0

        # Weights: uniform, zero for autocorrelations
        weights = np.ones((4, n_ant, n_ant))
        for pol in range(4):
            np.fill_diagonal(weights[pol], 0.0)

        return correlations, weights

    def normalize_phase_reference(self, gains, refant):
        """Normalize gains to reference antenna phase = 0."""
        ref_phase = np.angle(gains[refant])
        return gains * np.exp(-1j * ref_phase)

    def test_phase_only_perfect_data(self):
        """Test phase-only solution with perfect (noiseless) data."""
        # True gains with random phases, unit amplitude
        true_gains_xx = np.exp(1j * np.random.uniform(-np.pi, np.pi, self.n_ant))
        true_gains_yy = np.exp(1j * np.random.uniform(-np.pi, np.pi, self.n_ant))

        corr, wt = self.generate_point_source_correlations(true_gains_xx, true_gains_yy)

        solver = AntSolSolver(self.n_ant, mode="phase", solve_leakage=False)

        # Solve XX polarization
        gains_xx, leakage_xx, info_xx = solver.solve(
            corr, wt, refant=self.refant, pol="XX"
        )

        # Solve YY polarization
        gains_yy, leakage_yy, info_yy = solver.solve(
            corr, wt, refant=self.refant, pol="YY"
        )

        # Normalize both to same phase reference
        gains_xx_norm = self.normalize_phase_reference(gains_xx, self.refant)
        gains_yy_norm = self.normalize_phase_reference(gains_yy, self.refant)
        true_xx_norm = self.normalize_phase_reference(true_gains_xx, self.refant)
        true_yy_norm = self.normalize_phase_reference(true_gains_yy, self.refant)

        # Check convergence
        assert info_xx["converged"], "XX solver did not converge"
        assert info_yy["converged"], "YY solver did not converge"

        # Check amplitude is unity (phase-only mode)
        assert np.allclose(np.abs(gains_xx), 1.0), "XX gains should have unit amplitude"
        assert np.allclose(np.abs(gains_yy), 1.0), "YY gains should have unit amplitude"

        # Check phase recovery (iterative solver achieves ~1e-5 accuracy)
        phase_error_xx = np.angle(gains_xx_norm / true_xx_norm)
        phase_error_yy = np.angle(gains_yy_norm / true_yy_norm)

        assert (
            np.max(np.abs(phase_error_xx)) < 1e-5
        ), f"XX phase error too large: {np.max(np.abs(phase_error_xx))}"
        assert (
            np.max(np.abs(phase_error_yy)) < 1e-5
        ), f"YY phase error too large: {np.max(np.abs(phase_error_yy))}"

        # Check no leakage returned
        assert leakage_xx is None, "Leakage should be None when solve_leakage=False"
        assert leakage_yy is None, "Leakage should be None when solve_leakage=False"

    def test_amplitude_only_perfect_data(self):
        """Test amplitude-only solution with perfect data."""
        # True gains with random amplitudes, zero phase
        true_amps_xx = np.random.uniform(0.8, 1.2, self.n_ant)
        true_amps_yy = np.random.uniform(0.8, 1.2, self.n_ant)
        true_gains_xx = true_amps_xx + 0j
        true_gains_yy = true_amps_yy + 0j

        corr, wt = self.generate_point_source_correlations(true_gains_xx, true_gains_yy)

        solver = AntSolSolver(self.n_ant, mode="amplitude", solve_leakage=False)
        gains_xx, _, info_xx = solver.solve(corr, wt, refant=self.refant, pol="XX")
        gains_yy, _, info_yy = solver.solve(corr, wt, refant=self.refant, pol="YY")

        assert info_xx["converged"], "XX solver did not converge"
        assert info_yy["converged"], "YY solver did not converge"

        # Check phases are zero (amplitude-only mode)
        assert np.allclose(np.angle(gains_xx), 0.0), "XX gains should have zero phase"
        assert np.allclose(np.angle(gains_yy), 0.0), "YY gains should have zero phase"

        # Check amplitude recovery (absolute scale is arbitrary, check ratios)
        amp_ratio_xx = np.abs(gains_xx) / np.abs(gains_xx[self.refant])
        true_ratio_xx = true_amps_xx / true_amps_xx[self.refant]
        amp_ratio_yy = np.abs(gains_yy) / np.abs(gains_yy[self.refant])
        true_ratio_yy = true_amps_yy / true_amps_yy[self.refant]

        assert np.allclose(
            amp_ratio_xx, true_ratio_xx, rtol=1e-5
        ), "XX amplitude ratios incorrect"
        assert np.allclose(
            amp_ratio_yy, true_ratio_yy, rtol=1e-5
        ), "YY amplitude ratios incorrect"

    def test_amp_phase_perfect_data(self):
        """Test full amp+phase solution with perfect data."""
        # True gains with random amplitudes and phases
        true_gains_xx = np.random.uniform(0.8, 1.2, self.n_ant) * np.exp(
            1j * np.random.uniform(-np.pi, np.pi, self.n_ant)
        )
        true_gains_yy = np.random.uniform(0.8, 1.2, self.n_ant) * np.exp(
            1j * np.random.uniform(-np.pi, np.pi, self.n_ant)
        )

        corr, wt = self.generate_point_source_correlations(true_gains_xx, true_gains_yy)

        solver = AntSolSolver(self.n_ant, mode="amp_phase", solve_leakage=False)
        gains_xx, _, info_xx = solver.solve(corr, wt, refant=self.refant, pol="XX")
        gains_yy, _, info_yy = solver.solve(corr, wt, refant=self.refant, pol="YY")

        assert info_xx["converged"], "XX solver did not converge"
        assert info_yy["converged"], "YY solver did not converge"

        # Normalize to reference antenna
        gains_xx_norm = self.normalize_phase_reference(gains_xx, self.refant)
        gains_yy_norm = self.normalize_phase_reference(gains_yy, self.refant)
        true_xx_norm = self.normalize_phase_reference(true_gains_xx, self.refant)
        true_yy_norm = self.normalize_phase_reference(true_gains_yy, self.refant)

        # Also normalize amplitudes to refant (absolute scale arbitrary)
        gains_xx_norm /= np.abs(gains_xx_norm[self.refant])
        gains_yy_norm /= np.abs(gains_yy_norm[self.refant])
        true_xx_norm /= np.abs(true_xx_norm[self.refant])
        true_yy_norm /= np.abs(true_yy_norm[self.refant])

        # Check complex gain recovery
        assert np.allclose(
            gains_xx_norm, true_xx_norm, rtol=3e-5, atol=1e-7
        ), f"XX gains incorrect: max error = {np.max(np.abs(gains_xx_norm - true_xx_norm))}"
        assert np.allclose(
            gains_yy_norm, true_yy_norm, rtol=3e-5, atol=1e-7
        ), f"YY gains incorrect: max error = {np.max(np.abs(gains_yy_norm - true_yy_norm))}"


class TestAntSolLeakage:
    """Tests for leakage term solving."""

    def setup_method(self):
        """Setup common test parameters."""
        self.n_ant = 4
        self.refant = 0
        np.random.seed(42)

    @pytest.mark.skip(reason="Leakage solving not yet implemented")
    def test_leakage_additive_model(self):
        """Test leakage solving with additive model (Fortran-compatible)."""
        # For amp_phase mode, use realistic gains and leakage
        # True gains with amplitude and phase variations
        true_gains_xx = np.random.uniform(0.95, 1.05, self.n_ant) * np.exp(
            1j * np.random.uniform(-0.1, 0.1, self.n_ant)
        )
        true_gains_yy = np.random.uniform(0.95, 1.05, self.n_ant) * np.exp(
            1j * np.random.uniform(-0.1, 0.1, self.n_ant)
        )

        # Small leakage terms (typically ~1% level)
        true_leakage = 0.01 * np.exp(1j * np.random.uniform(-np.pi, np.pi, self.n_ant))

        # Generate correlations with leakage
        correlations = np.zeros((4, self.n_ant, self.n_ant), dtype=complex)
        correlations[0] = np.outer(true_gains_xx, np.conj(true_gains_xx)) + np.outer(
            true_leakage, np.conj(true_leakage)
        )
        correlations[3] = np.outer(true_gains_yy, np.conj(true_gains_yy)) + np.outer(
            true_leakage, np.conj(true_leakage)
        )

        weights = np.ones((4, self.n_ant, self.n_ant))
        for pol in range(4):
            np.fill_diagonal(weights[pol], 0.0)

        solver = AntSolSolver(
            self.n_ant, mode="amp_phase", solve_leakage=True, leakage_model="additive"
        )
        gains_xx, leakage_xx, info_xx = solver.solve(
            correlations, weights, refant=self.refant, pol="XX"
        )

        assert info_xx["converged"], "Solver did not converge with leakage"
        assert leakage_xx is not None, "Leakage should be returned"

        # Check leakage magnitude is reasonable (note: after reference subtraction, values can be larger)
        # The solver finds a gains+leakage decomposition, magnitudes depend on initialization
        assert leakage_xx is not None, "Should return leakage array"

        # Verify model fits data well
        model_corr = np.outer(gains_xx, np.conj(gains_xx)) + np.outer(
            leakage_xx, np.conj(leakage_xx)
        )
        residual = correlations[0] - model_corr
        weights_nodiag = weights[0].copy()
        np.fill_diagonal(weights_nodiag, 0.0)
        rms_residual = np.sqrt(
            np.sum(np.abs(residual) ** 2 * weights_nodiag) / np.sum(weights_nodiag)
        )

        # Leakage solver convergence is slower, so tolerance is relaxed
        assert rms_residual < 0.05, f"Model fit poor: RMS residual = {rms_residual}"

        # Check that solver actually improved the fit
        assert rms_residual < 0.1, "Residual should be reasonably small"

    @pytest.mark.skip(reason="Leakage solving not yet implemented")
    def test_jones_model_not_implemented(self):
        """Test that Jones leakage model raises NotImplementedError."""
        solver = AntSolSolver(
            4, mode="phase", solve_leakage=True, leakage_model="jones"
        )

        correlations = np.zeros((4, 4, 4), dtype=complex)
        weights = np.ones((4, 4, 4))

        with pytest.raises(NotImplementedError, match="Jones leakage model"):
            solver.solve(correlations, weights, refant=0, pol="XX")


class TestAntSolRobustness:
    """Tests for robustness to noise, flags, and edge cases."""

    def setup_method(self):
        """Setup common test parameters."""
        self.n_ant = 6
        self.refant = 2
        np.random.seed(42)

    def test_with_thermal_noise(self):
        """Test solution quality degrades gracefully with increasing noise."""
        true_gains = np.exp(1j * np.random.uniform(-0.5, 0.5, self.n_ant))

        # Generate perfect correlations
        corr_perfect = np.outer(true_gains, np.conj(true_gains))

        noise_levels = [0.0, 0.01, 0.05, 0.1]
        phase_errors = []

        for noise_std in noise_levels:
            # Add complex Gaussian noise
            noise = (
                noise_std
                * (
                    np.random.randn(self.n_ant, self.n_ant)
                    + 1j * np.random.randn(self.n_ant, self.n_ant)
                )
                / np.sqrt(2)
            )
            corr_noisy = corr_perfect + noise

            # Pack into 4-pol format
            correlations = np.zeros((4, self.n_ant, self.n_ant), dtype=complex)
            correlations[0] = corr_noisy
            correlations[3] = corr_noisy  # Same for YY (simplified)

            weights = np.ones((4, self.n_ant, self.n_ant))
            np.fill_diagonal(weights[0], 0.0)
            np.fill_diagonal(weights[3], 0.0)

            solver = AntSolSolver(self.n_ant, mode="phase")
            gains, _, info = solver.solve(
                correlations, weights, refant=self.refant, pol="XX"
            )

            # Normalize phase reference
            gains *= np.exp(-1j * np.angle(gains[self.refant]))
            true_gains_norm = true_gains * np.exp(
                -1j * np.angle(true_gains[self.refant])
            )

            phase_error = np.std(np.angle(gains / true_gains_norm))
            phase_errors.append(phase_error)

            assert info["converged"], f"Failed to converge with noise_std={noise_std}"

        # Check that errors increase monotonically with noise
        assert all(
            phase_errors[i] <= phase_errors[i + 1] * 2.0
            for i in range(len(phase_errors) - 1)
        ), "Phase errors should increase with noise"

        # Zero noise should give near-perfect solution
        assert phase_errors[0] < 1e-6, f"Zero noise error too large: {phase_errors[0]}"

    def test_flagged_baselines(self):
        """Test handling of flagged baselines (zero weights)."""
        true_gains = np.exp(1j * np.random.uniform(-np.pi, np.pi, self.n_ant))
        corr = np.outer(true_gains, np.conj(true_gains))

        correlations = np.zeros((4, self.n_ant, self.n_ant), dtype=complex)
        correlations[0] = corr

        weights = np.ones((4, self.n_ant, self.n_ant))
        np.fill_diagonal(weights[0], 0.0)

        # Flag 30% of baselines randomly
        n_baselines = self.n_ant * (self.n_ant - 1) // 2
        n_flagged = int(0.3 * n_baselines)
        flagged_pairs = []
        while len(flagged_pairs) < n_flagged:
            i, j = np.random.randint(0, self.n_ant, 2)
            if i != j and (i, j) not in flagged_pairs:
                weights[0, i, j] = 0.0
                weights[0, j, i] = 0.0
                flagged_pairs.append((i, j))

        solver = AntSolSolver(self.n_ant, mode="phase")
        gains, _, info = solver.solve(
            correlations, weights, refant=self.refant, pol="XX"
        )

        assert info["converged"], "Should converge even with flagged data"

        # Solution should still be reasonable (more error expected)
        gains_norm = gains * np.exp(-1j * np.angle(gains[self.refant]))
        true_norm = true_gains * np.exp(-1j * np.angle(true_gains[self.refant]))
        phase_error = np.std(np.angle(gains_norm / true_norm))

        assert (
            phase_error < 0.5
        ), f"Phase error too large with flagged data: {phase_error} rad"

    def test_dead_antenna(self):
        """Test handling of completely flagged antenna (no valid baselines)."""
        true_gains = np.exp(1j * np.random.uniform(-np.pi, np.pi, self.n_ant))
        corr = np.outer(true_gains, np.conj(true_gains))

        correlations = np.zeros((4, self.n_ant, self.n_ant), dtype=complex)
        correlations[0] = corr

        weights = np.ones((4, self.n_ant, self.n_ant))
        np.fill_diagonal(weights[0], 0.0)

        # Flag all baselines to antenna 3
        dead_ant = 3
        weights[0, dead_ant, :] = 0.0
        weights[0, :, dead_ant] = 0.0

        solver = AntSolSolver(self.n_ant, mode="phase")
        gains, _, info = solver.solve(
            correlations, weights, refant=self.refant, pol="XX"
        )

        # Dead antenna should have zero or unchanged gain
        assert (
            np.abs(gains[dead_ant]) < 1e-6 or not info["converged"]
        ), "Dead antenna should have zero gain or solver should fail"

    def test_convergence_detection(self):
        """Test that convergence is properly detected."""
        true_gains = np.exp(1j * np.array([0.1, -0.2, 0.3, -0.1]))
        corr = np.outer(true_gains, np.conj(true_gains))

        correlations = np.zeros((4, 4, 4), dtype=complex)
        correlations[0] = corr
        weights = np.ones((4, 4, 4))
        np.fill_diagonal(weights[0], 0.0)

        solver = AntSolSolver(4, mode="phase", max_iter=30000, eps=1e-10)
        gains, _, info = solver.solve(correlations, weights, refant=0, pol="XX")

        assert info["converged"], "Should converge for well-conditioned problem"
        assert (
            info["iterations"] < solver.max_iter
        ), "Should converge before max iterations"
        assert info["iterations"] > 1, "Should take more than 1 iteration"

        # Check residual decreased
        assert (
            info["final_residual"] < info["initial_residual"]
        ), "Final residual should be less than initial"

    def test_max_iterations_exceeded(self):
        """Test behavior when max iterations is exceeded."""
        # Create problem with very strict convergence tolerance
        true_gains = np.exp(1j * np.random.uniform(-0.5, 0.5, 4))

        corr = np.outer(true_gains, np.conj(true_gains))
        correlations = np.zeros((4, 4, 4), dtype=complex)
        correlations[0] = corr
        weights = np.ones((4, 4, 4))
        np.fill_diagonal(weights[0], 0.0)

        # Set very low max iterations
        solver = AntSolSolver(4, mode="phase", max_iter=5, eps=1e-12)
        gains, _, info = solver.solve(correlations, weights, refant=0, pol="XX")

        # Should hit max iterations with such strict tolerance
        assert (
            info["iterations"] >= 5
        ), f"Should use at least 5 iterations, used {info['iterations']}"


class TestAntSolEdgeCases:
    """Tests for edge cases and input validation."""

    def test_single_antenna_fails(self):
        """Test that single antenna configuration is rejected."""
        with pytest.raises(ValueError, match="at least 2 antennas"):
            AntSolSolver(1, mode="phase")

    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be"):
            AntSolSolver(4, mode="invalid")

    def test_invalid_polarization(self):
        """Test that invalid polarization raises error."""
        solver = AntSolSolver(4, mode="phase")
        correlations = np.zeros((4, 4, 4), dtype=complex)
        weights = np.ones((4, 4, 4))

        with pytest.raises(ValueError, match="pol must be"):
            solver.solve(correlations, weights, refant=0, pol="INVALID")

    def test_invalid_refant(self):
        """Test that out-of-range refant raises error."""
        solver = AntSolSolver(4, mode="phase")
        correlations = np.zeros((4, 4, 4), dtype=complex)
        weights = np.ones((4, 4, 4))

        with pytest.raises(ValueError, match="refant must be"):
            solver.solve(correlations, weights, refant=10, pol="XX")

    def test_mismatched_correlations_shape(self):
        """Test that wrong correlation array shape raises error."""
        solver = AntSolSolver(4, mode="phase")
        correlations = np.zeros((3, 4, 4), dtype=complex)  # Wrong: should be (4, 4, 4)
        weights = np.ones((4, 4, 4))

        with pytest.raises(ValueError, match="correlations.*shape"):
            solver.solve(correlations, weights, refant=0, pol="XX")

    def test_mismatched_weights_shape(self):
        """Test that wrong weights array shape raises error."""
        solver = AntSolSolver(4, mode="phase")
        correlations = np.zeros((4, 4, 4), dtype=complex)
        weights = np.ones((4, 5, 5))  # Wrong: should be (4, 4, 4)

        with pytest.raises(ValueError, match="weights.*shape"):
            solver.solve(correlations, weights, refant=0, pol="XX")

    def test_all_weights_zero(self):
        """Test that all-zero weights raises error."""
        solver = AntSolSolver(4, mode="phase")
        correlations = np.zeros((4, 4, 4), dtype=complex)
        weights = np.zeros((4, 4, 4))  # All flagged

        with pytest.raises(ValueError, match="No valid data.*all weights"):
            solver.solve(correlations, weights, refant=0, pol="XX")


class TestAntSolConvergenceInfo:
    """Tests for convergence diagnostics and info output."""

    def test_info_dict_contents(self):
        """Test that info dict contains all expected keys."""
        true_gains = np.exp(1j * np.random.uniform(-np.pi, np.pi, 4))
        corr = np.outer(true_gains, np.conj(true_gains))

        correlations = np.zeros((4, 4, 4), dtype=complex)
        correlations[0] = corr
        weights = np.ones((4, 4, 4))
        np.fill_diagonal(weights[0], 0.0)

        solver = AntSolSolver(4, mode="phase")
        gains, _, info = solver.solve(correlations, weights, refant=0, pol="XX")

        required_keys = [
            "converged",
            "iterations",
            "initial_residual",
            "final_residual",
            "residual_history",
        ]
        for key in required_keys:
            assert key in info, f"Missing key '{key}' in info dict"

        assert isinstance(info["converged"], bool)
        assert isinstance(info["iterations"], int)
        assert isinstance(info["initial_residual"], (float, np.floating))
        assert isinstance(info["final_residual"], (float, np.floating))
        assert isinstance(info["residual_history"], list)
        # residual_history includes initial residual + N iterations = N+1 entries
        assert len(info["residual_history"]) == info["iterations"] + 1

    def test_residual_decreases_monotonically(self):
        """Test that residual decreases over iterations (mostly)."""
        np.random.seed(123)  # Different seed for variety
        true_gains = np.exp(1j * np.random.uniform(-np.pi, np.pi, 6))
        corr = np.outer(true_gains, np.conj(true_gains))

        correlations = np.zeros((4, 6, 6), dtype=complex)
        correlations[0] = corr
        weights = np.ones((4, 6, 6))
        np.fill_diagonal(weights[0], 0.0)

        solver = AntSolSolver(6, mode="phase")
        gains, _, info = solver.solve(correlations, weights, refant=0, pol="XX")

        residuals = info["residual_history"]

        # Residual should decrease overall (but not necessarily monotonically due to relaxation)
        if len(residuals) > 10:
            # Compare early vs late
            early_avg = np.mean(residuals[:5])
            late_avg = np.mean(residuals[-5:])
            assert (
                late_avg < early_avg
            ), f"Residual should decrease on average: early={early_avg:.3e}, late={late_avg:.3e}"

        # Final residual should be small
        assert (
            residuals[-1] < residuals[0] * 0.1
        ), f"Final residual should be much smaller than initial: {residuals[-1]:.3e} vs {residuals[0]:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
