"""Jones matrix simulator with GPU acceleration"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import jax.numpy as jnp
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None


class JonesSimulator:
    """Jones matrix coordinator with GPU acceleration."""

    def __init__(self):
        self.effects: Dict[str, Any] = {}
        self.effect_order = [
            "parallactic",
            "leakage",
            "gains",
            "bandpass",
            "delays",  # Alias for bandpass delays
            "rotation_measure",
            "rl_delay",
            "crosshand_phase",
        ]

    def add_effect(self, name: str, effect_instance):
        """Add an effect instance to the simulator."""
        self.effects[name] = effect_instance

    def compute_jones_matrix(
        self, freq: float, time: float, antenna_id: int
    ) -> np.ndarray:
        """Compute total Jones matrix by multiplying effects in order."""
        result = np.eye(2, dtype=complex)

        for effect_name in self.effect_order:
            if effect_name in self.effects:
                effect_matrix = self.effects[effect_name].jones_matrix(
                    freq, time, antenna_id
                )
                result = result @ effect_matrix

        return result

    def corrupt_visibilities(
        self,
        ideal_visibilities: np.ndarray,
        frequencies: np.ndarray,
        times: np.ndarray,
        antenna1_ids: np.ndarray,
        antenna2_ids: np.ndarray,
        use_gpu: bool = False,
        batch_gpu_size: int = 100000,
        noise_params: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Apply Jones corruption: corrupted = J1 @ ideal_matrix @ J2†

        Parameters
        ----------
        ideal_visibilities : np.ndarray
            Input visibilities to corrupt
        frequencies : np.ndarray
            Frequency for each visibility
        times : np.ndarray
            Time for each visibility
        antenna1_ids : np.ndarray
            First antenna ID for each baseline
        antenna2_ids : np.ndarray
            Second antenna ID for each baseline
        use_gpu : bool
            Use GPU acceleration if available
        batch_gpu_size : int
            Batch size for GPU processing
        noise_params : Optional[Dict]
            Dictionary containing noise parameters:
                - 'tsys': System temperature (K)
                - 'aperture_eff': Aperture efficiency (0-1)
                - 'antenna_diameter': Antenna diameter (meters)
                - 'bandwidth': Channel bandwidth per visibility (Hz) - array
                - 'int_time': Integration time per visibility (seconds) - array
                - 'seed': Optional random seed for reproducibility
            If None, no noise is added.

        Returns
        -------
        np.ndarray
            Corrupted visibilities with optional noise added
        """
        n_vis = len(ideal_visibilities)
        corrupted = np.zeros_like(ideal_visibilities)

        if use_gpu and not CUPY_AVAILABLE:
            warnings.warn("CuPy not available, falling back to CPU")
            use_gpu = False

        # Set random seed if provided
        if noise_params is not None and "seed" in noise_params:
            np.random.seed(noise_params["seed"])
            if use_gpu and CUPY_AVAILABLE:
                cp.random.seed(noise_params["seed"])

        if not use_gpu:
            for i in range(n_vis):
                freq = frequencies[i]
                time = times[i]
                ant1 = antenna1_ids[i]
                ant2 = antenna2_ids[i]

                J1 = self.compute_jones_matrix(freq, time, ant1)
                J2 = self.compute_jones_matrix(freq, time, ant2)

                ideal_matrix = ideal_visibilities[i].reshape(2, 2)
                corrupted_matrix = J1 @ ideal_matrix @ J2.conj().T
                corrupted[i] = corrupted_matrix.flatten()

            # Add noise if requested (after all corruptions)
            if noise_params is not None:
                corrupted = self._add_thermal_noise_cpu(corrupted, noise_params)

            return corrupted

        else:
            print(f"    [GPU] Processing {n_vis:,} visibilities")

            n_batches = (n_vis + batch_gpu_size - 1) // batch_gpu_size

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_gpu_size
                batch_end = min(batch_start + batch_gpu_size, n_vis)
                if (batch_idx + 1) % max(1, n_batches // 10) == 0:
                    print(f"    [GPU] Batch {batch_idx + 1}/{n_batches}...", flush=True)

                ideal_batch = ideal_visibilities[batch_start:batch_end]
                freq_batch = frequencies[batch_start:batch_end]
                time_batch = times[batch_start:batch_end]
                ant1_batch = antenna1_ids[batch_start:batch_end]
                ant2_batch = antenna2_ids[batch_start:batch_end]

                ideal_gpu = cp.asarray(ideal_batch, dtype=cp.complex128)
                freq_gpu = cp.asarray(freq_batch, dtype=cp.float64)
                ant1_gpu = cp.asarray(ant1_batch, dtype=cp.int32)
                ant2_gpu = cp.asarray(ant2_batch, dtype=cp.int32)
                time_gpu = cp.asarray(time_batch, dtype=cp.float64)

                corrupted_batch_gpu = self._corrupt_batch_vectorized_delays(
                    ideal_gpu, freq_gpu, time_gpu, ant1_gpu, ant2_gpu
                )

                # Add noise on GPU if requested (after corruption)
                if noise_params is not None:
                    # Extract batch-specific noise parameters
                    noise_params_batch = noise_params.copy()
                    if "bandwidth" in noise_params:
                        noise_params_batch["bandwidth"] = noise_params["bandwidth"][
                            batch_start:batch_end
                        ]
                    if "int_time" in noise_params:
                        noise_params_batch["int_time"] = noise_params["int_time"][
                            batch_start:batch_end
                        ]

                    corrupted_batch_gpu = self._add_thermal_noise_gpu(
                        corrupted_batch_gpu, noise_params_batch
                    )

                corrupted[batch_start:batch_end] = cp.asnumpy(corrupted_batch_gpu)

            return corrupted

    def _calculate_noise_std(
        self,
        tsys: float,
        aperture_eff: float,
        antenna_diameter: float,
        bandwidth: np.ndarray,
        int_time: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate thermal noise standard deviation in Jy using radiometer equation.

        Parameters
        ----------
        tsys : float
            System temperature (K)
        aperture_eff : float
            Aperture efficiency (0-1)
        antenna_diameter : float
            Antenna diameter (meters)
        bandwidth : np.ndarray
            Channel bandwidth per visibility (Hz)
        int_time : np.ndarray
            Integration time per visibility (seconds)

        Returns
        -------
        np.ndarray
            Noise standard deviation in Jy for each visibility
        """
        # Boltzmann constant (J/K)
        k_B = 1.380649e-23

        # Calculate geometric area
        A_geo = np.pi * (antenna_diameter / 2.0) ** 2  # m²

        # Calculate SEFD (Jy)
        # 1 Jy = 1e-26 W/m²/Hz
        SEFD = (2 * k_B * tsys) / (aperture_eff * A_geo) / 1e-26

        # Calculate noise per visibility using radiometer equation
        # σ = SEFD / sqrt(Δν × Δt)
        noise_std = SEFD / np.sqrt(bandwidth * int_time)

        return noise_std

    def _add_thermal_noise_cpu(
        self, visibilities: np.ndarray, noise_params: Dict
    ) -> np.ndarray:
        """
        Add thermal noise to visibilities on CPU.
        Uses σ_real = σ_imag = σ/√2 convention.

        Parameters
        ----------
        visibilities : np.ndarray
            Complex visibilities to add noise to
        noise_params : Dict
            Noise parameters (tsys, aperture_eff, antenna_diameter, bandwidth, int_time)

        Returns
        -------
        np.ndarray
            Visibilities with added thermal noise
        """
        # Calculate noise standard deviation per visibility
        noise_std = self._calculate_noise_std(
            tsys=noise_params["tsys"],
            aperture_eff=noise_params["aperture_eff"],
            antenna_diameter=noise_params["antenna_diameter"],
            bandwidth=noise_params["bandwidth"],
            int_time=noise_params["int_time"],
        )

        # For complex noise: σ_real = σ_imag = σ_total/√2
        # This ensures total variance = σ_total²
        component_std = noise_std / np.sqrt(2)

        # Generate Gaussian noise for each correlation
        # Shape: (n_vis, 4) for [XX, XY, YX, YY]
        n_vis, n_corr = visibilities.shape

        # Expand component_std to match shape (n_vis, n_corr)
        component_std_expanded = np.repeat(component_std[:, np.newaxis], n_corr, axis=1)

        # Generate noise
        noise_real = np.random.normal(0, component_std_expanded)
        noise_imag = np.random.normal(0, component_std_expanded)
        noise = noise_real + 1j * noise_imag

        return visibilities + noise

    def _add_thermal_noise_gpu(self, visibilities_gpu, noise_params: Dict):
        """
        Add thermal noise to visibilities on GPU.
        Uses σ_real = σ_imag = σ/√2 convention.

        Parameters
        ----------
        visibilities_gpu : cupy.ndarray
            Complex visibilities on GPU to add noise to
        noise_params : Dict
            Noise parameters (tsys, aperture_eff, antenna_diameter, bandwidth, int_time)

        Returns
        -------
        cupy.ndarray
            Visibilities with added thermal noise
        """
        # Calculate noise standard deviation per visibility (on CPU first)
        bandwidth_cpu = (
            cp.asnumpy(noise_params["bandwidth"])
            if isinstance(noise_params["bandwidth"], cp.ndarray)
            else noise_params["bandwidth"]
        )
        int_time_cpu = (
            cp.asnumpy(noise_params["int_time"])
            if isinstance(noise_params["int_time"], cp.ndarray)
            else noise_params["int_time"]
        )

        noise_std = self._calculate_noise_std(
            tsys=noise_params["tsys"],
            aperture_eff=noise_params["aperture_eff"],
            antenna_diameter=noise_params["antenna_diameter"],
            bandwidth=bandwidth_cpu,
            int_time=int_time_cpu,
        )

        # Transfer to GPU
        noise_std_gpu = cp.asarray(noise_std)

        # For complex noise: σ_real = σ_imag = σ_total/√2
        component_std_gpu = noise_std_gpu / cp.sqrt(2)

        # Generate Gaussian noise for each correlation
        n_vis, n_corr = visibilities_gpu.shape

        # Expand component_std to match shape
        component_std_expanded = cp.repeat(
            component_std_gpu[:, cp.newaxis], n_corr, axis=1
        )

        # Generate noise
        noise_real = cp.random.normal(0, component_std_expanded)
        noise_imag = cp.random.normal(0, component_std_expanded)
        noise = noise_real + 1j * noise_imag

        return visibilities_gpu + noise

    def _corrupt_batch_vectorized_delays(
        self, ideal_gpu, freq_gpu, time_gpu, ant1_gpu, ant2_gpu
    ):
        """Vectorized delay corruption on GPU.

        Matches CPU path behavior for BandpassDelay effect.
        Uses formula: phase = 2π * tau * (freq - ref_freq)

        For delay solving where ref_freq=0, this simplifies to: phase = 2π * tau * freq
        """
        batch_size = len(ideal_gpu)
        ideal_matrix_gpu = ideal_gpu.reshape(batch_size, 2, 2)

        if "delays" not in self.effects:
            return ideal_gpu

        delay_effect = self.effects["delays"]

        # Get delay arrays - handle scalar, array, or callable
        # For GPU path, we only support array (per-antenna delays)
        tau_xx = delay_effect.tau_xx
        tau_yy = delay_effect.tau_yy
        ref_freq = getattr(delay_effect, 'ref_freq', 0.0)

        # Convert to arrays if scalar
        if not isinstance(tau_xx, np.ndarray):
            n_ant = int(cp.max(cp.maximum(ant1_gpu, ant2_gpu)).get()) + 1
            tau_xx = np.full(n_ant, tau_xx)
        if not isinstance(tau_yy, np.ndarray):
            n_ant = int(cp.max(cp.maximum(ant1_gpu, ant2_gpu)).get()) + 1
            tau_yy = np.full(n_ant, tau_yy)

        tau_xx_gpu = cp.asarray(tau_xx, dtype=cp.float64)
        tau_yy_gpu = cp.asarray(tau_yy, dtype=cp.float64)

        # Get delays for each antenna in batch
        tau_xx_1 = tau_xx_gpu[ant1_gpu]
        tau_yy_1 = tau_yy_gpu[ant1_gpu]
        tau_xx_2 = tau_xx_gpu[ant2_gpu]
        tau_yy_2 = tau_yy_gpu[ant2_gpu]

        # Compute phases using same formula as BandpassDelay.jones_matrix()
        # phase = 2π * tau * (freq - ref_freq)
        two_pi = 2.0 * cp.pi
        freq_shifted = freq_gpu - ref_freq

        phase_xx_1 = two_pi * tau_xx_1 * freq_shifted
        phase_yy_1 = two_pi * tau_yy_1 * freq_shifted
        phase_xx_2 = two_pi * tau_xx_2 * freq_shifted
        phase_yy_2 = two_pi * tau_yy_2 * freq_shifted

        # Build J1 matrices
        J1_batch = cp.zeros((batch_size, 2, 2), dtype=cp.complex128)
        J1_batch[:, 0, 0] = cp.exp(1j * phase_xx_1)
        J1_batch[:, 1, 1] = cp.exp(1j * phase_yy_1)

        # Build J2 matrices
        J2_batch = cp.zeros((batch_size, 2, 2), dtype=cp.complex128)
        J2_batch[:, 0, 0] = cp.exp(1j * phase_xx_2)
        J2_batch[:, 1, 1] = cp.exp(1j * phase_yy_2)

        # Apply corruption: J1 @ ideal @ J2†
        temp = cp.einsum("bij,bjk->bik", J1_batch, ideal_matrix_gpu)
        J2_conj_T = J2_batch.conj().transpose(0, 2, 1)
        corrupted_matrix = cp.einsum("bij,bjk->bik", temp, J2_conj_T)

        return corrupted_matrix.reshape(batch_size, 4)

    def predict_visibilities(
        self,
        model_vis: np.ndarray,
        frequencies: np.ndarray,
        antenna1: np.ndarray,
        antenna2: np.ndarray,
        params: Dict[str, np.ndarray],
        use_jax: bool = True,
    ) -> np.ndarray:
        """Vectorized forward model for solving.

        Parameters
        ----------
        model_vis : np.ndarray
            Model visibilities, shape (n_row, n_chan, n_corr)
        frequencies : np.ndarray
            Channel frequencies in Hz, shape (n_chan,)
        antenna1, antenna2 : np.ndarray
            Antenna indices, shape (n_row,)
        params : Dict[str, np.ndarray]
            Calibration parameters: K, G, B, D
        use_jax : bool
            Use JAX (default True, needed for autodiff in solver)

        Returns
        -------
        np.ndarray
            Predicted visibilities
        """
        if use_jax and not JAX_AVAILABLE:
            warnings.warn("JAX not available, falling back to NumPy")
            use_jax = False

        xp = jnp if use_jax else np

        # Transfer all to arrays
        vis_pred = xp.asarray(model_vis, dtype=xp.complex128)
        freqs = xp.asarray(frequencies)
        ant1 = xp.asarray(antenna1, dtype=xp.int32)
        ant2 = xp.asarray(antenna2, dtype=xp.int32)

        n_row, n_chan, n_corr = vis_pred.shape

        # Apply effects in order: K → B → G → D
        for effect_name in ['K', 'B', 'G', 'D']:
            if effect_name not in params:
                continue

            p = xp.asarray(params[effect_name])

            if effect_name == 'K':
                # Match corruption: phase = 2π * (τ1 - τ2) * freq
                # ref_freq=0 to match _corrupt_batch_vectorized_delays
                delay_diff = p[ant1] - p[ant2]
                phase = 2 * xp.pi * xp.outer(delay_diff, freqs)
                vis_pred = vis_pred * xp.exp(1j * phase)[:, :, None]

            elif effect_name == 'G':
                g = p[0] if p.ndim == 3 else p
                g1, g2 = g[ant1], g[ant2]
                if n_corr == 4:
                    gf = xp.stack([g1[:,0]*xp.conj(g2[:,0]), g1[:,0]*xp.conj(g2[:,1]),
                                   g1[:,1]*xp.conj(g2[:,0]), g1[:,1]*xp.conj(g2[:,1])], axis=1)
                else:
                    gf = xp.stack([g1[:,0]*xp.conj(g2[:,0]), g1[:,1]*xp.conj(g2[:,1])], axis=1)
                vis_pred = vis_pred * gf[:, None, :]

            elif effect_name == 'B':
                bp1, bp2 = p[ant1], p[ant2]
                if n_corr == 4:
                    bf = xp.stack([bp1[:,:,0]*xp.conj(bp2[:,:,0]), bp1[:,:,0]*xp.conj(bp2[:,:,1]),
                                   bp1[:,:,1]*xp.conj(bp2[:,:,0]), bp1[:,:,1]*xp.conj(bp2[:,:,1])], axis=2)
                else:
                    bf = xp.stack([bp1[:,:,0]*xp.conj(bp2[:,:,0]), bp1[:,:,1]*xp.conj(bp2[:,:,1])], axis=2)
                vis_pred = vis_pred * bf

            elif effect_name == 'D' and n_corr == 4:
                d1, d2 = p[ant1], p[ant2]
                v = vis_pred
                vis_pred = xp.stack([
                    v[:,:,0] + d1[:,0:1]*v[:,:,2] + xp.conj(d2[:,0:1])*v[:,:,1],
                    v[:,:,1] + d1[:,0:1]*v[:,:,3] + xp.conj(d2[:,1:2])*v[:,:,0],
                    v[:,:,2] + d1[:,1:2]*v[:,:,0] + xp.conj(d2[:,0:1])*v[:,:,3],
                    v[:,:,3] + d1[:,1:2]*v[:,:,1] + xp.conj(d2[:,1:2])*v[:,:,2]
                ], axis=2)

        return np.asarray(vis_pred) if use_jax else vis_pred

    def list_effects(self) -> List[str]:
        """Get list of active effect names."""
        return list(self.effects.keys())

    def remove_effect(self, name: str):
        """Remove an effect from the simulator."""
        if name in self.effects:
            del self.effects[name]

    def clear_effects(self):
        """Remove all effects."""
        self.effects.clear()
