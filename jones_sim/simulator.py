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


class JonesSimulator:
    """Jones matrix coordinator with GPU acceleration."""

    def __init__(self):
        self.effects: Dict[str, Any] = {}
        self.effect_order = [
            "parallactic",
            "leakage",
            "gains",
            "bandpass",
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

                corrupted_batch_gpu = self._corrupt_batch_vectorized_general(
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

    def _corrupt_batch_vectorized_general(
        self, ideal_gpu, freq_gpu, time_gpu, ant1_gpu, ant2_gpu
    ):
        """GENERAL matrix form - FULLY VECTORIZED on GPU."""
        batch_size = len(ideal_gpu)

        ideal_matrix_gpu = ideal_gpu.reshape(batch_size, 2, 2)

        if "delays" not in self.effects:
            return ideal_gpu

        delay_effect = self.effects["delays"]
        tau_xx = delay_effect.tau_xx

        tau_xx_gpu = cp.asarray(tau_xx, dtype=cp.float64)

        tau_1_gpu = tau_xx_gpu[ant1_gpu]
        tau_2_gpu = tau_xx_gpu[ant2_gpu]

        two_pi = 2.0 * cp.pi
        phases_1 = two_pi * tau_1_gpu * freq_gpu
        phases_2 = two_pi * tau_2_gpu * freq_gpu

        exp_phases_1 = cp.exp(1j * phases_1)
        exp_phases_2 = cp.exp(1j * phases_2)

        J1_batch = cp.zeros((batch_size, 2, 2), dtype=cp.complex128)
        J1_batch[:, 0, 0] = exp_phases_1
        J1_batch[:, 1, 1] = exp_phases_1

        J2_batch = cp.zeros((batch_size, 2, 2), dtype=cp.complex128)
        J2_batch[:, 0, 0] = exp_phases_2
        J2_batch[:, 1, 1] = exp_phases_2

        temp = cp.einsum("bij,bjk->bik", J1_batch, ideal_matrix_gpu)

        J2_conj_T = J2_batch.conj().transpose(0, 2, 1)
        corrupted_matrix = cp.einsum("bij,bjk->bik", temp, J2_conj_T)

        corrupted_gpu = corrupted_matrix.reshape(batch_size, 4)

        return corrupted_gpu

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
