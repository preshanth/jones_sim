"""Jones matrix simulator with GPU acceleration"""

from typing import Any, Dict, List, Optional
import warnings
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
    ) -> np.ndarray:
        """Apply Jones corruption: corrupted = J1 @ ideal_matrix @ J2†"""
        n_vis = len(ideal_visibilities)
        corrupted = np.zeros_like(ideal_visibilities)

        if use_gpu and not CUPY_AVAILABLE:
            warnings.warn("CuPy not available, falling back to CPU")
            use_gpu = False

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

            return corrupted

        else:
            print(f"    [GPU] Processing {n_vis:,} visibilities")

            n_batches = (n_vis + batch_gpu_size - 1) // batch_gpu_size

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_gpu_size
                batch_end = min(batch_start + batch_gpu_size, n_vis)
                batch_size_actual = batch_end - batch_start

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

                corrupted[batch_start:batch_end] = cp.asnumpy(corrupted_batch_gpu)

            return corrupted

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
