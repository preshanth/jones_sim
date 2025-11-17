"""Jones matrix chain for radio interferometry corruption.

Implements the full Jones chain in linear (X/Y) polarization basis:
    J = G · B(ν) · R(ν) · C · D · P

Physical order (Sky → Correlator): P → D → C → R → B → G

PURE MATH - NO MS I/O
"""

import warnings
from typing import Dict, Optional

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class JonesChain:
    """Jones matrix chain - pure math, no MS I/O."""

    def __init__(
        self,
        config_params: Dict,
        n_antennas: int,
        ref_frequencies: Dict[int, float],
        use_gpu: bool = False,
    ):
        """Initialize Jones chain.

        Args:
            config_params: Generated parameters from ConfigParser
            n_antennas: Number of antennas
            ref_frequencies: Dict mapping SPW ID to reference frequency (Hz)
            use_gpu: Use GPU acceleration
        """
        self.config_params = config_params
        self.n_antennas = n_antennas
        self.ref_frequencies = ref_frequencies
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        if use_gpu and not CUPY_AVAILABLE:
            warnings.warn("GPU requested but CuPy not available. Using CPU.")
            self.use_gpu = False

        self.xp = cp if self.use_gpu else np

        # Get chain order and enabled effects
        self.chain_order = config_params.get("_chain_order", [])
        self.enabled_effects = config_params.get("_enabled_effects", [])

        print(f"\n{'=' * 70}")
        print("JONES CHAIN INITIALIZED")
        print(f"{'=' * 70}")
        print(f"Antennas: {n_antennas}")
        print(f"Chain order: {self.chain_order}")
        print(f"Enabled effects: {self.enabled_effects}")
        print(f"GPU: {self.use_gpu}")
        print(f"{'=' * 70}\n")

    def compute_jones_matrix(
        self,
        freq: float,
        time: float,
        antenna_id: int,
        spw_id: int,
        parallactic_angle: float = 0.0,
    ) -> np.ndarray:
        """Compute 2x2 Jones matrix for one antenna.

        Applies effects in reverse chain order (right to left):
        J = G · B(ν) · R(ν) · C · D · P

        Args:
            freq: Frequency in Hz
            time: Time in MJD seconds
            antenna_id: Antenna index
            spw_id: Spectral window ID
            parallactic_angle: Parallactic angle in radians (pre-computed)

        Returns:
            2x2 complex Jones matrix
        """
        J = np.eye(2, dtype=complex)

        # Apply effects in REVERSE order (right to left in equation)
        for effect_name in reversed(self.chain_order):
            if effect_name in self.enabled_effects:
                J = self._apply_effect(
                    effect_name, J, freq, time, antenna_id, spw_id, parallactic_angle
                )

        return J

    def _apply_effect(
        self,
        effect_name: str,
        J_current: np.ndarray,
        freq: float,
        time: float,
        antenna_id: int,
        spw_id: int,
        parallactic_angle: float,
    ) -> np.ndarray:
        """Apply one Jones effect: J_new = J_effect @ J_current

        Args:
            effect_name: Name of effect to apply
            J_current: Current Jones matrix (2x2)
            freq: Frequency in Hz
            time: Time in MJD seconds
            antenna_id: Antenna index
            spw_id: Spectral window ID
            parallactic_angle: Parallactic angle in radians

        Returns:
            Updated Jones matrix (2x2)
        """
        if effect_name == "gain":
            J_effect = self._compute_gain_matrix(antenna_id)

        elif effect_name == "bandpass":
            J_effect = self._compute_bandpass_matrix(antenna_id, freq, spw_id)

        elif effect_name == "xy_delay":
            J_effect = self._compute_xy_delay_matrix(antenna_id, freq, spw_id)

        elif effect_name == "crosshand_phase":
            J_effect = self._compute_crosshand_phase_matrix(antenna_id)

        elif effect_name == "leakage":
            J_effect = self._compute_leakage_matrix(antenna_id)

        elif effect_name == "parallactic":
            J_effect = self._compute_parallactic_matrix(parallactic_angle)

        else:
            raise ValueError(f"Unknown effect: {effect_name}")

        return J_effect @ J_current

    def _compute_gain_matrix(self, antenna_id: int) -> np.ndarray:
        """Compute G_i matrix: diag(A_X e^{iφ_X}, A_Y e^{iφ_Y})"""
        params = self.config_params["gain"]

        A_x = params["amplitude_x"][antenna_id]
        A_y = params["amplitude_y"][antenna_id]
        phi_x = params["phase_x"][antenna_id]
        phi_y = params["phase_y"][antenna_id]

        g_xx = A_x * np.exp(1j * phi_x)
        g_yy = A_y * np.exp(1j * phi_y)

        return np.array([[g_xx, 0], [0, g_yy]], dtype=complex)

    def _compute_bandpass_matrix(
        self, antenna_id: int, freq: float, spw_id: int
    ) -> np.ndarray:
        """Compute B_i(ν) matrix: diag(b_X(ν) e^{i2πτ_X(ν-ν_c)}, b_Y(ν) e^{i2πτ_Y(ν-ν_c)})

        Note: Requires channel frequencies to be stored in config_params
        """
        params = self.config_params["bandpass"]

        # Get reference frequency for this SPW
        ref_freq = self.ref_frequencies[spw_id]

        # Get delay parameters
        tau_x = params["tau_x"][antenna_id]
        tau_y = params["tau_y"][antenna_id]

        # Get amplitude response
        # Stored as dict: {spw_id: (n_antennas, n_channels)}
        amp_x_spw = params["amplitude_x"][spw_id][antenna_id]  # (n_channels,)
        amp_y_spw = params["amplitude_y"][spw_id][antenna_id]

        # Get channel frequencies for this SPW (stored in config_params)
        chan_freqs = self.config_params["_spw_frequencies"][spw_id]

        # Find nearest channel
        channel_idx = np.argmin(np.abs(chan_freqs - freq))

        b_x = amp_x_spw[channel_idx]
        b_y = amp_y_spw[channel_idx]

        # Compute frequency-dependent phase
        phase_x = 2 * np.pi * tau_x * (freq - ref_freq)
        phase_y = 2 * np.pi * tau_y * (freq - ref_freq)

        return np.array(
            [[b_x * np.exp(1j * phase_x), 0], [0, b_y * np.exp(1j * phase_y)]],
            dtype=complex,
        )

    def _compute_xy_delay_matrix(
        self, antenna_id: int, freq: float, spw_id: int
    ) -> np.ndarray:
        """Compute R_i(ν) matrix: [[cos(Δθ/2), i·sin(Δθ/2)], [-i·sin(Δθ/2), cos(Δθ/2)]]

        Where Δθ = 2π·Δτ·(ν - ν_c)
        """
        params = self.config_params["xy_delay"]

        delta_tau = params["delta_tau"][antenna_id]
        ref_freq = self.ref_frequencies[spw_id]

        # Compute frequency-dependent phase
        delta_theta = 2 * np.pi * delta_tau * (freq - ref_freq)

        cos_half = np.cos(delta_theta / 2)
        sin_half = np.sin(delta_theta / 2)

        return np.array(
            [[cos_half, 1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex
        )

    def _compute_crosshand_phase_matrix(self, antenna_id: int) -> np.ndarray:
        """Compute C_i matrix: diag(1, e^{iφ})"""
        params = self.config_params["crosshand_phase"]

        phi = params["phi"][antenna_id]

        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)

    def _compute_leakage_matrix(self, antenna_id: int) -> np.ndarray:
        """Compute D_i matrix: [[1, d_XY], [d_YX, 1]]"""
        params = self.config_params["leakage"]

        d_xy = params["d_xy"][antenna_id]
        d_yx = params["d_yx"][antenna_id]

        return np.array([[1, d_xy], [d_yx, 1]], dtype=complex)

    def _compute_parallactic_matrix(self, parallactic_angle: float) -> np.ndarray:
        """Compute P_i matrix: [[cos(ψ), sin(ψ)], [-sin(ψ), cos(ψ)]]

        Args:
            parallactic_angle: Pre-computed parallactic angle in radians
        """
        cos_psi = np.cos(parallactic_angle)
        sin_psi = np.sin(parallactic_angle)

        return np.array([[cos_psi, sin_psi], [-sin_psi, cos_psi]], dtype=complex)

    def corrupt_visibilities(
        self,
        ideal_visibilities: np.ndarray,
        frequencies: np.ndarray,
        times: np.ndarray,
        antenna1_ids: np.ndarray,
        antenna2_ids: np.ndarray,
        spw_ids: np.ndarray,
        parallactic_angles1: np.ndarray,
        parallactic_angles2: np.ndarray,
        use_gpu: bool = False,
        batch_gpu_size: int = 10000,
        noise_params: Optional[Dict] = None,
    ) -> np.ndarray:
        """Apply Jones corruption: V_corrupted = J1 @ V_ideal @ J2†

        Args:
            ideal_visibilities: (n_vis, 4) array [XX, XY, YX, YY]
            frequencies: (n_vis,) array of frequencies in Hz
            times: (n_vis,) array of times in MJD seconds
            antenna1_ids: (n_vis,) array of first antenna IDs
            antenna2_ids: (n_vis,) array of second antenna IDs
            spw_ids: (n_vis,) array of SPW IDs
            parallactic_angles1: (n_vis,) array of parallactic angles for antenna1 (radians)
            parallactic_angles2: (n_vis,) array of parallactic angles for antenna2 (radians)
            use_gpu: Use GPU acceleration
            batch_gpu_size: Batch size for GPU processing
            noise_params: Optional noise parameters dict

        Returns:
            (n_vis, 4) array of corrupted visibilities
        """
        # n_vis = len(ideal_visibilities)

        use_gpu = use_gpu and self.use_gpu

        if use_gpu:
            return self._corrupt_gpu(
                ideal_visibilities,
                frequencies,
                times,
                antenna1_ids,
                antenna2_ids,
                spw_ids,
                parallactic_angles1,
                parallactic_angles2,
                batch_gpu_size,
                noise_params,
            )
        else:
            return self._corrupt_cpu(
                ideal_visibilities,
                frequencies,
                times,
                antenna1_ids,
                antenna2_ids,
                spw_ids,
                parallactic_angles1,
                parallactic_angles2,
                noise_params,
            )

    def _corrupt_cpu(
        self,
        ideal_visibilities: np.ndarray,
        frequencies: np.ndarray,
        times: np.ndarray,
        antenna1_ids: np.ndarray,
        antenna2_ids: np.ndarray,
        spw_ids: np.ndarray,
        parallactic_angles1: np.ndarray,
        parallactic_angles2: np.ndarray,
        noise_params: Optional[Dict],
    ) -> np.ndarray:
        """CPU-based corruption (loop over visibilities)."""
        n_vis = len(ideal_visibilities)
        corrupted = np.zeros_like(ideal_visibilities)

        for i in range(n_vis):
            freq = frequencies[i]
            time = times[i]
            ant1 = antenna1_ids[i]
            ant2 = antenna2_ids[i]
            spw = spw_ids[i]
            psi1 = parallactic_angles1[i]
            psi2 = parallactic_angles2[i]

            # Compute Jones matrices
            J1 = self.compute_jones_matrix(freq, time, ant1, spw, psi1)
            J2 = self.compute_jones_matrix(freq, time, ant2, spw, psi2)

            # Apply: V_corrupted = J1 @ V_ideal @ J2†
            ideal_matrix = ideal_visibilities[i].reshape(2, 2)
            corrupted_matrix = J1 @ ideal_matrix @ J2.conj().T
            corrupted[i] = corrupted_matrix.flatten()

        # Add noise if requested
        if noise_params is not None:
            corrupted = self._add_thermal_noise_cpu(corrupted, noise_params)

        return corrupted

    def _corrupt_gpu(
        self,
        ideal_visibilities: np.ndarray,
        frequencies: np.ndarray,
        times: np.ndarray,
        antenna1_ids: np.ndarray,
        antenna2_ids: np.ndarray,
        spw_ids: np.ndarray,
        parallactic_angles1: np.ndarray,
        parallactic_angles2: np.ndarray,
        batch_size: int,
        noise_params: Optional[Dict],
    ) -> np.ndarray:
        """GPU-based corruption with batching."""
        n_vis = len(ideal_visibilities)
        corrupted = np.zeros_like(ideal_visibilities)

        n_batches = (n_vis + batch_size - 1) // batch_size

        print(f"    [GPU] Processing {n_vis:,} visibilities in {n_batches} batches")

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_vis)

            if (batch_idx + 1) % max(1, n_batches // 10) == 0:
                print(f"    [GPU] Batch {batch_idx + 1}/{n_batches}...", flush=True)

            # Extract batch
            ideal_batch = ideal_visibilities[batch_start:batch_end]
            freq_batch = frequencies[batch_start:batch_end]
            time_batch = times[batch_start:batch_end]
            ant1_batch = antenna1_ids[batch_start:batch_end]
            ant2_batch = antenna2_ids[batch_start:batch_end]
            spw_batch = spw_ids[batch_start:batch_end]
            psi1_batch = parallactic_angles1[batch_start:batch_end]
            psi2_batch = parallactic_angles2[batch_start:batch_end]

            # Corrupt batch on CPU (GPU vectorization would require pre-computing all Jones matrices)
            corrupted_batch = self._corrupt_cpu(
                ideal_batch,
                freq_batch,
                time_batch,
                ant1_batch,
                ant2_batch,
                spw_batch,
                psi1_batch,
                psi2_batch,
                None,  # Add noise at the end
            )

            corrupted[batch_start:batch_end] = corrupted_batch

        # Add noise if requested
        if noise_params is not None:
            corrupted = self._add_thermal_noise_cpu(corrupted, noise_params)

        return corrupted

    def _add_thermal_noise_cpu(
        self, visibilities: np.ndarray, noise_params: Dict
    ) -> np.ndarray:
        """Add thermal noise using radiometer equation."""
        # Calculate noise standard deviation
        k_B = 1.380649e-23  # Boltzmann constant

        tsys = noise_params["tsys"]
        aperture_eff = noise_params["aperture_eff"]
        antenna_diameter = noise_params["antenna_diameter"]
        bandwidth = noise_params["bandwidth"]  # (n_vis,)
        int_time = noise_params["int_time"]  # (n_vis,)

        # Calculate SEFD
        A_geo = np.pi * (antenna_diameter / 2.0) ** 2
        SEFD = (2 * k_B * tsys) / (aperture_eff * A_geo) / 1e-26

        # Calculate noise per visibility
        noise_std = SEFD / np.sqrt(bandwidth * int_time)

        # For complex noise: σ_real = σ_imag = σ_total/√2
        component_std = noise_std / np.sqrt(2)

        # Generate noise for each correlation
        n_vis, n_corr = visibilities.shape
        component_std_expanded = np.repeat(component_std[:, np.newaxis], n_corr, axis=1)

        if "seed" in noise_params:
            np.random.seed(noise_params["seed"])

        noise_real = np.random.normal(0, component_std_expanded)
        noise_imag = np.random.normal(0, component_std_expanded)
        noise = noise_real + 1j * noise_imag

        return visibilities + noise
