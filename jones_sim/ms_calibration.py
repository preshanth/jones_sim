"""MS-based calibration utilities for running AntSol on real data.

Provides helpers to extract correlation matrices from Measurement Sets
and solve for gains in a gaincal-compatible way.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .antsol import AntSolSolver

try:
    from .casa_interface import CalibrationTableHandler, MeasurementSetHandler

    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False


class MSCalibrator:
    """Calibration solver that operates on Measurement Sets.

    Mimics CASA's gaincal functionality but using the AntSol algorithm.
    """

    def __init__(self, ms_path: str, use_gpu: bool = False):
        """Initialize MS calibrator.

        Args:
            ms_path: Path to measurement set
            use_gpu: Use GPU acceleration if available (requires CuPy)

        Raises:
            ImportError: If CASA tools not available
        """
        if not CASA_AVAILABLE:
            raise ImportError(
                "CASA tools required for MSCalibrator. "
                "Install casatools: pip install casatools"
            )

        self.ms_path = ms_path
        self.ms_handler = MeasurementSetHandler(ms_path)
        self.use_gpu = use_gpu

    def gaincal(
        self,
        caltable: str,
        field: Union[str, int, List] = "",
        spw: str = "",
        refant: Union[str, int] = 0,
        calmode: str = "p",
        solint: str = "int",
        combine: str = "",
        minsnr: float = 0.0,
        append: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Solve for antenna gains (mimics CASA gaincal).

        Args:
            caltable: Output calibration table path
            field: Field selection (name, ID, or list)
            spw: Spectral window and channel selection (e.g., '0:27~36')
            refant: Reference antenna (name or ID)
            calmode: Calibration mode - 'p' (phase), 'a' (amplitude), 'ap' (both)
            solint: Solution interval - 'int' (per integration), 'inf' (per scan), or time in seconds
            combine: Axes to combine ('scan', 'spw', etc.) - not yet implemented
            minsnr: Minimum SNR threshold for valid solutions
            append: Append to existing table (not yet implemented)

        Returns:
            Dictionary with solution statistics and diagnostics

        Raises:
            NotImplementedError: For unsupported gaincal features
        """
        # Parse parameters
        mode_map = {"p": "phase", "a": "amplitude", "ap": "amp_phase"}
        if calmode not in mode_map:
            raise ValueError(f"calmode must be 'p', 'a', or 'ap', got '{calmode}'")
        solver_mode = mode_map[calmode]

        # Parse spectral window selection
        spw_id, chan_sel = self._parse_spw(spw)

        # Convert refant name to index if needed
        refant_idx = self._parse_refant(refant)

        # Read visibility data
        print(f"Reading MS: {self.ms_path}")
        print(f"  Field: {field}, SPW: {spw}, RefAnt: {refant} (idx={refant_idx})")

        vis_data = self.ms_handler.read_visibilities(field=field, spw=spw_id)

        # Extract correlation matrices per solution interval
        print(f"Extracting correlations with solint={solint}")
        corr_blocks, metadata = self._extract_correlation_blocks(
            vis_data, solint, chan_sel, minsnr
        )

        print(f"Found {len(corr_blocks)} solution intervals")

        # Solve for gains per interval
        n_ant = metadata["n_antennas"]
        n_intervals = len(corr_blocks)

        # Storage for solutions
        solutions = {
            "gains_xx": np.zeros((n_intervals, n_ant), dtype=complex),
            "gains_yy": np.zeros((n_intervals, n_ant), dtype=complex),
            "flags": np.zeros((n_intervals, n_ant, 2), dtype=bool),  # [time, ant, pol]
            "times": np.zeros(n_intervals),
            "convergence": [],
        }

        # Solve per time interval (use GPU if requested)
        solver = AntSolSolver(
            n_ant, mode=solver_mode, solve_leakage=False, use_gpu=self.use_gpu
        )

        if self.use_gpu:
            from .antsol import CUPY_AVAILABLE

            if CUPY_AVAILABLE:
                print(f"  Using GPU acceleration (CuPy)")
            else:
                print(f"  GPU requested but CuPy not available, using CPU")

        for i, (correlations, weights, time_stamp) in enumerate(corr_blocks):
            solutions["times"][i] = time_stamp

            print(f"  Interval {i+1}/{n_intervals}: t={time_stamp:.1f}")

            # Debug first interval
            if i == 0:
                corr_rms = np.sqrt(np.mean(np.abs(correlations[0]) ** 2))
                weight_stats = weights[weights > 0]
                n_nonzero_corr = np.sum(np.abs(correlations[0]) > 0)

                # Check Hermitian property: C[i,j] should equal conj(C[j,i])
                is_hermitian = np.allclose(
                    correlations[0], correlations[0].conj().T, rtol=1e-5
                )

                # Check a few off-diagonal elements
                sample_vals = [correlations[0, 0, 1], correlations[0, 1, 0]]

                print(
                    f"    DEBUG: corr_rms={corr_rms:.2e}, n_nonzero={n_nonzero_corr}/{26*26}"
                )
                print(
                    f"    DEBUG: hermitian={is_hermitian}, C[0,1]={sample_vals[0]:.3f}, C[1,0]={sample_vals[1]:.3f}"
                )
                print(
                    f"    DEBUG: weights: {weight_stats.min():.1f}-{weight_stats.max():.1f} (mean={weight_stats.mean():.1f}), n_valid={len(weight_stats)}"
                )

            # Solve XX
            try:
                gains_xx, _, info_xx = solver.solve(
                    correlations, weights, refant=refant_idx, pol="XX"
                )
                solutions["gains_xx"][i] = gains_xx
                solutions["convergence"].append(info_xx)

                # Print convergence info
                conv_status = "✓" if info_xx["converged"] else "✗"
                print(
                    f"    XX: {conv_status} {info_xx['iterations']} iter, "
                    f"residual {info_xx['initial_residual']:.2e} → {info_xx['final_residual']:.2e}"
                )

                # Flag if didn't converge
                if not info_xx["converged"]:
                    solutions["flags"][i, :, 0] = True

            except Exception as e:
                print(f"    XX: ✗ FAILED - {e}")
                solutions["flags"][i, :, 0] = True
                solutions["gains_xx"][i] = 1.0 + 0j

            # Solve YY
            try:
                gains_yy, _, info_yy = solver.solve(
                    correlations, weights, refant=refant_idx, pol="YY"
                )
                solutions["gains_yy"][i] = gains_yy

                # Print convergence info
                conv_status = "✓" if info_yy["converged"] else "✗"
                print(
                    f"    YY: {conv_status} {info_yy['iterations']} iter, "
                    f"residual {info_yy['initial_residual']:.2e} → {info_yy['final_residual']:.2e}"
                )

                if not info_yy["converged"]:
                    solutions["flags"][i, :, 1] = True

            except Exception as e:
                print(f"    YY: ✗ FAILED - {e}")
                solutions["flags"][i, :, 1] = True
                solutions["gains_yy"][i] = 1.0 + 0j

        # Apply SNR threshold (simple: flag solutions with large residuals)
        if minsnr > 0:
            self._apply_snr_threshold(solutions, minsnr)

        # Write calibration table
        print(f"Writing calibration table: {caltable}")
        self._write_caltable(caltable, solutions, metadata, calmode)

        # Return diagnostics
        n_flagged = np.sum(solutions["flags"])
        n_total = solutions["flags"].size
        print(f"Solutions: {n_total - n_flagged}/{n_total} valid ({n_flagged} flagged)")

        return {
            "n_solutions": n_intervals,
            "n_antennas": n_ant,
            "n_flagged": int(n_flagged),
            "solutions": solutions,
            "metadata": metadata,
        }

    def _parse_spw(self, spw: str) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        """Parse spectral window selection string.

        Args:
            spw: SPW selection (e.g., '0:27~36', '0', '')

        Returns:
            (spw_id, (chan_start, chan_end)) or (None, None) if all
        """
        if not spw:
            return None, None

        if ":" in spw:
            spw_id_str, chan_str = spw.split(":", 1)
            spw_id = int(spw_id_str)

            if "~" in chan_str:
                chan_start, chan_end = chan_str.split("~")
                return spw_id, (int(chan_start), int(chan_end))
            else:
                return spw_id, None
        else:
            return int(spw), None

    def _parse_refant(self, refant: Union[str, int]) -> int:
        """Convert reference antenna to index.

        Args:
            refant: Antenna name (e.g., 'ea21') or index

        Returns:
            Antenna index (0-based)
        """
        if isinstance(refant, int):
            return refant

        # Get antenna names from MS
        summary = self.ms_handler.get_observation_summary()
        antenna_names = summary["antenna_names"]

        # Find matching antenna
        refant_lower = refant.lower()
        for i, name in enumerate(antenna_names):
            if name.lower() == refant_lower:
                return i

        raise ValueError(
            f"Reference antenna '{refant}' not found in MS. "
            f"Available: {antenna_names}"
        )

    def _extract_correlation_blocks(
        self,
        vis_data: Dict[str, np.ndarray],
        solint: str,
        chan_sel: Optional[Tuple[int, int]],
        minsnr: float,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], Dict]:
        """Extract correlation matrices grouped by solution interval.

        Args:
            vis_data: Visibility data from MS
            solint: Solution interval ('int', 'inf', or time string)
            chan_sel: Channel selection (start, end) or None for all
            minsnr: Minimum SNR for valid data

        Returns:
            List of (correlations, weights, timestamp) tuples
            Metadata dictionary
        """
        data = vis_data["data"]  # [n_corr, n_chan, n_row]
        flags = vis_data["flag"]  # [n_corr, n_chan, n_row]
        ant1 = vis_data["antenna1"]  # [n_row]
        ant2 = vis_data["antenna2"]  # [n_row]
        times = vis_data["time"]  # [n_row]

        # Get weights - MS stores as [n_row] or [n_corr, n_row]
        # WEIGHT is per-correlation (not per channel)
        if "weight" in vis_data:
            weights_ms = vis_data["weight"]
            # Check shape: could be [n_row] (single) or [n_corr, n_row]
            if weights_ms.ndim == 1:
                # Broadcast to all correlations
                weights_ms = np.tile(weights_ms, (data.shape[0], 1))
        else:
            # Fallback: uniform weights
            weights_ms = np.ones((data.shape[0], data.shape[2]))

        n_corr, n_chan, n_row = data.shape
        n_ant = max(np.max(ant1), np.max(ant2)) + 1

        # Select channels
        if chan_sel is not None:
            chan_start, chan_end = chan_sel
            data = data[:, chan_start : chan_end + 1, :]
            flags = flags[:, chan_start : chan_end + 1, :]
            n_chan = data.shape[1]

        # Group by time (solint='int' means one solution per unique time)
        unique_times = np.unique(times)

        if solint != "int":
            raise NotImplementedError(
                f"solint='{solint}' not yet supported. Use solint='int' for now."
            )

        correlation_blocks = []

        for t in unique_times:
            # Select data for this time
            time_mask = times == t
            data_t = data[:, :, time_mask]  # [n_corr, n_chan, n_baseline]
            flags_t = flags[:, :, time_mask]
            ant1_t = ant1[time_mask]
            ant2_t = ant2[time_mask]

            # Average over channels (frequency axis)
            # Shape after averaging: [n_corr, n_baseline]
            data_avg = np.mean(data_t, axis=1)
            flags_avg = np.any(flags_t, axis=1)  # Flag if any channel flagged

            # Extract weights for this time
            weights_t = weights_ms[:, time_mask]  # [n_corr, n_baseline]

            # Build correlation matrices [4, n_ant, n_ant]
            correlations = np.zeros((4, n_ant, n_ant), dtype=complex)
            weights = np.zeros((4, n_ant, n_ant))

            for baseline_idx in range(len(ant1_t)):
                i = ant1_t[baseline_idx]
                j = ant2_t[baseline_idx]

                for corr_idx in range(n_corr):
                    vis = data_avg[corr_idx, baseline_idx]
                    flagged = flags_avg[corr_idx, baseline_idx]
                    wt = weights_t[corr_idx, baseline_idx]

                    correlations[corr_idx, i, j] = vis
                    correlations[corr_idx, j, i] = np.conj(vis)  # Hermitian symmetry

                    if not flagged and np.isfinite(vis) and wt > 0:
                        # Use MS weight directly
                        weights[corr_idx, i, j] = wt
                        weights[corr_idx, j, i] = wt
                    else:
                        # Zero weight for flagged or invalid data
                        weights[corr_idx, i, j] = 0.0
                        weights[corr_idx, j, i] = 0.0

            # Zero out autocorrelations
            for corr_idx in range(4):
                np.fill_diagonal(correlations[corr_idx], 0.0)
                np.fill_diagonal(weights[corr_idx], 0.0)

            # Normalize by mean amplitude (source model = constant flux)
            # For point source calibrator: divide out the mean visibility amplitude
            for corr_idx in range(4):
                valid_mask = weights[corr_idx] > 0
                if np.any(valid_mask):
                    mean_amp = np.mean(np.abs(correlations[corr_idx][valid_mask]))
                    if mean_amp > 0:
                        correlations[corr_idx] /= mean_amp

            correlation_blocks.append((correlations, weights, float(t)))

        metadata = {
            "n_antennas": n_ant,
            "n_channels": n_chan,
            "n_correlations": n_corr,
            "spw_id": vis_data.get("spw_id", 0),
        }

        return correlation_blocks, metadata

    def _apply_snr_threshold(self, solutions: Dict, minsnr: float) -> None:
        """Flag solutions below SNR threshold (placeholder).

        Args:
            solutions: Solutions dictionary (modified in-place)
            minsnr: Minimum SNR threshold

        Note:
            This is a simplified implementation. Real SNR calculation requires
            noise estimates from the data.
        """
        # Placeholder: flag solutions with very small amplitudes
        for i in range(solutions["gains_xx"].shape[0]):
            for ant in range(solutions["gains_xx"].shape[1]):
                if np.abs(solutions["gains_xx"][i, ant]) < 1e-6:
                    solutions["flags"][i, ant, 0] = True
                if np.abs(solutions["gains_yy"][i, ant]) < 1e-6:
                    solutions["flags"][i, ant, 1] = True

    def _write_caltable(
        self,
        caltable: str,
        solutions: Dict,
        metadata: Dict,
        calmode: str,
    ) -> None:
        """Write solutions to CASA-format calibration table.

        Args:
            caltable: Output table path
            solutions: Solutions dictionary
            metadata: Metadata dictionary
            calmode: Calibration mode ('p', 'a', 'ap')

        Note:
            Currently writes to numpy .npz format. Full CASA table writing
            requires CalibrationTableHandler.write_synthetic_caltable()
            which is not yet implemented.
        """
        # For now, save as numpy file
        # TODO: Implement proper CASA table format writing
        output_file = caltable + ".npz"

        np.savez(
            output_file,
            gains_xx=solutions["gains_xx"],
            gains_yy=solutions["gains_yy"],
            flags=solutions["flags"],
            times=solutions["times"],
            n_antennas=metadata["n_antennas"],
            calmode=calmode,
        )

        print(f"  Saved solutions to {output_file} (numpy format)")
        print(f"  TODO: Convert to CASA table format for use with applycal")

    def compare_to_casa_caltable(
        self, our_solutions: Dict, casa_caltable: str
    ) -> Dict[str, np.ndarray]:
        """Compare our solutions to CASA gaincal output.

        Args:
            our_solutions: Solutions from our solver
            casa_caltable: Path to CASA calibration table

        Returns:
            Dictionary with comparison statistics
        """
        # Read CASA table
        cal_handler = CalibrationTableHandler(casa_caltable)
        casa_sols = cal_handler.read_gain_solutions()

        # Match times and antennas
        our_times = our_solutions["times"]
        casa_times = casa_sols["unique_times"]

        # Find common times (allowing small tolerance)
        time_tolerance = 0.1  # seconds
        matched_indices = []

        for i, t_our in enumerate(our_times):
            diffs = np.abs(casa_times - t_our)
            if np.min(diffs) < time_tolerance:
                casa_idx = np.argmin(diffs)
                matched_indices.append((i, casa_idx))

        print(f"Matched {len(matched_indices)}/{len(our_times)} time samples")

        # Extract matched solutions
        phase_diffs_xx = []
        phase_diffs_yy = []
        amp_diffs_xx = []
        amp_diffs_yy = []

        for our_idx, casa_idx in matched_indices:
            # CASA gains shape: [n_pol, n_chan, n_time_ant]
            # Need to reshape/index appropriately
            # This depends on CASA table structure

            our_g_xx = our_solutions["gains_xx"][our_idx]
            our_g_yy = our_solutions["gains_yy"][our_idx]

            # TODO: Extract corresponding CASA gains
            # This requires understanding CASA table indexing
            # Placeholder for now
            pass

        return {
            "n_matched": len(matched_indices),
            "phase_rms_xx": np.nan,  # Placeholder
            "phase_rms_yy": np.nan,
            "amp_rms_xx": np.nan,
            "amp_rms_yy": np.nan,
        }

    def close(self):
        """Close MS handler."""
        if hasattr(self, "ms_handler"):
            self.ms_handler.close()


# Convenience function for quick calibration


def quick_gaincal(
    ms_path: str,
    caltable: str,
    field: Union[str, int, List] = "",
    spw: str = "",
    refant: Union[str, int] = 0,
    calmode: str = "p",
    solint: str = "int",
) -> Dict:
    """Quick gaincal-style calibration.

    Args:
        ms_path: Path to measurement set
        caltable: Output calibration table
        field: Field selection
        spw: Spectral window selection (e.g., '0:27~36')
        refant: Reference antenna
        calmode: Calibration mode ('p', 'a', 'ap')
        solint: Solution interval

    Returns:
        Solution diagnostics dictionary
    """
    calibrator = MSCalibrator(ms_path)
    try:
        results = calibrator.gaincal(
            caltable=caltable,
            field=field,
            spw=spw,
            refant=refant,
            calmode=calmode,
            solint=solint,
        )
        return results
    finally:
        calibrator.close()
