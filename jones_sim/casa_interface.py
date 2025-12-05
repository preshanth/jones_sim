"""Interface classes for reading and writing CASA measurement sets and calibration tables.

This module provides classes for integrating jones_sim with real interferometric data
using casatools and casatasks. Handles visibility data, flags, and calibration solutions.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import casatools

    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False
    warnings.warn("casatools not available. CASA interface functionality disabled.")


class MeasurementSetHandler:
    """Handler for CASA measurement set reading and writing with flag support.

    Provides methods to read visibility data, flags, and metadata from measurement sets,
    and write back modified data while preserving data structure and flags.
    """

    def __init__(self, ms_path: str):
        """Initialize measurement set handler.

        Args:
            ms_path: Path to measurement set file

        Raises:
            ImportError: If casatools is not available
            RuntimeError: If measurement set cannot be opened
        """
        if not CASA_AVAILABLE:
            raise ImportError("casatools required for MeasurementSetHandler")

        self.ms_path = ms_path
        self.ms_tool = casatools.ms()
        self.msmd_tool = casatools.msmetadata()
        self.table_tool = casatools.table()

        # Verify MS can be opened
        try:
            self.ms_tool.open(ms_path)
            self.ms_tool.close()
            self.msmd_tool.open(ms_path)
            self.msmd_tool.close()
        except Exception as e:
            raise RuntimeError(f"Cannot open measurement set {ms_path}: {e}")

    def get_observation_summary(self) -> Dict[str, Any]:
        """Get high-level summary of observation parameters.

        Returns:
            Dictionary with observation metadata
        """
        try:
            self.msmd_tool.open(self.ms_path)

            summary = {
                "n_antennas": self.msmd_tool.nantennas(),
                "antenna_names": self.msmd_tool.antennanames(),
                "n_spw": self.msmd_tool.nspw(),
                "field_names": self.msmd_tool.fieldnames(),
                "source_names": self.msmd_tool.sourcenames(),
                "scan_numbers": self.msmd_tool.scannumbers(),
                "n_observations": self.msmd_tool.nobservations(),
                "observatory_names": self.msmd_tool.observatorynames(),
            }

            # Get frequency information for each spectral window
            freq_info = {}
            for spw in range(summary["n_spw"]):
                freq_info[spw] = {
                    "chan_freqs": self.msmd_tool.chanfreqs(spw),
                    "ref_freq": self.msmd_tool.reffreq(spw),
                    "n_channels": len(self.msmd_tool.chanfreqs(spw)),
                }
            summary["frequency_info"] = freq_info

            # Get time range information
            if summary["n_observations"] > 0:
                time_ranges = []
                for obs in range(summary["n_observations"]):
                    time_range = self.msmd_tool.timerangeforobs(obs)
                    time_ranges.append(time_range)
                summary["time_ranges"] = time_ranges

            return summary

        finally:
            self.msmd_tool.close()

    def get_antenna_positions(self) -> np.ndarray:
        """Get antenna positions in ITRF coordinates.

        Returns:
            Array of shape (n_antenna, 3) with [x, y, z] positions in meters
        """
        try:
            self.msmd_tool.open(self.ms_path)
            n_ant = self.msmd_tool.nantennas()

            positions = np.zeros((n_ant, 3))
            for ant_id in range(n_ant):
                pos = self.msmd_tool.antennaposition(ant_id)
                positions[ant_id] = pos["value"]

            return positions

        finally:
            self.msmd_tool.close()

    def read_visibilities(
        self,
        field: Optional[Union[str, int, List]] = None,
        spw: Optional[Union[str, int, List]] = None,
        antenna: Optional[Union[str, int, List]] = None,
        time_range: Optional[str] = None,
        correlation: Optional[Union[str, List]] = None,
    ) -> Dict[str, np.ndarray]:
        """Read visibility data and flags from measurement set.

        Uses CASA's ms.msselect() method with high-level MSSelection syntax.
        This is the same selection syntax used in tasks like gaincal, split, etc.

        Args:
            field: Field selection (name, ID, or list; e.g., 0, '0,1,9', 'J1820-2528')
            spw: Spectral window selection (e.g., '0:27~36', '0,1,2')
            antenna: Antenna selection (e.g., 'ea21', '0~25')
            time_range: Time range string (CASA format)
            correlation: Correlation selection

        Returns:
            Dictionary containing:
                - 'data': Complex visibility array [n_corr, n_chan, n_row]
                - 'flag': Boolean flag array [n_corr, n_chan, n_row]
                - 'weight': Weight array [n_corr, n_row] or [n_row]
                - 'uvw': UVW coordinates [3, n_row] in meters
                - 'antenna1': First antenna indices [n_row]
                - 'antenna2': Second antenna indices [n_row]
                - 'time': Time stamps [n_row] in MJD seconds
                - 'frequency': Channel frequencies [n_chan] in Hz (if single SPW)
                - 'field_id': Field ID for each row [n_row]
                - 'scan_number': Scan number for each row [n_row]
        """
        try:
            self.ms_tool.open(self.ms_path)

            # Build selection dictionary for msselect (high-level MSSelection syntax)
            selection = {}
            if field is not None:
                selection["field"] = str(field) if not isinstance(field, str) else field
            if spw is not None:
                selection["spw"] = str(spw) if not isinstance(spw, str) else spw
            if antenna is not None:
                selection["baseline"] = (
                    str(antenna) if not isinstance(antenna, str) else antenna
                )
            if time_range is not None:
                selection["time"] = time_range

            # Apply selection if any criteria specified
            # Use msselect (not select) for high-level MSSelection syntax
            if selection:
                self.ms_tool.msselect(selection)

            # Get data items
            data_items = [
                "data",
                "flag",
                "weight",
                "uvw",
                "antenna1",
                "antenna2",
                "time",
                "field_id",
                "scan_number",
            ]

            if correlation is not None:
                # Handle correlation selection
                data_items.append("axis_info")

            data = self.ms_tool.getdata(data_items)

            # Add frequency information if single SPW selected
            try:
                self.msmd_tool.open(self.ms_path)
                if spw is not None and isinstance(spw, (int, str)):
                    # Single SPW selected
                    spw_id = int(spw) if isinstance(spw, str) else spw
                    data["frequency"] = self.msmd_tool.chanfreqs(spw_id)
                elif "axis_info" in data:
                    # Extract frequency from axis info
                    freq_axis = data["axis_info"]["freq_axis"]
                    if "chan_freq" in freq_axis:
                        data["frequency"] = freq_axis["chan_freq"]["value"]

            except Exception as e:
                warnings.warn(f"Could not extract frequency information: {e}")

            return data

        finally:
            self.ms_tool.close()
            try:
                self.msmd_tool.close()
            except Exception:
                pass  # Already closed

    def write_visibilities(
        self,
        modified_data: np.ndarray,
        flags: Optional[np.ndarray] = None,
        column: str = "CORRECTED_DATA",
        selection: Optional[Dict] = None,
    ) -> None:
        """Write modified visibility data back to measurement set.

        Args:
            modified_data: Modified visibility array [n_corr, n_chan, n_row]
            flags: Optional modified flag array [n_corr, n_chan, n_row]
            column: Target column name ('DATA', 'CORRECTED_DATA', 'MODEL_DATA')
            selection: Selection criteria used when reading data
        """
        try:
            # Use table tool for direct column access
            self.table_tool.open(self.ms_path, nomodify=False)

            # Apply same selection if provided
            if selection:
                # Convert to table query format
                query_str = self._build_taql_query(selection)
                if query_str:
                    selected_table = self.table_tool.query(query_str)
                    selected_table.putcol(column, modified_data)

                    if flags is not None:
                        selected_table.putcol("FLAG", flags)

                    selected_table.close()
                else:
                    # No selection, write to entire table
                    self.table_tool.putcol(column, modified_data)
                    if flags is not None:
                        self.table_tool.putcol("FLAG", flags)
            else:
                # Write to entire table
                self.table_tool.putcol(column, modified_data)
                if flags is not None:
                    self.table_tool.putcol("FLAG", flags)

        finally:
            self.table_tool.close()

    def _build_taql_query(self, selection: Dict) -> str:
        """Build TaQL query string from selection dictionary.

        Args:
            selection: Selection criteria dictionary

        Returns:
            TaQL query string
        """
        conditions = []

        if "field" in selection:
            field = selection["field"]
            if isinstance(field, str):
                conditions.append(
                    f"FIELD_ID IN (SELECT ROWID() FROM ::FIELD WHERE NAME=='{field}')"
                )
            else:
                conditions.append(f"FIELD_ID=={field}")

        if "antenna" in selection:
            antenna = selection["antenna"]
            if isinstance(antenna, (list, tuple)):
                ant_list = ",".join(map(str, antenna))
                conditions.append(
                    f"ANTENNA1 IN [{ant_list}] OR ANTENNA2 IN [{ant_list}]"
                )
            else:
                conditions.append(f"ANTENNA1=={antenna} OR ANTENNA2=={antenna}")

        # Add more selection criteria as needed

        return " AND ".join(conditions) if conditions else ""

    def get_baseline_info(self, exclude_autocorr: bool = True) -> Dict[str, np.ndarray]:
        """Get baseline information for the measurement set.

        Args:
            exclude_autocorr: Whether to exclude autocorrelations

        Returns:
            Dictionary with baseline information
        """
        try:
            data = self.read_visibilities()

            ant1 = data["antenna1"]
            ant2 = data["antenna2"]

            if exclude_autocorr:
                baseline_mask = ant1 != ant2
                ant1 = ant1[baseline_mask]
                ant2 = ant2[baseline_mask]

            # Get unique baselines
            baselines = list(set(zip(ant1, ant2)))
            baselines.sort()

            n_ant = max(np.max(ant1), np.max(ant2)) + 1
            n_baselines = len(baselines)

            return {
                "baselines": np.array(baselines),
                "n_antennas": n_ant,
                "n_baselines": n_baselines,
                "baseline_mask": baseline_mask if exclude_autocorr else None,
            }

        except Exception as e:
            warnings.warn(f"Could not extract baseline information: {e}")
            return {}

    def summary(self) -> None:
        """Print summary of measurement set contents."""
        try:
            obs_summary = self.get_observation_summary()

            print(f"Measurement Set: {self.ms_path}")
            print(f"  Observatory: {obs_summary.get('observatory_names', 'Unknown')}")
            print(f"  Antennas: {obs_summary['n_antennas']}")
            print(f"  Fields: {obs_summary['field_names']}")
            print(f"  Sources: {obs_summary['source_names']}")
            print(f"  Spectral Windows: {obs_summary['n_spw']}")

            for spw, freq_info in obs_summary["frequency_info"].items():
                n_chan = freq_info["n_channels"]
                freq_range = freq_info["chan_freqs"]
                if len(freq_range) > 0:
                    freq_min = freq_range[0] / 1e9  # Convert to GHz
                    freq_max = freq_range[-1] / 1e9
                    print(
                        f"    SPW {spw}: {n_chan} channels, {freq_min:.3f}-{freq_max:.3f} GHz"
                    )

            if "time_ranges" in obs_summary:
                print(f"  Time ranges: {len(obs_summary['time_ranges'])} observations")

        except Exception as e:
            print(f"Error generating summary: {e}")

    def close(self):
        """Close all CASA tools."""
        try:
            if hasattr(self, "ms_tool"):
                self.ms_tool.close()
            if hasattr(self, "msmd_tool"):
                self.msmd_tool.close()
            if hasattr(self, "table_tool"):
                self.table_tool.close()
        except Exception:
            pass


class CalibrationTableHandler:
    """Handler for CASA calibration table reading and writing.

    Provides methods to read calibration solutions from CASA calibration tables
    and write synthetic calibration tables from jones_sim effects.
    """

    def __init__(self, cal_path: Optional[str] = None):
        """Initialize calibration table handler.

        Args:
            cal_path: Path to calibration table (optional)
        """
        if not CASA_AVAILABLE:
            raise ImportError("casatools required for CalibrationTableHandler")

        self.cal_path = cal_path
        self.table_tool = casatools.table()
        self.calibrater_tool = casatools.calibrater()

    def read_gain_solutions(
        self, cal_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Read gain solutions from calibration table.

        Args:
            cal_path: Path to calibration table (overrides instance path)

        Returns:
            Dictionary containing gain solutions and metadata
        """
        table_path = cal_path or self.cal_path
        if not table_path:
            raise ValueError("No calibration table path specified")

        try:
            self.table_tool.open(table_path)

            # Read main calibration data columns
            gains = self.table_tool.getcol("GAIN")  # Complex gains
            flags = self.table_tool.getcol("FLAG")  # Gain flags
            times = self.table_tool.getcol("TIME")  # Time stamps
            antennas = self.table_tool.getcol("ANTENNA1")  # Antenna IDs

            # Try to read additional columns
            try:
                spw_ids = self.table_tool.getcol("SPECTRAL_WINDOW_ID")
            except Exception:
                spw_ids = np.zeros(len(times), dtype=int)

            try:
                field_ids = self.table_tool.getcol("FIELD_ID")
            except Exception:
                field_ids = np.zeros(len(times), dtype=int)

            # Get unique values for organization
            unique_times = np.unique(times)
            unique_antennas = np.unique(antennas)
            unique_spws = np.unique(spw_ids)

            return {
                "gains": gains,
                "flags": flags,
                "times": times,
                "antennas": antennas,
                "spw_ids": spw_ids,
                "field_ids": field_ids,
                "unique_times": unique_times,
                "unique_antennas": unique_antennas,
                "unique_spws": unique_spws,
                "n_pol": gains.shape[0] if len(gains.shape) > 1 else 1,
                "n_chan": gains.shape[1] if len(gains.shape) > 2 else 1,
            }

        finally:
            self.table_tool.close()

    def read_bandpass_solutions(
        self, cal_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Read bandpass solutions from calibration table.

        Args:
            cal_path: Path to bandpass calibration table

        Returns:
            Dictionary containing bandpass solutions and metadata
        """
        # Similar structure to gain solutions but with frequency dependence
        return self.read_gain_solutions(cal_path)

    def write_synthetic_caltable(
        self,
        jones_effects: Dict[str, Any],
        output_path: str,
        times: np.ndarray,
        antennas: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        cal_type: str = "G",
    ) -> None:
        """Write synthetic calibration table from jones_sim effects.

        Args:
            jones_effects: Dictionary of jones_sim effect instances
            output_path: Path for output calibration table
            times: Time array for solutions
            antennas: Antenna array
            frequencies: Frequency array (for bandpass calibration)
            cal_type: Calibration type ('G' for gains, 'B' for bandpass)
        """
        # This is a complex implementation that would require:
        # 1. Creating proper cal table structure
        # 2. Converting jones_sim effects to gain solutions
        # 3. Writing in CASA cal table format

        # For now, provide framework
        raise NotImplementedError("Synthetic cal table writing not yet implemented")

    def list_cal_contents(self, cal_path: Optional[str] = None) -> None:
        """Print summary of calibration table contents.

        Args:
            cal_path: Path to calibration table
        """
        table_path = cal_path or self.cal_path
        if not table_path:
            raise ValueError("No calibration table path specified")

        try:
            solutions = self.read_gain_solutions(table_path)

            print(f"Calibration Table: {table_path}")
            print(f"  Antennas: {len(solutions['unique_antennas'])}")
            print(f"  Time samples: {len(solutions['unique_times'])}")
            print(f"  Spectral windows: {len(solutions['unique_spws'])}")
            print(f"  Polarizations: {solutions['n_pol']}")
            print(f"  Channels: {solutions['n_chan']}")

            # Show gain statistics
            gains = solutions["gains"]
            flags = solutions["flags"]

            unflagged_gains = gains[~flags]
            if len(unflagged_gains) > 0:
                amp_stats = np.abs(unflagged_gains)
                phase_stats = np.angle(unflagged_gains)

                print(
                    f"  Gain amplitudes: {np.mean(amp_stats):.3f} ± {np.std(amp_stats):.3f}"
                )
                print(
                    f"  Gain phases: {np.mean(phase_stats):.3f} ± {np.std(phase_stats):.3f} rad"
                )
                print(f"  Flagged fraction: {np.sum(flags) / flags.size:.1%}")

        except Exception as e:
            print(f"Error reading calibration table: {e}")

    def close(self):
        """Close all CASA tools."""
        try:
            if hasattr(self, "table_tool"):
                self.table_tool.close()
            if hasattr(self, "calibrater_tool"):
                self.calibrater_tool.close()
        except Exception:
            pass


# Convenience functions for quick MS/cal table operations


def quick_ms_summary(ms_path: str) -> None:
    """Quickly print measurement set summary."""
    handler = MeasurementSetHandler(ms_path)
    try:
        handler.summary()
    finally:
        handler.close()


def quick_cal_summary(cal_path: str) -> None:
    """Quickly print calibration table summary."""
    handler = CalibrationTableHandler(cal_path)
    try:
        handler.list_cal_contents()
    finally:
        handler.close()


def compare_ms_structure(ms1_path: str, ms2_path: str) -> None:
    """Compare structure of two measurement sets."""
    print("Comparing measurement set structures:")
    print("\nFirst MS:")
    quick_ms_summary(ms1_path)
    print("\nSecond MS:")
    quick_ms_summary(ms2_path)


# =============================================================================
# DELAY TABLE UNWRAPPING
# =============================================================================


def read_delay_table(
    ktable_path: str,
    n_antennas: int,
    unwrap: bool = False,
    ms_path: str = None,
    max_wraps: int = 20,
    apply_sign_flip: bool = True,
    spw: Union[str, int] = 0,
) -> np.ndarray:
    """Read CASA K-table delays with optional unwrapping.

    Args:
        ktable_path: Path to CASA K-table
        n_antennas: Number of antennas
        unwrap: Whether to unwrap phase wraps
        ms_path: Path to MS (required if unwrap=True)
        max_wraps: Maximum wrap offset to search (±max_wraps)
        apply_sign_flip: Apply sign flip (CASA returns corrections, not corruptions)
        spw: Spectral window selection for unwrapping (CASA MSSelection syntax)

    Returns:
        delays_ns: Delays in nanoseconds [n_antennas]
    """
    if not CASA_AVAILABLE:
        raise ImportError("casatools required for reading delay tables")

    # Read raw delays from table
    tb = casatools.table()
    tb.open(ktable_path)
    fparam = tb.getcol("FPARAM")  # nanoseconds
    antennas = tb.getcol("ANTENNA1")
    flags = tb.getcol("FLAG")
    tb.close()

    casa_delays_ns = np.zeros(n_antennas)
    delay_counts = np.zeros(n_antennas)

    # Average over polarizations and channels
    if fparam.ndim == 3:
        n_pol, n_chan, n_rows = fparam.shape
        for row in range(n_rows):
            ant = antennas[row]
            for pol in range(n_pol):
                for chan in range(n_chan):
                    if not flags[pol, chan, row]:
                        casa_delays_ns[ant] += fparam[pol, chan, row]
                        delay_counts[ant] += 1
    elif fparam.ndim == 2:
        n_pol, n_rows = fparam.shape
        for row in range(n_rows):
            ant = antennas[row]
            for pol in range(n_pol):
                if not flags[pol, row]:
                    casa_delays_ns[ant] += fparam[pol, row]
                    delay_counts[ant] += 1

    # Average
    nonzero_mask = delay_counts > 0
    casa_delays_ns[nonzero_mask] /= delay_counts[nonzero_mask]

    # Unwrap if requested
    if unwrap:
        if ms_path is None:
            raise ValueError("ms_path required for unwrapping")
        casa_delays_ns = unwrap_delay_table(
            ms_path, casa_delays_ns, max_wraps, apply_sign_flip, spw
        )
    elif apply_sign_flip:
        # Just apply sign flip without unwrapping
        casa_delays_ns = -casa_delays_ns

    return casa_delays_ns


def unwrap_delay_table(
    ms_path: str,
    casa_delays_ns: np.ndarray,
    max_wraps: int = 20,
    apply_sign_flip: bool = True,
    spw: Union[str, int] = 0,
) -> np.ndarray:
    """Unwrap CASA delay table using phase residual minimization.

    CASA's delay solver can wrap into different 2π branches for different antennas.
    This function finds the correct unwrapping by testing wrap offsets and picking
    the one that minimizes phase residuals across frequency.

    Algorithm:
        For each antenna:
        1. Try wrap offsets n = -max_wraps to +max_wraps
        2. For each offset: delay_test = casa_delay + n * (1/freq_center)
        3. Apply this delay to baselines involving this antenna
        4. Measure RMS of phase residuals across frequency
        5. Pick offset with minimum RMS

    Args:
        ms_path: Path to measurement set
        casa_delays_ns: Raw CASA delays in nanoseconds [n_antennas]
        max_wraps: Maximum wrap offset to try (±max_wraps)
        apply_sign_flip: Apply sign flip (CASA corrections vs corruptions)
        spw: Spectral window selection (CASA MSSelection syntax)

    Returns:
        unwrapped_delays_ns: Unwrapped delays in nanoseconds [n_antennas]
    """
    if not CASA_AVAILABLE:
        raise ImportError("casatools required for unwrapping")

    print(f"\nUnwrapping delays using phase residuals (±{max_wraps} wraps)...")

    # Use MeasurementSetHandler for all MS access
    ms_handler = MeasurementSetHandler(ms_path)

    # Read DATA column
    print("  Reading DATA column...")
    data_dict = ms_handler.read_visibilities(spw=str(spw))
    data = data_dict["data"]
    ant1 = data_dict["antenna1"]
    ant2 = data_dict["antenna2"]
    flag = data_dict["flag"]
    freqs = data_dict["frequency"]

    # Read MODEL_DATA column separately (to avoid double RAM)
    print("  Reading MODEL_DATA column...")
    # Temporarily read MODEL_DATA via table tool (until we extend read_visibilities)
    # TODO: Add column selection to read_visibilities() to avoid this
    tb = casatools.table()
    tb.open(ms_path)

    # Apply same SPW selection as DATA read
    # For now, assume MODEL_DATA has same structure
    model = tb.getcol("MODEL_DATA")
    tb.close()

    # Validate shapes match
    if data.shape != model.shape:
        raise ValueError(
            f"DATA and MODEL_DATA shape mismatch: {data.shape} vs {model.shape}"
        )

    n_antennas = len(casa_delays_ns)
    freq_center = np.mean(freqs)
    wrap_period_ns = 1e9 / freq_center

    print(f"  Frequency range: {freqs[0]/1e9:.3f} - {freqs[-1]/1e9:.3f} GHz")
    print(f"  Wrap period: {wrap_period_ns:.3f} ns at {freq_center/1e9:.3f} GHz")

    # Apply sign flip if requested (CASA returns corrections)
    casa_with_sign = -casa_delays_ns if apply_sign_flip else casa_delays_ns

    # Unwrap each antenna
    unwrapped_ns = np.zeros(n_antennas)

    for ant in range(n_antennas):
        # Find baselines involving this antenna
        baseline_mask = (ant1 == ant) | (ant2 == ant)

        if not np.any(baseline_mask):
            # No baselines for this antenna
            unwrapped_ns[ant] = casa_with_sign[ant]
            continue

        # Get data for these baselines
        data_subset = data[:, :, baseline_mask]  # (n_corr, n_chan, n_baselines)
        model_subset = model[:, :, baseline_mask]
        flag_subset = flag[:, :, baseline_mask]
        ant1_subset = ant1[baseline_mask]
        ant2_subset = ant2[baseline_mask]

        # Test different wrap offsets
        best_rms = np.inf
        best_offset = 0

        for n_wrap in range(-max_wraps, max_wraps + 1):
            # Test delay with this wrap offset
            test_delays_ns = casa_with_sign.copy()
            test_delays_ns[ant] = casa_with_sign[ant] + n_wrap * wrap_period_ns

            # Compute phase residuals for baselines involving this antenna
            rms = _compute_phase_residual_rms(
                data_subset,
                model_subset,
                flag_subset,
                ant1_subset,
                ant2_subset,
                test_delays_ns,
                freqs,
            )

            if rms < best_rms:
                best_rms = rms
                best_offset = n_wrap

        unwrapped_ns[ant] = casa_with_sign[ant] + best_offset * wrap_period_ns

        if best_offset != 0:
            print(
                f"  Ant {ant}: offset {best_offset:+3d} wraps "
                f"({casa_delays_ns[ant]:.2f} → {unwrapped_ns[ant]:.2f} ns, RMS={best_rms:.3f} rad)"
            )

    print("✓ Unwrapping complete")
    return unwrapped_ns


def _compute_phase_residual_rms(
    data: np.ndarray,
    model: np.ndarray,
    flag: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    delays_ns: np.ndarray,
    freqs: np.ndarray,
) -> float:
    """Compute RMS phase residual after applying delay corrections.

    Args:
        data: Observed visibility data (n_corr, n_chan, n_baselines)
        model: Model visibility data (n_corr, n_chan, n_baselines)
        flag: Flags (n_corr, n_chan, n_baselines)
        ant1: Antenna 1 indices (n_baselines,)
        ant2: Antenna 2 indices (n_baselines,)
        delays_ns: Delays to test in nanoseconds (n_antennas,)
        freqs: Channel frequencies in Hz (n_chan,)

    Returns:
        rms: RMS phase residual in radians
    """
    n_corr, n_chan, n_baselines = data.shape
    delays_sec = delays_ns * 1e-9

    # Compute delay phase for each baseline
    tau1 = delays_sec[ant1]  # (n_baselines,)
    tau2 = delays_sec[ant2]
    delay_phase = (
        2 * np.pi * (tau1[:, None] - tau2[:, None]) * freqs[None, :]
    )  # (n_baselines, n_chan)

    # Apply delay correction to data
    # V_corrected = V_data * exp(-i * delay_phase)
    phase_correction = np.exp(-1j * delay_phase)  # (n_baselines, n_chan)

    # Broadcast to all correlations: (n_baselines, n_chan) -> (n_corr, n_chan, n_baselines)
    # data shape: (n_corr, n_chan, n_baselines)
    # Need: (1, n_chan, n_baselines)
    phase_correction_bc = phase_correction.T[
        None, :, :
    ]  # Transpose then add corr dimension

    data_corrected = data * phase_correction_bc

    # Compute phase residual: angle(V_corrected / V_model)
    ratio = data_corrected / (model + 1e-10)  # Avoid division by zero
    phase_residual = np.angle(ratio)

    # Mask flagged data
    phase_residual_masked = phase_residual[~flag]

    if len(phase_residual_masked) == 0:
        return np.inf

    # RMS across all unflagged data
    rms = np.sqrt(np.mean(phase_residual_masked**2))
    return rms
