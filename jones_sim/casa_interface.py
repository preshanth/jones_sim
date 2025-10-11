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

        Args:
            field: Field selection (name, ID, or list)
            spw: Spectral window selection
            antenna: Antenna selection
            time_range: Time range string (CASA format)
            correlation: Correlation selection

        Returns:
            Dictionary containing:
                - 'data': Complex visibility array [n_corr, n_chan, n_row]
                - 'flag': Boolean flag array [n_corr, n_chan, n_row]
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

            # Build selection dictionary
            selection = {}
            if field is not None:
                selection["field"] = field
            if spw is not None:
                selection["spw"] = spw
            if antenna is not None:
                selection["antenna"] = antenna
            if time_range is not None:
                selection["time"] = time_range

            # Apply selection if any criteria specified
            if selection:
                self.ms_tool.select(selection)

            # Get data items
            data_items = [
                "data",
                "flag",
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
            if self.msmd_tool.isopen():
                self.msmd_tool.close()

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
