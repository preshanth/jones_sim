#!/usr/bin/env python3
"""Apply full Jones matrix corruption to MS using config file.

Complete A-to-Z implementation:
- Parse config
- Open MS
- Compute parallactic angles
- Generate Jones parameters
- Corrupt in chunks
- Write to DATA column
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict
import casatools

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config_parser import ConfigParser
from jones_chain import JonesChain
from casa_interface import MeasurementSetHandler


def compute_parallactic_angles(
    source_ra: float,
    source_dec: float,
    antenna_latitudes: np.ndarray,
    observatory_longitude: float,
    times_mjd: np.ndarray,
    antenna_ids: np.ndarray,
) -> np.ndarray:
    """Compute parallactic angles for given times and antennas.

    Args:
        source_ra: Source RA in radians
        source_dec: Source Dec in radians
        antenna_latitudes: Array of antenna latitudes in radians (n_antennas,)
        observatory_longitude: Observatory longitude in radians
        times_mjd: Array of times in MJD seconds (n_vis,)
        antenna_ids: Array of antenna indices (n_vis,)

    Returns:
        Array of parallactic angles in radians (n_vis,)
    """
    # Compute LST for all times
    mjd_days = times_mjd / 86400.0
    d = mjd_days - 51544.5  # Days since J2000
    gst_deg = (280.46061837 + 360.98564736629 * d) % 360.0
    gst_rad = np.radians(gst_deg)
    lst = gst_rad + observatory_longitude

    # Hour angle for all times
    ha = lst - source_ra

    # Get latitude for each visibility's antenna
    latitudes = antenna_latitudes[antenna_ids]

    # Parallactic angle formula (vectorized)
    sin_psi = np.sin(ha) * np.cos(latitudes) / np.cos(source_dec)
    cos_psi = np.sin(latitudes) * np.cos(source_dec) - np.cos(latitudes) * np.sin(
        source_dec
    ) * np.cos(ha)

    psi = np.arctan2(sin_psi, cos_psi)

    return psi


def load_observatory_metadata(ms_path: str) -> Dict:
    """Load source position and antenna positions from MS.

    Args:
        ms_path: Path to measurement set

    Returns:
        Dict with: source_ra, source_dec, antenna_latitudes, observatory_longitude
    """
    tb = casatools.table()

    # Get source position (FIELD table)
    tb.open(f"{ms_path}/FIELD")
    phase_dir = tb.getcol("PHASE_DIR")  # Shape: (2, n_poly, n_fields)
    tb.close()

    source_ra = float(phase_dir[0, 0, 0])  # radians
    source_dec = float(phase_dir[1, 0, 0])  # radians

    # Get antenna positions (ANTENNA table)
    tb.open(f"{ms_path}/ANTENNA")
    positions = tb.getcol("POSITION")  # (3, n_antennas) - ITRF XYZ
    tb.close()

    # Convert ITRF XYZ to lat/lon
    x = positions[0, :]
    y = positions[1, :]
    z = positions[2, :]

    r = np.sqrt(x**2 + y**2 + z**2)
    antenna_latitudes = np.arcsin(z / r)

    observatory_longitude = np.arctan2(y[0], x[0])

    return {
        "source_ra": source_ra,
        "source_dec": source_dec,
        "antenna_latitudes": antenna_latitudes,
        "observatory_longitude": observatory_longitude,
    }


def corrupt_ms_with_jones(
    ms_path: str,
    config_path: str,
    output_column: str = "DATA",
    chunk_size: int = 100000,
    batch_gpu_size: int = 10000,
):
    """Apply Jones corruption to MS - Complete A-to-Z implementation.

    Args:
        ms_path: Path to measurement set
        config_path: Path to JSON config file
        output_column: Column to write corrupted data
        chunk_size: Rows to process per chunk
        batch_gpu_size: Visibilities per GPU batch
    """

    print(f"\n{'=' * 70}")
    print(f"JONES MATRIX CORRUPTION")
    print(f"{'=' * 70}")
    print(f"MS: {ms_path}")
    print(f"Config: {config_path}")
    print(f"Output column: {output_column}")

    # ========================================================================
    # STEP 1: Load and parse config (ONCE!)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"STEP 1: LOADING CONFIGURATION")
    print(f"{'=' * 70}")

    parser = ConfigParser(config_path)
    parser.print_summary()

    proc_config = parser.get_processing_config()
    use_gpu = proc_config.get("use_gpu", False)

    # Override defaults with config values
    chunk_size = proc_config.get("chunk_size_rows", chunk_size)
    batch_gpu_size = proc_config.get("batch_gpu_size", batch_gpu_size)

    print(f"Chunk size: {chunk_size:,} rows")
    print(f"Batch size (GPU): {batch_gpu_size:,} visibilities")

    # ========================================================================
    # STEP 2: Open MS and get metadata (ONCE!)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"STEP 2: OPENING MS")
    print(f"{'=' * 70}")

    ms_handler = MeasurementSetHandler(ms_path)
    summary = ms_handler.get_observation_summary()

    n_antennas = summary["n_antennas"]
    n_spw = summary["n_spw"]

    print(f"Antennas: {n_antennas}")
    print(f"Spectral Windows: {n_spw}")
    print(f"Fields: {len(summary['field_names'])}")

    # ========================================================================
    # STEP 3: Build SPW map (ONCE!)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"STEP 3: BUILDING SPW MAP")
    print(f"{'=' * 70}")

    spw_map: Dict = {}
    n_channels_per_spw = {}
    ref_frequencies = {}
    spw_frequencies = {}

    for spw_id in range(n_spw):
        spw_info = summary["frequency_info"][spw_id]
        n_chan = spw_info["n_channels"]
        chan_freqs = spw_info["chan_freqs"]
        chan_width = spw_info.get("chan_width", 0)

        # Safety check
        if chan_width == 0 or chan_width is None:
            if len(chan_freqs) > 1:
                chan_width = np.abs(chan_freqs[1] - chan_freqs[0])
            else:
                chan_width = 1e6

        spw_map[spw_id] = {
            "n_chan": n_chan,
            "freqs": chan_freqs,
            "chan_width": chan_width,
        }
        n_channels_per_spw[spw_id] = n_chan
        ref_frequencies[spw_id] = np.median(chan_freqs)
        spw_frequencies[spw_id] = chan_freqs

        print(
            f"  SPW {spw_id}: {n_chan} channels, "
            f"{chan_freqs[0] / 1e9:.3f} - {chan_freqs[-1] / 1e9:.3f} GHz, "
            f"Δν = {chan_width / 1e6:.3f} MHz"
        )

    # ========================================================================
    # STEP 4: Load observatory metadata for parallactic angles (ONCE!)
    # ========================================================================
    obs_metadata = None
    if "parallactic" in parser.get_enabled_effects():
        print(f"\n{'=' * 70}")
        print(f"STEP 4: LOADING OBSERVATORY METADATA")
        print(f"{'=' * 70}")

        obs_metadata = load_observatory_metadata(ms_path)

        print(
            f"Source: RA={np.degrees(obs_metadata['source_ra']):.2f}°, "
            f"Dec={np.degrees(obs_metadata['source_dec']):.2f}°"
        )
        print(
            f"Observatory: Lat={np.degrees(obs_metadata['antenna_latitudes'][0]):.2f}°, "
            f"Lon={np.degrees(obs_metadata['observatory_longitude']):.2f}°"
        )

    # ========================================================================
    # STEP 5: Generate Jones parameters (ONCE!)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"STEP 5: GENERATING JONES PARAMETERS")
    print(f"{'=' * 70}")

    params = parser.generate_all_parameters(n_antennas, n_channels_per_spw)

    # Add metadata to params
    params["_chain_order"] = parser.get_chain_order()
    params["_enabled_effects"] = parser.get_enabled_effects()
    params["_spw_frequencies"] = spw_frequencies

    # ========================================================================
    # STEP 6: Create Jones chain (ONCE!)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"STEP 6: CREATING JONES CHAIN")
    print(f"{'=' * 70}")

    jones_chain = JonesChain(
        config_params=params,
        n_antennas=n_antennas,
        ref_frequencies=ref_frequencies,
        use_gpu=use_gpu,
    )

    # ========================================================================
    # STEP 7: Setup noise (if enabled)
    # ========================================================================
    noise_config = parser.get_noise_config()
    add_noise = noise_config.get("enabled", False)

    if add_noise:
        thermal = noise_config.get("thermal_noise", {})
        tsys = thermal.get("tsys_kelvin")
        aperture_eff = thermal.get("aperture_efficiency")
        antenna_diameter = thermal.get("antenna_diameter_meters")
        noise_seed = noise_config.get("random_seed")

        print(f"\n{'=' * 70}")
        print(f"THERMAL NOISE ENABLED")
        print(f"{'=' * 70}")
        print(f"Tsys: {tsys} K")
        print(f"Aperture efficiency: {aperture_eff}")
        print(f"Antenna diameter: {antenna_diameter} m")

        k_B = 1.380649e-23
        A_geo = np.pi * (antenna_diameter / 2.0) ** 2
        SEFD = (2 * k_B * tsys) / (aperture_eff * A_geo) / 1e-26
        print(f"SEFD: {SEFD:.1f} Jy")

    # ========================================================================
    # STEP 8: Get MS size and integration time
    # ========================================================================
    print(f"\n{'=' * 70}")
    print(f"STEP 8: MS SIZE AND INTEGRATION TIME")
    print(f"{'=' * 70}")

    tb = casatools.table()
    tb.open(ms_path)
    n_row_total = tb.nrows()
    times_all = tb.getcol("TIME")
    tb.close()

    print(f"Total rows: {n_row_total:,}")

    # Calculate integration time
    unique_times = np.unique(times_all)
    if len(unique_times) > 1:
        int_time = np.median(np.diff(unique_times))
    else:
        tb.open(ms_path)
        if "INTERVAL" in tb.colnames():
            int_time = np.median(tb.getcol("INTERVAL"))
        else:
            int_time = 1.0
        tb.close()

    print(f"Integration time: {int_time:.3f} s")

    # ========================================================================
    # STEP 9: Process in chunks
    # ========================================================================
    n_chunks = (n_row_total + chunk_size - 1) // chunk_size

    print(f"\n{'=' * 70}")
    print(f"STEP 9: PROCESSING {n_chunks} CHUNKS")
    print(f"{'=' * 70}")

    n_vis_total = 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_row_total)
        chunk_rows = chunk_end - chunk_start

        print(f"\n--- Chunk {chunk_idx + 1}/{n_chunks} ---")
        print(f"  Rows: {chunk_start:,} - {chunk_end:,} ({chunk_rows:,})")

        # Read chunk
        print(f"  Reading MODEL_DATA...", end="", flush=True)

        tb.open(ms_path)
        data_chunk = tb.getcol("MODEL_DATA", startrow=chunk_start, nrow=chunk_rows)
        data_desc_ids = tb.getcol("DATA_DESC_ID", startrow=chunk_start, nrow=chunk_rows)
        antenna1 = tb.getcol("ANTENNA1", startrow=chunk_start, nrow=chunk_rows)
        antenna2 = tb.getcol("ANTENNA2", startrow=chunk_start, nrow=chunk_rows)
        times_chunk = tb.getcol("TIME", startrow=chunk_start, nrow=chunk_rows)
        tb.close()

        n_corr, n_chan_max, n_row = data_chunk.shape
        print(f" Done ({n_corr}×{n_chan_max}×{n_row})")

        # Flatten
        print(f"  Flattening...", end="", flush=True)

        ideal_visibilities = []
        frequencies_expanded = []
        times_expanded = []
        antenna1_expanded = []
        antenna2_expanded = []
        spw_ids_expanded = []
        bandwidth_expanded = []

        for row_idx in range(n_row):
            spw_id = int(data_desc_ids[row_idx])
            n_chan_spw = spw_map[spw_id]["n_chan"]
            freqs_spw = spw_map[spw_id]["freqs"]
            chan_width_spw = spw_map[spw_id]["chan_width"]

            for chan_idx in range(n_chan_spw):
                vis = data_chunk[:, chan_idx, row_idx]
                ideal_visibilities.append(vis)
                frequencies_expanded.append(freqs_spw[chan_idx])
                times_expanded.append(times_chunk[row_idx])
                antenna1_expanded.append(antenna1[row_idx])
                antenna2_expanded.append(antenna2[row_idx])
                spw_ids_expanded.append(spw_id)
                bandwidth_expanded.append(chan_width_spw)

        ideal_visibilities = np.array(ideal_visibilities, dtype=complex)
        frequencies_expanded = np.array(frequencies_expanded)
        times_expanded = np.array(times_expanded)
        antenna1_expanded = np.array(antenna1_expanded, dtype=int)
        antenna2_expanded = np.array(antenna2_expanded, dtype=int)
        spw_ids_expanded = np.array(spw_ids_expanded, dtype=int)
        bandwidth_expanded = np.array(bandwidth_expanded)

        n_vis_chunk = len(ideal_visibilities)
        n_vis_total += n_vis_chunk

        print(f" Done ({n_vis_chunk:,} visibilities)")

        # Compute parallactic angles
        if obs_metadata is not None:
            parallactic_angles1 = compute_parallactic_angles(
                obs_metadata["source_ra"],
                obs_metadata["source_dec"],
                obs_metadata["antenna_latitudes"],
                obs_metadata["observatory_longitude"],
                times_expanded,
                antenna1_expanded,
            )
            parallactic_angles2 = compute_parallactic_angles(
                obs_metadata["source_ra"],
                obs_metadata["source_dec"],
                obs_metadata["antenna_latitudes"],
                obs_metadata["observatory_longitude"],
                times_expanded,
                antenna2_expanded,
            )
        else:
            # No parallactic angles needed
            parallactic_angles1 = np.zeros(n_vis_chunk)
            parallactic_angles2 = np.zeros(n_vis_chunk)

        # Corrupt
        print(f"  Corrupting...", end="", flush=True)

        noise_params = None
        if add_noise:
            int_time_expanded = np.full(n_vis_chunk, int_time)
            noise_params = {
                "tsys": tsys,
                "aperture_eff": aperture_eff,
                "antenna_diameter": antenna_diameter,
                "bandwidth": bandwidth_expanded,
                "int_time": int_time_expanded,
            }
            if noise_seed is not None:
                noise_params["seed"] = noise_seed

        corrupted_visibilities = jones_chain.corrupt_visibilities(
            ideal_visibilities,
            frequencies_expanded,
            times_expanded,
            antenna1_expanded,
            antenna2_expanded,
            spw_ids_expanded,
            parallactic_angles1,
            parallactic_angles2,
            use_gpu=use_gpu,
            batch_gpu_size=batch_gpu_size,
            noise_params=noise_params,
        )

        print(f" Done")

        # Reshape
        print(f"  Reshaping...", end="", flush=True)

        corrupted_data = np.zeros_like(data_chunk)
        vis_idx = 0

        for row_idx in range(n_row):
            spw_id = int(data_desc_ids[row_idx])
            n_chan_spw = spw_map[spw_id]["n_chan"]

            for chan_idx in range(n_chan_spw):
                corrupted_data[:, chan_idx, row_idx] = corrupted_visibilities[
                    vis_idx, :
                ]
                vis_idx += 1

        print(f" Done")

        # Write
        print(f"  Writing to {output_column}...", end="", flush=True)

        try:
            tb.open(ms_path, nomodify=False)
            tb.putcol(
                output_column, corrupted_data, startrow=chunk_start, nrow=chunk_rows
            )
            tb.close()
            print(f" Done")
        except Exception as e:
            print(f" Error: {e}")
            continue

    # Summary
    print(f"\n{'=' * 70}")
    print(f" COMPLETE!")
    print(f"{'=' * 70}")
    print(f"MS: {ms_path}")
    print(f"Config: {config_path}")
    print(f"Output column: {output_column}")
    print(f"Processing:")
    print(f"  Total rows: {n_row_total:,}")
    print(f"  Total visibilities: {n_vis_total:,}")
    print(f"  Chunks: {n_chunks}")
    print(f"Jones Chain:")
    print(f"  Order: {parser.get_chain_order()}")
    print(f"  Enabled: {parser.get_enabled_effects()}")
    if add_noise:
        print(f"Thermal Noise: ENABLED (SEFD={SEFD:.1f} Jy)")
    print(f"GPU: {'ENABLED' if use_gpu else 'DISABLED'}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python corrupt_jones.py <ms_path> <config_json>")
        print("\nExample:")
        print("  python corrupt_jones.py test.ms jones_config.json")
        sys.exit(1)

    ms_file = sys.argv[1]
    config_file = sys.argv[2]

    corrupt_ms_with_jones(ms_file, config_file)
