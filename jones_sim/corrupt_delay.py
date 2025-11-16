#!/usr/bin/env python3
"""Apply delays to entire MS - chunked processing (read/corrupt/write per chunk)."""

import sys
from typing import Callable, Dict, Optional

import numpy as np
from casa_interface import MeasurementSetHandler
from effects import BandpassDelay
from simulator import JonesSimulator


def corrupt_ms_with_delays(
    ms_path: str,
    output_column: str = "DATA",
    ref_antenna: int = 0,
    delay_range_ns: float = 10.0,
    delay_func: Optional[Callable] = None,
    random_seed: Optional[int] = None,
    use_gpu: bool = False,
    chunk_size: int = 100000,
    batch_gpu_size: int = 10000,
    add_noise: bool = False,
    tsys: Optional[float] = None,
    aperture_eff: Optional[float] = None,
    antenna_diameter: Optional[float] = None,
    noise_seed: Optional[int] = None,
) -> None:
    """
    Apply delays to entire MS - chunked processing with GPU batching.

    For each chunk of rows:
    1. Read chunk from MODEL_DATA
    2. Track which rows belong to which SPW
    3. Get correct frequencies per SPW
    4. Corrupt chunk (with GPU batching if enabled)
    5. Add thermal noise if requested
    6. Write chunk to DATA

    Args:
        ms_path: Path to measurement set
        output_column: Column to write to (default: DATA)
        ref_antenna: Reference antenna (gets 0 delay)
        delay_range_ns: Range for random delays (±nanoseconds)
        delay_func: Optional function(antenna_id) -> delay_in_seconds
        random_seed: Random seed for reproducibility
        use_gpu: If True, use GPU acceleration via CuPy
        chunk_size: Rows to process per chunk (default: 100,000)
        batch_gpu_size: Visibilities per GPU batch (default: 10,000)
        add_noise: If True, add thermal noise based on radiometer equation
        tsys: System temperature in Kelvin (required if add_noise=True)
        aperture_eff: Aperture efficiency 0-1 (required if add_noise=True)
        antenna_diameter: Antenna diameter in meters (required if add_noise=True)
        noise_seed: Random seed for noise generation (default: None)
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Validate noise parameters
    if add_noise:
        if tsys is None or aperture_eff is None or antenna_diameter is None:
            raise ValueError(
                "When add_noise=True, must provide tsys, aperture_eff, and antenna_diameter"
            )

    print(f"\nOpening MS: {ms_path}")
    ms_handler = MeasurementSetHandler(ms_path)
    summary = ms_handler.get_observation_summary()

    n_antennas = summary["n_antennas"]
    n_spw = summary["n_spw"]
    n_fields = len(summary["field_names"])

    print(f"\n{'=' * 70}")
    print("MS METADATA")
    print(f"{'=' * 70}")
    print(f"Antennas: {n_antennas}")
    print(f"Spectral Windows: {n_spw}")
    print(f"Fields: {n_fields}")

    for spw_id in range(n_spw):
        spw_info = summary["frequency_info"][spw_id]
        n_chan_spw = spw_info["n_channels"]
        freq_min = spw_info["chan_freqs"][0] / 1e9
        freq_max = spw_info["chan_freqs"][-1] / 1e9
        chan_width = spw_info.get("chan_width", 0)
        print(
            f"  SPW {spw_id}: {n_chan_spw} channels, {freq_min:.3f} - {freq_max:.3f} GHz"
        )
        print(
            f"           Channel width (Δν): {chan_width / 1e6:.3f} MHz = {chan_width:.0f} Hz"
        )

    print(f"\n{'=' * 70}")
    print("BUILDING SPW FREQUENCY MAP")
    print(f"{'=' * 70}")

    spw_map: Dict = {}
    for spw_id in range(n_spw):
        spw_info = summary["frequency_info"][spw_id]
        chan_width = spw_info.get("chan_width", 0)

        # Safety check for bandwidth
        if chan_width == 0 or chan_width is None:
            print(
                f"WARNING: SPW {spw_id} has invalid channel width, trying to calculate from frequencies"
            )
            freqs = spw_info["chan_freqs"]
            if len(freqs) > 1:
                chan_width = np.abs(freqs[1] - freqs[0])
                print(f"  Calculated channel width: {chan_width / 1e6:.3f} MHz")
            else:
                chan_width = 1e6  # Default 1 MHz
                print("  WARNING: Using default 1 MHz bandwidth")

        spw_map[spw_id] = {
            "n_chan": spw_info["n_channels"],
            "freqs": spw_info["chan_freqs"],
            "chan_width": chan_width,
        }

    print(f"✓ Built map for {n_spw} SPWs")

    print(f"\n{'=' * 70}")
    print("GENERATING ANTENNA DELAYS")
    print(f"{'=' * 70}")

    delay_range_sec = delay_range_ns * 1e-9

    if delay_func is not None:
        antenna_delays = np.array([delay_func(ant_id) for ant_id in range(n_antennas)])
        print("Using custom delay function")
    else:
        antenna_delays = np.random.uniform(
            -delay_range_sec, delay_range_sec, n_antennas
        )
        print(f"Using random delays: ±{delay_range_ns} ns")

    antenna_delays[ref_antenna] = 0.0

    print(f"\nAntenna delays ({n_antennas} antennas):")
    for ant_id in range(n_antennas):
        delay_ns = antenna_delays[ant_id] * 1e9
        print(f"  Antenna {ant_id}: {delay_ns:8.3f} ns")

    print(f"\n{'=' * 70}")
    print("CREATING JONES SIMULATOR")
    print(f"{'=' * 70}")

    delay_effect = BandpassDelay(
        tau_xx=antenna_delays,
        tau_yy=antenna_delays,
    )

    jones_sim = JonesSimulator()
    jones_sim.add_effect("delays", delay_effect)
    print("✓ BandpassDelay effect created")

    if use_gpu:
        print("✓ GPU ACCELERATION ENABLED")

    if add_noise:
        print("✓ THERMAL NOISE ENABLED")
        print(f"  Tsys: {tsys} K")
        print(f"  Aperture efficiency: {aperture_eff}")
        print(f"  Antenna diameter: {antenna_diameter} m")

        # Calculate SEFD for reference
        k_B = 1.380649e-23
        A_geo = np.pi * (antenna_diameter / 2.0) ** 2
        SEFD = (2 * k_B * tsys) / (aperture_eff * A_geo) / 1e-26
        print(f"  Calculated SEFD: {SEFD:.1f} Jy")

        if noise_seed is not None:
            print(f"  Noise seed: {noise_seed}")

    print(f"\n{'=' * 70}")
    print("DETERMINING MS SIZE")
    print(f"{'=' * 70}")

    try:
        import casatools

        table_tool = casatools.table()
        table_tool.open(ms_path)
        n_row_total = table_tool.nrows()
        table_tool.close()
        print(f"Total rows in MS: {n_row_total:,}")
    except Exception as e:
        print(f"Error getting MS size: {e}")
        raise

    # Calculate integration time from TIME column
    print(f"\n{'=' * 70}")
    print("CALCULATING INTEGRATION TIME")
    print(f"{'=' * 70}")

    try:
        table_tool = casatools.table()
        table_tool.open(ms_path)
        times_all = table_tool.getcol("TIME")
        table_tool.close()

        # Calculate time differences between consecutive rows
        unique_times = np.unique(times_all)
        if len(unique_times) > 1:
            time_diffs = np.diff(unique_times)
            int_time = np.median(time_diffs)
        else:
            # Only one time step, try INTERVAL column
            table_tool.open(ms_path)
            if "INTERVAL" in table_tool.colnames():
                intervals = table_tool.getcol("INTERVAL")
                int_time = np.median(intervals)
            else:
                int_time = 1.0  # Default fallback
                print(
                    "Warning: Cannot determine integration time, using default 1.0 s"
                )
            table_tool.close()

        print(f"✓ Integration time (Δt): {int_time:.3f} seconds")

    except Exception as e:
        print(f"Error calculating integration time: {e}")
        int_time = 1.0
        print(f"Using default integration time: {int_time} s")

    # Calculate and report expected noise levels
    if add_noise:
        print(f"\n{'=' * 70}")
        print("EXPECTED NOISE LEVELS (per SPW)")
        print(f"{'=' * 70}")
        print("Formula: σ = SEFD / sqrt(Δν × Δt)")
        print(f"         = {SEFD:.1f} / sqrt(bandwidth × {int_time:.3f})")
        print()

        for spw_id in range(n_spw):
            bw = spw_map[spw_id]["chan_width"]
            sigma = SEFD / np.sqrt(bw * int_time)
            print(f"  SPW {spw_id}:")
            print(f"    Bandwidth: {bw / 1e6:.3f} MHz")
            print(f"    σ_noise: {sigma * 1e3:.3f} mJy = {sigma:.6f} Jy")
            print(f"    (σ_real = σ_imag = {sigma / np.sqrt(2) * 1e3:.3f} mJy)")

    print(f"\n{'=' * 70}")
    print(f"PROCESSING IN CHUNKS (chunk_size={chunk_size:,})")
    print(f"{'=' * 70}")

    n_chunks = (n_row_total + chunk_size - 1) // chunk_size
    n_vis_total = 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_row_total)
        chunk_rows = chunk_end - chunk_start

        print(f"\n{'─' * 70}")
        print(
            f"Chunk {chunk_idx + 1}/{n_chunks}: Rows {chunk_start:,} - {chunk_end:,} ({chunk_rows:,} rows)"
        )
        print(f"{'─' * 70}")

        print("  Reading chunk...", end="", flush=True)

        try:
            table_tool = casatools.table()
            table_tool.open(ms_path)

            # Read MODEL_DATA column (ideal/clean data)
            data_chunk = table_tool.getcol(
                "MODEL_DATA", startrow=chunk_start, nrow=chunk_rows
            )
            antenna1 = table_tool.getcol(
                "ANTENNA1", startrow=chunk_start, nrow=chunk_rows
            )
            antenna2 = table_tool.getcol(
                "ANTENNA2", startrow=chunk_start, nrow=chunk_rows
            )
            times = table_tool.getcol("TIME", startrow=chunk_start, nrow=chunk_rows)
            data_desc_ids = table_tool.getcol(
                "DATA_DESC_ID", startrow=chunk_start, nrow=chunk_rows
            )

            table_tool.close()

            n_corr, n_chan, n_row = data_chunk.shape
            print(f" Done ({n_corr}×{n_chan}×{n_row})")

            # Check for NaNs in input data
            n_nans_input = np.sum(np.isnan(data_chunk))
            if n_nans_input > 0:
                print(f"  WARNING: Input data contains {n_nans_input} NaNs!")

        except Exception as e:
            print(f" Error: {e}")
            continue

        print("  Reshaping for corruption...", end="", flush=True)

        ideal_visibilities_list = []
        frequencies_expanded_list = []
        times_expanded_list = []
        antenna1_expanded_list = []
        antenna2_expanded_list = []
        bandwidth_list = []

        for row_idx in range(n_row):
            spw_id = int(data_desc_ids[row_idx])
            n_chan_spw = spw_map[spw_id]["n_chan"]
            freq_spw = spw_map[spw_id]["freqs"]
            chan_width = spw_map[spw_id]["chan_width"]

            for chan_idx in range(n_chan_spw):
                ideal_visibilities_list.append(data_chunk[:, chan_idx, row_idx])
                frequencies_expanded_list.append(freq_spw[chan_idx])
                times_expanded_list.append(times[row_idx])
                antenna1_expanded_list.append(antenna1[row_idx])
                antenna2_expanded_list.append(antenna2[row_idx])
                bandwidth_list.append(chan_width)

        ideal_visibilities = np.array(ideal_visibilities_list, dtype=complex)
        frequencies_expanded = np.array(frequencies_expanded_list, dtype=float)
        times_expanded = np.array(times_expanded_list, dtype=float)
        antenna1_expanded = np.array(antenna1_expanded_list, dtype=int)
        antenna2_expanded = np.array(antenna2_expanded_list, dtype=int)
        bandwidth_expanded = np.array(bandwidth_list, dtype=float)

        n_vis_chunk = len(ideal_visibilities)
        n_vis_total += n_vis_chunk
        print(f" Done ({n_vis_chunk:,} visibilities)")

        # Debug: Check arrays for issues
        if chunk_idx == 0:  # Only for first chunk
            print(
                f"  [DEBUG] Frequency range: {frequencies_expanded.min() / 1e9:.3f} - {frequencies_expanded.max() / 1e9:.3f} GHz"
            )
            print(
                f"  [DEBUG] Time range: {times_expanded.min():.1f} - {times_expanded.max():.1f} s"
            )
            print(
                f"  [DEBUG] Bandwidth range: {bandwidth_expanded.min() / 1e6:.3f} - {bandwidth_expanded.max() / 1e6:.3f} MHz"
            )
            print(
                f"  [DEBUG] Ideal vis stats: mean amp = {np.mean(np.abs(ideal_visibilities)):.3e}"
            )
            if add_noise:
                print(f"  [DEBUG] Integration time: {int_time:.3f} s")
                print(
                    f"  [DEBUG] Expected noise range: {(SEFD / np.sqrt(bandwidth_expanded.max() * int_time)) * 1e3:.3f} - {(SEFD / np.sqrt(bandwidth_expanded.min() * int_time)) * 1e3:.3f} mJy"
                )

        print("  Corrupting...", end="", flush=True)

        # Prepare noise parameters if needed
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

        corrupted_visibilities = jones_sim.corrupt_visibilities(
            ideal_visibilities,
            frequencies_expanded,
            times_expanded,
            antenna1_expanded,
            antenna2_expanded,
            use_gpu=use_gpu,
            batch_gpu_size=batch_gpu_size,
            noise_params=noise_params,
        )

        print(" Done")

        # Check for NaNs in corrupted data
        n_nans_output = np.sum(np.isnan(corrupted_visibilities))
        if n_nans_output > 0:
            print(f"  ERROR: Corrupted data contains {n_nans_output} NaNs!")
            print("  [DEBUG] Checking which step produced NaNs...")

            # Try without noise to isolate the issue
            test_corrupted = jones_sim.corrupt_visibilities(
                ideal_visibilities,
                frequencies_expanded,
                times_expanded,
                antenna1_expanded,
                antenna2_expanded,
                use_gpu=use_gpu,
                batch_gpu_size=batch_gpu_size,
                noise_params=None,  # No noise
            )

            n_nans_no_noise = np.sum(np.isnan(test_corrupted))
            if n_nans_no_noise > 0:
                print(
                    f"  ERROR: NaNs from Jones corruption itself ({n_nans_no_noise} NaNs)"
                )
            else:
                print("  ERROR: NaNs from noise addition")

        if chunk_idx == 0:  # Report stats for first chunk
            print(
                f"  [DEBUG] Corrupted vis stats: mean amp = {np.mean(np.abs(corrupted_visibilities)):.3e}"
            )

        print("  Reshaping back...", end="", flush=True)

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

        print(" Done")

        print("  Writing chunk...", end="", flush=True)

        try:
            table_tool = casatools.table()
            table_tool.open(ms_path, nomodify=False)
            table_tool.putcol(
                output_column, corrupted_data, startrow=chunk_start, nrow=chunk_rows
            )
            table_tool.close()
            print(" Done")
        except Exception as e:
            print(f" Error: {e}")
            continue

    print(f"\n{'=' * 70}")
    print("✓ COMPLETE!")
    print(f"{'=' * 70}")
    print(f"MS: {ms_path}")
    print(f"Output column: {output_column}")
    print("Processing:")
    print(f"  Total rows: {n_row_total:,}")
    print(f"  Total visibilities: {n_vis_total:,}")
    print(f"  Chunk size (disk): {chunk_size:,} rows")
    print(f"  Chunks processed: {n_chunks}")
    print(f"  Integration time (Δt): {int_time:.3f} s")
    if use_gpu:
        print(f"  Batch size (GPU): {batch_gpu_size:,} visibilities")
    print("Delays:")
    print(f"  Antennas: {n_antennas} (ref antenna: {ref_antenna})")
    print("  Effect: BandpassDelay (φ = 2π·τ·ν)")
    if add_noise:
        print("Thermal Noise:")
        print(f"  Tsys: {tsys} K")
        print(f"  Aperture efficiency: {aperture_eff}")
        print(f"  Antenna diameter: {antenna_diameter} m")
        print(f"  SEFD: {SEFD:.1f} Jy")
        print("  Method: Radiometer equation (σ_real = σ_imag = σ/√2)")
        if noise_seed is not None:
            print(f"  Seed: {noise_seed}")
    else:
        print("Thermal Noise: DISABLED")
    print(f"GPU: {'ENABLED' if use_gpu else 'DISABLED'}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python corrupt_ms_clean.py <ms_path> [options]")
        print("\nOptions (in order):")
        print("  delay_ns         Delay range in nanoseconds (default: 10.0)")
        print("  ref_ant          Reference antenna (default: 0)")
        print("  seed             Random seed for delays (default: 42)")
        print("  chunk_size       Rows per disk chunk (default: 100,000)")
        print("  batch_gpu_size   Visibilities per GPU batch (default: 10,000)")
        print("  --use-gpu        Enable GPU acceleration")
        print("  --add-noise      Enable thermal noise")
        print("  --tsys VALUE     System temperature in K (required with --add-noise)")
        print(
            "  --aperture-eff VALUE  Aperture efficiency 0-1 (required with --add-noise)"
        )
        print(
            "  --diameter VALUE Antenna diameter in meters (required with --add-noise)"
        )
        print("  --noise-seed VALUE    Random seed for noise")
        print("\nExamples:")
        print("  # No noise")
        print("  python corrupt_ms_clean.py test.ms --use-gpu")
        print("\n  # With thermal noise (Tsys=50K, 25m dishes, 70% efficiency)")
        print(
            "  python corrupt_ms_clean.py test.ms --use-gpu --add-noise --tsys 50 --aperture-eff 0.7 --diameter 25"
        )
        print("\n  # Full options")
        print(
            "  python corrupt_ms_clean.py test.ms 5.0 0 42 100000 10000 --use-gpu --add-noise --tsys 50 --aperture-eff 0.7 --diameter 25 --noise-seed 123"
        )
        sys.exit(1)

    ms_file = sys.argv[1]

    # Parse flags
    use_gpu = "--use-gpu" in sys.argv
    add_noise = "--add-noise" in sys.argv

    # Get positional arguments (exclude flags)
    args = [arg for arg in sys.argv[2:] if not arg.startswith("--")]

    delay_ns = float(args[0]) if len(args) > 0 else 10.0
    ref_ant = int(args[1]) if len(args) > 1 else 0
    seed = int(args[2]) if len(args) > 2 else 42
    chunk_size = int(args[3]) if len(args) > 3 else 100000
    batch_gpu_size = int(args[4]) if len(args) > 4 else 10000

    # Parse noise parameters
    tsys = None
    aperture_eff = None
    antenna_diameter = None
    noise_seed = None

    for i, arg in enumerate(sys.argv):
        if arg == "--tsys" and i + 1 < len(sys.argv):
            tsys = float(sys.argv[i + 1])
        elif arg == "--aperture-eff" and i + 1 < len(sys.argv):
            aperture_eff = float(sys.argv[i + 1])
        elif arg == "--diameter" and i + 1 < len(sys.argv):
            antenna_diameter = float(sys.argv[i + 1])
        elif arg == "--noise-seed" and i + 1 < len(sys.argv):
            noise_seed = int(sys.argv[i + 1])

    corrupt_ms_with_delays(
        ms_path=ms_file,
        output_column="DATA",
        ref_antenna=ref_ant,
        delay_range_ns=delay_ns,
        random_seed=seed,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        batch_gpu_size=batch_gpu_size,
        add_noise=add_noise,
        tsys=tsys,
        aperture_eff=aperture_eff,
        antenna_diameter=antenna_diameter,
        noise_seed=noise_seed,
    )
