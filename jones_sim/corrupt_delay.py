#!/usr/bin/env python3
"""Apply delays to entire MS - chunked processing (read/corrupt/write per chunk)."""

import numpy as np
from typing import Optional, Callable, Dict
import sys

from casa_interface import MeasurementSetHandler
from simulator import JonesSimulator
from effects import BandpassDelay


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
) -> None:
    """
    Apply delays to entire MS - chunked processing with GPU batching.

    For each chunk of rows:
    1. Read chunk from MODEL_DATA
    2. Track which rows belong to which SPW
    3. Get correct frequencies per SPW
    4. Corrupt chunk (with GPU batching if enabled)
    5. Write chunk to DATA

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
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    print(f"\nOpening MS: {ms_path}")
    ms_handler = MeasurementSetHandler(ms_path)
    summary = ms_handler.get_observation_summary()

    n_antennas = summary["n_antennas"]
    n_spw = summary["n_spw"]
    n_fields = len(summary["field_names"])

    print(f"\n{'=' * 70}")
    print(f"MS METADATA")
    print(f"{'=' * 70}")
    print(f"Antennas: {n_antennas}")
    print(f"Spectral Windows: {n_spw}")
    print(f"Fields: {n_fields}")

    for spw_id in range(n_spw):
        spw_info = summary["frequency_info"][spw_id]
        n_chan_spw = spw_info["n_channels"]
        freq_min = spw_info["chan_freqs"][0] / 1e9
        freq_max = spw_info["chan_freqs"][-1] / 1e9
        print(
            f"  SPW {spw_id}: {n_chan_spw} channels, {freq_min:.3f} - {freq_max:.3f} GHz"
        )

    print(f"\n{'=' * 70}")
    print(f"BUILDING SPW FREQUENCY MAP")
    print(f"{'=' * 70}")

    spw_map: Dict = {}
    for spw_id in range(n_spw):
        spw_info = summary["frequency_info"][spw_id]
        spw_map[spw_id] = {
            "n_chan": spw_info["n_channels"],
            "freqs": spw_info["chan_freqs"],
        }

    print(f"✓ Built map for {n_spw} SPWs")

    print(f"\n{'=' * 70}")
    print(f"GENERATING ANTENNA DELAYS")
    print(f"{'=' * 70}")

    delay_range_sec = delay_range_ns * 1e-9

    if delay_func is not None:
        antenna_delays = np.array([delay_func(ant_id) for ant_id in range(n_antennas)])
        print(f"Using custom delay function")
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
    print(f"CREATING JONES SIMULATOR")
    print(f"{'=' * 70}")

    delay_effect = BandpassDelay(
        tau_xx=antenna_delays,
        tau_yy=antenna_delays,
    )

    jones_sim = JonesSimulator()
    jones_sim.add_effect("delays", delay_effect)
    print(f"✓ BandpassDelay effect created")

    if use_gpu:
        print(f"✓ GPU ACCELERATION ENABLED")

    print(f"\n{'=' * 70}")
    print(f"DETERMINING MS SIZE")
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

        print(f"  Reading chunk...", end="", flush=True)

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
        except Exception as e:
            print(f" Error: {e}")
            continue

        print(f"  Reshaping for corruption...", end="", flush=True)

        ideal_visibilities_list = []
        frequencies_expanded_list = []
        times_expanded_list = []
        antenna1_expanded_list = []
        antenna2_expanded_list = []

        for row_idx in range(n_row):
            spw_id = int(data_desc_ids[row_idx])
            n_chan_spw = spw_map[spw_id]["n_chan"]
            freq_spw = spw_map[spw_id]["freqs"]

            for chan_idx in range(n_chan_spw):
                ideal_visibilities_list.append(data_chunk[:, chan_idx, row_idx])
                frequencies_expanded_list.append(freq_spw[chan_idx])
                times_expanded_list.append(times[row_idx])
                antenna1_expanded_list.append(antenna1[row_idx])
                antenna2_expanded_list.append(antenna2[row_idx])

        ideal_visibilities = np.array(ideal_visibilities_list, dtype=complex)
        frequencies_expanded = np.array(frequencies_expanded_list, dtype=float)
        times_expanded = np.array(times_expanded_list, dtype=float)
        antenna1_expanded = np.array(antenna1_expanded_list, dtype=int)
        antenna2_expanded = np.array(antenna2_expanded_list, dtype=int)

        n_vis_chunk = len(ideal_visibilities)
        n_vis_total += n_vis_chunk
        print(f" Done ({n_vis_chunk:,} visibilities)")

        print(f"  Corrupting...", end="", flush=True)

        corrupted_visibilities = jones_sim.corrupt_visibilities(
            ideal_visibilities,
            frequencies_expanded,
            times_expanded,
            antenna1_expanded,
            antenna2_expanded,
            use_gpu=use_gpu,
            batch_gpu_size=batch_gpu_size,
        )

        print(f" Done")

        print(f"  Reshaping back...", end="", flush=True)

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

        print(f"  Writing chunk...", end="", flush=True)

        try:
            table_tool = casatools.table()
            table_tool.open(ms_path, nomodify=False)
            table_tool.putcol(
                output_column, corrupted_data, startrow=chunk_start, nrow=chunk_rows
            )
            table_tool.close()
            print(f" Done")
        except Exception as e:
            print(f" Error: {e}")
            continue

    print(f"\n{'=' * 70}")
    print(f"✓ COMPLETE!")
    print(f"{'=' * 70}")
    print(f"MS: {ms_path}")
    print(f"Output column: {output_column}")
    print(f"Processing:")
    print(f"  Total rows: {n_row_total:,}")
    print(f"  Total visibilities: {n_vis_total:,}")
    print(f"  Chunk size (disk): {chunk_size:,} rows")
    print(f"  Chunks processed: {n_chunks}")
    if use_gpu:
        print(f"  Batch size (GPU): {batch_gpu_size:,} visibilities")
    print(f"Delays:")
    print(f"  Antennas: {n_antennas} (ref antenna: {ref_antenna})")
    print(f"  Effect: BandpassDelay (φ = 2π·τ·ν)")
    print(f"GPU: {'ENABLED' if use_gpu else 'DISABLED'}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python corrupt_ms_clean.py <ms_path> [delay_ns] [ref_ant] [seed] [chunk_size] [batch_gpu_size] [--use-gpu]"
        )
        print("\nExamples:")
        print("  python corrupt_ms_clean.py test.ms --use-gpu")
        print("  python corrupt_ms_clean.py test.ms 5.0 0 42 100000 10000 --use-gpu")
        print("\nParameters:")
        print("  ms_path: Path to MS file")
        print("  delay_ns: Delay range in nanoseconds (default: 10.0)")
        print("  ref_ant: Reference antenna (default: 0)")
        print("  seed: Random seed (default: 42)")
        print("  chunk_size: Rows per disk chunk (default: 100,000)")
        print("  batch_gpu_size: Visibilities per GPU batch (default: 10,000)")
        print("  --use-gpu: Enable GPU acceleration")
        sys.exit(1)

    ms_file = sys.argv[1]
    use_gpu = "--use-gpu" in sys.argv

    args = [arg for arg in sys.argv[2:] if not arg.startswith("--")]

    delay_ns = float(args[0]) if len(args) > 0 else 10.0
    ref_ant = int(args[1]) if len(args) > 1 else 0
    seed = int(args[2]) if len(args) > 2 else 42
    chunk_size = int(args[3]) if len(args) > 3 else 100000
    batch_gpu_size = int(args[4]) if len(args) > 4 else 10000

    corrupt_ms_with_delays(
        ms_path=ms_file,
        output_column="DATA",
        ref_antenna=ref_ant,
        delay_range_ns=delay_ns,
        random_seed=seed,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        batch_gpu_size=batch_gpu_size,
    )
