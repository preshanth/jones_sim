#!/usr/bin/env python3
"""Common validation utilities for Jones calibration testing.

This module provides reusable functions for:
- Ground truth generation for all Jones effects
- MS corruption using JonesSimulator
- CASA calibration tasks
- Result comparison and validation

Follows the established pattern from validate_delay_recovery.py.
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple, List

from casatools import table
from jones_sim import JonesSimulator
from jones_sim.casa_interface import MeasurementSetHandler


# =============================================================================
# GROUND TRUTH GENERATION
# =============================================================================


def generate_delays(
    n_antennas: int, delay_range_ns: float = 10.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ground truth delays.

    Args:
        n_antennas: Number of antennas
        delay_range_ns: Delay range in nanoseconds (±)
        seed: Random seed

    Returns:
        delays_ns: Delays in nanoseconds
        delays_sec: Delays in seconds
    """
    np.random.seed(seed)
    delays_ns = np.random.uniform(-delay_range_ns, delay_range_ns, n_antennas)
    delays_ns[0] = 0.0  # Reference antenna
    delays_sec = delays_ns * 1e-9
    return delays_ns, delays_sec


def generate_gains(
    n_antennas: int,
    amp_std: float = 0.1,
    phase_std: float = 0.1,
    seed: int = 43,
) -> np.ndarray:
    """Generate ground truth complex gains.

    Args:
        n_antennas: Number of antennas
        amp_std: Standard deviation of amplitude (log-normal)
        phase_std: Standard deviation of phase (radians)
        seed: Random seed

    Returns:
        gains: Complex array [n_antennas, 2] for XX, YY
    """
    np.random.seed(seed)
    amp = np.exp(np.random.normal(0, amp_std, (n_antennas, 2)))
    phase = np.random.normal(0, phase_std, (n_antennas, 2))
    gains = amp * np.exp(1j * phase)
    gains[0, :] = 1.0 + 0j  # Reference antenna
    return gains


def generate_bandpass(
    n_antennas: int,
    n_channels: int,
    freqs: np.ndarray,
    delay_range_ns: float = 5.0,
    amp_variation: float = 0.05,
    seed: int = 44,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ground truth bandpass.

    Creates frequency-dependent complex gains with delay and amplitude ripple.

    Args:
        n_antennas: Number of antennas
        n_channels: Number of channels
        freqs: Channel frequencies (Hz)
        delay_range_ns: Delay range (ns)
        amp_variation: Amplitude ripple amplitude
        seed: Random seed

    Returns:
        bandpass: Complex bandpass [n_ant, 2, n_chan]
        delays_sec: Delays per antenna (seconds)
    """
    np.random.seed(seed)

    # Generate delays
    delays_ns = np.random.uniform(-delay_range_ns, delay_range_ns, n_antennas)
    delays_ns[0] = 0.0
    delays_sec = delays_ns * 1e-9

    bandpass = np.zeros((n_antennas, 2, n_channels), dtype=complex)
    ref_freq = np.mean(freqs)

    for ant in range(n_antennas):
        for pol in range(2):
            # Delay phase
            delay = delays_sec[ant] if pol == 0 else delays_sec[ant] * 1.02
            phase = 2 * np.pi * (freqs - ref_freq) * delay

            # Amplitude with edge rolloff and ripple
            freq_norm = (freqs - freqs[0]) / (freqs[-1] - freqs[0])
            edge_frac = 0.1
            amp = np.ones(n_channels)

            for i, f_norm in enumerate(freq_norm):
                if f_norm < edge_frac:
                    amp[i] = 0.8 + 0.2 * (f_norm / edge_frac)
                elif f_norm > (1 - edge_frac):
                    amp[i] = 0.8 + 0.2 * ((1 - f_norm) / edge_frac)

            if ant > 0:
                ripple = amp_variation * np.sin(2 * np.pi * freq_norm * 5)
                amp += ripple

            bandpass[ant, pol, :] = amp * np.exp(1j * phase)

    return bandpass, delays_sec


def generate_dterms(
    n_antennas: int, leakage_level: float = 0.05, seed: int = 45
) -> np.ndarray:
    """Generate ground truth D-terms (polarization leakage).

    Args:
        n_antennas: Number of antennas
        leakage_level: RMS leakage level (typical: 0.01-0.1)
        seed: Random seed

    Returns:
        d_terms: Complex D-terms [n_ant, 2] where [:, 0] = d_xy, [:, 1] = d_yx
    """
    np.random.seed(seed)
    d_terms = np.zeros((n_antennas, 2), dtype=complex)

    for ant in range(n_antennas):
        for pol in range(2):
            real_part = np.random.normal(0, leakage_level)
            imag_part = np.random.normal(0, leakage_level)
            d_terms[ant, pol] = real_part + 1j * imag_part

    d_terms[0, :] = 0.0 + 0j  # Reference antenna
    return d_terms


def compute_parallactic_angles(
    times_hours: np.ndarray,
    latitude_deg: float,
    source_dec_deg: float,
    n_antennas: int,
) -> np.ndarray:
    """Compute parallactic angles from source/telescope geometry.

    Parallactic angle is the angle between the direction of North Celestial
    Pole and zenith, measured at the source position.

    Formula: tan(PA) = sin(HA) / (cos(lat)*tan(dec) - sin(lat)*cos(HA))
    where HA = hour angle, lat = telescope latitude, dec = source declination

    Args:
        times_hours: Observation times in hours from transit
        latitude_deg: Telescope latitude (degrees)
        source_dec_deg: Source declination (degrees)
        n_antennas: Number of antennas (all see same parallactic angle)

    Returns:
        parallactic_angles: [n_antennas, n_times] in radians
    """
    lat = np.deg2rad(latitude_deg)
    dec = np.deg2rad(source_dec_deg)

    # Hour angle (HA) - negative before transit, positive after
    ha = np.deg2rad(times_hours * 15.0)  # 15 deg/hour

    # Parallactic angle formula
    numerator = np.sin(ha)
    denominator = np.cos(lat) * np.tan(dec) - np.sin(lat) * np.cos(ha)
    pa = np.arctan2(numerator, denominator)

    # All antennas see same PA (ignoring baseline effects)
    pa_array = np.tile(pa, (n_antennas, 1))  # [n_ant, n_time]

    return pa_array


# =============================================================================
# MS CORRUPTION
# =============================================================================


def corrupt_ms(
    ms_path: str,
    effects: Dict,
    add_noise: bool = False,
    sefd: float = 420.0,
) -> None:
    """Corrupt MS DATA column with Jones effects using JonesSimulator.

    Args:
        ms_path: Path to measurement set
        effects: Dict mapping effect names to effect objects
                 Example: {"delays": BandpassDelay(...), "gains": ElectronicGains(...)}
        add_noise: Whether to add thermal noise
        sefd: System Equivalent Flux Density in Jy (VLA L-band ~420)
    """
    print(f"\n{'=' * 70}")
    print("CORRUPTING MS WITH EFFECTS")
    print(f"{'=' * 70}")

    # Get MS info
    ms_handler = MeasurementSetHandler(ms_path)
    summary = ms_handler.get_observation_summary()

    print(f"Antennas: {summary['n_antennas']}")
    print(f"SPWs: {summary['n_spw']}")
    print(f"Effects: {', '.join(effects.keys())}")

    # Read data
    tb = table()
    tb.open(ms_path, nomodify=False)

    model_data = tb.getcol("MODEL_DATA")
    antenna1 = tb.getcol("ANTENNA1")
    antenna2 = tb.getcol("ANTENNA2")
    times_col = tb.getcol("TIME")

    n_corr, n_chan, n_row = model_data.shape
    print(f"\nData shape: {n_corr} corr × {n_chan} chan × {n_row} rows")

    # Get frequency info
    spw_info = summary["frequency_info"][0]
    freqs = spw_info["chan_freqs"]
    chan_width = spw_info.get(
        "chan_width", freqs[1] - freqs[0] if len(freqs) > 1 else 1e6
    )

    # Create simulator and add effects
    sim = JonesSimulator()
    for name, effect in effects.items():
        sim.add_effect(name, effect)
        print(f"  Added: {name}")

    # Reshape for simulator: (n_vis, 4)
    n_vis = n_chan * n_row
    ideal_vis = model_data.transpose(2, 1, 0).reshape(-1, n_corr)
    frequencies = np.tile(freqs, n_row)
    times = np.repeat(times_col, n_chan)
    ant1_arr = np.repeat(antenna1, n_chan)
    ant2_arr = np.repeat(antenna2, n_chan)

    # Corrupt using simulator
    print("\nCorrupting visibilities...", end=" ", flush=True)
    corrupted_vis = sim.corrupt_visibilities(
        ideal_vis, frequencies, times, ant1_arr, ant2_arr, use_gpu=False
    )
    print("Done")

    # Reshape back
    corrupted_data = corrupted_vis.reshape(n_row, n_chan, n_corr).transpose(2, 1, 0)

    # Add thermal noise if requested
    if add_noise:
        unique_times = np.unique(times_col)
        int_time = np.median(np.diff(unique_times)) if len(unique_times) > 1 else 2.0
        sigma = sefd / np.sqrt(2 * chan_width * int_time)

        print(f"\nAdding thermal noise:")
        print(f"  SEFD: {sefd} Jy")
        print(f"  Channel width: {chan_width/1e6:.2f} MHz")
        print(f"  Int time: {int_time} s")
        print(f"  Sigma: {sigma:.4f} Jy per visibility")

        sigma_complex = sigma / np.sqrt(2)
        noise = (
            np.random.normal(0, sigma_complex, corrupted_data.shape)
            + 1j * np.random.normal(0, sigma_complex, corrupted_data.shape)
        )
        corrupted_data += noise
    else:
        print("\nNo thermal noise (exact recovery test)")

    # Write back
    tb.putcol("DATA", corrupted_data)

    # Verify corruption actually happened
    data_check = tb.getcol("DATA")
    model_check = tb.getcol("MODEL_DATA")
    diff = np.abs(data_check - model_check)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    tb.close()

    print("✓ DATA column corrupted")
    print(f"  Corruption verification: max diff = {max_diff:.3e}, mean diff = {mean_diff:.3e}")

    if max_diff < 1e-10:
        print("  WARNING: DATA and MODEL_DATA are identical - corruption may have failed!")


# =============================================================================
# CASA CALIBRATION
# =============================================================================


def run_casa_calibration(
    ms_path: str,
    cal_type: str,
    caltable: str,
    refant: str = "0",
    solint: str = "inf",
    gaintables: Optional[List[str]] = None,
) -> str:
    """Run CASA calibration task.

    Args:
        ms_path: Path to MS
        cal_type: Calibration type: 'K', 'G', 'B', or 'D'
        caltable: Output caltable path
        refant: Reference antenna
        solint: Solution interval
        gaintables: Prior calibration tables to apply

    Returns:
        caltable path
    """
    from casatasks import gaincal, bandpass, polcal

    print(f"\n{'=' * 70}")
    print(f"RUNNING CASA {cal_type} CALIBRATION")
    print(f"{'=' * 70}")

    # Remove existing table
    os.system(f"rm -rf {caltable}")

    gaintables = gaintables or []

    try:
        if cal_type == "K":
            gaincal(
                vis=ms_path,
                caltable=caltable,
                refant=refant,
                gaintype="K",
                solint=solint,
                combine="scan",
                gaintable=gaintables,
            )
        elif cal_type == "G":
            gaincal(
                vis=ms_path,
                caltable=caltable,
                refant=refant,
                gaintype="G",
                solint=solint,
                calmode="ap",
                gaintable=gaintables,
            )
        elif cal_type == "B":
            bandpass(
                vis=ms_path,
                caltable=caltable,
                refant=refant,
                solint=solint,
                combine="scan",
                gaintable=gaintables,
            )
        elif cal_type == "D":
            polcal(
                vis=ms_path,
                caltable=caltable,
                refant=refant,
                poltype="D",
                gaintable=gaintables,
                # polcal doesn't have datacolumn parameter, uses DATA by default
            )
        else:
            raise ValueError(f"Unknown cal_type: {cal_type}")

        print(f"✓ {cal_type} calibration complete: {caltable}")
        return caltable

    except Exception as e:
        print(f"✗ {cal_type} calibration failed: {e}")
        raise


def read_casa_delays(caltable: str, n_antennas: int) -> np.ndarray:
    """Read delays from CASA K-table.

    Args:
        caltable: Path to K-table
        n_antennas: Number of antennas

    Returns:
        delays_ns: Delays in nanoseconds [n_antennas]
    """
    tb = table()
    tb.open(caltable)
    fparam = tb.getcol("FPARAM")
    antennas = tb.getcol("ANTENNA1")
    flags = tb.getcol("FLAG")
    tb.close()

    casa_delays_ns = np.zeros(n_antennas)

    if fparam.ndim == 3:
        n_pol, n_chan, n_rows = fparam.shape
        for row in range(n_rows):
            ant = antennas[row]
            count = 0
            for pol in range(n_pol):
                for chan in range(n_chan):
                    if not flags[pol, chan, row]:
                        casa_delays_ns[ant] += fparam[pol, chan, row]
                        count += 1
            if count > 0:
                casa_delays_ns[ant] /= count
    elif fparam.ndim == 2:
        n_pol, n_rows = fparam.shape
        for row in range(n_rows):
            ant = antennas[row]
            count = 0
            for pol in range(n_pol):
                if not flags[pol, row]:
                    casa_delays_ns[ant] += fparam[pol, row]
                    count += 1
            if count > 0:
                casa_delays_ns[ant] /= count

    return casa_delays_ns


def read_casa_gains(caltable: str, n_antennas: int) -> np.ndarray:
    """Read gains from CASA G-table.

    Args:
        caltable: Path to G-table
        n_antennas: Number of antennas

    Returns:
        gains: Complex gains [n_antennas, 2] (XX, YY)
    """
    tb = table()
    tb.open(caltable)
    cparam = tb.getcol("CPARAM")  # Complex gains
    antennas = tb.getcol("ANTENNA1")
    tb.close()

    gains = np.ones((n_antennas, 2), dtype=complex)

    # cparam shape: (n_pol, n_chan, n_rows) or (n_pol, n_rows)
    if cparam.ndim == 3:
        # Average over channels if frequency-dependent
        for row in range(len(antennas)):
            ant = antennas[row]
            gains[ant, 0] = np.mean(cparam[0, :, row])  # XX
            gains[ant, 1] = np.mean(cparam[1, :, row])  # YY
    else:
        for row in range(len(antennas)):
            ant = antennas[row]
            gains[ant, 0] = cparam[0, row]  # XX
            gains[ant, 1] = cparam[1, row]  # YY

    return gains


def read_casa_bandpass(caltable: str, n_antennas: int, n_channels: int) -> np.ndarray:
    """Read bandpass from CASA B-table.

    Args:
        caltable: Path to B-table
        n_antennas: Number of antennas
        n_channels: Number of channels

    Returns:
        bandpass: Complex bandpass [n_antennas, 2, n_channels]
    """
    tb = table()
    tb.open(caltable)
    cparam = tb.getcol("CPARAM")  # Complex bandpass
    antennas = tb.getcol("ANTENNA1")
    tb.close()

    bandpass = np.ones((n_antennas, 2, n_channels), dtype=complex)

    # cparam shape: (n_pol, n_chan, n_rows)
    n_pol, n_chan, n_rows = cparam.shape

    for row in range(n_rows):
        ant = antennas[row]
        bandpass[ant, 0, :] = cparam[0, :, row]  # XX
        bandpass[ant, 1, :] = cparam[1, :, row]  # YY

    return bandpass


def read_casa_dterms(caltable: str, n_antennas: int) -> np.ndarray:
    """Read D-terms from CASA D-table (polcal).

    Args:
        caltable: Path to D-table
        n_antennas: Number of antennas

    Returns:
        dterms: Complex D-terms [n_antennas, 2] where [:, 0] = d_xy, [:, 1] = d_yx
    """
    tb = table()
    tb.open(caltable)
    cparam = tb.getcol("CPARAM")  # Complex D-terms
    antennas = tb.getcol("ANTENNA1")
    tb.close()

    dterms = np.zeros((n_antennas, 2), dtype=complex)

    # cparam for polcal D-terms: (n_pol=2, n_chan, n_rows)
    # The two pols are d_xy and d_yx
    if cparam.ndim == 3:
        # Average over channels if needed
        for row in range(len(antennas)):
            ant = antennas[row]
            dterms[ant, 0] = np.mean(cparam[0, :, row])  # d_xy
            dterms[ant, 1] = np.mean(cparam[1, :, row])  # d_yx
    else:
        for row in range(len(antennas)):
            ant = antennas[row]
            dterms[ant, 0] = cparam[0, row]  # d_xy
            dterms[ant, 1] = cparam[1, row]  # d_yx

    return dterms


# =============================================================================
# STATE MANAGEMENT
# =============================================================================


def save_ground_truth(filepath: str, ground_truth: Dict) -> None:
    """Save ground truth values to .npz file.

    Args:
        filepath: Path to .npz file
        ground_truth: Dict containing ground truth arrays
    """
    np.savez(filepath, **ground_truth)
    print(f"✓ Ground truth saved: {filepath}")


def load_ground_truth(filepath: str) -> Dict:
    """Load ground truth values from .npz file.

    Args:
        filepath: Path to .npz file

    Returns:
        Dictionary with ground truth values
    """
    data = np.load(filepath)
    ground_truth = {key: data[key] for key in data.files}
    print(f"✓ Ground truth loaded: {filepath}")
    return ground_truth


def save_solver_results(filepath: str, solver, effect_name: str) -> None:
    """Save solver results to .npz file.

    Args:
        filepath: Path to .npz file
        solver: CalibrationSolver instance
        effect_name: Name of effect (K, G, B, etc.)
    """
    solution = solver.get_solution(effect_name)
    trace_data = {key: val for key, val in solver.trace.items()}

    np.savez(
        filepath,
        solution=solution,
        effect_name=effect_name,
        **trace_data
    )
    print(f"✓ Solver results saved: {filepath}")


# =============================================================================
# COMPARISON AND VALIDATION
# =============================================================================


def print_section_header(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")


def compare_delays(
    truth_delays_ns: np.ndarray,
    casa_delays_ns: np.ndarray,
    recovered_delays_ns: np.ndarray,
) -> Dict:
    """Compare delay solutions (returns dict, no printing).

    Args:
        truth_delays_ns: Ground truth delays (ns)
        casa_delays_ns: CASA delays (ns)
        recovered_delays_ns: Our recovered delays (ns)

    Returns:
        Dictionary with error statistics
    """
    n_antennas = len(truth_delays_ns)

    casa_errors = []
    our_errors = []

    for ant in range(n_antennas):
        truth = truth_delays_ns[ant]
        casa = casa_delays_ns[ant]
        ours = recovered_delays_ns[ant]

        casa_diff = casa - truth
        our_diff = ours - truth

        if ant > 0:  # Skip reference antenna
            casa_errors.append(casa_diff)
            our_errors.append(our_diff)

    # Statistics
    casa_errors = np.array(casa_errors)
    our_errors = np.array(our_errors)

    casa_rms = np.sqrt(np.mean(casa_errors**2))
    our_rms = np.sqrt(np.mean(our_errors**2))

    # Identify problematic antennas (errors > 0.1 ns)
    bad_ants = np.where(np.abs(our_errors) > 0.1)[0] + 1  # +1 to skip ref ant

    return {
        "effect": "K",
        "truth": truth_delays_ns,
        "casa": casa_delays_ns,
        "ours": recovered_delays_ns,
        "casa_errors": casa_errors,
        "our_errors": our_errors,
        "casa_rms": casa_rms,
        "our_rms": our_rms,
        "casa_mean": np.mean(casa_errors),
        "our_mean": np.mean(our_errors),
        "casa_std": np.std(casa_errors),
        "our_std": np.std(our_errors),
        "casa_max": np.max(np.abs(casa_errors)),
        "our_max": np.max(np.abs(our_errors)),
        "bad_antennas": bad_ants.tolist() if len(bad_ants) > 0 else [],
    }


def print_delay_comparison(results: Dict) -> None:
    """Print delay comparison results from dict.

    Args:
        results: Dictionary from compare_delays()
    """
    print_section_header("DELAY COMPARISON: TRUTH vs CASA vs OURS")

    truth = results["truth"]
    casa = results["casa"]
    ours = results["ours"]
    n_antennas = len(truth)

    # Print table
    print(f"\n{'Ant':<5} {'Truth':<12} {'CASA':<12} {'Ours':<12} {'CASA-Truth':<12} {'Ours-Truth':<12}")
    print("-" * 70)

    for ant in range(n_antennas):
        casa_diff = casa[ant] - truth[ant]
        our_diff = ours[ant] - truth[ant]
        print(f"{ant:<5} {truth[ant]:>11.3f} {casa[ant]:>11.3f} {ours[ant]:>11.3f} "
              f"{casa_diff:>11.3f} {our_diff:>11.3f}")

    # Statistics
    print(f"\n{'=' * 70}")
    print("ERROR STATISTICS (excluding reference antenna)")
    print(f"{'=' * 70}")

    print(f"\nCASA errors (ns):")
    print(f"  Mean: {results['casa_mean']:.4f}")
    print(f"  Std:  {results['casa_std']:.4f}")
    print(f"  RMS:  {results['casa_rms']:.4f}")
    print(f"  Max:  {results['casa_max']:.4f}")

    print(f"\nOur errors (ns):")
    print(f"  Mean: {results['our_mean']:.4f}")
    print(f"  Std:  {results['our_std']:.4f}")
    print(f"  RMS:  {results['our_rms']:.4f}")
    print(f"  Max:  {results['our_max']:.4f}")

    if results["bad_antennas"]:
        print(f"\nAntennas with errors > 0.1 ns: {results['bad_antennas']}")
        for ant_idx in results["bad_antennas"]:
            error = results["our_errors"][ant_idx - 1]
            print(f"  Ant {ant_idx}: {error:.3f} ns")


def compare_gains(
    truth_gains: np.ndarray,
    casa_gains: np.ndarray,
    recovered_gains: np.ndarray,
) -> Dict:
    """Compare gain solutions (returns dict, no printing).

    Args:
        truth_gains: Ground truth gains [n_ant, 2] complex
        casa_gains: CASA gains [n_ant, 2] complex
        recovered_gains: Our recovered gains [n_ant, 2] complex

    Returns:
        Dictionary with error statistics
    """
    n_antennas = truth_gains.shape[0]

    # Compute amplitude and phase errors
    casa_amp_errors = []
    our_amp_errors = []
    casa_phase_errors = []
    our_phase_errors = []

    for ant in range(n_antennas):
        if ant == 0:  # Skip reference antenna
            continue

        for pol in range(2):
            truth_amp = np.abs(truth_gains[ant, pol])
            truth_phase = np.angle(truth_gains[ant, pol])

            casa_amp = np.abs(casa_gains[ant, pol])
            casa_phase = np.angle(casa_gains[ant, pol])

            ours_amp = np.abs(recovered_gains[ant, pol])
            ours_phase = np.angle(recovered_gains[ant, pol])

            casa_amp_errors.append(casa_amp - truth_amp)
            our_amp_errors.append(ours_amp - truth_amp)

            # Phase errors (handle wrapping)
            casa_phase_diff = (casa_phase - truth_phase + np.pi) % (2 * np.pi) - np.pi
            our_phase_diff = (ours_phase - truth_phase + np.pi) % (2 * np.pi) - np.pi

            casa_phase_errors.append(casa_phase_diff * 180 / np.pi)  # Convert to degrees
            our_phase_errors.append(our_phase_diff * 180 / np.pi)

    casa_amp_errors = np.array(casa_amp_errors)
    our_amp_errors = np.array(our_amp_errors)
    casa_phase_errors = np.array(casa_phase_errors)
    our_phase_errors = np.array(our_phase_errors)

    # Find problematic antennas (amp error > 0.05 or phase error > 5 deg)
    amp_threshold = 0.05
    phase_threshold = 5.0
    bad_ants = set()

    for ant in range(1, n_antennas):
        for pol in range(2):
            idx = (ant - 1) * 2 + pol
            if np.abs(our_amp_errors[idx]) > amp_threshold or np.abs(our_phase_errors[idx]) > phase_threshold:
                bad_ants.add(ant)

    return {
        "effect": "G",
        "truth": truth_gains,
        "casa": casa_gains,
        "ours": recovered_gains,
        "casa_amp_errors": casa_amp_errors,
        "our_amp_errors": our_amp_errors,
        "casa_phase_errors": casa_phase_errors,
        "our_phase_errors": our_phase_errors,
        "casa_amp_rms": np.sqrt(np.mean(casa_amp_errors**2)),
        "our_amp_rms": np.sqrt(np.mean(our_amp_errors**2)),
        "casa_phase_rms": np.sqrt(np.mean(casa_phase_errors**2)),
        "our_phase_rms": np.sqrt(np.mean(our_phase_errors**2)),
        "bad_antennas": sorted(list(bad_ants)),
    }


def print_gains_comparison(results: Dict) -> None:
    """Print gains comparison results from dict.

    Args:
        results: Dictionary from compare_gains()
    """
    print_section_header("GAINS COMPARISON: TRUTH vs CASA vs OURS")

    truth = results["truth"]
    casa = results["casa"]
    ours = results["ours"]
    n_antennas = truth.shape[0]

    # Print table for XX polarization
    print("\nXX Polarization:")
    print(f"{'Ant':<5} {'Truth Amp':<12} {'CASA Amp':<12} {'Ours Amp':<12} {'Truth Phase':<12} {'CASA Phase':<12} {'Ours Phase':<12}")
    print("-" * 90)

    for ant in range(min(n_antennas, 10)):  # First 10 antennas
        t_amp = np.abs(truth[ant, 0])
        c_amp = np.abs(casa[ant, 0])
        o_amp = np.abs(ours[ant, 0])
        t_ph = np.angle(truth[ant, 0]) * 180 / np.pi
        c_ph = np.angle(casa[ant, 0]) * 180 / np.pi
        o_ph = np.angle(ours[ant, 0]) * 180 / np.pi
        print(f"{ant:<5} {t_amp:>11.4f} {c_amp:>11.4f} {o_amp:>11.4f} {t_ph:>11.2f} {c_ph:>11.2f} {o_ph:>11.2f}")

    if n_antennas > 10:
        print(f"... ({n_antennas - 10} more antennas)")

    # Statistics
    print(f"\n{'=' * 70}")
    print("ERROR STATISTICS (excluding reference antenna)")
    print(f"{'=' * 70}")

    print(f"\nCASA amplitude errors:")
    print(f"  RMS: {results['casa_amp_rms']:.4f}")
    print(f"\nOur amplitude errors:")
    print(f"  RMS: {results['our_amp_rms']:.4f}")

    print(f"\nCASA phase errors (deg):")
    print(f"  RMS: {results['casa_phase_rms']:.4f}")
    print(f"\nOur phase errors (deg):")
    print(f"  RMS: {results['our_phase_rms']:.4f}")

    if results["bad_antennas"]:
        print(f"\nAntennas with significant errors: {results['bad_antennas']}")


def compare_bandpass(
    truth_bp: np.ndarray,
    casa_bp: np.ndarray,
    recovered_bp: np.ndarray,
    freqs: np.ndarray,
) -> Dict:
    """Compare bandpass solutions (returns dict, no printing).

    Args:
        truth_bp: Ground truth bandpass [n_ant, 2, n_chan] complex
        casa_bp: CASA bandpass [n_ant, 2, n_chan] complex
        recovered_bp: Our recovered bandpass [n_ant, 2, n_chan] complex
        freqs: Channel frequencies

    Returns:
        Dictionary with error statistics
    """
    n_antennas, n_pol, n_chan = truth_bp.shape

    # Compute amplitude and phase errors averaged over frequency
    casa_amp_errors = []
    our_amp_errors = []
    casa_phase_errors = []
    our_phase_errors = []

    for ant in range(1, n_antennas):  # Skip reference antenna
        for pol in range(n_pol):
            truth_amp = np.abs(truth_bp[ant, pol, :])
            truth_phase = np.angle(truth_bp[ant, pol, :])

            casa_amp = np.abs(casa_bp[ant, pol, :])
            casa_phase = np.angle(casa_bp[ant, pol, :])

            ours_amp = np.abs(recovered_bp[ant, pol, :])
            ours_phase = np.angle(recovered_bp[ant, pol, :])

            # Mean error across channels
            casa_amp_errors.append(np.mean(casa_amp - truth_amp))
            our_amp_errors.append(np.mean(ours_amp - truth_amp))

            # Phase errors (handle wrapping)
            casa_phase_diff = (casa_phase - truth_phase + np.pi) % (2 * np.pi) - np.pi
            our_phase_diff = (ours_phase - truth_phase + np.pi) % (2 * np.pi) - np.pi

            casa_phase_errors.append(np.mean(casa_phase_diff) * 180 / np.pi)
            our_phase_errors.append(np.mean(our_phase_diff) * 180 / np.pi)

    casa_amp_errors = np.array(casa_amp_errors)
    our_amp_errors = np.array(our_amp_errors)
    casa_phase_errors = np.array(casa_phase_errors)
    our_phase_errors = np.array(our_phase_errors)

    return {
        "effect": "B",
        "truth": truth_bp,
        "casa": casa_bp,
        "ours": recovered_bp,
        "freqs": freqs,
        "casa_amp_rms": np.sqrt(np.mean(casa_amp_errors**2)),
        "our_amp_rms": np.sqrt(np.mean(our_amp_errors**2)),
        "casa_phase_rms": np.sqrt(np.mean(casa_phase_errors**2)),
        "our_phase_rms": np.sqrt(np.mean(our_phase_errors**2)),
    }


def print_bandpass_comparison(results: Dict) -> None:
    """Print bandpass comparison results from dict.

    Args:
        results: Dictionary from compare_bandpass()
    """
    print_section_header("BANDPASS COMPARISON: TRUTH vs CASA vs OURS")

    print(f"\nAmplitude RMS error:")
    print(f"  CASA: {results['casa_amp_rms']:.4f}")
    print(f"  Ours: {results['our_amp_rms']:.4f}")

    print(f"\nPhase RMS error (deg):")
    print(f"  CASA: {results['casa_phase_rms']:.4f}")
    print(f"  Ours: {results['our_phase_rms']:.4f}")

    print(f"\n(See plots for per-channel details)")


def compare_leakage(
    truth_dterms: np.ndarray,
    casa_dterms: np.ndarray,
    recovered_dterms: np.ndarray,
) -> Dict:
    """Compare leakage (D-term) solutions (returns dict, no printing).

    Args:
        truth_dterms: Ground truth D-terms [n_ant, 2] complex
        casa_dterms: CASA D-terms [n_ant, 2] complex
        recovered_dterms: Our recovered D-terms [n_ant, 2] complex

    Returns:
        Dictionary with error statistics
    """
    n_antennas = truth_dterms.shape[0]

    casa_errors = []
    our_errors = []

    for ant in range(1, n_antennas):  # Skip reference antenna
        for pol in range(2):
            truth_val = truth_dterms[ant, pol]
            casa_val = casa_dterms[ant, pol]
            ours_val = recovered_dterms[ant, pol]

            # Magnitude of complex difference
            casa_errors.append(np.abs(casa_val - truth_val))
            our_errors.append(np.abs(ours_val - truth_val))

    casa_errors = np.array(casa_errors)
    our_errors = np.array(our_errors)

    return {
        "effect": "D",
        "truth": truth_dterms,
        "casa": casa_dterms,
        "ours": recovered_dterms,
        "casa_rms": np.sqrt(np.mean(casa_errors**2)),
        "our_rms": np.sqrt(np.mean(our_errors**2)),
        "casa_mean": np.mean(casa_errors),
        "our_mean": np.mean(our_errors),
    }


def print_leakage_comparison(results: Dict) -> None:
    """Print leakage comparison results from dict.

    Args:
        results: Dictionary from compare_leakage()
    """
    print_section_header("LEAKAGE (D-TERMS) COMPARISON: TRUTH vs CASA vs OURS")

    truth = results["truth"]
    casa = results["casa"]
    ours = results["ours"]
    n_antennas = truth.shape[0]

    # Print table
    print(f"\n{'Ant':<5} {'Truth |d_xy|':<15} {'CASA |d_xy|':<15} {'Ours |d_xy|':<15} {'Truth |d_yx|':<15} {'CASA |d_yx|':<15} {'Ours |d_yx|':<15}")
    print("-" * 95)

    for ant in range(min(n_antennas, 10)):
        t_xy = np.abs(truth[ant, 0])
        c_xy = np.abs(casa[ant, 0])
        o_xy = np.abs(ours[ant, 0])
        t_yx = np.abs(truth[ant, 1])
        c_yx = np.abs(casa[ant, 1])
        o_yx = np.abs(ours[ant, 1])
        print(f"{ant:<5} {t_xy:>14.4f} {c_xy:>14.4f} {o_xy:>14.4f} {t_yx:>14.4f} {c_yx:>14.4f} {o_yx:>14.4f}")

    if n_antennas > 10:
        print(f"... ({n_antennas - 10} more antennas)")

    print(f"\n{'=' * 70}")
    print("ERROR STATISTICS (excluding reference antenna)")
    print(f"{'=' * 70}")

    print(f"\nCASA errors:")
    print(f"  Mean: {results['casa_mean']:.6f}")
    print(f"  RMS:  {results['casa_rms']:.6f}")

    print(f"\nOur errors:")
    print(f"  Mean: {results['our_mean']:.6f}")
    print(f"  RMS:  {results['our_rms']:.6f}")


def report_metric(metric_name: str, value: float, unit: str = "") -> None:
    """Report a metric value without pass/fail logic.

    Args:
        metric_name: Name of metric (e.g., "Delay RMS")
        value: Metric value
        unit: Optional unit string
    """
    print(f"\n{metric_name}: {value:.4f} {unit}".strip())
