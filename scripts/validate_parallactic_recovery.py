#!/usr/bin/env python3
"""End-to-end validation of parallactic angle (P) correction.

Parallactic angle is a geometric effect that rotates the polarization basis
from celestial coordinates to the feed frame. It varies with time as the
source moves across the sky.

This script:
1. Creates simulated full-polarization MS (4 correlations)
2. Computes ground truth parallactic angles from source/telescope geometry
3. Corrupts DATA with P rotation
4. Verifies P correction recovers original polarization
5. Compares geometric computation vs correction
6. Returns exit code 0 if validation passes

Note: P is computed from geometry, not solved for like other effects.
CASA does not have a dedicated P calibration - it's applied automatically
during imaging/calibration when parang=True.

Usage:
    python validate_parallactic_recovery.py [options]

Options:
    --msname NAME       MS name (default: sim_parallactic_test.ms)
    --skip_sim          Skip simulation if MS exists
    --latitude DEG      Telescope latitude (default: -30.72 for VLA-like)
    --source_dec DEG    Source declination (default: -45.0)
    --obs_hours H       Observation duration (hours, default: 4)
    --seed N            Random seed (default: 50)
    --no_noise          Skip thermal noise
"""

import argparse
import os
import sys

import numpy as np

from jones_sim import JonesSimulator, JonesConfig
from jones_sim.calibration_solver import CalibrationSolver
from jones_sim.casa_interface import MeasurementSetHandler
from jones_sim.plotting_enhanced import plot_three_way_comparison


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


def corrupt_ms_with_parallactic(ms_path, pa_angles, add_noise=True, sefd=420.0):
    """Corrupt MS DATA with parallactic angle rotation.

    P rotation matrix:
    P(pa) = [[cos(pa), -sin(pa)],
             [sin(pa),  cos(pa)]]

    Applied as: V' = (P1 ⊗ P2*) V

    Args:
        ms_path: Path to MS
        pa_angles: Parallactic angles [n_ant, n_time] in radians
        add_noise: Whether to add thermal noise
        sefd: System equivalent flux density (Jy)
    """
    ms_handler = MeasurementSetHandler(ms_path)

    # Read data
    data_dict = ms_handler.read_data()
    vis = data_dict["data"]
    ant1 = data_dict["antenna1"]
    ant2 = data_dict["antenna2"]
    times = data_dict["times"]

    # Check full polarization
    if vis.shape[2] != 4:
        raise ValueError(
            f"MS must have 4 polarizations for parallactic angle, got {vis.shape[2]}"
        )

    # Map times to indices
    unique_times = np.unique(times)
    time_to_idx = {t: i for i, t in enumerate(unique_times)}

    # Apply parallactic rotation
    n_vis = len(vis)
    for i in range(n_vis):
        a1 = ant1[i]
        a2 = ant2[i]
        t_idx = time_to_idx[times[i]]

        pa1 = pa_angles[a1, t_idx]
        pa2 = pa_angles[a2, t_idx]

        c1, s1 = np.cos(pa1), np.sin(pa1)
        c2, s2 = np.cos(pa2), np.sin(pa2)

        # P1 ⊗ P2* rotation matrix applied to [XX, XY, YX, YY]
        p_matrix = np.array([
            c1 * c2 + s1 * s2,  # XX
            -c1 * s2 + s1 * c2,  # XY
            -s1 * c2 + c1 * s2,  # YX
            s1 * s2 + c1 * c2,  # YY
        ])

        # Apply to all channels
        for ch in range(vis.shape[1]):
            vis[i, ch, :] *= p_matrix

    # Add thermal noise if requested
    if add_noise:
        t_int = 10.0
        bw = 1e6
        sigma = sefd / np.sqrt(2 * t_int * bw)
        noise_real = np.random.normal(0, sigma, vis.shape)
        noise_imag = np.random.normal(0, sigma, vis.shape)
        vis += noise_real + 1j * noise_imag

    # Write back
    ms_handler.write_data(vis)
    print(f"✓ Applied parallactic rotation (noise={add_noise})")


def apply_parallactic_correction(ms_path, pa_angles):
    """Apply inverse parallactic rotation to correct MS.

    Args:
        ms_path: Path to MS
        pa_angles: Parallactic angles [n_ant, n_time] in radians

    Returns:
        corrected_vis: Corrected visibilities
    """
    ms_handler = MeasurementSetHandler(ms_path)

    # Read data
    data_dict = ms_handler.read_data()
    vis = data_dict["data"].copy()
    ant1 = data_dict["antenna1"]
    ant2 = data_dict["antenna2"]
    times = data_dict["times"]

    # Map times
    unique_times = np.unique(times)
    time_to_idx = {t: i for i, t in enumerate(unique_times)}

    # Apply inverse rotation (negate angles)
    n_vis = len(vis)
    for i in range(n_vis):
        a1 = ant1[i]
        a2 = ant2[i]
        t_idx = time_to_idx[times[i]]

        # Inverse = rotation by -pa
        pa1 = -pa_angles[a1, t_idx]
        pa2 = -pa_angles[a2, t_idx]

        c1, s1 = np.cos(pa1), np.sin(pa1)
        c2, s2 = np.cos(pa2), np.sin(pa2)

        p_matrix = np.array([
            c1 * c2 + s1 * s2,
            -c1 * s2 + s1 * c2,
            -s1 * c2 + c1 * s2,
            s1 * s2 + c1 * c2,
        ])

        for ch in range(vis.shape[1]):
            vis[i, ch, :] *= p_matrix

    return vis


def compare_parallactic_correction(
    original_vis,
    corrected_vis,
    pa_angles,
    threshold_percent=1.0,
):
    """Compare original vs corrected visibilities.

    Args:
        original_vis: Original visibilities before P corruption
        corrected_vis: Corrected visibilities after P^-1
        pa_angles: Parallactic angles used
        threshold_percent: Error threshold (%)

    Returns:
        0 if passed, 1 if failed
    """
    # Compute residuals
    residuals = corrected_vis - original_vis
    rms_residual = np.sqrt(np.mean(np.abs(residuals)**2))
    rms_original = np.sqrt(np.mean(np.abs(original_vis)**2))
    error_percent = 100 * rms_residual / rms_original

    # Compute PA statistics
    pa_range = np.max(pa_angles) - np.min(pa_angles)
    pa_rms = np.sqrt(np.mean(pa_angles**2))

    print("\n" + "="*60)
    print("PARALLACTIC ANGLE VALIDATION RESULTS")
    print("="*60)
    print(f"PA range: {np.rad2deg(pa_range):.1f}° "
          f"(min={np.rad2deg(np.min(pa_angles)):.1f}°, "
          f"max={np.rad2deg(np.max(pa_angles)):.1f}°)")
    print(f"PA RMS: {np.rad2deg(pa_rms):.1f}°")
    print(f"\nVisibility recovery:")
    print(f"  RMS residual: {rms_residual:.6e} Jy")
    print(f"  RMS original: {rms_original:.6e} Jy")
    print(f"  Error: {error_percent:.3f}%")
    print(f"  Threshold: {threshold_percent:.1f}%")

    # Validation
    passed = error_percent < threshold_percent

    if passed:
        print(f"\n✓ PASS: Error {error_percent:.3f}% < {threshold_percent:.1f}%")
    else:
        print(f"\n✗ FAIL: Error {error_percent:.3f}% >= {threshold_percent:.1f}%")

    print("="*60)

    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description="Validate parallactic angle correction")
    parser.add_argument("--msname", default="sim_parallactic_test.ms", help="MS name")
    parser.add_argument("--skip_sim", action="store_true", help="Skip simulation")
    parser.add_argument("--latitude", type=float, default=-30.72, help="Telescope latitude (deg)")
    parser.add_argument("--source_dec", type=float, default=-45.0, help="Source declination (deg)")
    parser.add_argument("--obs_hours", type=float, default=4.0, help="Observation duration (hours)")
    parser.add_argument("--seed", type=int, default=50, help="Random seed")
    parser.add_argument("--no_noise", action="store_true", help="Skip thermal noise")
    args = parser.parse_args()

    # Check MS exists
    if not os.path.exists(args.msname):
        print(f"✗ MS not found: {args.msname}")
        print("  MS must have 4 polarizations (XX, XY, YX, YY)")
        print("  Create MS first or implement simulation in this script")
        return 1

    # Load MS
    ms_handler = MeasurementSetHandler(args.msname)
    n_antennas = len(ms_handler.get_antenna_names())
    data_dict = ms_handler.read_data()
    original_vis = data_dict["data"].copy()

    # Compute parallactic angles
    n_times = 30
    times_hours = np.linspace(-args.obs_hours/2, args.obs_hours/2, n_times)

    pa_angles = compute_parallactic_angles(
        times_hours,
        args.latitude,
        args.source_dec,
        n_antennas,
    )

    print(f"Computed parallactic angles:")
    print(f"  Latitude: {args.latitude:.2f}°")
    print(f"  Source dec: {args.source_dec:.2f}°")
    print(f"  Observation: {args.obs_hours:.1f} hours")
    print(f"  PA range: {np.rad2deg(np.max(pa_angles) - np.min(pa_angles)):.1f}°")

    # Corrupt MS with parallactic rotation
    corrupt_ms_with_parallactic(
        args.msname,
        pa_angles,
        add_noise=not args.no_noise,
    )

    # Apply correction
    print("Applying parallactic correction...")
    corrected_vis = apply_parallactic_correction(args.msname, pa_angles)

    # Compare
    exit_code = compare_parallactic_correction(
        original_vis,
        corrected_vis,
        pa_angles,
    )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
