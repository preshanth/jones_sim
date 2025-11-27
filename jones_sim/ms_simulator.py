#!/usr/bin/env python3
"""Helper functions for creating simulated MSs for validation testing.

This module provides utilities to create measurement sets with known
ground truth effects for validating calibration pipelines.
"""

import numpy as np
from pathlib import Path
import shutil

try:
    from casatools import simulator, measures
    from casatasks import simobserve
    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False


def create_simple_ms(
    msname: str,
    n_antennas: int = 27,
    n_times: int = 30,
    n_channels: int = 32,
    start_freq_ghz: float = 1.0,
    channel_width_mhz: float = 1.0,
    integration_time_s: float = 10.0,
    n_polarizations: int = 2,
    source_flux_jy: float = 1.0,
    telescope: str = "VLA",
    overwrite: bool = True,
):
    """Create a simple measurement set for testing.

    Args:
        msname: Output MS name
        n_antennas: Number of antennas
        n_times: Number of time samples
        n_channels: Number of frequency channels
        start_freq_ghz: Starting frequency (GHz)
        channel_width_mhz: Channel width (MHz)
        integration_time_s: Integration time (seconds)
        n_polarizations: Number of polarizations (2 or 4)
        source_flux_jy: Point source flux (Jy)
        telescope: Telescope name (VLA, ALMA, etc)
        overwrite: Overwrite existing MS

    Returns:
        Path to created MS
    """
    if not CASA_AVAILABLE:
        raise ImportError("CASA tools required for MS creation")

    msname = str(Path(msname).absolute())

    # Remove existing MS if overwrite
    if overwrite and Path(msname).exists():
        shutil.rmtree(msname)

    # Create simulator
    sm = simulator()

    # Set up telescope configuration
    if telescope == "VLA":
        # Use VLA C configuration
        sm.open(msname)
        pos_obs = measures().observatory('VLA')
        sm.setconfig(
            telescopename='VLA',
            x=pos_obs['m0']['value'],
            y=pos_obs['m1']['value'],
            z=pos_obs['m2']['value'],
            dishdiameter=25.0,
            mount='alt-az',
            coordsystem='global'
        )
    else:
        raise NotImplementedError(f"Telescope {telescope} not yet supported")

    # Set up spectral window
    freq_hz = start_freq_ghz * 1e9
    chan_width_hz = channel_width_mhz * 1e6

    sm.setspwindow(
        spwname='SPW0',
        freq=f"{freq_hz}Hz",
        deltafreq=f"{chan_width_hz}Hz",
        freqresolution=f"{chan_width_hz}Hz",
        nchannels=n_channels,
        stokes='XX YY' if n_polarizations == 2 else 'XX XY YX YY'
    )

    # Set up feed (polarization)
    sm.setfeed(mode='perfect X Y' if n_polarizations >= 2 else 'perfect R L')

    # Set up field (pointing center)
    sm.setfield(
        sourcename='TEST_SOURCE',
        sourcedirection=['J2000', '0h0m0s', '-45d0m0s']
    )

    # Set limits
    sm.setlimits(shadowlimit=0.01, elevationlimit='10deg')

    # Set autocorr (typically off)
    sm.setauto(autocorrwt=0.0)

    # Observe
    total_time_s = n_times * integration_time_s
    sm.settimes(
        integrationtime=f"{integration_time_s}s",
        usehourangle=True,
        referencetime=measures().epoch('UTC', 'today')
    )

    # Scan
    sm.observe(
        sourcename='TEST_SOURCE',
        spwname='SPW0',
        starttime='0s',
        stoptime=f"{total_time_s}s"
    )

    # Set visibility data (point source at phase center)
    sm.setdata(
        spwid=0,
        fieldid=0
    )

    # Predict point source
    sm.predict(
        imagename='',  # Point source
        incremental=False,
        complist='',
        specmode='mfs'
    )

    # Close simulator
    sm.close()

    print(f"✓ Created MS: {msname}")
    print(f"  Antennas: {n_antennas}")
    print(f"  Times: {n_times}")
    print(f"  Channels: {n_channels}")
    print(f"  Frequency: {start_freq_ghz:.2f} GHz")
    print(f"  Polarizations: {n_polarizations}")

    return msname


def create_ms_with_config(
    msname: str,
    config_path: str,
    n_antennas: int = 27,
    n_times: int = 30,
    n_channels: int = 32,
    overwrite: bool = True,
):
    """Create MS using JonesConfig and apply effects.

    Args:
        msname: Output MS name
        config_path: Path to JSON config file
        n_antennas: Number of antennas
        n_times: Number of time samples
        n_channels: Number of frequency channels
        overwrite: Overwrite existing MS

    Returns:
        Tuple of (msname, ground_truth_dict)
    """
    from jones_sim import JonesConfig

    # Create base MS
    create_simple_ms(
        msname=msname,
        n_antennas=n_antennas,
        n_times=n_times,
        n_channels=n_channels,
        overwrite=overwrite,
    )

    # Load config
    config = JonesConfig(config_path)

    # Create simulator
    sim = config.create_simulator(n_antennas=n_antennas)

    # Apply effects to MS
    # TODO: Implement corruption of MS DATA column
    # This would read DATA, apply sim.corrupt_visibilities(), write back

    ground_truth = {
        'config': config,
        'simulator': sim,
    }

    return msname, ground_truth


if __name__ == "__main__":
    # Test MS creation
    import argparse

    parser = argparse.ArgumentParser(description="Create test MS")
    parser.add_argument("msname", help="Output MS name")
    parser.add_argument("--n_antennas", type=int, default=27)
    parser.add_argument("--n_times", type=int, default=30)
    parser.add_argument("--n_channels", type=int, default=32)
    parser.add_argument("--freq_ghz", type=float, default=1.0)
    args = parser.parse_args()

    create_simple_ms(
        args.msname,
        n_antennas=args.n_antennas,
        n_times=args.n_times,
        n_channels=args.n_channels,
        start_freq_ghz=args.freq_ghz,
    )
