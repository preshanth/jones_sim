#!/usr/bin/env python3
"""Simulate VLA observation of 3C286 for delay validation.

Creates a measurement set with:
- 3C286 as point source
- VLA D-config, 27 antennas
- L-band: 1-2 GHz
- Configurable channels and observation time
- 4 correlations (RR, RL, LR, LL)

Usage:
    python simulate_3c286.py [options]

Options:
    --msname NAME       Output MS name (default: sim_3c286.ms)
    --n_channels N      Number of channels (default: 64)
    --obs_time MIN      Observation time in minutes (default: 5)
    --int_time SEC      Integration time in seconds (default: 2)
    --antconfig PATH    Antenna config file (default: auto-detect)

After creation, run setjy to get proper 3C286 flux model.
"""

import os
import argparse
import numpy as np

from casatools import componentlist, quanta, simulator, measures, table
from casatasks import ft, flagdata, setjy
from casatasks.private import simutil


def find_antenna_config():
    """Find VLA D-config antenna file.

    Searches common CASA data locations.

    Returns:
        Path to vla.d.cfg or None if not found
    """
    # Common locations for CASA data
    possible_paths = [
        # User's CASA data directory (most common for modular CASA)
        os.path.expanduser("~/.casa/data/alma/simmos/vla.d.cfg"),
        # Standard CASA installations
        "/home/casa/data/trunk/alma/simmos/vla.d.cfg",
        # Environment variable
        os.path.join(os.environ.get("CASADATA", ""), "alma/simmos/vla.d.cfg"),
        # Common Linux locations
        "/usr/share/casa/data/alma/simmos/vla.d.cfg",
        "/opt/casa/data/alma/simmos/vla.d.cfg",
    ]

    # Try casatools to get data path
    try:
        import casatools
        casa_data = casatools.ctsys.resolve("alma/simmos/vla.d.cfg")
        if os.path.exists(casa_data):
            return casa_data
    except:
        pass

    for path in possible_paths:
        if path and os.path.exists(path):
            return path

    return None

# CASA tools
tb = table()
cl = componentlist()
qa = quanta()
sm = simulator()
me = measures()
mysu = simutil.simutil()

# 3C286 position (J2000)
SOURCE_RA = '13h31m08.288s'
SOURCE_DEC = '+30d30m32.96s'
SOURCE_NAME = '3C286'

# Frequency setup
FREQ_START = 1.0e9  # 1 GHz
FREQ_END = 2.0e9    # 2 GHz
REF_FREQ = 1.5e9    # Center frequency


def create_componentlist(flux_jy: float = 17.0, clname: str = 'temp_3c286.cl'):
    """Create component list for 3C286.

    Args:
        flux_jy: Initial flux density in Jy (will be replaced by setjy)
        clname: Component list filename

    Returns:
        Path to component list
    """
    os.system(f'rm -rf {clname}')
    cl.done()

    cl.addcomponent(
        dir=f'J2000 {SOURCE_RA} {SOURCE_DEC}',
        flux=flux_jy,
        fluxunit='Jy',
        freq=f'{REF_FREQ/1e9}GHz',
        shape='point',
        spectrumtype='spectral index',
        index=0  # Flat spectrum initially, setjy will fix
    )

    cl.rename(clname)
    cl.done()

    return clname


def create_ms(
    msname: str = 'sim_3c286.ms',
    n_channels: int = 64,
    obs_time_min: float = 5.0,
    int_time_sec: float = 2.0,
    antconfig: str = None,
):
    """Create empty measurement set with VLA D-config.

    Args:
        msname: Output MS name
        n_channels: Number of frequency channels
        obs_time_min: Total observation time in minutes
        int_time_sec: Integration time in seconds
        antconfig: Path to antenna config file (auto-detect if None)

    Returns:
        Path to created MS
    """
    print(f"\n{'=' * 70}")
    print("CREATING MEASUREMENT SET")
    print(f"{'=' * 70}")

    os.system(f'rm -rf {msname}')
    sm.open(ms=msname)

    # VLA D-config
    if antconfig is None:
        antconfig = find_antenna_config()
        if antconfig is None:
            raise FileNotFoundError(
                "Could not find VLA antenna config file. "
                "Please specify --antconfig /path/to/vla.d.cfg"
            )

    print(f"Antenna config: {antconfig}")
    x, y, z, d, an, an2, telname, obspos = mysu.readantenna(antconfig)

    print(f"Telescope: {telname}")
    print(f"Antennas: {len(an)}")

    sm.setconfig(
        telescopename=telname,
        x=np.array(x),
        y=np.array(y),
        z=np.array(z),
        dishdiameter=np.array(d),
        mount=['alt-az'],
        antname=np.array(an).tolist(),
        coordsystem='global',
        referencelocation=me.observatory('VLA')
    )

    # Pointing center = source position
    dir_pointing = me.direction('J2000', SOURCE_RA, SOURCE_DEC)

    # Feed setup - circular polarization for VLA
    sm.setfeed(mode='perfect R L', pol=[''])

    # Spectral window
    # Total bandwidth
    total_bw = FREQ_END - FREQ_START  # 1 GHz
    chan_width = total_bw / n_channels

    print(f"\nSpectral setup:")
    print(f"  Frequency range: {FREQ_START/1e9:.1f} - {FREQ_END/1e9:.1f} GHz")
    print(f"  Channels: {n_channels}")
    print(f"  Channel width: {chan_width/1e6:.3f} MHz")

    sm.setspwindow(
        spwname="LBand",
        freq=f'{FREQ_START}Hz',
        deltafreq=f'{chan_width}Hz',
        freqresolution=f'{chan_width}Hz',
        nchannels=n_channels,
        stokes='RR RL LR LL'  # Full polarization
    )

    # Field
    sm.setfield(sourcename=SOURCE_NAME, sourcedirection=dir_pointing)

    # Limits
    sm.setlimits(shadowlimit=0.01, elevationlimit='1deg')
    sm.setauto(autocorrwt=0.0)

    # Time setup
    # Convert observation time to hour angle range
    obs_time_hr = obs_time_min / 60.0
    ha_start = -obs_time_hr / 2.0
    ha_end = obs_time_hr / 2.0

    print(f"\nTime setup:")
    print(f"  Observation time: {obs_time_min:.1f} min")
    print(f"  Integration time: {int_time_sec:.1f} s")
    print(f"  Hour angle range: {ha_start:.3f} to {ha_end:.3f} hr")

    sm.settimes(
        integrationtime=f'{int_time_sec}s',
        usehourangle=True,
        referencetime=me.epoch('UTC', '2020/10/4/00:00:00')
    )

    # Observe
    sm.observe(
        sourcename=SOURCE_NAME,
        spwname="LBand",
        starttime=f'{ha_start}h',
        stoptime=f'{ha_end}h'
    )

    sm.close()

    # Unflag all data
    flagdata(vis=msname, mode='unflag')

    print(f"\n✓ Empty MS created: {msname}")

    return msname


def predict_visibilities(msname: str, clname: str):
    """Predict visibilities from component list.

    Args:
        msname: Measurement set path
        clname: Component list path
    """
    print(f"\n{'=' * 70}")
    print("PREDICTING VISIBILITIES")
    print(f"{'=' * 70}")

    ft(vis=msname, model='', complist=clname, usescratch=True, nterms=1)

    # Copy MODEL_DATA to DATA
    tb.open(msname, nomodify=False)
    tb.putcol('DATA', tb.getcol('MODEL_DATA'))
    tb.close()

    print("✓ Visibilities predicted and copied to DATA column")


def apply_setjy(msname: str):
    """Apply setjy to get proper 3C286 flux model.

    Args:
        msname: Measurement set path
    """
    print(f"\n{'=' * 70}")
    print("APPLYING SETJY FOR 3C286")
    print(f"{'=' * 70}")

    # Run setjy with 3C286 standard
    result = setjy(
        vis=msname,
        field=SOURCE_NAME,
        standard='Perley-Butler 2017',
        model='',
        usescratch=True,
        scalebychan=True,
    )

    print("✓ setjy applied with Perley-Butler 2017 standard")

    # Copy updated MODEL_DATA to DATA
    tb.open(msname, nomodify=False)
    model_data = tb.getcol('MODEL_DATA')
    tb.putcol('DATA', model_data)
    tb.close()

    # Report flux values
    print("\nFlux density per channel (sample):")
    tb.open(msname)
    model = tb.getcol('MODEL_DATA')
    tb.close()

    # Get mean amplitude for first few channels
    n_chan = model.shape[1]
    for i in [0, n_chan//4, n_chan//2, 3*n_chan//4, n_chan-1]:
        amp = np.mean(np.abs(model[0, i, :]))  # RR
        freq = FREQ_START + i * (FREQ_END - FREQ_START) / n_chan
        print(f"  Chan {i} ({freq/1e9:.3f} GHz): {amp:.3f} Jy")

    return result


def get_ms_summary(msname: str):
    """Print MS summary.

    Args:
        msname: Measurement set path
    """
    print(f"\n{'=' * 70}")
    print("MS SUMMARY")
    print(f"{'=' * 70}")

    tb.open(msname)
    n_rows = tb.nrows()
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    data = tb.getcol('DATA')
    times = tb.getcol('TIME')
    tb.close()

    n_baselines = len(np.unique(list(zip(ant1, ant2)), axis=0))
    n_times = len(np.unique(times))
    n_corr, n_chan, _ = data.shape

    print(f"MS: {msname}")
    print(f"Rows: {n_rows:,}")
    print(f"Baselines: {n_baselines}")
    print(f"Time samples: {n_times}")
    print(f"Channels: {n_chan}")
    print(f"Correlations: {n_corr}")
    print(f"Total visibilities: {n_rows * n_chan:,}")

    # Data statistics
    amp = np.abs(data)
    print(f"\nData amplitude statistics:")
    print(f"  Mean: {np.mean(amp):.3f} Jy")
    print(f"  Min: {np.min(amp):.3f} Jy")
    print(f"  Max: {np.max(amp):.3f} Jy")


def simulate_3c286(
    msname: str = 'sim_3c286.ms',
    n_channels: int = 64,
    obs_time_min: float = 5.0,
    int_time_sec: float = 2.0,
    antconfig: str = None,
):
    """Main function to create simulated 3C286 observation.

    Args:
        msname: Output MS name
        n_channels: Number of frequency channels
        obs_time_min: Observation time in minutes
        int_time_sec: Integration time in seconds
        antconfig: Path to antenna config file (auto-detect if None)

    Returns:
        Path to created MS
    """
    print(f"\n{'=' * 70}")
    print("3C286 SIMULATION")
    print(f"{'=' * 70}")
    print(f"Source: {SOURCE_NAME}")
    print(f"Position: {SOURCE_RA} {SOURCE_DEC}")
    print(f"Output: {msname}")

    # Create component list with placeholder flux
    clname = create_componentlist(flux_jy=17.0)

    # Create empty MS
    create_ms(
        msname=msname,
        n_channels=n_channels,
        obs_time_min=obs_time_min,
        int_time_sec=int_time_sec,
        antconfig=antconfig,
    )

    # Predict visibilities
    predict_visibilities(msname, clname)

    # Apply setjy for proper 3C286 model
    apply_setjy(msname)

    # Print summary
    get_ms_summary(msname)

    # Cleanup
    os.system(f'rm -rf {clname}')

    print(f"\n{'=' * 70}")
    print("✓ SIMULATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"MS: {msname}")
    print(f"MODEL_DATA: 3C286 flux from Perley-Butler 2017")
    print(f"DATA: Copy of MODEL_DATA (clean, no corruption)")
    print(f"\nNext step: Corrupt DATA with delays using corrupt_delay.py")

    return msname


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simulate VLA observation of 3C286"
    )
    parser.add_argument(
        '--msname',
        default='sim_3c286.ms',
        help='Output MS name (default: sim_3c286.ms)'
    )
    parser.add_argument(
        '--n_channels',
        type=int,
        default=64,
        help='Number of channels (default: 64)'
    )
    parser.add_argument(
        '--obs_time',
        type=float,
        default=5.0,
        help='Observation time in minutes (default: 5)'
    )
    parser.add_argument(
        '--int_time',
        type=float,
        default=2.0,
        help='Integration time in seconds (default: 2)'
    )
    parser.add_argument(
        '--antconfig',
        default=None,
        help='Antenna config file (default: auto-detect from ~/.casa/data)'
    )

    args = parser.parse_args()

    simulate_3c286(
        msname=args.msname,
        n_channels=args.n_channels,
        obs_time_min=args.obs_time,
        int_time_sec=args.int_time,
        antconfig=args.antconfig,
    )
