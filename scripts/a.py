import numpy as np
from daskms.experimental.zarr import xds_from_zarr

# Load the G-term gains
gains_list = xds_from_zarr("gains.qc::G")

# Pick the first dataset (most likely contains all antennas)
gains = gains_list[0]

# Choose first correlation (RR)
for ant in gains.antenna.values:
    g_ant = gains.sel(antenna=ant, correlation='RR')  # pick RR
    g_value = g_ant.gains.compute().item()           # single complex number
    print(f"Antenna {ant}: {g_value}")
