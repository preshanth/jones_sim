Configuration System
====================

JSON Configuration Format
-------------------------

The simulator uses JSON configuration files to define effects and parameters.

Basic Structure
~~~~~~~~~~~~~~~

::

    {
      "jones_chain": {
        "enabled_effects": ["bandpass", "gain"]
      },
      "effects": {
        "bandpass": { ... },
        "gain": { ... }
      },
      "noise": { ... },
      "validation": { ... }
    }

Effect Types
------------

K Effects (Delays)
~~~~~~~~~~~~~~~~~~

::

    "bandpass": {
      "type": "bandpass_amplitude_delay",
      "delay": {
        "tau_x": {
          "distribution": "uniform",
          "min": -10e-9,
          "max": 10e-9
        },
        "tau_y": {
          "distribution": "uniform",
          "min": -10e-9,
          "max": 10e-9
        }
      },
      "seed": 42
    }

G Effects (Gains)
~~~~~~~~~~~~~~~~~

::

    "gain": {
      "type": "time_variable_gain",
      "timescale_minutes": 10.0,
      "amplitude": {
        "mean": {
          "distribution": "log_normal",
          "mean": 0.0,
          "std": 0.05
        }
      },
      "phase": {
        "std": {
          "distribution": "uniform",
          "min": 1.0,
          "max": 10.0
        }
      },
      "seed": 43
    }

B Effects (Bandpass)
~~~~~~~~~~~~~~~~~~~~

::

    "bandpass": {
      "type": "bandpass_ripple",
      "amplitude_ripple": 0.1,
      "phase_ripple_deg": 5.0,
      "n_ripples": 3,
      "seed": 44
    }

D Effects (Leakage)
~~~~~~~~~~~~~~~~~~~

::

    "leakage": {
      "type": "polarization_leakage",
      "d_terms": {
        "real": {
          "distribution": "gaussian",
          "mean": 0.0,
          "std": 0.01
        },
        "imag": {
          "distribution": "gaussian",
          "mean": 0.0,
          "std": 0.01
        }
      },
      "seed": 45
    }

Distribution Types
------------------

Uniform
~~~~~~~

::

    "param": {
      "distribution": "uniform",
      "min": -1.0,
      "max": 1.0
    }

Gaussian
~~~~~~~~

::

    "param": {
      "distribution": "gaussian",
      "mean": 0.0,
      "std": 0.1
    }

Log-Normal
~~~~~~~~~~

::

    "param": {
      "distribution": "log_normal",
      "mean": 0.0,
      "std": 0.2
    }

Complex
~~~~~~~

::

    "param": {
      "distribution": "complex",
      "amplitude": {
        "distribution": "log_normal",
        "mean": 0.0,
        "std": 0.1
      },
      "phase": {
        "distribution": "uniform",
        "min": -180.0,
        "max": 180.0
      }
    }

Noise Configuration
-------------------

::

    "noise": {
      "add_thermal_noise": true,
      "sefd": 420.0,
      "channel_width_hz": 1e6,
      "integration_time_s": 10.0
    }

Validation Thresholds
---------------------

::

    "validation": {
      "delay_threshold_ns": 0.1,
      "amplitude_threshold": 0.01,
      "phase_threshold_deg": 1.0
    }

Complete Examples
-----------------

See ``configs/`` directory for complete configuration examples:

- ``test_k_delays.json`` - Delay calibration
- ``test_g_gains.json`` - Gain calibration
- ``test_b_bandpass.json`` - Bandpass calibration
- ``test_d_leakage.json`` - Leakage calibration
- ``test_k_g_combined.json`` - Sequential K+G
- ``test_k_g_b_full.json`` - Full calibration chain

Configuration Loading
---------------------

From file::

    from jones_sim import JonesConfig

    config = JonesConfig("path/to/config.json")
    sim = config.create_simulator(n_antennas=27)

From dictionary::

    config_dict = {
        "jones_chain": {"enabled_effects": ["bandpass"]},
        "effects": { ... }
    }
    config = JonesConfig(config_dict)

Default configuration::

    config = JonesConfig()  # Uses default minimal config
