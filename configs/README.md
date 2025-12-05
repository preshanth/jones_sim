# Jones Simulator Test Configurations

This directory contains JSON configuration files for calibration testing and validation.

## Test Configuration Files

### Single-Effect Tests
- **`test_k_delays.json`** - Delay (K) calibration only
  - Delay range: ±10 ns
  - No noise for exact recovery
  - Expected RMS: < 0.01 ns

- **`test_g_gains.json`** - Gain (G) calibration only
  - Amplitude variation: σ = 0.1
  - Phase variation: ±0.5 rad
  - Expected amp RMS: < 0.01
  - Expected phase RMS: < 1°

- **`test_b_bandpass.json`** - Bandpass (B) calibration
  - Trapezoidal bandpass shape
  - Edge rolloff: 80%
  - Ripple amplitude: 2%
  - Requires: ≥32 channels

- **`test_d_leakage.json`** - Leakage (D) calibration
  - D-term magnitude: ~0.05
  - Expected RMS: < 0.001
  - Requires: Full Stokes (4 corr)

### Combined-Effect Tests
- **`test_k_g_combined.json`** - Delay + Gain
  - Sequential solving: K → G
  - Moderate corruption levels
  - No noise

- **`test_k_g_b_full.json`** - Full calibration chain
  - All effects: K + G + B
  - With thermal noise
  - Realistic validation scenario

## Configuration Schema

### Top-Level Structure
```json
{
  "metadata": {
    "description": "Human-readable description",
    "version": "1.0",
    "use_case": "What this config tests"
  },
  "jones_chain": {
    "order": ["effect1", "effect2", ...],
    "enabled_effects": ["effect1", ...]
  },
  "effects": { ... },
  "noise": { ... },
  "processing": { ... },
  "validation": { ... }
}
```

### Effect Types
- `bandpass_amplitude_delay` - Frequency-dependent delays and amplitude
- `complex_electronic_gain` - Antenna-based complex gains
- `feed_leakage` - Polarization leakage (D-terms)
- `xy_phase_offset` - Cross-hand phase
- `xy_differential_delay` - Cross-hand delay

### Distribution Types
- `constant` - Fixed value
- `uniform` - Uniform random [min, max]
- `gaussian` / `normal` - Normal distribution
- `log_normal` - Log-normal distribution
- `complex_gaussian` - Complex Gaussian (separate real/imag)

### Validation Section
Each test config includes validation criteria:
```json
"validation": {
  "solve_sequence": ["K", "G"],
  "expected_rms_ns": 0.01,
  "expected_amp_rms": 0.02,
  "expected_phase_rms_deg": 2.0,
  "casa_comparison_tolerance": 0.1,
  "min_channels": 32,
  "with_noise": false
}
```

## Usage in Validation Scripts

### Python
```python
from jones_sim import JonesConfig

# Load test configuration
config = JonesConfig("configs/test_k_delays.json")

# Create simulator with ground truth
sim = config.create_simulator(n_antennas=27)

# Get validation thresholds
val_config = config.config["validation"]
assert rms_error < val_config["expected_rms_ns"]
```

### Command Line
```bash
# Run validation with specific config
python scripts/validate_delay_recovery.py \
    --config configs/test_k_delays.json \
    --msname test.ms
```

## Creating New Test Configs

1. Copy an existing config as template
2. Update `metadata` section
3. Configure `effects` for your test case
4. Set appropriate `validation` thresholds
5. Document in this README

## Testing Philosophy

- **No Noise Tests**: For exact recovery validation
  - Expected RMS should be very small (< 0.01)
  - Tests solver correctness

- **With Noise Tests**: For realistic scenarios
  - Expected RMS reflects noise floor
  - Tests solver robustness

- **Combined Tests**: For pipeline validation
  - Tests sequential solving (K → G → B)
  - Tests error propagation between stages

## Random Seeds

Each config uses a unique random seed to ensure:
- Reproducible test scenarios
- Independent test cases
- Consistent validation across runs

Seeds are sequential:
- 42: K (delays)
- 43: G (gains)
- 44: B (bandpass)
- 45: D (leakage)
- 46: K+G combined
- 47: K+G+B full
