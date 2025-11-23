# Simulation-Based Inference (SBI) for Radio Calibration

This guide explains how to use the SBI framework in `jones_sim` for solving calibration problems with **full uncertainty quantification**.

## Overview

Traditional calibration gives you **point estimates**:
```
Bandpass gain for channel 0 = 1.21
```

SBI gives you **posterior distributions** with credible intervals:
```
Bandpass gain for channel 0 = 1.21 ± 0.05 (95% CI)
```

## Why SBI?

**Advantages:**
- ✓ Full uncertainty quantification (credible intervals)
- ✓ Fast inference after training (~seconds instead of hours)
- ✓ Handles complex, non-Gaussian posteriors
- ✓ No need for likelihood derivation
- ✓ Works with any forward simulator

**When to use:**
- You need uncertainty estimates on calibration solutions
- You'll solve many similar problems (amortized inference)
- Traditional MCMC is too slow
- Likelihood function is intractable

## Quick Start

### Installation

```bash
# Install jones_sim with SBI dependencies
pip install -e '.[sbi]'
```

### Basic Usage

```python
from jones_sim.sbi_solver import BandpassSBISimulator, SBICalibrationSolver
from jones_sim.solvable_effects import BandpassEffect
import numpy as np

# 1. Define your visibility model
def point_source_model(ant1, ant2, freqs, n_antennas):
    """Simple point source at phase center."""
    n_baselines = len(ant1)
    n_channels = len(freqs)
    return np.ones((n_baselines, n_channels, 4), dtype=complex)

# 2. Create simulator
effect = BandpassEffect()
simulator = BandpassSBISimulator(
    effect=effect,
    visibility_model=point_source_model,
    n_antennas=4,
    n_channels=16,
    noise_std=0.02,
)

# 3. Train neural network
solver = SBICalibrationSolver(
    simulator=simulator,
    n_rounds=2,
    density_estimator="maf",
)

solver.train(n_simulations=10000)

# 4. Perform inference
observed_data = ... # Your actual visibility measurements
samples, summary = solver.infer(observed_data, num_samples=10000)

# 5. Extract results
print(f"Mean: {summary['mean']}")
print(f"Std:  {summary['std']}")
print(f"95% CI: [{summary['credible_interval_95'][0]}, "
      f"{summary['credible_interval_95'][1]}]")
```

## Architecture

### Components

1. **SBISimulator**: Abstract base class defining the simulation interface
   - `simulate(params)`: Generate observations from parameters
   - `get_prior()`: Return prior distribution
   - `get_param_dim()`: Parameter dimensionality
   - `get_obs_dim()`: Observation dimensionality

2. **Effect-Specific Simulators**:
   - `BandpassSBISimulator`: For bandpass calibration
   - `GainSBISimulator`: For gain calibration
   - Easy to add more (delays, leakage, etc.)

3. **SBICalibrationSolver**: General solver for any effect
   - Trains neural density estimators
   - Performs amortized inference
   - Returns full posteriors with credible intervals

### Workflow

```
┌─────────────────────┐
│  Prior p(θ)         │
│  (parameters)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Simulator          │
│  θ → x              │
│  (params → vis)     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Training Data      │
│  {(θᵢ, xᵢ)}         │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Neural Network     │
│  learns p(θ|x)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Posterior p(θ|x_obs)│
│  with uncertainties │
└─────────────────────┘
```

## Neural Density Estimators

The solver supports three types:

1. **MAF** (Masked Autoregressive Flow)
   - Best overall performance
   - Good for high dimensions
   - Recommended default

2. **NSF** (Neural Spline Flow)
   - Excellent for complex distributions
   - Slightly slower than MAF

3. **MDN** (Mixture Density Network)
   - Fastest training
   - Good for simple problems
   - Use for quick prototyping

Example:
```python
solver = SBICalibrationSolver(
    simulator=simulator,
    density_estimator="maf",  # or "nsf", "mdn"
)
```

## Examples

### Example 1: Bandpass Calibration

```bash
# Run demo script
python scripts/demo_sbi_bandpass.py \
    --n-antennas 4 \
    --n-channels 16 \
    --n-train 10000 \
    --n-rounds 2 \
    --output-dir results/
```

This will:
1. Train an SBI network for bandpass calibration
2. Generate test observations
3. Perform inference
4. Plot results with credible intervals

### Example 2: Gain Calibration

```python
from jones_sim.sbi_solver import GainSBISimulator, SBICalibrationSolver
from jones_sim.solvable_effects import GainEffect

effect = GainEffect()
simulator = GainSBISimulator(
    effect=effect,
    visibility_model=your_vis_model,
    n_antennas=27,
    noise_std=0.01,
)

solver = SBICalibrationSolver(simulator, n_rounds=3)
solver.train(n_simulations=20000)

# Save for reuse
solver.save("trained_gain_posterior.pkl")
```

### Example 3: Loading Pre-Trained Posterior

```python
solver = SBICalibrationSolver(simulator, n_rounds=1)
solver.load("trained_gain_posterior.pkl")

# Now inference is instant!
samples, summary = solver.infer(new_observation)
```

## Advanced Topics

### Sequential Neural Posterior Estimation (SNPE)

By default, we use SNPE-C which refines the posterior over multiple rounds:

```python
solver = SBICalibrationSolver(
    simulator=simulator,
    n_rounds=3,  # More rounds = better accuracy, longer training
)
```

**Round 1**: Sample from prior → Train network → Get approximate posterior
**Round 2**: Sample from approximate posterior → Refine network
**Round 3**: Final refinement

### Custom Priors

Override the simulator's `get_prior()` method:

```python
class CustomBandpassSim(BandpassSBISimulator):
    def get_prior(self):
        # Custom prior with tighter bounds
        return sbi_utils.BoxUniform(
            low=torch.tensor([...]),
            high=torch.tensor([...]),
        )
```

### Multi-Effect Joint Inference

For joint inference over multiple effects (e.g., gains + bandpass):

```python
# Coming soon! Design in progress.
# Will support joint posteriors p(gains, bandpass | data)
```

## Performance Tips

1. **Start small**: Test with 1000 simulations, 1 round
2. **Scale up**: Use 10k-100k simulations for production
3. **Use GPU**: Set `device="cuda"` for faster training
4. **Save posteriors**: Reuse trained networks
5. **Batch observations**: Process multiple datasets with same trained network

## Comparison with MCMC

| Method | Training Time | Inference Time | Uncertainty | Use Case |
|--------|--------------|----------------|-------------|----------|
| MCMC (NumPyro) | 0 | Hours | ✓ Full | One-off analysis |
| SBI | Minutes-Hours | Seconds | ✓ Full | Many similar problems |
| Traditional | 0 | Seconds | ✗ None | No uncertainties needed |

**Rule of thumb**: If you're solving >10 similar problems, SBI is faster overall.

## Troubleshooting

### Training diverges
- Reduce learning rate
- Use simpler density estimator (MDN)
- Check simulator outputs are finite

### Poor posterior quality
- Increase n_simulations
- Add more rounds
- Check prior bounds are reasonable
- Verify simulator is deterministic (seeding)

### Slow training
- Reduce parameter dimensionality
- Use MDN instead of MAF
- Enable GPU: `device="cuda"`

## References

- SBI Library: https://github.com/mackelab/sbi
- Paper: "The frontier of simulation-based inference" (Cranmer et al. 2020)
- Tutorial: https://sbi-dev.github.io/sbi/

## Future Directions

- [ ] Joint multi-effect inference
- [ ] Time-dependent effects (gains over time)
- [ ] Active learning for efficient simulation budget
- [ ] Validation metrics and diagnostics
- [ ] Integration with CASA measurement sets

## Support

For questions or issues:
- GitHub Issues: https://github.com/yourusername/jones_sim/issues
- Example notebooks: `examples/sbi_*.ipynb`
