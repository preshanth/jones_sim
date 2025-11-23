#!/bin/bash
# Setup script for GPU-enabled jones_sim environment
# Compatible with GTX 1080 Ti and CUDA 12.x

set -e

echo "=========================================="
echo "Jones Sim GPU Environment Setup"
echo "=========================================="

# Create fresh micromamba environment
ENV_NAME="jones_sim_gpu"

echo ""
echo "Step 1: Creating micromamba environment '$ENV_NAME'..."
micromamba create -n $ENV_NAME python=3.11 -y

echo ""
echo "Step 2: Activating environment..."
eval "$(micromamba shell hook --shell bash)"
micromamba activate $ENV_NAME

echo ""
echo "Step 3: Installing PyTorch with CUDA 12.4..."
# Use your existing PyTorch 2.5.1 with CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "Step 4: Installing JAX with CUDA 12..."
# Install JAX with CUDA 12 support (compatible with NumPyro and your cudnn 9.1)
pip install "jax[cuda12]==0.4.29" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "Step 5: Installing NumPyro..."
pip install numpyro==0.15.0

echo ""
echo "Step 6: Installing SBI..."
pip install sbi==0.23.2

echo ""
echo "Step 7: Installing PyMC stack..."
pip install pymc==5.18.0 pytensor==2.25.5 arviz==0.20.0

echo ""
echo "Step 8: Installing jones_sim..."
cd /home/pjaganna/Software/jones_sim
pip install -e .

echo ""
echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying GPU support..."

python << 'EOF'
import sys
print("\n" + "="*50)
print("GPU Support Verification")
print("="*50)

# PyTorch
try:
    import torch
    print(f"\n✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")
    sys.exit(1)

# JAX
try:
    import jax
    import jax.numpy as jnp
    print(f"\n✓ JAX: {jax.__version__}")
    # Test GPU
    x = jnp.ones(10)
    print(f"  Default backend: {x.devices()}")
except Exception as e:
    print(f"✗ JAX error: {e}")
    sys.exit(1)

# NumPyro
try:
    import numpyro
    print(f"\n✓ NumPyro: {numpyro.__version__}")
except Exception as e:
    print(f"✗ NumPyro error: {e}")
    sys.exit(1)

# SBI
try:
    import sbi
    print(f"\n✓ SBI: {sbi.__version__}")
except Exception as e:
    print(f"✗ SBI error: {e}")
    sys.exit(1)

# Jones Sim
try:
    from jones_sim.sbi_bandpass_solver import BandpassSBISimulator
    print(f"\n✓ Jones Sim SBI: Imported successfully")
except Exception as e:
    print(f"✗ Jones Sim error: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("All checks passed! Environment ready.")
print("="*50)
print(f"\nActivate with: micromamba activate {ENV_NAME}")
print("Test with: python scripts/validate_sbi_bandpass.py --test basic")
EOF

echo ""
echo "Environment '$ENV_NAME' is ready!"
echo ""
echo "To use:"
echo "  micromamba activate $ENV_NAME"
echo "  python scripts/validate_sbi_bandpass.py --test basic"
