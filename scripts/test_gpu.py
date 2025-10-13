#!/usr/bin/env python
"""Quick GPU availability and performance test."""

import time

import numpy as np

print("=" * 70)
print("GPU Availability Test")
print("=" * 70)

# Test 1: Check CuPy
print("\nTest 1: CuPy availability")
try:
    import cupy as cp

    print("  ✓ CuPy is installed")

    # Check GPU
    print(f"  ✓ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"  ✓ GPU device: {cp.cuda.Device().name.decode()}")
    mem_info = cp.cuda.Device().mem_info
    print(
        f"  ✓ GPU memory: {mem_info[1]/1e9:.1f} GB total, {mem_info[0]/1e9:.1f} GB free"
    )

    cupy_available = True
except ImportError as e:
    print(f"  ✗ CuPy not available: {e}")
    print("  Install with: pip install cupy-cuda11x")
    cupy_available = False
except Exception as e:
    print(f"  ✗ Error checking GPU: {e}")
    cupy_available = False

if not cupy_available:
    print("\nSkipping GPU tests - CuPy not available")
    exit(0)

# Test 2: Basic GPU array operations
print("\nTest 2: Basic GPU array operations")
try:
    # Create array on GPU
    x_gpu = cp.random.randn(1000, 1000).astype(cp.complex64)
    y_gpu = cp.random.randn(1000, 1000).astype(cp.complex64)

    # Matrix multiply on GPU
    z_gpu = cp.dot(x_gpu, y_gpu)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU

    print("  ✓ GPU array creation and multiplication works")
except Exception as e:
    print(f"  ✗ GPU operations failed: {e}")

# Test 3: CPU vs GPU performance
print("\nTest 3: CPU vs GPU performance comparison")
n = 2000
print(f"  Problem size: {n}x{n} complex matrix operations")

# CPU test
x_cpu = np.random.randn(n, n).astype(np.complex64)
y_cpu = np.random.randn(n, n).astype(np.complex64)

t0_cpu = time.time()
for _ in range(10):
    z_cpu = np.dot(x_cpu, y_cpu)
t_cpu = time.time() - t0_cpu
print(f"  CPU: {t_cpu:.3f} seconds (10 iterations)")

# GPU test
x_gpu = cp.asarray(x_cpu)
y_gpu = cp.asarray(y_cpu)

# Warmup
z_gpu = cp.dot(x_gpu, y_gpu)
cp.cuda.Stream.null.synchronize()

t0_gpu = time.time()
for _ in range(10):
    z_gpu = cp.dot(x_gpu, y_gpu)
cp.cuda.Stream.null.synchronize()
t_gpu = time.time() - t0_gpu
print(f"  GPU: {t_gpu:.3f} seconds (10 iterations)")

speedup = t_cpu / t_gpu
print(f"  Speedup: {speedup:.1f}x")

# Test 4: AntSol GPU solver
print("\nTest 4: AntSol solver GPU test")
try:
    from jones_sim import AntSolSolver

    # Small synthetic problem
    n_ant = 10
    true_gains = np.exp(1j * np.random.uniform(-np.pi, np.pi, n_ant))
    corr = np.outer(true_gains, np.conj(true_gains))

    correlations = np.zeros((4, n_ant, n_ant), dtype=complex)
    correlations[0] = corr
    weights = np.ones((4, n_ant, n_ant))
    np.fill_diagonal(weights[0], 0.0)

    # CPU solve
    solver_cpu = AntSolSolver(n_ant, mode="phase", use_gpu=False)
    t0 = time.time()
    gains_cpu, _, info_cpu = solver_cpu.solve(correlations, weights, refant=0, pol="XX")
    t_cpu_solve = time.time() - t0

    # GPU solve
    solver_gpu = AntSolSolver(n_ant, mode="phase", use_gpu=True)
    t0 = time.time()
    gains_gpu, _, info_gpu = solver_gpu.solve(correlations, weights, refant=0, pol="XX")
    t_gpu_solve = time.time() - t0

    print(
        f"  ✓ CPU solve: {t_cpu_solve:.4f} seconds, {info_cpu['iterations']} iterations"
    )
    print(
        f"  ✓ GPU solve: {t_gpu_solve:.4f} seconds, {info_gpu['iterations']} iterations"
    )
    print(f"  ✓ GPU used: {info_gpu['used_gpu']}")

    # Check results match
    phase_diff = np.angle(gains_cpu / gains_gpu)
    max_diff = np.max(np.abs(phase_diff))
    print(f"  ✓ Max phase difference: {np.degrees(max_diff):.6f} degrees")

    if max_diff < 1e-5:
        print("  ✓ CPU and GPU results match!")
    else:
        print(f"  ⚠ Results differ by {np.degrees(max_diff):.3f} degrees")

except Exception as e:
    print(f"  ✗ AntSol GPU test failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)
print("GPU test complete!")
print("=" * 70)
print("\nYou can now use --gpu flag with:")
print("  python scripts/test_ms_read.py --gpu")
print("  python scripts/run_gaincal_comparison.py --gpu ...")
