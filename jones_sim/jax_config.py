"""JAX configuration utilities for GPU/CPU setup."""

import os
import warnings


def configure_jax(max_cpu_fraction: float = 0.5, gpu_device: int = 0):
    """Configure JAX for GPU or CPU with appropriate settings.

    Args:
        max_cpu_fraction: Maximum fraction of CPU cores to use (default: 0.5 = 50%)
        gpu_device: GPU device ID to use if available (default: 0)

    Returns:
        str: 'gpu' if GPU configured, 'cpu' if CPU configured
    """
    try:
        import jax
    except ImportError:
        warnings.warn("JAX not available, skipping configuration")
        return None

    # Check for GPU availability
    try:
        gpus = jax.devices("gpu")
        if gpus and len(gpus) > 0:
            # GPU available - use it
            if gpu_device < len(gpus):
                jax.config.update("jax_default_device", gpus[gpu_device])
                print(f"JAX configured for GPU {gpu_device} ({len(gpus)} GPUs available)")
                return "gpu"
            else:
                warnings.warn(
                    f"GPU device {gpu_device} not available (only {len(gpus)} GPUs). "
                    "Falling back to CPU."
                )
    except (RuntimeError, IndexError):
        pass  # No GPU available, fall through to CPU config

    # Configure for CPU
    cpu_count = os.cpu_count() or 1
    max_threads = max(1, int(cpu_count * max_cpu_fraction))

    # Set JAX CPU thread limit
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={max_threads}"
    )
    jax.config.update("jax_platform_name", "cpu")

    print(
        f"JAX configured for CPU: using {max_threads}/{cpu_count} cores "
        f"({max_cpu_fraction*100:.0f}% limit)"
    )
    return "cpu"
