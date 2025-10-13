#!/usr/bin/env python
"""Quick test to verify MS reading and solver works.

Tests reading the 3C391 MS file and running a single solve.
"""

import sys
import traceback

try:
    from jones_sim import MSCalibrator
except ImportError as e:
    print(f"Error importing jones_sim: {e}")
    sys.exit(1)

# Your MS file path
ms_path = "/home/pjaganna/Data/3C391/3c391_ctm_mosaic_10s_spw0.ms"

print("=" * 70)
print("MS Read and Solver Test")
print("=" * 70)

# Check for GPU

use_gpu = "--gpu" in sys.argv

# Test 1: Open MS and get summary
print(f"\nTest 1: Opening MS: {ms_path}")
print(f"  GPU mode: {use_gpu}")
try:
    calibrator = MSCalibrator(ms_path, use_gpu=use_gpu)
    print("  ✓ MS opened successfully")
except Exception as e:
    print(f"  ✗ Failed to open MS: {e}")

    traceback.print_exc()
    sys.exit(1)

# Test 2: Get observation summary
print("\nTest 2: Reading observation summary")
try:
    summary = calibrator.ms_handler.get_observation_summary()
    print(f"  ✓ Observatory: {summary['observatory_names']}")
    print(f"  ✓ Antennas: {summary['n_antennas']} ({summary['antenna_names'][:5]}...)")
    print(f"  ✓ Fields: {summary['field_names']}")
    print(f"  ✓ SPWs: {summary['n_spw']}")

    # Show SPW 0 info
    spw0_info = summary["frequency_info"][0]
    print(
        f"  ✓ SPW 0: {spw0_info['n_channels']} channels, "
        f"{spw0_info['chan_freqs'][0]/1e9:.3f}-{spw0_info['chan_freqs'][-1]/1e9:.3f} GHz"
    )
except Exception as e:
    print(f"  ✗ Failed to get summary: {e}")

    traceback.print_exc()

# Test 3: Parse refant
print("\nTest 3: Parsing reference antenna 'ea21'")
try:
    refant_idx = calibrator._parse_refant("ea21")
    print(f"  ✓ Reference antenna 'ea21' → index {refant_idx}")
except Exception as e:
    print(f"  ✗ Failed to parse refant: {e}")

    traceback.print_exc()

# Test 4: Read visibility data (small selection)
print("\nTest 4: Reading visibility data (field 0, spw 0:27~36)")
try:
    vis_data = calibrator.ms_handler.read_visibilities(field=0, spw=0)
    print(f"  ✓ Data shape: {vis_data['data'].shape}")
    print(f"  ✓ Flag shape: {vis_data['flag'].shape}")
    print(f"  ✓ Weight shape: {vis_data['weight'].shape}")

    # Calculate baseline statistics
    n_rows = vis_data["antenna1"].shape[0]
    n_times = len(set(vis_data["time"]))
    unique_baselines = set(zip(vis_data["antenna1"], vis_data["antenna2"]))
    n_baselines = len(unique_baselines)

    print(f"  ✓ Rows: {n_rows} (time × baseline)")
    print(f"  ✓ Unique times: {n_times}")
    print(f"  ✓ Unique baselines: {n_baselines}")
    print(f"  ✓ Baselines per time: ~{n_rows / n_times:.0f}")

    # Check for flagged data
    flag_frac = vis_data["flag"].sum() / vis_data["flag"].size
    print(f"  ✓ Flagged data: {flag_frac*100:.1f}%")

    # Check weight statistics
    if "weight" in vis_data:
        wt = vis_data["weight"]
        print(f"  ✓ Weight range: {wt.min():.2e} to {wt.max():.2e}")
        print(f"  ✓ Weight mean: {wt.mean():.2e}")

except Exception as e:
    print(f"  ✗ Failed to read visibilities: {e}")

    traceback.print_exc()
    calibrator.close()
    sys.exit(1)

# Test 5: Run a single gaincal solve
print("\nTest 5: Running gaincal (small test: field 0 only)")
try:
    results = calibrator.gaincal(
        caltable="test_gaincal.npz",
        field=0,  # Just one field for quick test
        spw="0:27~36",
        refant="ea21",
        calmode="p",
        solint="int",
        minsnr=0.0,
    )

    print(f"  ✓ Solved {results['n_solutions']} time intervals")
    print(f"  ✓ Antennas: {results['n_antennas']}")
    print(f"  ✓ Flagged: {results['n_flagged']} solutions")

    # Check convergence
    if results["solutions"]["convergence"]:
        n_converged = sum(
            info["converged"] for info in results["solutions"]["convergence"]
        )
        print(f"  ✓ Converged: {n_converged}/{results['n_solutions']} intervals")

        # Show example convergence info
        example_info = results["solutions"]["convergence"][0]
        print(
            f"  ✓ Example interval: {example_info['iterations']} iterations, "
            f"residual {example_info['initial_residual']:.2e} → {example_info['final_residual']:.2e}"
        )

    print("\n  Solutions saved to: test_gaincal.npz")

except Exception as e:
    print(f"  ✗ Failed to run gaincal: {e}")

    traceback.print_exc()
finally:
    calibrator.close()

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
print("\nNext step: Run full gaincal with all fields:")
print("  python scripts/run_gaincal_comparison.py \\")
print("      --ms /home/pjaganna/Data/3C391/3c391_ctm_mosaic_10s_spw0.ms \\")
print("      --field '0,1,9' \\")
print("      --spw '0:27~36' \\")
print("      --refant ea21 \\")
print("      --calmode p \\")
print("      --output antsol_G0all.npz")
