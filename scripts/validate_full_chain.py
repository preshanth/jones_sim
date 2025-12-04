#!/usr/bin/env python3
"""Validate full Jones calibration chain using ValidationPipeline.

This is a thin wrapper around ValidationPipeline that handles:
- Command-line arguments
- Pipeline execution
- Results printing
- Cleanup

The heavy lifting is in ValidationPipeline class.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from validation_pipeline import ValidationPipeline
from validation_lib import (
    print_delay_comparison,
    print_gains_comparison,
    print_bandpass_comparison,
    print_leakage_comparison,
    print_section_header,
)
from simulate_3c286 import simulate_3c286


def print_results(pipeline):
    """Print all comparison results from pipeline.

    Args:
        pipeline: ValidationPipeline instance with comparison_results
    """
    for effect, results in pipeline.comparison_results.items():
        if effect == "K":
            print_delay_comparison(results)
        elif effect == "G":
            print_gains_comparison(results)
        elif effect == "B":
            print_bandpass_comparison(results)
        elif effect == "D":
            print_leakage_comparison(results)

def main():
    parser = argparse.ArgumentParser(
        description="Validate full Jones calibration chain (K, B, G, D, P)"
    )
    parser.add_argument(
        "--msname", default="sim_full_chain.ms", help="MS name"
    )
    parser.add_argument(
        "--skip_sim", action="store_true", help="Skip MS simulation"
    )
    parser.add_argument(
        "--effects", default="K,G", help="Comma-separated list of effects (default: K,G)"
    )
    parser.add_argument(
        "--n_channels", type=int, default=64, help="Number of channels"
    )
    parser.add_argument(
        "--seed", type=int, default=100, help="Random seed"
    )
    parser.add_argument(
        "--no_noise", action="store_true", help="Skip thermal noise"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from saved state"
    )
    parser.add_argument(
        "--no_plots", action="store_true", help="Skip plot generation"
    )
    parser.add_argument(
        "--output_dir", default=".", help="Output directory"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output during optimization"
    )
    args = parser.parse_args()

    effects = args.effects.split(",")

    print_section_header("FULL JONES CHAIN VALIDATION")
    print(f"MS: {args.msname}")
    print(f"Effects: {', '.join(effects)}")
    print(f"Output: {args.output_dir}")
    print(f"Noise: {not args.no_noise}")
    print(f"Resume: {args.resume}")

    # Create MS if needed
    if not args.skip_sim and not os.path.exists(args.msname):
        print_section_header("CREATING MEASUREMENT SET")
        simulate_3c286(
            msname=args.msname,
            n_channels=args.n_channels,
            obs_time_min=5.0,
            int_time_sec=2.0,
        )

    # Create and run pipeline
    pipeline = ValidationPipeline(
        ms_path=args.msname,
        output_dir=args.output_dir,
        effects=effects,
        seed=args.seed,
        debug=args.debug,
    )

    results = pipeline.run(
        resume=args.resume,
        generate_plots=not args.no_plots,
        add_noise=not args.no_noise,
    )

    # Print all results
    print_section_header("VALIDATION RESULTS")
    print_results(pipeline)

    # Save final state
    pipeline.save_state()

    print_section_header("VALIDATION COMPLETE")
    print(f"✓ Results saved to: {args.output_dir}")

    return results


if __name__ == "__main__":
    results = main()
