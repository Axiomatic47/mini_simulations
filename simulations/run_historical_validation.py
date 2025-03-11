#!/usr/bin/env python
"""
Run script for the improved historical validation module.
This script allows testing the axiomatic model against historical data and optimizing parameters.
With added numerical stability options.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path to find modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the historical validation module
try:
    # First try to import our improved version
    from config.historical_validation_improved import HistoricalValidation as HistoricalValidation
except ImportError:
    # Fall back to the original implementation if available
    try:
        from config.historical_validation import HistoricalValidation
    except ImportError:
        print("Error: Neither historical_validation.py nor historical_validation_improved.py found.")
        print("Please create one of these files in the config directory.")
        sys.exit(1)


def main():
    """Main function to run historical validation."""
    parser = argparse.ArgumentParser(
        description="Run historical validation for the Axiomatic Intelligence Growth Simulation.")

    parser.add_argument("--data", type=str, help="Path to historical data CSV file (optional)")
    parser.add_argument("--start-year", type=int, default=1000, help="Start year for validation")
    parser.add_argument("--end-year", type=int, default=2020, help="End year for validation")
    parser.add_argument("--interval", type=int, default=10, help="Year interval between data points")
    parser.add_argument("--output-dir", type=str, default="outputs/historical_validation",
                        help="Output directory for results")
    parser.add_argument("--optimize", action="store_true", help="Optimize parameters to fit historical data")
    parser.add_argument("--optimize-params", type=str, help="Comma-separated list of parameters to optimize")
    parser.add_argument("--default", action="store_true", help="Run with default parameters only (no optimization)")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic historical data")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information during execution")

    # New numerical stability parameters
    parser.add_argument("--max-knowledge", type=float, default=100.0, help="Maximum knowledge value")
    parser.add_argument("--max-suppression", type=float, default=100.0, help="Maximum suppression value")
    parser.add_argument("--max-intelligence", type=float, default=100.0, help="Maximum intelligence value")
    parser.add_argument("--max-truth", type=float, default=100.0, help="Maximum truth value")
    parser.add_argument("--enable-circuit-breaker", action="store_true", help="Enable circuit breaker for stability")
    parser.add_argument("--stability-threshold", type=float, default=1e-6,
                        help="Threshold for stability detection")
    parser.add_argument("--enable-adaptive-timestep", action="store_true",
                        help="Enable adaptive timestep for stability")
    parser.add_argument("--min-timestep", type=float, default=0.1, help="Minimum timestep for adaptive calculations")
    parser.add_argument("--max-timestep", type=float, default=5.0, help="Maximum timestep for adaptive calculations")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine data source
    data_source = None
    if args.data:
        data_source = args.data
    elif args.generate_data:
        print("Generating synthetic historical data...")
    else:
        print("No data provided. Generating synthetic historical data by default...")

    # Initialize historical validation with numerical stability parameters
    print(f"Initializing historical validation ({args.start_year}-{args.end_year}, interval: {args.interval} years)...")
    validator = HistoricalValidation(
        data_source=data_source,
        start_year=args.start_year,
        end_year=args.end_year,
        interval=args.interval,
        # New numerical stability parameters
        max_knowledge=args.max_knowledge,
        max_suppression=args.max_suppression,
        max_intelligence=args.max_intelligence,
        max_truth=args.max_truth,
        enable_circuit_breaker=args.enable_circuit_breaker,
        stability_threshold=args.stability_threshold,
        enable_adaptive_timestep=args.enable_adaptive_timestep,
        min_timestep=args.min_timestep,
        max_timestep=args.max_timestep
    )

    # Parse parameters to optimize
    if args.optimize_params:
        params_to_optimize = args.optimize_params.split(",")
        # Verify that parameters exist
        for param in params_to_optimize:
            if param not in validator.default_params:
                print(f"Warning: Parameter '{param}' not found. Ignoring.")
                params_to_optimize.remove(param)
    else:
        # Default parameters to optimize
        params_to_optimize = [
            "K_0", "S_0", "knowledge_growth_rate", "truth_adoption_rate",
            "suppression_decay", "medieval_knowledge_mult", "renaissance_knowledge_mult",
            "enlightenment_knowledge_mult", "industrial_knowledge_mult",
            "modern_knowledge_mult", "scientific_revolution_effect",
            "cultural_diffusion_rate", "truth_knowledge_synergy",
            "war_suppression_multiplier"
        ]

    # Run comprehensive analysis
    if args.default:
        # Run without optimization
        print("Running simulation with default parameters only...")
        validator.run_simulation()
        validator.save_results(output_dir=str(output_dir / "default"))
        validator.visualize_comparison(save_path=str(output_dir / "default" / "comparison.png"))
        validator.visualize_periods(save_path=str(output_dir / "default" / "periods.png"))
        validator.visualize_events(save_path=str(output_dir / "default" / "events.png"))

        # Save stability metrics if circuit breaker was enabled
        if args.enable_circuit_breaker and hasattr(validator, 'circuit_breaker'):
            validator.save_stability_metrics(save_path=str(output_dir / "default" / "stability_metrics.csv"))
    else:
        # Run comprehensive analysis
        print("Running comprehensive analysis...")
        results = validator.run_comprehensive_analysis(
            output_dir=str(output_dir),
            optimize=args.optimize
        )

    print("\nValidation complete!")
    print(f"Results saved to: {output_dir}")

    # Print stability information if circuit breaker was enabled
    if args.enable_circuit_breaker and hasattr(validator, 'circuit_breaker'):
        if validator.circuit_breaker.was_triggered:
            print("WARNING: Circuit breaker was triggered during simulation!")
            print(f"Number of instabilities detected: {validator.circuit_breaker.trigger_count}")
        else:
            print("Simulation completed without numerical instabilities.")


if __name__ == "__main__":
    main()