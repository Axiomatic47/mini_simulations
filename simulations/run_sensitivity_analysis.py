from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer
from physics_domains.thermodynamics.intelligence_growth import intelligence_growth
from physics_domains.relativity.truth_adoption import truth_adoption
from physics_domains.electromagnetism.wisdom_field import wisdom_field


# Import other needed functions

def main():
    # Create analyzer with appropriate simulation functions
    analyzer = ParameterSensitivityAnalyzer(...)

    # Define parameter ranges
    analyzer.define_parameter_ranges({
        'K_0': (0.1, 5.0, 5),
        'S_0': (5.0, 20.0, 5),
        # Other parameters...
    })

    # Run analysis
    results = analyzer.run_one_at_a_time_sensitivity()

    # Generate report
    analyzer.generate_comprehensive_report("validation/reports/sensitivity_report")

    print("Sensitivity analysis complete. Report generated in validation/reports/sensitivity_report")


if __name__ == "__main__":
    main()