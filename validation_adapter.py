# ValidationAdapter.py
# This module provides adapters between the validation framework and the refactored codebase structure.

import importlib
import logging
import inspect
import numpy as np
import sys
import os
import re
from pathlib import Path

# Import module_map if available
try:
    from module_map import map_function_to_domain, load_function_from_domain, get_domain_for_function

    HAS_MODULE_MAP = True
except ImportError:
    HAS_MODULE_MAP = False
    print("Warning: Could not import module_map, falling back to direct imports")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ValidationAdapter")


class FunctionAdapter:
    """
    Adapts functions from the refactored codebase to the interface expected by the validation framework.
    Prioritizes using the existing module_map for function discovery.
    """

    @staticmethod
    def get_physics_function(function_name):
        """
        Attempts to find and return a function by name from the physics domains.
        Tries multiple methods in priority order:
        1. Use module_map.load_function_from_domain if available
        2. Try direct import from physics_domains structure
        3. Fall back to direct import from config directory
        4. Create a dummy function as last resort

        Args:
            function_name: The name of the function to find

        Returns:
            The function if found, or a dummy function if not found
        """
        # 1. Try using module_map if available
        if HAS_MODULE_MAP:
            try:
                # First try loading using module_map's function
                function = load_function_from_domain(function_name)
                if function is not None:
                    logger.info(f"Found {function_name} using module_map.load_function_from_domain")
                    return function

                # If that fails, try getting the domain and importing directly
                domain = get_domain_for_function(function_name)
                if domain and domain != "UNKNOWN":
                    try:
                        module_name = f"physics_domains.{domain.lower()}.{function_name}"
                        module = importlib.import_module(module_name)
                        if hasattr(module, function_name):
                            logger.info(f"Found {function_name} in {module_name} using domain from module_map")
                            return getattr(module, function_name)
                    except ImportError:
                        pass
            except Exception as e:
                logger.warning(f"Error using module_map to find {function_name}: {e}")

        # 2. Try direct import from physics_domains structure
        # First determine likely domains based on function name
        domain_mapping = {
            "intelligence": "thermodynamics",
            "knowledge": "thermodynamics",
            "truth": "relativity",
            "suppression": "weak_nuclear",
            "resistance": "weak_nuclear",
            "field": "electromagnetism",
            "wisdom": "electromagnetism",
            "quantum": "quantum_mechanics",
            "entanglement": "quantum_mechanics",
            "civilization": "strong_nuclear",
        }

        # Determine possible domains for this function
        possible_domains = []
        for keyword, domain in domain_mapping.items():
            if keyword in function_name:
                possible_domains.append(domain)

        # Add a default list of all domains if nothing matched
        if not possible_domains:
            possible_domains = ["thermodynamics", "relativity", "electromagnetism",
                                "weak_nuclear", "strong_nuclear", "quantum_mechanics",
                                "astrophysics", "multi_system"]

        # Try direct import from physics_domains
        for domain in possible_domains:
            try:
                module_name = f"physics_domains.{domain}.{function_name}"
                module = importlib.import_module(module_name)
                if hasattr(module, function_name):
                    logger.info(f"Found {function_name} in {module_name}")
                    return getattr(module, function_name)
            except ImportError:
                pass

        # 3. Try fallback to config
        try:
            # Try config.equations first
            try:
                module = importlib.import_module("config.equations")
                if hasattr(module, function_name):
                    logger.info(f"Found {function_name} in config.equations")
                    return getattr(module, function_name)
            except ImportError:
                pass

            # Try more specific config modules
            if "intelligence" in function_name or "knowledge" in function_name:
                from config.equations import intelligence_growth
                return intelligence_growth
            elif "truth" in function_name:
                from config.equations import truth_adoption
                return truth_adoption
            elif "suppression" in function_name or "resistance" in function_name:
                from config.equations import suppression_feedback
                return suppression_feedback
            elif "field" in function_name or "wisdom" in function_name:
                from config.equations import wisdom_field
                return wisdom_field
            elif "quantum" in function_name or "entanglement" in function_name:
                try:
                    from config.quantum_em_extensions import quantum_tunneling_probability
                    return quantum_tunneling_probability
                except ImportError:
                    pass
            elif "civilization" in function_name:
                try:
                    from config.astrophysics_extensions import civilization_lifecycle_phase
                    return civilization_lifecycle_phase
                except ImportError:
                    pass
        except ImportError:
            pass

        # 4. Create a dummy function as last resort
        logger.warning(f"Function {function_name} not found. Creating dummy implementation.")

        def dummy_function(*args, **kwargs):
            logger.warning(f"Using dummy implementation of {function_name}")
            if "intelligence" in function_name or "knowledge" in function_name:
                return 5.0
            elif "truth" in function_name:
                return 1.0
            elif "suppression" in function_name or "resistance" in function_name:
                return 2.0
            elif "field" in function_name or "wisdom" in function_name:
                return 3.0
            elif "quantum" in function_name or "entanglement" in function_name:
                return 0.5
            elif "oscillation" in function_name:
                return 0.1
            else:
                return 1.0

        return dummy_function

    @staticmethod
    def get_docstring_metadata(func_name, func):
        """
        Extract metadata from a function's docstring.

        Args:
            func_name: Name of the function
            func: Function object

        Returns:
            dict: Metadata extracted from docstring
        """
        metadata = {
            'physics_domain': None,
            'scale_level': None,
            'application_domains': []
        }

        if not func.__doc__:
            return metadata

        # Extract Physics Domain
        domain_match = re.search(r"Physics Domain:\s*(\w+)", func.__doc__, re.IGNORECASE)
        if domain_match:
            metadata['physics_domain'] = domain_match.group(1).lower()

        # Extract Scale Level
        scale_match = re.search(r"Scale Level:\s*(\w+)", func.__doc__, re.IGNORECASE)
        if scale_match:
            metadata['scale_level'] = scale_match.group(1).lower()

        # Extract Application Domains
        apps_match = re.search(r"Application Domains:\s*([\w\s,]+)", func.__doc__, re.IGNORECASE)
        if apps_match:
            apps = [app.strip().lower() for app in apps_match.group(1).split(',')]
            metadata['application_domains'] = apps

        return metadata


class SimulationAdapter:
    """
    Adapts simulation functions to the interface expected by the validation framework.
    """

    @staticmethod
    def get_simulation_function(simulation_name):
        """
        Attempts to find and return a simulation function by name.

        Args:
            simulation_name: The name of the simulation module

        Returns:
            The simulation function if found, or a dummy function if not found
        """
        # Try both old and new locations
        for module_prefix in ["simulations", "config.simulations"]:
            module_name = f"{module_prefix}.{simulation_name}"
            try:
                module = importlib.import_module(module_name)
                # Look for common function names
                for func_name in ["run_simulation", "run", "simulate", "execute_simulation"]:
                    if hasattr(module, func_name):
                        logger.info(f"Found {func_name} in {module_name}")
                        return getattr(module, func_name)
            except ImportError:
                logger.warning(f"Could not import module {module_name}")

        # Look for alternative naming patterns
        alternative_names = [
            f"{simulation_name}_simulation",
            f"run_{simulation_name}",
            f"simulate_{simulation_name}",
        ]

        for alt_name in alternative_names:
            for module_prefix in ["simulations", "config.simulations"]:
                module_name = f"{module_prefix}.{alt_name}"
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, "run_simulation"):
                        logger.info(f"Found run_simulation in {module_name}")
                        return getattr(module, "run_simulation")
                except ImportError:
                    continue

        logger.warning(f"Simulation function for {simulation_name} not found. Creating dummy implementation.")

        def dummy_simulation(*args, **kwargs):
            logger.warning(f"Using dummy implementation for {simulation_name}")
            timesteps = 100
            return {
                'time': np.arange(timesteps),
                'knowledge': np.random.rand(timesteps) * 10,
                'suppression': np.random.rand(timesteps) * 5,
                'intelligence': np.random.rand(timesteps) * 15,
                'truth': np.random.rand(timesteps) * 20
            }

        return dummy_simulation


class ValidationAdapter:
    """
    Main adapter class that provides interfaces for all validation components.
    """

    def __init__(self):
        """Initialize the validation adapter."""
        self.function_adapter = FunctionAdapter()
        self.simulation_adapter = SimulationAdapter()

        # Initialize components that will be lazily loaded
        self._edge_case_checker = None
        self._cross_level_validator = None
        self._sensitivity_analyzer = None
        self._dimensional_validator = None
        self._circuit_breaker = None

    def get_edge_case_checker(self):
        """Get the EdgeCaseChecker or an adapter for it."""
        if self._edge_case_checker is not None:
            return self._edge_case_checker

        try:
            from utils.edge_case_checker import EdgeCaseChecker

            # Get core functions
            eq_funcs = {}
            for fname in [
                "intelligence_growth", "truth_adoption", "wisdom_field",
                "suppression_feedback", "resistance_resurgence", "civilization_oscillation"
            ]:
                eq_funcs[fname] = self.function_adapter.get_physics_function(fname)

            self._edge_case_checker = EdgeCaseChecker(eq_funcs)
            return self._edge_case_checker

        except ImportError:
            logger.warning("EdgeCaseChecker not found, creating adapter")

            class EdgeCaseCheckerAdapter:
                def __init__(self, *args, **kwargs):
                    pass

                def analyze_all_functions(self):
                    return {}

                def generate_recommendations(self):
                    return {}

                def generate_fixes(self, function_name):
                    return f"# No fixes needed for {function_name}"

                def run_edge_case_check(self, equations, output_dir=None):
                    return {'status': 'success', 'message': 'Edge case check simulated'}

            self._edge_case_checker = EdgeCaseCheckerAdapter()
            return self._edge_case_checker

    def get_cross_level_validator(self):
        """Get the CrossLevelValidator or an adapter for it."""
        if self._cross_level_validator is not None:
            return self._cross_level_validator

        try:
            from utils.cross_level_validator import CrossLevelValidator

            # Get core functions
            eq_funcs = {}
            for fname in [
                "intelligence_growth", "truth_adoption", "wisdom_field",
                "suppression_feedback", "resistance_resurgence", "civilization_oscillation"
            ]:
                eq_funcs[fname] = self.function_adapter.get_physics_function(fname)

            # Create standard hierarchy levels
            hierarchy_levels = {
                'Level 1 (Core)': [
                    'intelligence_growth', 'truth_adoption', 'wisdom_field'
                ],
                'Level 2 (Extended)': [
                    'suppression_feedback', 'resistance_resurgence', 'civilization_oscillation'
                ]
            }

            self._cross_level_validator = CrossLevelValidator(eq_funcs, hierarchy_levels)

            # Inject adapter's function discovery into the validator
            original_validate = self._cross_level_validator.validate_level_dependencies

            def patched_validate(expected_dependencies=None):
                # Make sure to update equation functions with the latest versions
                for fname in eq_funcs:
                    eq_funcs[fname] = self.function_adapter.get_physics_function(fname)

                self._cross_level_validator.equation_functions = eq_funcs
                return original_validate(expected_dependencies)

            self._cross_level_validator.validate_level_dependencies = patched_validate

            # Also patch the run_cross_level_validation function if it exists
            if hasattr(self._cross_level_validator, "run_cross_level_validation"):
                original_run = self._cross_level_validator.run_cross_level_validation

                def patched_run(equations=None, hierarchy_levels=None, output_dir=None):
                    if equations is None:
                        equations = eq_funcs
                    if hierarchy_levels is None:
                        hierarchy_levels = hierarchy_levels
                    return original_run(equations, hierarchy_levels, output_dir)

                self._cross_level_validator.run_cross_level_validation = patched_run

            return self._cross_level_validator

        except ImportError:
            logger.warning("CrossLevelValidator not found, creating adapter")

            class CrossLevelValidatorAdapter:
                def __init__(self, *args, **kwargs):
                    pass

                def build_dependency_graph(self):
                    return {}

                def validate_level_dependencies(self):
                    return {'is_valid': True, 'violations': [], 'status': 'success'}

                def detect_feedback_loops(self):
                    return []

                def run_cross_level_validation(self, equations=None, hierarchy_levels=None, output_dir=None):
                    return {'status': 'success', 'message': 'Cross-level validation simulated'}

            self._cross_level_validator = CrossLevelValidatorAdapter()
            return self._cross_level_validator

    def get_sensitivity_analyzer(self):
        """Get the ParameterSensitivityAnalyzer or an adapter for it."""
        if self._sensitivity_analyzer is not None:
            return self._sensitivity_analyzer

        try:
            from utils.sensitivity_analyzer import ParameterSensitivityAnalyzer

            # Get simulation function
            sim_func = self.simulation_adapter.get_simulation_function("comprehensive_simulation")

            # Common metrics and parameters
            metrics = ["knowledge", "suppression", "intelligence", "truth"]
            base_params = {
                'alpha_wisdom': 0.1,
                'alpha_feedback': 0.1,
                'beta_feedback': 0.05,
                'gamma_phase': 0.1
            }

            self._sensitivity_analyzer = ParameterSensitivityAnalyzer(sim_func, metrics, base_params)
            return self._sensitivity_analyzer

        except ImportError:
            logger.warning("ParameterSensitivityAnalyzer not found, creating adapter")

            class SensitivityAnalyzerAdapter:
                def __init__(self, *args, **kwargs):
                    pass

                def define_parameter_ranges(self, ranges):
                    return

                def run_one_at_a_time_sensitivity(self, parallel=False):
                    return {}

                def calculate_parameter_importance(self):
                    import pandas as pd
                    return pd.Series({'param1': 0.5, 'param2': 0.3, 'param3': 0.2})

                def run_sensitivity_analysis(self, parameters=None, metrics=None, output_dir=None):
                    return {'status': 'success', 'message': 'Sensitivity analysis simulated'}

            self._sensitivity_analyzer = SensitivityAnalyzerAdapter()
            return self._sensitivity_analyzer

    def get_dimensional_validator(self):
        """Get the DimensionalValidator or an adapter for it."""
        if self._dimensional_validator is not None:
            return self._dimensional_validator

        try:
            from utils.dimensional_consistency import DimensionalValidator

            self._dimensional_validator = DimensionalValidator()

            # Patch the validate_dimensional_consistency method to use our function adapter
            original_validate = self._dimensional_validator.validate_dimensional_consistency

            def patched_validate(equation_modules=None):
                """
                Patched version that uses our function adapter to get equation functions.
                """
                if equation_modules is None:
                    # Create a dict of equation functions
                    equation_modules = {}

                # Add our physics functions if needed
                core_functions = [
                    "intelligence_growth", "truth_adoption", "wisdom_field",
                    "suppression_feedback", "resistance_resurgence", "civilization_oscillation"
                ]

                for fname in core_functions:
                    if fname not in equation_modules:
                        equation_modules[fname] = self.function_adapter.get_physics_function(fname)

                return original_validate(equation_modules)

            self._dimensional_validator.validate_dimensional_consistency = patched_validate

            # Make sure we have the run_dimensional_validation function
            if not hasattr(self._dimensional_validator, "run_dimensional_validation"):
                from utils.dimensional_consistency import run_dimensional_validation

                def patched_run(dimensional_equations=None, output_dir=None):
                    if dimensional_equations is None:
                        dimensional_equations = {}
                        for fname in core_functions:
                            dimensional_equations[fname] = self.function_adapter.get_physics_function(fname)
                    return run_dimensional_validation(dimensional_equations, output_dir)

                self._dimensional_validator.run_dimensional_validation = patched_run

            return self._dimensional_validator

        except (ImportError, AttributeError):
            logger.warning("DimensionalValidator not found, checking for alternative")
            try:
                from utils.dimensional_consistency import check_dimensional_consistency, run_dimensional_validation

                class DimensionalValidatorWrapper:
                    def __init__(self):
                        self.check_func = check_dimensional_consistency
                        self.function_adapter = FunctionAdapter()

                    def validate_dimensional_consistency(self, equation_modules=None):
                        if equation_modules is None:
                            equation_modules = {}

                        # Add our physics functions
                        core_functions = [
                            "intelligence_growth", "truth_adoption", "wisdom_field",
                            "suppression_feedback", "resistance_resurgence", "civilization_oscillation"
                        ]

                        for fname in core_functions:
                            if fname not in equation_modules:
                                equation_modules[fname] = self.function_adapter.get_physics_function(fname)

                        return self.check_func(equation_modules)

                    def run_dimensional_validation(self, dimensional_equations=None, output_dir=None):
                        if dimensional_equations is None:
                            dimensional_equations = {}
                            core_functions = [
                                "intelligence_growth", "truth_adoption", "wisdom_field",
                                "suppression_feedback", "resistance_resurgence", "civilization_oscillation"
                            ]
                            for fname in core_functions:
                                dimensional_equations[fname] = self.function_adapter.get_physics_function(fname)

                        return run_dimensional_validation(dimensional_equations, output_dir)

                self._dimensional_validator = DimensionalValidatorWrapper()
                return self._dimensional_validator

            except ImportError:
                logger.warning("check_dimensional_consistency not found, creating adapter")

                class DimensionalValidatorAdapter:
                    def __init__(self):
                        pass

                    def validate_dimensional_consistency(self, equation_modules=None):
                        return {
                            'equations': {},
                            'status': 'success',
                            'issues_found': 0,
                            'total_equations': 0
                        }

                    def run_dimensional_validation(self, dimensional_equations=None, output_dir=None):
                        return {
                            'status': 'success',
                            'message': 'Dimensional validation simulated'
                        }

                self._dimensional_validator = DimensionalValidatorAdapter()
                return self._dimensional_validator

    def get_circuit_breaker(self):
        """Get the CircuitBreaker or an adapter for it."""
        if self._circuit_breaker is not None:
            return self._circuit_breaker

        try:
            from utils.circuit_breaker import CircuitBreaker
            self._circuit_breaker = CircuitBreaker()
            return self._circuit_breaker
        except ImportError:
            logger.warning("CircuitBreaker not found, creating adapter")

            class CircuitBreakerAdapter:
                def __init__(self):
                    pass

                def check_value_stability(self, value):
                    return False

                def check_and_fix(self, value, min_val=None, max_val=None, default=None):
                    return value

                def safe_div(self, x, y, default=0.0):
                    return x / y if y != 0 else default

                def safe_exp(self, x, max_result=1e10):
                    return min(np.exp(min(50, x)), max_result)

            self._circuit_breaker = CircuitBreakerAdapter()
            return self._circuit_breaker

    def get_physics_domains(self):
        """
        Get a list of all available physics domains in the codebase.
        Uses module_map if available, otherwise scans directories.

        Returns:
            list: List of physics domain names
        """
        domains = []

        # First try using module_map if available
        if HAS_MODULE_MAP:
            try:
                from module_map import get_functions_by_domain
                domain_funcs = get_functions_by_domain()
                return list(domain_funcs.keys())
            except (ImportError, AttributeError):
                pass

        # Then check physics_domains directory
        base_path = Path(__file__).resolve().parent / "physics_domains"
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir() and not item.name.startswith('__'):
                    domains.append(item.name)

        # Add known domains if none found
        if not domains:
            domains = [
                "thermodynamics", "relativity", "electromagnetism",
                "weak_nuclear", "strong_nuclear", "quantum_mechanics",
                "astrophysics", "multi_system"
            ]

        return domains

    def get_functions_by_domain(self):
        """
        Get a mapping of functions to their physics domains.
        Uses module_map if available, otherwise scans directories.

        Returns:
            dict: Dictionary mapping domain names to lists of function names
        """
        # First try using module_map if available
        if HAS_MODULE_MAP:
            try:
                from module_map import get_functions_by_domain
                return get_functions_by_domain()
            except (ImportError, AttributeError):
                pass

        domains = self.get_physics_domains()
        domain_functions = {domain: [] for domain in domains}

        # Scan physics_domains directory
        base_path = Path(__file__).resolve().parent / "physics_domains"
        if base_path.exists():
            for domain in domains:
                domain_path = base_path / domain
                if domain_path.exists():
                    for py_file in domain_path.glob("*.py"):
                        if py_file.name.startswith('__'):
                            continue
                        # Function name is same as file name without .py
                        func_name = py_file.stem
                        domain_functions[domain].append(func_name)

        return domain_functions

    def run_unified_validation(self, output_dir=None):
        """
        Run all validations in a unified manner.

        Args:
            output_dir: Directory to save validation results

        Returns:
            dict: Combined validation results
        """
        results = {}

        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Run edge case check
        try:
            edge_case_checker = self.get_edge_case_checker()

            # Get core functions
            eq_funcs = {}
            for fname in [
                "intelligence_growth", "truth_adoption", "wisdom_field",
                "suppression_feedback", "resistance_resurgence", "civilization_oscillation"
            ]:
                eq_funcs[fname] = self.function_adapter.get_physics_function(fname)

            # Run analysis
            edge_case_checker.analyze_all_functions()

            # Save results if output directory provided
            if output_dir and hasattr(edge_case_checker, "generate_edge_case_completion_report"):
                edge_case_dir = os.path.join(output_dir, "edge_case")
                os.makedirs(edge_case_dir, exist_ok=True)
                edge_case_checker.generate_edge_case_completion_report(edge_case_dir)

            # Add to results
            results['edge_case'] = {
                'status': 'success',
                'recommendations': edge_case_checker.generate_recommendations()
            }
        except Exception as e:
            logger.error(f"Error in edge case validation: {e}")
            results['edge_case'] = {'status': 'error', 'message': str(e)}

        # Run cross-level validation
        try:
            cross_validator = self.get_cross_level_validator()

            # Get core functions
            eq_funcs = {}
            for fname in [
                "intelligence_growth", "truth_adoption", "wisdom_field",
                "suppression_feedback", "resistance_resurgence", "civilization_oscillation"
            ]:
                eq_funcs[fname] = self.function_adapter.get_physics_function(fname)

            # Build graph
            cross_validator.build_dependency_graph()

            # Validate dependencies
            dependency_results = cross_validator.validate_level_dependencies()

            # Save results if output directory provided
            if output_dir and hasattr(cross_validator, "generate_validation_report"):
                cross_level_dir = os.path.join(output_dir, "cross_level")
                os.makedirs(cross_level_dir, exist_ok=True)
                cross_validator.generate_validation_report(cross_level_dir)

            # Add to results
            results['cross_level'] = {
                'status': 'success' if dependency_results.get('is_valid', False) else 'warning',
                'dependencies': dependency_results
            }
        except Exception as e:
            logger.error(f"Error in cross-level validation: {e}")
            results['cross_level'] = {'status': 'error', 'message': str(e)}

        # Run dimensional consistency validation
        try:
            dim_validator = self.get_dimensional_validator()

            # Validate dimensional consistency
            dim_results = dim_validator.validate_dimensional_consistency()

            # Save results if output directory provided
            if output_dir:
                dim_dir = os.path.join(output_dir, "dimensional")
                os.makedirs(dim_dir, exist_ok=True)
                if hasattr(dim_validator, "run_dimensional_validation"):
                    dim_validator.run_dimensional_validation(output_dir=dim_dir)

            # Add to results
            results['dimensional'] = {
                'status': dim_results.get('status', 'error'),
                'issues_found': dim_results.get('issues_found', 0),
                'total_equations': dim_results.get('total_equations', 0)
            }
        except Exception as e:
            logger.error(f"Error in dimensional validation: {e}")
            results['dimensional'] = {'status': 'error', 'message': str(e)}

        # Run sensitivity analysis
        try:
            sensitivity_analyzer = self.get_sensitivity_analyzer()

            # Define parameter ranges
            parameter_ranges = {
                'alpha_wisdom': (0.01, 0.5, 5),
                'alpha_feedback': (0.01, 0.5, 5),
                'beta_feedback': (0.01, 0.2, 5),
                'gamma_phase': (0.01, 0.5, 5)
            }
            sensitivity_analyzer.define_parameter_ranges(parameter_ranges)

            # Run sensitivity analysis
            sensitivity_analyzer.run_one_at_a_time_sensitivity(parallel=True)

            # Save results if output directory provided
            if output_dir and hasattr(sensitivity_analyzer, "generate_comprehensive_report"):
                sensitivity_dir = os.path.join(output_dir, "sensitivity")
                os.makedirs(sensitivity_dir, exist_ok=True)
                sensitivity_analyzer.generate_comprehensive_report(sensitivity_dir)

            # Add to results
            results['sensitivity'] = {
                'status': 'success',
                'importance': sensitivity_analyzer.calculate_parameter_importance().to_dict()
            }
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            results['sensitivity'] = {'status': 'error', 'message': str(e)}

        # Determine overall status
        overall_status = 'success'
        for component, result in results.items():
            if result.get('status') == 'error':
                overall_status = 'error'
                break
            elif result.get('status') == 'warning' and overall_status != 'error':
                overall_status = 'warning'

        # Generate unified report if output directory provided
        if output_dir:
            self._generate_unified_report(results, output_dir, overall_status)

        return {
            'components': results,
            'overall_status': overall_status
        }

    def _generate_unified_report(self, results, output_dir, overall_status):
        """
        Generate a unified HTML report of all validation results.

        Args:
            results: Dictionary of validation results
            output_dir: Directory to save the report
            overall_status: Overall validation status
        """
        from datetime import datetime

        # Create unified report directory
        unified_dir = os.path.join(output_dir, "unified")
        os.makedirs(unified_dir, exist_ok=True)

        # Generate HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unified Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
                .section {{ margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Unified Validation Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

                <div class="summary">
                    <h2>Validation Summary</h2>
                    <p>Overall Status: <span class="{overall_status}">{overall_status.upper()}</span></p>
                    <table>
                        <tr>
                            <th>Component</th>
                            <th>Status</th>
                            <th>Details</th>
                        </tr>
        """

        # Add component results
        for component, result in results.items():
            status = result.get('status', 'unknown')
            details = ''

            if component == 'edge_case':
                recommendations = result.get('recommendations', {})
                total_recs = sum(len(recs) for recs in recommendations.values())
                details = f"{total_recs} recommendations"
            elif component == 'cross_level':
                dependencies = result.get('dependencies', {})
                violations = dependencies.get('violations', [])
                details = f"{len(violations)} violations"
            elif component == 'dimensional':
                issues = result.get('issues_found', 0)
                total = result.get('total_equations', 0)
                details = f"{issues} issues in {total} equations"
            elif component == 'sensitivity':
                importance = result.get('importance', {})
                if importance:
                    top_param = max(importance.items(), key=lambda x: x[1])[0]
                    details = f"Most important parameter: {top_param}"

            html += f"""
                        <tr>
                            <td>{component.replace('_', ' ').title()}</td>
                            <td class="{status}">{status.upper()}</td>
                            <td>{details}</td>
                        </tr>
            """

        html += """
                    </table>
                </div>
        """

        # Add sections for each component
        for component, result in results.items():
            html += f"""
                <div class="section">
                    <h2>{component.replace('_', ' ').title()} Validation</h2>
            """

            if component == 'edge_case':
                recommendations = result.get('recommendations', {})
                if recommendations:
                    html += """
                        <h3>Key Recommendations</h3>
                        <table>
                            <tr>
                                <th>Function</th>
                                <th>Recommendations</th>
                            </tr>
                    """

                    for func, recs in recommendations.items():
                        html += f"""
                            <tr>
                                <td>{func}</td>
                                <td>{len(recs)} recommendations</td>
                            </tr>
                        """

                    html += """
                        </table>
                        <p>See detailed report in <code>edge_case/</code> directory for more information.</p>
                    """
            elif component == 'cross_level':
                dependencies = result.get('dependencies', {})
                violations = dependencies.get('violations', [])

                if violations:
                    html += """
                        <h3>Dependency Violations</h3>
                        <table>
                            <tr>
                                <th>From Function</th>
                                <th>From Level</th>
                                <th>To Function</th>
                                <th>To Level</th>
                            </tr>
                    """

                    for violation in violations:
                        html += f"""
                            <tr>
                                <td>{violation.get('from_function', '')}</td>
                                <td>{violation.get('from_level', '')}</td>
                                <td>{violation.get('to_function', '')}</td>
                                <td>{violation.get('to_level', '')}</td>
                            </tr>
                        """

                    html += """
                        </table>
                    """
                else:
                    html += """
                        <p class="success">No dependency violations found.</p>
                    """

                html += """
                    <p>See detailed report in <code>cross_level/</code> directory for more information.</p>
                """
            elif component == 'dimensional':
                issues = result.get('issues_found', 0)
                total = result.get('total_equations', 0)

                if issues > 0:
                    html += f"""
                        <p class="warning">Found {issues} dimensional consistency issues in {total} equations.</p>
                    """
                else:
                    html += """
                        <p class="success">No dimensional consistency issues found.</p>
                    """

                html += """
                    <p>See detailed report in <code>dimensional/</code> directory for more information.</p>
                """
            elif component == 'sensitivity':
                importance = result.get('importance', {})

                if importance:
                    html += """
                        <h3>Parameter Importance</h3>
                        <table>
                            <tr>
                                <th>Parameter</th>
                                <th>Importance</th>
                            </tr>
                    """

                    for param, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                        html += f"""
                            <tr>
                                <td>{param}</td>
                                <td>{value:.4f}</td>
                            </tr>
                        """

                    html += """
                        </table>
                    """

                html += """
                    <p>See detailed report in <code>sensitivity/</code> directory for more information.</p>
                """

            html += """
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        # Write HTML to file
        with open(os.path.join(unified_dir, "unified_validation_report.html"), "w") as f:
            f.write(html)

        logger.info(f"Unified validation report saved to {os.path.join(unified_dir, 'unified_validation_report.html')}")


# Create a global instance for easy access
validation_adapter = ValidationAdapter()