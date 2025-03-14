# utils/edge_case_checker.py
import inspect
import re
import ast
import sys
import numpy as np
from pathlib import Path


class EdgeCaseChecker:
    """
    Utility for checking and fixing potential edge cases in equation functions.
    This helps identify and resolve numerical stability issues.
    """

    def __init__(self, equation_functions):
        """
        Initialize with a dictionary of equation functions.

        Args:
            equation_functions (dict): Dictionary mapping function names to function objects
        """
        self.equation_functions = equation_functions
        self.analysis_results = {}
        self.results = {}  # Adding this attribute that's referenced in visualization code
        self.analyzed_functions = {}  # Initialize this attribute
        self.function_code = {}  # Initialize other attributes
        self.numeric_patterns = {
            'division_by_zero': {
                'pattern': r'\/\s*([a-zA-Z_][a-zA-Z0-9_]*|\d*\.?\d+)',
                'description': 'Division that might lead to division by zero',
                'severity': 'high'
            },
            'log_of_non_positive': {
                'pattern': r'(np\.log|np\.log10|np\.log2|np\.log1p|math\.log|math\.log10|math\.log2)\s*\(',
                'description': 'Logarithm that might receive zero or negative input',
                'severity': 'high'
            },
            'sqrt_of_negative': {
                'pattern': r'(np\.sqrt|math\.sqrt)\s*\(',
                'description': 'Square root that might receive negative input',
                'severity': 'high'
            },
            'exponent_overflow': {
                'pattern': r'(np\.exp|math\.exp)\s*\(',
                'description': 'Exponential that might overflow',
                'severity': 'medium'
            },
            'array_bounds': {
                'pattern': r'\[[a-zA-Z_][a-zA-Z0-9_]*\]',
                'description': 'Array indexing that might go out of bounds',
                'severity': 'high'
            },
            'conditional_logic': {
                'pattern': r'if\s+[^:]+:',
                'description': 'Conditional logic that might need additional testing',
                'severity': 'low'
            }
        }

    def analyze_function(self, func_name):
        """
        Analyze a single function for potential edge cases.

        Args:
            func_name (str): Name of the function to analyze

        Returns:
            dict: Analysis results
        """
        if func_name not in self.equation_functions:
            return {'error': f"Function {func_name} not found"}

        func = self.equation_functions[func_name]

        # Get function source code
        try:
            source = inspect.getsource(func)
        except (TypeError, OSError):
            return {'error': f"Could not retrieve source code for {func_name}"}

        # Initialize result
        result = {
            'function_name': func_name,
            'patterns_found': {},
            'edge_case_count': 0,
            'safety_score': 100,
            'recommendations': []
        }

        # Check for each numeric pattern
        for pattern_name, pattern_info in self.numeric_patterns.items():
            matches = re.finditer(pattern_info['pattern'], source)
            match_lines = []

            for match in matches:
                # Get line information
                line_start = source[:match.start()].count('\n') + 1
                line_content = source.split('\n')[line_start - 1].strip()

                # Check if this line already has a safety check
                has_safety_check = self._has_safety_check(pattern_name, line_content)

                match_info = {
                    'line': line_start,
                    'content': line_content,
                    'has_safety_check': has_safety_check
                }

                match_lines.append(match_info)

            if match_lines:
                unprotected_matches = [m for m in match_lines if not m['has_safety_check']]
                result['patterns_found'][pattern_name] = {
                    'description': pattern_info['description'],
                    'severity': pattern_info['severity'],
                    'matches': match_lines,
                    'total_matches': len(match_lines),
                    'unprotected_matches': len(unprotected_matches)
                }
                result['edge_case_count'] += len(unprotected_matches)

                # Reduce safety score based on severity
                if unprotected_matches:
                    if pattern_info['severity'] == 'high':
                        result['safety_score'] -= 15 * len(unprotected_matches)
                    elif pattern_info['severity'] == 'medium':
                        result['safety_score'] -= 8 * len(unprotected_matches)
                    else:  # low
                        result['safety_score'] -= 3 * len(unprotected_matches)

        # Generate recommendations
        result['recommendations'] = self._generate_function_recommendations(func_name, result)

        # Cap safety score at 0-100
        result['safety_score'] = max(0, min(100, result['safety_score']))

        # Assign a risk level
        if result['safety_score'] >= 80:
            result['risk_level'] = 'low'
        elif result['safety_score'] >= 50:
            result['risk_level'] = 'medium'
        else:
            result['risk_level'] = 'high'

        self.analysis_results[func_name] = result
        return result

    def analyze_all_functions(self):
        """
        Analyze all functions in the equation_functions dictionary for potential edge cases.

        Returns:
            dict: Analysis results for all functions
        """
        for func_name, func in self.equation_functions.items():
            self.analyzed_functions[func_name] = self.analyze_function(func_name)

            # Update the results dictionary that's used by visualization code
            self.results[func_name] = {
                'issues': self.analyzed_functions[func_name],
                'score': self._calculate_score(self.analyzed_functions[func_name])
            }

        return self.analyzed_functions

    def _calculate_score(self, analysis_result):
        """
        Calculate a safety score based on analysis results.

        Args:
            analysis_result (dict): Analysis results for a function

        Returns:
            float: Safety score between 0-100
        """
        # Return the safety score directly if it exists
        if 'safety_score' in analysis_result:
            return analysis_result['safety_score']

        # Otherwise calculate a basic score
        score = 100

        # Reduce score based on unprotected patterns
        for pattern_name, pattern_result in analysis_result.get('patterns_found', {}).items():
            unprotected_count = pattern_result.get('unprotected_matches', 0)
            severity = pattern_result.get('severity', 'medium')

            # Adjust score based on severity
            if severity == 'high':
                score -= 15 * unprotected_count
            elif severity == 'medium':
                score -= 8 * unprotected_count
            else:  # low
                score -= 3 * unprotected_count

        # Ensure score is between 0-100
        return max(0, min(100, score))

    def generate_recommendations(self):
        """
        Generate recommendations for improving all analyzed functions.

        Returns:
            dict: Recommendations for each function
        """
        if not self.analysis_results:
            self.analyze_all_functions()

        recommendations = {}

        for func_name, result in self.analysis_results.items():
            if 'error' in result:
                recommendations[func_name] = []
                continue

            func_recommendations = []

            for pattern_name, pattern_result in result.get('patterns_found', {}).items():
                if pattern_result['unprotected_matches'] > 0:
                    for match in pattern_result['matches']:
                        if not match['has_safety_check']:
                            rec = {
                                'issue': f"{pattern_result['description']} at line {match['line']}",
                                'line': match['line'],
                                'content': match['content'],
                                'severity': pattern_result['severity'],
                                'recommendation': self._get_recommendation_text(pattern_name, match['content'])
                            }
                            func_recommendations.append(rec)

            # Sort recommendations by severity
            severity_order = {'high': 0, 'medium': 1, 'low': 2}
            func_recommendations.sort(key=lambda x: severity_order[x['severity']])

            recommendations[func_name] = func_recommendations

        return recommendations

    def generate_fixes(self, func_name):
        """
        Generate fixed code for a function with edge cases.

        Args:
            func_name (str): Name of the function to fix

        Returns:
            str: Fixed function code
        """
        if func_name not in self.equation_functions:
            return f"Function {func_name} not found"

        if func_name not in self.analysis_results:
            self.analyze_function(func_name)

        result = self.analysis_results[func_name]
        if 'error' in result:
            return f"Error analyzing function: {result['error']}"

        if result['edge_case_count'] == 0:
            return "No edge cases found that need fixing"

        # Get original function
        func = self.equation_functions[func_name]
        try:
            source = inspect.getsource(func)
        except (TypeError, OSError):
            return f"Could not retrieve source code for {func_name}"

        # Try to parse as AST
        try:
            tree = ast.parse(source)
            function_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                    function_node = node
                    break

            if not function_node:
                # Fallback to regex-based approach
                return self._fix_with_regex(func_name, source)

            # Apply fixes using AST transformer
            transformer = EdgeCaseTransformer(self.analysis_results[func_name])
            fixed_tree = transformer.visit(tree)

            # Convert back to source
            fixed_source = ast.unparse(fixed_tree)
            return fixed_source

        except (SyntaxError, AttributeError) as e:
            # Fallback to regex-based approach
            return self._fix_with_regex(func_name, source)

    def add_circuit_breaker(self, func_name):
        """
        Add circuit breaker integration to a function.

        Args:
            func_name (str): Name of the function to enhance

        Returns:
            str: Modified function code with circuit breaker
        """
        if func_name not in self.equation_functions:
            return f"Function {func_name} not found"

        # Get original function
        func = self.equation_functions[func_name]
        try:
            source = inspect.getsource(func)
        except (TypeError, OSError):
            return f"Could not retrieve source code for {func_name}"

        # Parse function signature and docstring
        try:
            tree = ast.parse(source)
            function_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                    function_node = node
                    break

            if not function_node:
                return "Could not parse function structure"

            # Extract function details
            func_name = function_node.name
            args = [arg.arg for arg in function_node.args.args]

            # Generate circuit breaker version
            cb_source = self._generate_circuit_breaker_version(source, func_name, args)
            return cb_source

        except (SyntaxError, AttributeError) as e:
            return f"Error parsing function: {str(e)}"

    def generate_test_cases(self, func_name):
        """
        Generate test cases targeting edge cases for a function.

        Args:
            func_name (str): Name of the function to test

        Returns:
            str: Test code for the function
        """
        if func_name not in self.equation_functions:
            return f"Function {func_name} not found"

        if func_name not in self.analysis_results:
            self.analyze_function(func_name)

        result = self.analysis_results[func_name]
        if 'error' in result:
            return f"Error analyzing function: {result['error']}"

        # Get original function
        func = self.equation_functions[func_name]
        try:
            signature = inspect.signature(func)
            parameters = signature.parameters
        except (TypeError, ValueError):
            return f"Could not retrieve signature for {func_name}"

        # Generate test case code
        test_code = [
            f"def test_{func_name}_edge_cases():",
            f"    \"\"\"Test edge cases for {func_name} function.\"\"\"",
            "    # Import function",
            f"    from config.equations import {func_name}",
            "    import numpy as np",
            "",
            "    # Test normal case first",
            "    try:"
        ]

        # Generate standard test case
        arg_list = []
        for name, param in parameters.items():
            if param.default is not param.empty:
                arg_list.append(f"{name}={param.default}")
            else:
                # Generate reasonable default based on parameter name
                if 'max' in name.lower():
                    arg_list.append(f"{name}=100.0")
                elif any(x in name.lower() for x in ['rate', 'factor', 'alpha', 'beta']):
                    arg_list.append(f"{name}=0.1")
                else:
                    arg_list.append(f"{name}=1.0")

        args_str = ", ".join(arg_list)
        test_code.extend([
            f"        result = {func_name}({args_str})",
            f"        assert result is not None, \"Function should return a value\"",
            "    except Exception as e:",
            "        assert False, f\"Function failed with standard inputs: {e}\"",
            ""
        ])

        # Add edge case tests based on patterns found
        for pattern_name, pattern_result in result.get('patterns_found', {}).items():
            if pattern_name == 'division_by_zero':
                # Generate division by zero test
                for param in parameters:
                    if param != 'self':  # Skip self for class methods
                        test_code.extend([
                            f"    # Test with {param} = 0 (potential division by zero)",
                            "    try:",
                            f"        result = {func_name}({', '.join([f'{p}=0.0' if p == param else arg for p, arg in zip(parameters, arg_list)])})",
                            f"        # Function should either handle zero or raise controlled exception",
                            "    except ZeroDivisionError:",
                            "        assert False, \"Function should handle division by zero\"",
                            "    except ValueError:",
                            "        # Controlled exception is acceptable",
                            "        pass",
                            ""
                        ])

            elif pattern_name == 'log_of_non_positive':
                # Generate log of negative/zero test
                for param in parameters:
                    if param != 'self':  # Skip self for class methods
                        test_code.extend([
                            f"    # Test with {param} = 0 (potential log of zero)",
                            "    try:",
                            f"        result = {func_name}({', '.join([f'{p}=0.0' if p == param else arg for p, arg in zip(parameters, arg_list)])})",
                            f"        # Function should either handle zero or raise controlled exception",
                            "    except ValueError:",
                            "        # Controlled exception is acceptable",
                            "        pass",
                            "",
                            f"    # Test with {param} = -1 (potential log of negative)",
                            "    try:",
                            f"        result = {func_name}({', '.join([f'{p}=-1.0' if p == param else arg for p, arg in zip(parameters, arg_list)])})",
                            f"        # Function should either handle negative or raise controlled exception",
                            "    except ValueError:",
                            "        # Controlled exception is acceptable",
                            "        pass",
                            ""
                        ])

            elif pattern_name == 'sqrt_of_negative':
                # Generate sqrt of negative test
                for param in parameters:
                    if param != 'self':  # Skip self for class methods
                        test_code.extend([
                            f"    # Test with {param} = -1 (potential sqrt of negative)",
                            "    try:",
                            f"        result = {func_name}({', '.join([f'{p}=-1.0' if p == param else arg for p, arg in zip(parameters, arg_list)])})",
                            f"        # Function should either handle negative or raise controlled exception",
                            "    except ValueError:",
                            "        # Controlled exception is acceptable",
                            "        pass",
                            ""
                        ])

            elif pattern_name == 'exponent_overflow':
                # Generate exponent overflow test
                for param in parameters:
                    if param != 'self':  # Skip self for class methods
                        test_code.extend([
                            f"    # Test with {param} = 100 (potential exponent overflow)",
                            "    try:",
                            f"        result = {func_name}({', '.join([f'{p}=100.0' if p == param else arg for p, arg in zip(parameters, arg_list)])})",
                            f"        # Function should handle large values",
                            "        assert np.isfinite(result), \"Function should return finite value\"",
                            "    except OverflowError:",
                            "        assert False, \"Function should handle exponent overflow\"",
                            ""
                        ])

        # Add test for NaN handling
        test_code.extend([
            "    # Test with NaN input",
            "    try:",
            f"        result = {func_name}({', '.join([f'{list(parameters.keys())[0]}=float(\"nan\")'] + arg_list[1:])})",
            "        # Function should either handle NaN or raise controlled exception",
            "        assert not (isinstance(result, float) and np.isnan(result)), \"Function should handle NaN inputs\"",
            "    except ValueError:",
            "        # Controlled exception is acceptable",
            "        pass",
            ""
        ])

        return "\n".join(test_code)

    def generate_edge_case_completion_report(self, output_dir):
        """
        Generate a comprehensive report on edge case completion.

        Args:
            output_dir (str): Directory to save the report
        """
        try:
            from pathlib import Path
        except ImportError:
            print("pathlib is required for report generation")
            return

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Ensure we have analyzed all functions
        if not self.analysis_results:
            self.analyze_all_functions()

        # Prepare data for report
        total_edge_cases = 0
        total_protected = 0
        function_scores = {}
        pattern_counts = {}

        for func_name, result in self.analysis_results.items():
            if 'error' in result:
                continue

            function_scores[func_name] = result['safety_score']

            for pattern_name, pattern_result in result.get('patterns_found', {}).items():
                total_matches = pattern_result['total_matches']
                unprotected = pattern_result['unprotected_matches']
                protected = total_matches - unprotected

                total_edge_cases += total_matches
                total_protected += protected

                if pattern_name not in pattern_counts:
                    pattern_counts[pattern_name] = {
                        'total': 0,
                        'protected': 0,
                        'description': pattern_result['description'],
                        'severity': pattern_result['severity']
                    }

                pattern_counts[pattern_name]['total'] += total_matches
                pattern_counts[pattern_name]['protected'] += protected

        # Calculate overall metrics
        protection_rate = (total_protected / total_edge_cases) * 100 if total_edge_cases > 0 else 100
        avg_safety_score = sum(function_scores.values()) / len(function_scores) if function_scores else 0

        # Sort functions by safety score
        sorted_functions = sorted(function_scores.items(), key=lambda x: x[1])

        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Edge Case Completion Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #2980b9;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 5px;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .metric {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .progress {{
                    height: 20px;
                    background-color: #ecf0f1;
                    border-radius: 10px;
                    margin: 10px 0;
                    overflow: hidden;
                }}
                .progress-bar {{
                    height: 100%;
                    background-color: #3498db;
                    border-radius: 10px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .high {{
                    color: #e74c3c;
                }}
                .medium {{
                    color: #f39c12;
                }}
                .low {{
                    color: #27ae60;
                }}
                .score-cell {{
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Edge Case Completion Report</h1>

                <div class="summary">
                    <h2>Executive Summary</h2>
                    <p>This report analyzes the edge case handling in the equation functions.</p>
                    <p>Overall protection rate: <span class="metric">{protection_rate:.1f}%</span></p>
                    <div class="progress">
                        <div class="progress-bar" style="width: {protection_rate}%;"></div>
                    </div>
                    <p>Average safety score: <span class="metric">{avg_safety_score:.1f}/100</span></p>
                    <div class="progress">
                        <div class="progress-bar" style="width: {avg_safety_score}%;"></div>
                    </div>
                    <p>Total edge cases identified: <span class="metric">{total_edge_cases}</span></p>
                    <p>Edge cases with protection: <span class="metric">{total_protected}</span></p>
                </div>

                <h2>Edge Cases by Category</h2>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Description</th>
                        <th>Severity</th>
                        <th>Total Cases</th>
                        <th>Protected</th>
                        <th>Protection Rate</th>
                    </tr>
        """

        # Add pattern rows
        for pattern_name, data in pattern_counts.items():
            protection_rate = (data['protected'] / data['total']) * 100 if data['total'] > 0 else 100
            html_content += f"""
                    <tr>
                        <td>{pattern_name}</td>
                        <td>{data['description']}</td>
                        <td class="{data['severity']}">{data['severity'].upper()}</td>
                        <td>{data['total']}</td>
                        <td>{data['protected']}</td>
                        <td>{protection_rate:.1f}%</td>
                    </tr>
            """

        html_content += """
                </table>

                <h2>Function Safety Scores</h2>
                <table>
                    <tr>
                        <th>Function</th>
                        <th>Safety Score</th>
                        <th>Risk Level</th>
                        <th>Edge Cases</th>
                        <th>Protected</th>
                    </tr>
        """

        # Add function rows
        for func_name, score in sorted_functions:
            result = self.analysis_results[func_name]

            total_cases = 0
            protected_cases = 0
            for pattern_result in result.get('patterns_found', {}).values():
                total_cases += pattern_result['total_matches']
                protected_cases += pattern_result['total_matches'] - pattern_result['unprotected_matches']

            risk_level = result['risk_level']

            html_content += f"""
                    <tr>
                        <td>{func_name}</td>
                        <td class="score-cell {risk_level}">{score:.1f}</td>
                        <td class="{risk_level}">{risk_level.upper()}</td>
                        <td>{total_cases}</td>
                        <td>{protected_cases}</td>
                    </tr>
            """

        html_content += """
                </table>

                <h2>Improvement Priorities</h2>
                <p>Based on risk level and edge case protection, these functions should be prioritized for improvement:</p>
                <ol>
        """

        # Add improvement priorities (focus on high risk functions)
        priority_functions = [func for func, _ in sorted_functions if
                              self.analysis_results[func]['risk_level'] == 'high']
        for func_name in priority_functions[:5]:  # Top 5 priorities
            html_content += f"""
                    <li><strong>{func_name}</strong> (Score: {function_scores[func_name]:.1f})</li>
            """

        # If there are no high risk functions, include medium risk ones
        if len(priority_functions) == 0:
            medium_risk = [func for func, _ in sorted_functions if
                           self.analysis_results[func]['risk_level'] == 'medium']
            for func_name in medium_risk[:3]:  # Top 3 medium risk
                html_content += f"""
                        <li><strong>{func_name}</strong> (Score: {function_scores[func_name]:.1f})</li>
                """

        html_content += """
                </ol>
            </div>
        </body>
        </html>
        """

        # Write HTML to file
        with open(Path(output_dir) / "edge_case_completion_report.html", "w") as f:
            f.write(html_content)

        print(f"Edge case completion report generated in {output_dir}")

    def _has_safety_check(self, pattern_name, line_content):
        """Check if a line already has a safety check for the given pattern."""
        if pattern_name == 'division_by_zero':
            return any(check in line_content for check in [
                'max(', 'np.maximum(', 'max(1e-', 'safe_div', 'if', 'else'
            ])
        elif pattern_name == 'log_of_non_positive':
            return any(check in line_content for check in [
                'max(', 'np.maximum(', 'abs(', 'np.abs(', 'if', 'else'
            ])
        elif pattern_name == 'sqrt_of_negative':
            return any(check in line_content for check in [
                'max(', 'np.maximum(', 'abs(', 'np.abs(', 'if', 'else'
            ])
        elif pattern_name == 'exponent_overflow':
            return any(check in line_content for check in [
                'min(', 'np.minimum(', 'clip(', 'np.clip(', 'if', 'else'
            ])
        elif pattern_name == 'array_bounds':
            return any(check in line_content for check in [
                'try', 'except', 'if', 'else', 'len(', 'shape'
            ])
        return False

    def _get_recommendation_text(self, pattern_name, line_content):
        """Generate recommendation text for a specific pattern and line."""
        if pattern_name == 'division_by_zero':
            return "Use 'max(divisor, epsilon)' or similar to prevent division by zero"
        elif pattern_name == 'log_of_non_positive':
            return "Use 'max(x, epsilon)' to ensure positive input to logarithm"
        elif pattern_name == 'sqrt_of_negative':
            return "Use 'max(0, x)' or 'abs(x)' to ensure non-negative input to square root"
        elif pattern_name == 'exponent_overflow':
            return "Use 'min(x, max_exponent)' to prevent exponential overflow"
        elif pattern_name == 'array_bounds':
            return "Add bounds checking before indexing arrays"
        elif pattern_name == 'conditional_logic':
            return "Consider adding tests for this conditional logic branch"
        return "Consider adding appropriate safety checks"

    def _generate_function_recommendations(self, func_name, analysis_result):
        """Generate specific recommendations for improving a function."""
        recommendations = []

        # Check for division by zero
        if 'division_by_zero' in analysis_result['patterns_found']:
            div_zero = analysis_result['patterns_found']['division_by_zero']
            if div_zero['unprotected_matches'] > 0:
                recommendations.append({
                    'issue': 'Division by zero risk',
                    'recommendation': 'Add a small epsilon or use max() to prevent division by zero',
                    'severity': 'high',
                    'example': 'denominator = max(1e-10, original_denominator)'
                })

        # Check for logarithm of non-positive values
        if 'log_of_non_positive' in analysis_result['patterns_found']:
            log_issue = analysis_result['patterns_found']['log_of_non_positive']
            if log_issue['unprotected_matches'] > 0:
                recommendations.append({
                    'issue': 'Logarithm of zero or negative value risk',
                    'recommendation': 'Ensure inputs to logarithm functions are positive',
                    'severity': 'high',
                    'example': 'np.log(max(1e-10, value))'
                })

        # Check for square root of negative values
        if 'sqrt_of_negative' in analysis_result['patterns_found']:
            sqrt_issue = analysis_result['patterns_found']['sqrt_of_negative']
            if sqrt_issue['unprotected_matches'] > 0:
                recommendations.append({
                    'issue': 'Square root of negative value risk',
                    'recommendation': 'Ensure inputs to square root functions are non-negative',
                    'severity': 'high',
                    'example': 'np.sqrt(max(0, value))'
                })

        # Check for exponential overflow
        if 'exponent_overflow' in analysis_result['patterns_found']:
            exp_issue = analysis_result['patterns_found']['exponent_overflow']
            if exp_issue['unprotected_matches'] > 0:
                recommendations.append({
                    'issue': 'Exponential overflow risk',
                    'recommendation': 'Limit inputs to exponential functions',
                    'severity': 'medium',
                    'example': 'np.exp(min(50, value))'
                })

        # Check for array bounds issues
        if 'array_bounds' in analysis_result['patterns_found']:
            array_issue = analysis_result['patterns_found']['array_bounds']
            if array_issue['unprotected_matches'] > 0:
                recommendations.append({
                    'issue': 'Array index out of bounds risk',
                    'recommendation': 'Add bounds checking before indexing arrays',
                    'severity': 'high',
                    'example': 'if idx < len(array): value = array[idx]'
                })

        # General recommendation for low safety score
        if analysis_result['safety_score'] < 60:
            recommendations.append({
                'issue': 'Overall numerical stability',
                'recommendation': 'Consider integrating with CircuitBreaker for better stability',
                'severity': 'medium',
                'example': 'from utils.circuit_breaker import CircuitBreaker'
            })

        return recommendations

    def _fix_with_regex(self, func_name, source):
        """Fix edge cases using regex-based approach."""
        if func_name not in self.analysis_results:
            return source

        result = self.analysis_results[func_name]
        if 'error' in result:
            return source

        fixed_source = source

        # Extract indentation from function definition line
        match = re.search(r'^(\s*)def\s+', source, re.MULTILINE)
        base_indent = match.group(1) if match else ''

        # Extract function body's indentation
        func_lines = source.split('\n')
        body_start_idx = next((i for i, line in enumerate(func_lines) if re.match(r'^\s*def\s+', line)), 0)
        if body_start_idx < len(func_lines) - 1:
            body_indent_match = re.match(r'^(\s+)', func_lines[body_start_idx + 1])
            body_indent = body_indent_match.group(1) if body_indent_match else base_indent + '    '
        else:
            body_indent = base_indent + '    '

        # Apply fixes for each pattern
        for pattern_name, pattern_info in result.get('patterns_found', {}).items():
            if pattern_info['unprotected_matches'] == 0:
                continue

            # Get all lines to fix
            lines_to_fix = sorted(
                [match for match in pattern_info['matches'] if not match['has_safety_check']],
                key=lambda x: x['line']
            )

            for line_info in lines_to_fix:
                line_num = line_info['line'] - 1  # Convert to 0-based index
                if line_num >= len(func_lines):
                    continue

                line = func_lines[line_num]
                fixed_line = None

                if pattern_name == 'division_by_zero':
                    # Fix division by zero
                    # Look for the denominator
                    div_match = re.search(r'\/\s*([a-zA-Z_][a-zA-Z0-9_]*|\d*\.?\d+)', line)
                    if div_match:
                        denominator = div_match.group(1)
                        if denominator.replace('.', '', 1).isdigit():
                            # If it's a literal number, probably doesn't need fixing
                            continue

                        # Check if this is an array operation
                        if '[' in line and ']' in line:
                            # Array operation - more complex to fix
                            continue

                        # Replace with safe max expression
                        fixed_line = line.replace(
                            f"/ {denominator}",
                            f"/ max(1e-10, {denominator})"
                        ).replace(
                            f"/{denominator}",
                            f"/max(1e-10, {denominator})"
                        )

                elif pattern_name == 'log_of_non_positive':
                    # Fix logarithm of non-positive
                    log_match = re.search(
                        r'(np\.log|np\.log10|np\.log2|np\.log1p|math\.log|math\.log10|math\.log2)\s*\(([^)]+)\)',
                        line
                    )
                    if log_match:
                        log_func = log_match.group(1)
                        log_arg = log_match.group(2)

                        # Check if this is an array operation
                        if '[' in log_arg and ']' in log_arg:
                            # Array operation - more complex to fix
                            continue

                        # Replace with safe max expression
                        fixed_line = line.replace(
                            f"{log_func}({log_arg})",
                            f"{log_func}(max(1e-10, {log_arg}))"
                        )

                elif pattern_name == 'sqrt_of_negative':
                    # Fix square root of negative
                    sqrt_match = re.search(r'(np\.sqrt|math\.sqrt)\s*\(([^)]+)\)', line)
                    if sqrt_match:
                        sqrt_func = sqrt_match.group(1)
                        sqrt_arg = sqrt_match.group(2)

                        # Check if this is an array operation
                        if '[' in sqrt_arg and ']' in sqrt_arg:
                            # Array operation - more complex to fix
                            continue

                        # Replace with safe max expression
                        fixed_line = line.replace(
                            f"{sqrt_func}({sqrt_arg})",
                            f"{sqrt_func}(max(0, {sqrt_arg}))"
                        )

                elif pattern_name == 'exponent_overflow':
                    # Fix exponential overflow
                    exp_match = re.search(r'(np\.exp|math\.exp)\s*\(([^)]+)\)', line)
                    if exp_match:
                        exp_func = exp_match.group(1)
                        exp_arg = exp_match.group(2)

                        # Check if this is an array operation
                        if '[' in exp_arg and ']' in exp_arg:
                            # Array operation - more complex to fix
                            continue

                        # Replace with safe min expression
                        fixed_line = line.replace(
                            f"{exp_func}({exp_arg})",
                            f"{exp_func}(min(50, {exp_arg}))"
                        )

                elif pattern_name == 'array_bounds':
                    # Fix array bounds checking
                    array_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\[([^]]+)\]', line)
                    if array_match:
                        array_name = array_match.group(1)
                        index = array_match.group(2)

                        # Create a try-except block
                        leading_spaces = re.match(r'^\s*', line).group(0)
                        indent_str = leading_spaces

                        # Extract the part of the line that uses the array
                        line_prefix = line[:array_match.start()]
                        line_suffix = line[array_match.end():]

                        # Create a variable name for the result
                        result_var = f"{array_name}_value"

                        # Replace with try-except block
                        fixed_line = f"{indent_str}try:\n"
                        fixed_line += f"{indent_str}    {line_prefix}{array_name}[{index}]{line_suffix}\n"
                        fixed_line += f"{indent_str}except IndexError:\n"
                        fixed_line += f"{indent_str}    # Handle index out of bounds\n{indent_str}    pass"

                if fixed_line:
                    func_lines[line_num] = fixed_line

        # Reassemble the function
        fixed_source = '\n'.join(func_lines)
        return fixed_source

    def _generate_circuit_breaker_version(self, source, func_name, args):
        """Generate a version of the function with circuit breaker integration."""
        lines = source.split('\n')

        # Find function definition line
        func_def_idx = next((i for i, line in enumerate(lines) if re.match(r'^\s*def\s+' + func_name, line)), -1)
        if func_def_idx == -1:
            return source

        # Extract indentation
        indent_match = re.match(r'^(\s*)', lines[func_def_idx])
        base_indent = indent_match.group(1) if indent_match else ''
        body_indent = base_indent + '    '

        # Find docstring if present
        has_docstring = False
        docstring_end = func_def_idx + 1
        if docstring_end < len(lines) and '"""' in lines[docstring_end]:
            has_docstring = True
            docstring_end = next((i for i, line in enumerate(lines[func_def_idx + 1:], func_def_idx + 1)
                                  if '"""' in line and i > func_def_idx), func_def_idx + 1)

        # Insert circuit breaker imports at the top of the file
        import_lines = [
            "import numpy as np",
            "from utils.circuit_breaker import CircuitBreaker"
        ]

        # Add imports if not already present
        for imp in import_lines:
            if not any(imp in line for line in lines[:func_def_idx]):
                lines.insert(0, imp)
                func_def_idx += 1
                docstring_end += 1

        # Create circuit breaker initialization
        cb_init = [
            f"{body_indent}# Initialize circuit breaker for numerical stability",
            f"{body_indent}circuit_breaker = CircuitBreaker(",
            f"{body_indent}    threshold=1e-10,",
            f"{body_indent}    max_value=1e10,",
            f"{body_indent}    min_value=1e-10,",
            f"{body_indent}    max_rate_of_change=1.0",
            f"{body_indent})"
        ]

        # Insert circuit breaker initialization after docstring
        for i, line in enumerate(cb_init):
            lines.insert(docstring_end + 1 + i, line)

        # Shift indices to account for insertions
        func_body_start = docstring_end + 1 + len(cb_init)

        # Wrap the function's return value with circuit breaker
        # Find return statements
        for i, line in enumerate(lines[func_body_start:], func_body_start):
            if re.match(r'^\s*return\s+', line):
                # Extract the returned expression
                match = re.match(r'^\s*return\s+(.+)', line)
                if match:
                    return_expr = match.group(1)
                    # Replace with circuit breaker check
                    lines[i] = line.replace(
                        f"return {return_expr}",
                        f"result = {return_expr}\n{body_indent}return circuit_breaker.check_and_fix(result)"
                    )

        # Add safety checks for division, exponents, etc.
        for i, line in enumerate(lines[func_body_start:], func_body_start):
            # Check for division operations
            if '/' in line and not 'safe_div' in line:
                div_match = re.search(r'([^=]+)\s*=\s*([^/]+)\/([^;]+)', line)
                if div_match:
                    target = div_match.group(1).strip()
                    numerator = div_match.group(2).strip()
                    denominator = div_match.group(3).strip()

                    # Replace with safe division
                    indent = re.match(r'^\s*', line).group(0)
                    lines[i] = f"{indent}{target} = circuit_breaker.safe_div({numerator}, {denominator})"

            # Check for exponential operations
            if 'exp(' in line or 'np.exp(' in line:
                exp_match = re.search(r'([^=]+)\s*=\s*([^(]*)(exp\s*\([^)]+\))', line)
                if exp_match:
                    target = exp_match.group(1).strip()
                    prefix = exp_match.group(2).strip()
                    exp_expr = exp_match.group(3)

                    # Replace with safe exponent
                    indent = re.match(r'^\s*', line).group(0)
                    lines[i] = f"{indent}{target} = {prefix}circuit_breaker.safe_exp({exp_expr[4:-1]})"

        # Join lines back together
        return '\n'.join(lines)


class EdgeCaseTransformer(ast.NodeTransformer):
    """AST transformer for fixing edge cases."""

    def __init__(self, analysis_result):
        self.analysis_result = analysis_result
        self.fixes_applied = 0

    def visit_BinOp(self, node):
        # First recursively process child nodes
        self.generic_visit(node)

        # Check for division operations
        if isinstance(node.op, ast.Div):
            # Fix division by zero
            if isinstance(node.right, ast.Name) or isinstance(node.right, ast.Attribute):
                # Replace divisor with max(1e-10, divisor)
                return ast.BinOp(
                    left=node.left,
                    op=ast.Div(),
                    right=ast.Call(
                        func=ast.Name(id='max', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=1e-10),
                            node.right
                        ],
                        keywords=[]
                    )
                )
                self.fixes_applied += 1

        return node

    def visit_Call(self, node):
        # First recursively process child nodes
        self.generic_visit(node)

        # Check for log operations
        if isinstance(node.func, ast.Attribute) and hasattr(node.func, 'attr'):
            if node.func.attr in ['log', 'log10', 'log2', 'log1p'] and node.args:
                # Replace log(arg) with log(max(1e-10, arg))
                node.args[0] = ast.Call(
                    func=ast.Name(id='max', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=1e-10),
                        node.args[0]
                    ],
                    keywords=[]
                )
                self.fixes_applied += 1

            # Check for sqrt operations
            elif node.func.attr == 'sqrt' and node.args:
                # Replace sqrt(arg) with sqrt(max(0, arg))
                node.args[0] = ast.Call(
                    func=ast.Name(id='max', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=0),
                        node.args[0]
                    ],
                    keywords=[]
                )
                self.fixes_applied += 1

            # Check for exp operations
            elif node.func.attr == 'exp' and node.args:
                # Replace exp(arg) with exp(min(50, arg))
                node.args[0] = ast.Call(
                    func=ast.Name(id='min', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=50),
                        node.args[0]
                    ],
                    keywords=[]
                )
                self.fixes_applied += 1

        return node

    def visit_Subscript(self, node):
        # First recursively process child nodes
        self.generic_visit(node)

        # Can't easily fix array indexing with AST transformation
        # as it would require restructuring control flow
        return node

    def run_edge_case_check(equations, output_dir=None):
        """
        Run edge case validation on the provided equations.

        Args:
            equations (dict): Dictionary of equation functions
            output_dir (str, optional): Directory to save output reports

        Returns:
            dict: Results of edge case validation
        """
        checker = EdgeCaseChecker(equations)
        checker.analyze_all_functions()

        results = {
            'recommendations': checker.generate_recommendations(),
            'coverage': {
                'total_edge_cases': sum(
                    result.get('edge_case_count', 0) for result in checker.analysis_results.values()),
                'protected_cases': sum(
                    result.get('edge_case_count', 0) - sum(
                        pattern.get('unprotected_matches', 0)
                        for pattern in result.get('patterns_found', {}).values()
                    )
                    for result in checker.analysis_results.values()
                )
            },
            'status': 'success'
        }

        if output_dir:
            try:
                checker.generate_edge_case_completion_report(output_dir)
            except Exception as e:
                print(f"Warning: Could not generate edge case report: {e}")
                results['status'] = 'warning'

        return results