"""
Validation Report Generator
This module generates comprehensive HTML reports from validation results,
combining insights from all validation components.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import logging
import base64
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ReportGenerator")


class ReportGenerator:
    """
    Generates comprehensive HTML reports from validation results.
    """

    def __init__(self):
        """Initialize the report generator."""
        self.report_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                h1 {{
                    text-align: center;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                }}
                h2 {{
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
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
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .metric-good {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .metric-warning {{
                    color: #f39c12;
                    font-weight: bold;
                }}
                .metric-bad {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .summary-box {{
                    padding: 15px;
                    margin: 20px 0;
                    border-radius: 5px;
                    background-color: #f8f9fa;
                    border-left: 5px solid #3498db;
                }}
                .good-box {{
                    border-left-color: #27ae60;
                    background-color: #ebfaf0;
                }}
                .warning-box {{
                    border-left-color: #f39c12;
                    background-color: #fef5e9;
                }}
                .bad-box {{
                    border-left-color: #e74c3c;
                    background-color: #fdedeb;
                }}
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart {{
                    max-width: 100%;
                    height: auto;
                    margin: 10px auto;
                    display: block;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .gap-table {{
                    margin: 20px 0;
                }}
                .gap-high {{
                    background-color: #ffdddd;
                }}
                .gap-medium {{
                    background-color: #ffffcc;
                }}
                .gap-low {{
                    background-color: #ddffdd;
                }}
                .footer {{
                    margin-top: 40px;
                    padding-top: 10px;
                    border-top: 1px solid #ddd;
                    font-size: 0.8em;
                    text-align: center;
                }}
                .toc {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .toc ul {{
                    list-style-type: none;
                    padding-left: 20px;
                }}
                .toc li {{
                    margin-bottom: 5px;
                }}
                .toc a {{
                    text-decoration: none;
                    color: #3498db;
                }}
                .toc a:hover {{
                    text-decoration: underline;
                }}
                .opportunities-list {{
                    list-style-type: none;
                    padding-left: 0;
                }}
                .opportunities-list li {{
                    margin-bottom: 15px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-left: 3px solid #3498db;
                    border-radius: 3px;
                }}
                .opportunity-high {{
                    border-left-color: #e74c3c !important;
                }}
                .opportunity-medium {{
                    border-left-color: #f39c12 !important;
                }}
                .opportunity-low {{
                    border-left-color: #27ae60 !important;
                }}
                .recommendations {{
                    font-style: italic;
                    margin-top: 5px;
                    color: #333;
                }}
                .integration-matrix {{
                    margin: 20px 0;
                }}
                .badge {{
                    display: inline-block;
                    padding: 3px 7px;
                    font-size: 12px;
                    font-weight: bold;
                    line-height: 1;
                    color: #fff;
                    text-align: center;
                    white-space: nowrap;
                    vertical-align: baseline;
                    border-radius: 10px;
                }}
                .badge-success {{
                    background-color: #27ae60;
                }}
                .badge-warning {{
                    background-color: #f39c12;
                }}
                .badge-danger {{
                    background-color: #e74c3c;
                }}
                .badge-info {{
                    background-color: #3498db;
                }}
                .recommendations-section {{
                    margin-top: 40px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    border-left: 5px solid #3498db;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on: {generation_time}</p>

            <div class="toc">
                <h2>Table of Contents</h2>
                <ul>
                    <li><a href="#summary">Executive Summary</a></li>
                    <li><a href="#equation-coverage">Equation Coverage Analysis</a></li>
                    <li><a href="#cross-scale">Cross-Scale Interaction Analysis</a></li>
                    <li><a href="#comparative">Comparative Simulation Analysis</a></li>
                    <li><a href="#gaps">Identified Gaps</a></li>
                    <li><a href="#optimization">Optimization Opportunities</a></li>
                    <li><a href="#recommendations">Recommendations</a></li>
                </ul>
            </div>

            <div id="summary" class="summary-box {summary_box_class}">
                <h2>Executive Summary</h2>
                {summary_content}
            </div>

            <div id="equation-coverage">
                <h2>Equation Coverage Analysis</h2>
                {coverage_content}

                <div class="chart-container">
                    <h3>Physics Domain Coverage</h3>
                    <img src="data:image/png;base64,{physics_domain_chart}" class="chart" alt="Physics Domain Coverage Chart">
                </div>

                <div class="chart-container">
                    <h3>Scale Level Coverage</h3>
                    <img src="data:image/png;base64,{scale_level_chart}" class="chart" alt="Scale Level Coverage Chart">
                </div>

                <div class="chart-container">
                    <h3>Cross-Domain Coverage Matrix</h3>
                    <img src="data:image/png;base64,{cross_domain_chart}" class="chart" alt="Cross-Domain Coverage Matrix">
                </div>
            </div>

            <div id="cross-scale">
                <h2>Cross-Scale Interaction Analysis</h2>
                {cross_scale_content}

                <div class="chart-container">
                    <h3>Dependency Graph</h3>
                    <img src="data:image/png;base64,{dependency_graph}" class="chart" alt="Equation Dependency Graph">
                </div>

                <div class="chart-container">
                    <h3>Scale Adjacency Matrix</h3>
                    <img src="data:image/png;base64,{scale_adjacency}" class="chart" alt="Scale Adjacency Matrix">
                </div>

                <div class="chart-container">
                    <h3>Signal Propagation</h3>
                    <img src="data:image/png;base64,{signal_propagation}" class="chart" alt="Signal Propagation Across Scales">
                </div>
            </div>

            <div id="comparative">
                <h2>Comparative Simulation Analysis</h2>
                {comparative_content}

                <div class="chart-container">
                    <h3>Metric Comparison</h3>
                    <img src="data:image/png;base64,{metric_comparison}" class="chart" alt="Metric Comparison Chart">
                </div>

                <div class="chart-container">
                    <h3>Configuration Scores</h3>
                    <img src="data:image/png;base64,{config_scores}" class="chart" alt="Configuration Scores Chart">
                </div>

                <div class="chart-container">
                    <h3>Knowledge Comparison</h3>
                    <img src="data:image/png;base64,{knowledge_comparison}" class="chart" alt="Knowledge Comparison Chart">
                </div>

                <div class="chart-container">
                    <h3>Suppression Comparison</h3>
                    <img src="data:image/png;base64,{suppression_comparison}" class="chart" alt="Suppression Comparison Chart">
                </div>
            </div>

            <div id="gaps">
                <h2>Identified Gaps</h2>
                {gaps_content}

                <div class="chart-container">
                    <h3>Gap Analysis</h3>
                    <img src="data:image/png;base64,{gap_chart}" class="chart" alt="Gap Analysis Chart">
                </div>
            </div>

            <div id="optimization">
                <h2>Optimization Opportunities</h2>
                {optimization_content}
            </div>

            <div id="recommendations" class="recommendations-section">
                <h2>Recommendations</h2>
                {recommendations_content}
            </div>

            <div class="footer">
                <p>Generated by Axiomatic Intelligence Growth Simulation Framework</p>
                <p>Unified Validation System v1.0</p>
            </div>
        </body>
        </html>
        """

    def generate_unified_report(self, validation_results, output_path, include_visualizations=True):
        """
        Generate a comprehensive HTML report from validation results.

        Args:
            validation_results: Dictionary containing all validation results
            output_path: Path to save the HTML report
            include_visualizations: Whether to include visualizations in the report

        Returns:
            Path to the generated report
        """
        logger.info("Generating unified validation report")

        # Create placeholder content
        content = {
            "title": "Unified Validation Report",
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary_box_class": "good-box",
            "summary_content": self._generate_summary_content(validation_results),
            "coverage_content": self._generate_coverage_content(validation_results),
            "cross_scale_content": self._generate_cross_scale_content(validation_results),
            "comparative_content": self._generate_comparative_content(validation_results),
            "gaps_content": self._generate_gaps_content(validation_results),
            "optimization_content": self._generate_optimization_content(validation_results),
            "recommendations_content": self._generate_recommendations_content(validation_results)
        }

        # Generate charts if requested
        if include_visualizations:
            content.update(self._generate_chart_content(validation_results))
        else:
            # Use placeholder images
            for key in ["physics_domain_chart", "scale_level_chart", "cross_domain_chart",
                        "dependency_graph", "scale_adjacency", "signal_propagation",
                        "metric_comparison", "config_scores", "knowledge_comparison",
                        "suppression_comparison", "gap_chart"]:
                content[key] = ""

        # Determine summary box class based on overall results
        summary_score = self._calculate_summary_score(validation_results)
        if summary_score >= 0.7:
            content["summary_box_class"] = "good-box"
        elif summary_score >= 0.4:
            content["summary_box_class"] = "warning-box"
        else:
            content["summary_box_class"] = "bad-box"

        # Generate the HTML report
        html_content = self.report_template.format(**content)

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Unified report saved to {output_path}")

        return output_path

    def _calculate_summary_score(self, validation_results):
        """
        Calculate an overall score for the validation results.

        Args:
            validation_results: Dictionary containing all validation results

        Returns:
            Float between 0 and 1 indicating overall quality
        """
        score_components = []

        # Coverage metrics (25%)
        if "equation_coverage" in validation_results:
            coverage = validation_results["equation_coverage"].get("coverage", {})
            # Calculate weighted average of coverage percentages
            coverage_pct = 0.0
            if "physics_coverage_pct" in coverage:
                coverage_pct += 0.4 * coverage["physics_coverage_pct"] / 100.0
            if "application_coverage_pct" in coverage:
                coverage_pct += 0.3 * coverage["application_coverage_pct"] / 100.0
            if "scale_coverage_pct" in coverage:
                coverage_pct += 0.3 * coverage["scale_coverage_pct"] / 100.0

            score_components.append((coverage_pct, 0.25))

        # Cross-scale integration (25%)
        if "cross_scale" in validation_results:
            cross_scale = validation_results["cross_scale"]
            integration_score = 0.0

            # Overall integration score
            if "overall_integration_score" in cross_scale:
                integration_score = cross_scale["overall_integration_score"]

            # Signal propagation
            elif "signal_propagation" in cross_scale:
                integration_score = cross_scale["signal_propagation"].get("average_efficiency", 0.0)

            score_components.append((integration_score, 0.25))

        # Comparative simulation (30%)
        if "comparative" in validation_results:
            comparative = validation_results["comparative"]

            comparative_score = 0.0
            if "comparison" in comparative:
                comparison = comparative["comparison"]

                # Use best overall score if available
                if "best_overall" in comparison:
                    best_score = comparison["best_overall"].get("score", 0)
                    total_metrics = sum(1 for key in comparison if key.endswith("_by_config"))
                    if total_metrics > 0:
                        comparative_score = best_score / total_metrics

                # Otherwise, use average of best metrics
                else:
                    best_count = sum(1 for key in comparison if key.startswith("best_"))
                    total_metrics = sum(1 for key in comparison if key.endswith("_by_config"))
                    if total_metrics > 0:
                        comparative_score = best_count / total_metrics

            score_components.append((comparative_score, 0.30))

        # Gap analysis (20% - inverse score, fewer gaps is better)
        if "equation_coverage" in validation_results:
            gaps = validation_results["equation_coverage"].get("gaps", [])
            gaps_score = 0.0

            # Calculate weighted gap score (high severity gaps count more)
            if gaps:
                weighted_sum = 0
                total_weight = 0

                for gap in gaps:
                    severity = gap.get("severity", "medium")
                    weight = 3 if severity == "high" else (2 if severity == "medium" else 1)
                    total_weight += weight

                if total_weight > 0:
                    # Invert score - fewer gaps means higher score
                    # Scale based on expected number of gaps
                    expected_max_gaps = 10
                    gap_ratio = min(1.0, len(gaps) / expected_max_gaps)
                    gaps_score = 1.0 - gap_ratio
            else:
                # No gaps is perfect score
                gaps_score = 1.0

            score_components.append((gaps_score, 0.20))

        # Calculate weighted average score
        if score_components:
            total_score = sum(score * weight for score, weight in score_components)
            total_weight = sum(weight for _, weight in score_components)

            if total_weight > 0:
                return total_score / total_weight

        # Default if no components available
        return 0.5

    def _generate_summary_content(self, validation_results):
        """
        Generate executive summary content for the report.

        Args:
            validation_results: Dictionary containing all validation results

        Returns:
            HTML string with summary content
        """
        summary_parts = []

        # Overall score
        summary_score = self._calculate_summary_score(validation_results)
        score_class = "metric-good" if summary_score >= 0.7 else (
            "metric-warning" if summary_score >= 0.4 else "metric-bad")
        summary_parts.append(f"<p>Overall Validation Score: <span class='{score_class}'>{summary_score:.2f}</span></p>")

        # Coverage summary
        if "equation_coverage" in validation_results:
            coverage = validation_results["equation_coverage"].get("coverage", {})

            if "physics_coverage_pct" in coverage:
                physics_pct = coverage["physics_coverage_pct"]
                physics_class = "metric-good" if physics_pct >= 70 else (
                    "metric-warning" if physics_pct >= 40 else "metric-bad")
                summary_parts.append(
                    f"<p>Physics Domain Coverage: <span class='{physics_class}'>{physics_pct:.1f}%</span></p>")

            if "scale_coverage_pct" in coverage:
                scale_pct = coverage["scale_coverage_pct"]
                scale_class = "metric-good" if scale_pct >= 70 else (
                    "metric-warning" if scale_pct >= 40 else "metric-bad")
                summary_parts.append(
                    f"<p>Scale Level Coverage: <span class='{scale_class}'>{scale_pct:.1f}%</span></p>")

        # Integration summary
        if "cross_scale" in validation_results:
            cross_scale = validation_results["cross_scale"]

            if "overall_integration_score" in cross_scale:
                integration = cross_scale["overall_integration_score"]
                integration_class = "metric-good" if integration >= 0.7 else (
                    "metric-warning" if integration >= 0.4 else "metric-bad")
                summary_parts.append(
                    f"<p>Cross-Scale Integration: <span class='{integration_class}'>{integration:.2f}</span></p>")

        # Gaps summary
        if "equation_coverage" in validation_results:
            gaps = validation_results["equation_coverage"].get("gaps", [])

            high_severity = sum(1 for gap in gaps if gap.get("severity", "") == "high")
            medium_severity = sum(1 for gap in gaps if gap.get("severity", "") == "medium")

            gaps_class = "metric-good" if high_severity == 0 and medium_severity <= 2 else (
                "metric-warning" if high_severity <= 2 and medium_severity <= 5 else "metric-bad")

            summary_parts.append(f"<p>Identified Gaps: <span class='{gaps_class}'>{len(gaps)} total, "
                                 f"{high_severity} high priority, {medium_severity} medium priority</span></p>")

        # Comparative summary
        if "comparative" in validation_results:
            comparative = validation_results["comparative"]

            if "comparison" in comparative and "best_overall" in comparative["comparison"]:
                best = comparative["comparison"]["best_overall"]
                summary_parts.append(f"<p>Best Configuration: <span class='metric-good'>{best['configuration']}</span> "
                                     f"with score {best['score']}</p>")

        # Add key insights
        summary_parts.append("<h3>Key Insights</h3><ul>")

        # Add coverage insights
        if "equation_coverage" in validation_results:
            coverage = validation_results["equation_coverage"].get("coverage", {})

            # Find best and worst covered domains
            if "physics_domains" in coverage:
                physics = coverage["physics_domains"]
                if physics:
                    best_domain = max(physics.items(), key=lambda x: x[1])
                    worst_domain = min(physics.items(), key=lambda x: x[1])

                    summary_parts.append(f"<li>Best covered physics domain: <strong>{best_domain[0]}</strong> "
                                         f"with {best_domain[1]} equations</li>")

                    if worst_domain[1] == 0:
                        summary_parts.append(f"<li>No equations found for physics domain: <strong class='metric-bad'>"
                                             f"{worst_domain[0]}</strong></li>")
                    elif worst_domain[1] < 2:
                        summary_parts.append(f"<li>Limited coverage of physics domain: <strong class='metric-warning'>"
                                             f"{worst_domain[0]}</strong> with only {worst_domain[1]} equation</li>")

            # Add gap insights
            gaps = validation_results["equation_coverage"].get("gaps", [])
            if gaps:
                gap_types = {}
                for gap in gaps:
                    gap_type = gap.get("type", "unknown")
                    gap_types[gap_type] = gap_types.get(gap_type, 0) + 1

                most_common_gap = max(gap_types.items(), key=lambda x: x[1])
                summary_parts.append(f"<li>Most common gap type: <strong>{most_common_gap[0]}</strong> "
                                     f"with {most_common_gap[1]} instances</li>")

        # Add cross-scale insights
        if "cross_scale" in validation_results:
            cross_scale = validation_results["cross_scale"]

            if "scale_transitions" in cross_scale and "transitions" in cross_scale["scale_transitions"]:
                transitions = cross_scale["scale_transitions"]["transitions"]

                # Find best and worst transitions
                if transitions:
                    best_transition = max(transitions.items(), key=lambda x: x[1]["quality"])
                    worst_transition = min(transitions.items(), key=lambda x: x[1]["quality"])

                    summary_parts.append(f"<li>Strongest scale transition: <strong>{best_transition[0][0]}</strong> to "
                                         f"<strong>{best_transition[0][1]}</strong> with quality score "
                                         f"{best_transition[1]['quality']:.2f}</li>")

                    if worst_transition[1]["quality"] < 0.4:
                        summary_parts.append(f"<li>Weakest scale transition: <strong class='metric-warning'>"
                                             f"{worst_transition[0][0]}</strong> to <strong class='metric-warning'>"
                                             f"{worst_transition[0][1]}</strong> with quality score "
                                             f"{worst_transition[1]['quality']:.2f}</li>")

        # Add comparative insights
        if "comparative" in validation_results:
            comparative = validation_results["comparative"]

            if "comparison" in comparative:
                comparison = comparative["comparison"]

                for metric in ["final_knowledge", "knowledge_growth_rate", "suppression_decay_rate"]:
                    if f"best_{metric}" in comparison:
                        best = comparison[f"best_{metric}"]
                        summary_parts.append(f"<li>Best {metric.replace('_', ' ')}: <strong>"
                                             f"{best['configuration']}</strong> configuration with value "
                                             f"{best['value']:.3f}</li>")

        summary_parts.append("</ul>")

        # Add overall assessment
        if summary_score >= 0.7:
            summary_parts.append("<h3>Overall Assessment</h3>"
                                 "<p>The equation system shows <strong>strong coverage</strong> across physics domains and scale levels. "
                                 "Cross-scale integration is effective, with good signal propagation between levels. "
                                 "There are only minor gaps that should be addressed to further strengthen the model.</p>")
        elif summary_score >= 0.4:
            summary_parts.append("<h3>Overall Assessment</h3>"
                                 "<p>The equation system shows <strong>adequate coverage</strong> but has some notable gaps. "
                                 "Cross-scale integration needs improvement, particularly between certain transition points. "
                                 "Addressing the identified high-priority gaps and strengthening integration would "
                                 "significantly improve model performance.</p>")
        else:
            summary_parts.append("<h3>Overall Assessment</h3>"
                                 "<p>The equation system has <strong>significant gaps</strong> in coverage and integration. "
                                 "Several physics domains or scale levels have limited or no equations. "
                                 "Cross-scale integration is weak, leading to poor signal propagation. "
                                 "A major overhaul is recommended to address the high-priority gaps and strengthen the "
                                 "foundational equations.</p>")

        return "".join(summary_parts)

    def _generate_coverage_content(self, validation_results):
        """
        Generate content for equation coverage analysis.

        Args:
            validation_results: Dictionary containing all validation results

        Returns:
            HTML string with coverage content
        """
        if "equation_coverage" not in validation_results:
            return "<p>No equation coverage analysis results available.</p>"

        equation_coverage = validation_results["equation_coverage"]
        coverage = equation_coverage.get("coverage", {})

        content_parts = []

        # Overview statistics
        content_parts.append("<h3>Coverage Overview</h3>")

        if "physics_coverage_pct" in coverage:
            physics_pct = coverage["physics_coverage_pct"]
            physics_class = "metric-good" if physics_pct >= 70 else (
                "metric-warning" if physics_pct >= 40 else "metric-bad")
            content_parts.append(
                f"<p>Physics Domain Coverage: <span class='{physics_class}'>{physics_pct:.1f}%</span></p>")

        if "application_coverage_pct" in coverage:
            app_pct = coverage["application_coverage_pct"]
            app_class = "metric-good" if app_pct >= 70 else ("metric-warning" if app_pct >= 40 else "metric-bad")
            content_parts.append(f"<p>Application Domain Coverage: <span class='{app_class}'>{app_pct:.1f}%</span></p>")

        if "scale_coverage_pct" in coverage:
            scale_pct = coverage["scale_coverage_pct"]
            scale_class = "metric-good" if scale_pct >= 70 else ("metric-warning" if scale_pct >= 40 else "metric-bad")
            content_parts.append(f"<p>Scale Level Coverage: <span class='{scale_class}'>{scale_pct:.1f}%</span></p>")

        # Physics domain coverage table
        if "physics_domains" in coverage:
            content_parts.append("<h3>Physics Domain Coverage</h3>")
            content_parts.append("<table>")
            content_parts.append("<tr><th>Physics Domain</th><th>Number of Equations</th><th>Coverage Status</th></tr>")

            for domain, count in coverage["physics_domains"].items():
                status_class = "metric-good" if count >= 3 else ("metric-warning" if count >= 1 else "metric-bad")
                status_text = "Good" if count >= 3 else ("Limited" if count >= 1 else "None")

                content_parts.append(f"<tr>")
                content_parts.append(f"<td>{domain.replace('_', ' ').title()}</td>")
                content_parts.append(f"<td>{count}</td>")
                content_parts.append(f"<td class='{status_class}'>{status_text}</td>")
                content_parts.append(f"</tr>")

            content_parts.append("</table>")

        # Scale level coverage table
        if "scale_levels" in coverage:
            content_parts.append("<h3>Scale Level Coverage</h3>")
            content_parts.append("<table>")
            content_parts.append("<tr><th>Scale Level</th><th>Number of Equations</th><th>Coverage Status</th></tr>")

            for scale, count in coverage["scale_levels"].items():
                status_class = "metric-good" if count >= 3 else ("metric-warning" if count >= 1 else "metric-bad")
                status_text = "Good" if count >= 3 else ("Limited" if count >= 1 else "None")

                content_parts.append(f"<tr>")
                content_parts.append(f"<td>{scale.replace('_', ' ').title()}</td>")
                content_parts.append(f"<td>{count}</td>")
                content_parts.append(f"<td class='{status_class}'>{status_text}</td>")
                content_parts.append(f"</tr>")

            content_parts.append("</table>")

        # Integration quality
        if "integration" in equation_coverage:
            integration = equation_coverage["integration"]

            content_parts.append("<h3>Integration Quality</h3>")

            if "integration_quality" in integration:
                quality = integration["integration_quality"]

                content_parts.append("<table>")
                content_parts.append("<tr><th>Category</th><th>Integration Score</th><th>Status</th></tr>")

                for category, score in quality.items():
                    status_class = "metric-good" if score >= 0.7 else (
                        "metric-warning" if score >= 0.4 else "metric-bad")
                    status_text = "Good" if score >= 0.7 else ("Moderate" if score >= 0.4 else "Poor")

                    content_parts.append(f"<tr>")
                    content_parts.append(f"<td>{category.replace('_', ' ').title()}</td>")
                    content_parts.append(f"<td>{score:.2f}</td>")
                    content_parts.append(f"<td class='{status_class}'>{status_text}</td>")
                    content_parts.append(f"</tr>")

                content_parts.append("</table>")

        return "".join(content_parts)

    def _generate_cross_scale_content(self, validation_results):
        """
        Generate content for cross-scale interaction analysis.

        Args:
            validation_results: Dictionary containing all validation results

        Returns:
            HTML string with cross-scale content
        """
        if "cross_scale" not in validation_results:
            return "<p>No cross-scale interaction analysis results available.</p>"

        cross_scale = validation_results["cross_scale"]
        content_parts = []

        # Overall integration score
        if "overall_integration_score" in cross_scale:
            score = cross_scale["overall_integration_score"]
            score_class = "metric-good" if score >= 0.7 else ("metric-warning" if score >= 0.4 else "metric-bad")
            score_text = "Strong" if score >= 0.7 else ("Moderate" if score >= 0.4 else "Weak")

            content_parts.append("<h3>Overall Integration</h3>")
            content_parts.append(
                f"<p>Cross-Scale Integration Score: <span class='{score_class}'>{score:.2f}</span></p>")
            content_parts.append(f"<p>Integration Status: <span class='{score_class}'>{score_text}</span></p>")

        # Scale transitions
        if "scale_transitions" in cross_scale and "transitions" in cross_scale["scale_transitions"]:
            transitions = cross_scale["scale_transitions"]["transitions"]

            content_parts.append("<h3>Scale Transitions</h3>")
            content_parts.append("<table>")
            content_parts.append("<tr><th>Transition</th><th>Quality</th><th>Bridge Equations</th><th>Status</th></tr>")

            for (scale1, scale2), transition in transitions.items():
                quality = transition["quality"]
                status_class = "metric-good" if quality >= 0.7 else (
                    "metric-warning" if quality >= 0.4 else "metric-bad")
                status_text = "Strong" if quality >= 0.7 else ("Moderate" if quality >= 0.4 else "Weak")

                bridge_equations = transition.get("bridge_equations", [])
                bridge_text = ", ".join(bridge_equations) if bridge_equations else "None"

                content_parts.append(f"<tr>")
                content_parts.append(
                    f"<td>{scale1.replace('_', ' ').title()} → {scale2.replace('_', ' ').title()}</td>")
                content_parts.append(f"<td>{quality:.2f}</td>")
                content_parts.append(f"<td>{bridge_text}</td>")
                content_parts.append(f"<td class='{status_class}'>{status_text}</td>")
                content_parts.append(f"</tr>")

            content_parts.append("</table>")

        # Key transitions
        if "transition_quality" in cross_scale:
            transition_quality = cross_scale["transition_quality"]

            content_parts.append("<h3>Key Equation Transitions</h3>")
            content_parts.append("<table>")
            content_parts.append(
                "<tr><th>Source Equation</th><th>Target Equation</th><th>Quality</th><th>Status</th></tr>")

            for (eq1, eq2), quality in transition_quality.items():
                status_class = "metric-good" if quality >= 0.7 else (
                    "metric-warning" if quality >= 0.4 else "metric-bad")
                status_text = "Strong" if quality >= 0.7 else ("Moderate" if quality >= 0.4 else "Weak")

                content_parts.append(f"<tr>")
                content_parts.append(f"<td>{eq1}</td>")
                content_parts.append(f"<td>{eq2}</td>")
                content_parts.append(f"<td>{quality:.2f}</td>")
                content_parts.append(f"<td class='{status_class}'>{status_text}</td>")
                content_parts.append(f"</tr>")

            content_parts.append("</table>")

        # Signal propagation
        if "signal_propagation" in cross_scale and "by_source_scale" in cross_scale["signal_propagation"]:
            signal = cross_scale["signal_propagation"]
            by_source = signal["by_source_scale"]

            content_parts.append("<h3>Signal Propagation</h3>")
            content_parts.append("<p>How effectively signals propagate from one scale to others:</p>")
            content_parts.append("<table class='integration-matrix'>")
            content_parts.append(
                "<tr><th>Source Scale</th><th>Average Signal</th><th>Max Signal</th><th>Status</th></tr>")

            for source, metrics in by_source.items():
                avg_signal = metrics.get("average_signal", 0)
                max_signal = metrics.get("max_signal", 0)

                status_class = "metric-good" if avg_signal >= 0.5 else (
                    "metric-warning" if avg_signal >= 0.2 else "metric-bad")
                status_text = "Good" if avg_signal >= 0.5 else ("Moderate" if avg_signal >= 0.2 else "Poor")

                content_parts.append(f"<tr>")
                content_parts.append(f"<td>{source.replace('_', ' ').title()}</td>")
                content_parts.append(f"<td>{avg_signal:.2f}</td>")
                content_parts.append(f"<td>{max_signal:.2f}</td>")
                content_parts.append(f"<td class='{status_class}'>{status_text}</td>")
                content_parts.append(f"</tr>")

            content_parts.append("</table>")

            if "average_efficiency" in signal:
                efficiency = signal["average_efficiency"]
                efficiency_class = "metric-good" if efficiency >= 0.5 else (
                    "metric-warning" if efficiency >= 0.2 else "metric-bad")

                content_parts.append(
                    f"<p>Overall Propagation Efficiency: <span class='{efficiency_class}'>{efficiency:.2f}</span></p>")

        return "".join(content_parts)

    def _generate_comparative_content(self, validation_results):
        """
        Generate content for comparative simulation analysis.

        Args:
            validation_results: Dictionary containing all validation results

        Returns:
            HTML string with comparative content
        """
        if "comparative" not in validation_results:
            return "<p>No comparative simulation analysis results available.</p>"

        comparative = validation_results["comparative"]
        content_parts = []

        # Best overall configuration
        if "comparison" in comparative and "best_overall" in comparative["comparison"]:
            best = comparative["comparison"]["best_overall"]

            content_parts.append("<h3>Best Configuration</h3>")
            content_parts.append(f"<p>The best overall configuration is <span class='metric-good'>"
                                 f"{best['configuration']}</span> with a score of {best['score']}.</p>")

        # Configuration scores
        if "comparison" in comparative and "scores" in comparative["comparison"]:
            scores = comparative["comparison"]["scores"]

            content_parts.append("<h3>Configuration Scores</h3>")
            content_parts.append("<table>")
            content_parts.append("<tr><th>Configuration</th><th>Score</th><th>Rank</th></tr>")

            # Sort configurations by score
            sorted_configs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            for i, (config, score) in enumerate(sorted_configs, 1):
                rank_class = "metric-good" if i == 1 else ("metric-warning" if i <= 3 else "")

                content_parts.append(f"<tr>")
                content_parts.append(f"<td>{config}</td>")
                content_parts.append(f"<td>{score}</td>")
                content_parts.append(f"<td class='{rank_class}'>{i}</td>")
                content_parts.append(f"</tr>")

            content_parts.append("</table>")

        # Metric comparison
        if "comparison" in comparative:
            comparison = comparative["comparison"]
            metrics = [key.replace("_by_config", "") for key in comparison.keys() if key.endswith("_by_config")]

            if metrics:
                content_parts.append("<h3>Metric Comparison</h3>")
                content_parts.append("<table>")

                # Header row with configuration names
                config_names = list(comparison.get(f"{metrics[0]}_by_config", {}).keys())
                content_parts.append("<tr><th>Metric</th>")
                for config in config_names:
                    content_parts.append(f"<th>{config}</th>")
                content_parts.append("<th>Best Configuration</th></tr>")

                # Metric rows
                for metric in metrics:
                    metric_by_config = comparison.get(f"{metric}_by_config", {})
                    best_metric = comparison.get(f"best_{metric}", {})
                    best_config = best_metric.get("configuration", "")

                    content_parts.append(f"<tr>")
                    content_parts.append(f"<td>{metric.replace('_', ' ').title()}</td>")

                    for config in config_names:
                        value = metric_by_config.get(config, "")
                        cell_class = "metric-good" if config == best_config else ""

                        if isinstance(value, (int, float)):
                            content_parts.append(f"<td class='{cell_class}'>{value:.3f}</td>")
                        else:
                            content_parts.append(f"<td class='{cell_class}'>{value}</td>")

                    content_parts.append(f"<td class='metric-good'>{best_config}</td>")
                    content_parts.append(f"</tr>")

                content_parts.append("</table>")

        return "".join(content_parts)

    def _generate_gaps_content(self, validation_results):
        """
        Generate content for identified gaps analysis.

        Args:
            validation_results: Dictionary containing all validation results

        Returns:
            HTML string with gaps content
        """
        if "equation_coverage" not in validation_results or "gaps" not in validation_results["equation_coverage"]:
            return "<p>No gap analysis results available.</p>"

        gaps = validation_results["equation_coverage"]["gaps"]
        content_parts = []

        # Summary of gaps
        content_parts.append("<h3>Gap Summary</h3>")

        if not gaps:
            content_parts.append("<p class='metric-good'>No gaps identified in equation coverage!</p>")
        else:
            # Count gaps by type and severity
            gap_types = {}
            gap_severity = {"high": 0, "medium": 0, "low": 0}

            for gap in gaps:
                gap_type = gap.get("type", "unknown")
                severity = gap.get("severity", "medium")

                gap_types[gap_type] = gap_types.get(gap_type, 0) + 1
                gap_severity[severity] = gap_severity.get(severity, 0) + 1

            content_parts.append("<table>")
            content_parts.append("<tr><th>Category</th><th>Count</th></tr>")

            for gap_type, count in gap_types.items():
                content_parts.append(f"<tr><td>{gap_type.replace('_', ' ').title()}</td><td>{count}</td></tr>")

            content_parts.append("<tr><th>Severity</th><th>Count</th></tr>")

            for severity, count in gap_severity.items():
                severity_class = "metric-bad" if severity == "high" else (
                    "metric-warning" if severity == "medium" else "")
                content_parts.append(f"<tr><td class='{severity_class}'>{severity.title()}</td><td>{count}</td></tr>")

            content_parts.append("</table>")

        # Detailed gaps
        if gaps:
            content_parts.append("<h3>Detailed Gaps</h3>")
            content_parts.append("<table class='gap-table'>")
            content_parts.append("<tr><th>Description</th><th>Type</th><th>Severity</th><th>Recommendation</th></tr>")

            for gap in gaps:
                severity = gap.get("severity", "medium")
                row_class = "gap-high" if severity == "high" else ("gap-medium" if severity == "medium" else "gap-low")

                content_parts.append(f"<tr class='{row_class}'>")
                content_parts.append(f"<td>{gap.get('description', '')}</td>")
                content_parts.append(f"<td>{gap.get('type', '').replace('_', ' ').title()}</td>")
                content_parts.append(f"<td>{severity.title()}</td>")
                content_parts.append(f"<td>{gap.get('recommendation', '')}</td>")
                content_parts.append("</tr>")

            content_parts.append("</table>")

        return "".join(content_parts)

    def _generate_optimization_content(self, validation_results):
        """
        Generate content for optimization opportunities.

        Args:
            validation_results: Dictionary containing all validation results

        Returns:
            HTML string with optimization content
        """
        if "opportunities" not in validation_results:
            return "<p>No optimization opportunities identified.</p>"

        opportunities = validation_results["opportunities"]
        content_parts = []

        # High priority opportunities
        high_priority = []
        for category, items in opportunities.items():
            for item in items:
                if item.get("priority") == "High":
                    high_priority.append({
                        "category": category,
                        "item": item
                    })

        if high_priority:
            content_parts.append("<h3>High Priority Optimizations</h3>")
            content_parts.append("<ul class='opportunities-list'>")

            for opportunity in high_priority:
                category = opportunity["category"].replace("_", " ").title()
                item = opportunity["item"]

                description = item.get('description', item.get('equation', item.get('parameter',
                                                                                    item.get('transition',
                                                                                             item.get('best_config',
                                                                                                      'Optimization')))))

                content_parts.append(f"<li class='opportunity-high'>")
                content_parts.append(f"<strong>{category}:</strong> {description}")
                content_parts.append(
                    f"<div class='recommendations'>Recommendation: {item.get('recommendation', '')}</div>")
                content_parts.append("</li>")

            content_parts.append("</ul>")

        # Other opportunities by category
        for category, items in opportunities.items():
            if items:
                # Skip high priority items already listed
                category_items = [item for item in items if item.get("priority") != "High"]

                if category_items:
                    category_title = category.replace("_", " ").title()
                    content_parts.append(f"<h3>{category_title}</h3>")
                    content_parts.append("<ul class='opportunities-list'>")

                    for item in category_items:
                        priority = item.get("priority", "Medium")
                        priority_class = f"opportunity-{priority.lower()}"

                        description = item.get('description', item.get('equation', item.get('parameter',
                                                                                            item.get('transition',
                                                                                                     item.get(
                                                                                                         'best_config',
                                                                                                         'Optimization')))))

                        content_parts.append(f"<li class='{priority_class}'>")
                        content_parts.append(f"<strong>{description}</strong>")
                        content_parts.append(
                            f"<div class='recommendations'>Recommendation: {item.get('recommendation', '')}</div>")
                        content_parts.append("</li>")

                    content_parts.append("</ul>")

        return "".join(content_parts)

    def _generate_recommendations_content(self, validation_results):
        """
        Generate content for recommendations section.

        Args:
            validation_results: Dictionary containing all validation results

        Returns:
            HTML string with recommendations content
        """
        content_parts = []
        recommendation_steps = []

        # Generate recommendations based on gap analysis
        if "equation_coverage" in validation_results and "gaps" in validation_results["equation_coverage"]:
            gaps = validation_results["equation_coverage"]["gaps"]

            # Prioritize high severity gaps
            high_severity_gaps = [gap for gap in gaps if gap.get("severity", "") == "high"]

            if high_severity_gaps:
                recommendation_steps.append("<h3>Fill High Priority Gaps</h3>")
                recommendation_steps.append("<ol>")

                for gap in high_severity_gaps:
                    recommendation_steps.append(f"<li><strong>{gap.get('description', '')}</strong><br>")
                    recommendation_steps.append(f"{gap.get('recommendation', '')}</li>")

                recommendation_steps.append("</ol>")

        # Generate recommendations based on cross-scale analysis
        if "cross_scale" in validation_results:
            cross_scale = validation_results["cross_scale"]

            if "scale_transitions" in cross_scale and "transitions" in cross_scale["scale_transitions"]:
                transitions = cross_scale["scale_transitions"]["transitions"]

                # Find weak transitions
                weak_transitions = [(scales, details) for scales, details in transitions.items()
                                    if details["quality"] < 0.4]

                if weak_transitions:
                    recommendation_steps.append("<h3>Strengthen Scale Transitions</h3>")
                    recommendation_steps.append("<ol>")

                    for (scale1, scale2), details in weak_transitions:
                        recommendation_steps.append(f"<li><strong>Improve {scale1.replace('_', ' ').title()} to "
                                                    f"{scale2.replace('_', ' ').title()} transition</strong><br>")

                        if "bridge_equations" in details and details["bridge_equations"]:
                            bridges = ", ".join(details["bridge_equations"])
                            recommendation_steps.append(f"Enhance existing bridge equations: {bridges}</li>")
                        else:
                            recommendation_steps.append(f"Create new bridge equations connecting these scales</li>")

                    recommendation_steps.append("</ol>")

        # Generate recommendations based on comparative analysis
        if "comparative" in validation_results and "comparison" in validation_results["comparative"]:
            comparison = validation_results["comparative"]["comparison"]

            best_configs = {}
            for key, value in comparison.items():
                if key.startswith("best_") and not key == "best_overall":
                    metric = key.replace("best_", "")
                    if "configuration" in value:
                        best_configs[metric] = value["configuration"]

            if "best_overall" in comparison:
                best_overall = comparison["best_overall"]["configuration"]

                different_bests = [metric for metric, config in best_configs.items()
                                   if config != best_overall]

                if different_bests:
                    recommendation_steps.append("<h3>Optimize Equation Configuration</h3>")
                    recommendation_steps.append("<ol>")

                    recommendation_steps.append(
                        f"<li><strong>Analyze why {best_overall} configuration performs best overall</strong><br>")
                    recommendation_steps.append(
                        f"Identify key strengths that can be applied to other configurations</li>")

                    for metric in different_bests:
                        config = best_configs[metric]
                        recommendation_steps.append(f"<li><strong>Incorporate strengths from {config} "
                                                    f"configuration for {metric.replace('_', ' ')}</strong><br>")
                        recommendation_steps.append(
                            f"Study how {config} handles {metric.replace('_', ' ')} more effectively</li>")

                    recommendation_steps.append("</ol>")

        # Add optimization opportunities as recommendations
        if "opportunities" in validation_results:
            opportunities = validation_results["opportunities"]

            # Group opportunities by category
            categories = {}
            for category, items in opportunities.items():
                if items:
                    categories[category] = items

            if categories:
                recommendation_steps.append("<h3>Implementation Plan</h3>")
                recommendation_steps.append("<ol>")

                # Equation gaps
                if "equation_gaps" in categories:
                    recommendation_steps.append("<li><strong>Address Equation Gaps</strong><ul>")
                    for item in categories["equation_gaps"]:
                        recommendation_steps.append(f"<li>{item.get('description', '')}</li>")
                    recommendation_steps.append("</ul></li>")

                # Cross-scale improvements
                if "cross_scale_improvements" in categories:
                    recommendation_steps.append("<li><strong>Improve Cross-Scale Integration</strong><ul>")
                    for item in categories["cross_scale_improvements"]:
                        recommendation_steps.append(f"<li>{item.get('description', '')}</li>")
                    recommendation_steps.append("</ul></li>")

                # Stability enhancements
                if "stability_enhancements" in categories:
                    recommendation_steps.append("<li><strong>Enhance Numerical Stability</strong><ul>")
                    for item in categories["stability_enhancements"]:
                        recommendation_steps.append(f"<li>{item.get('description', '')}</li>")
                    recommendation_steps.append("</ul></li>")

                # Parameter optimizations
                if "parameter_optimizations" in categories:
                    recommendation_steps.append("<li><strong>Optimize Parameters</strong><ul>")
                    for item in categories["parameter_optimizations"]:
                        recommendation_steps.append(f"<li>{item.get('description', '')}</li>")
                    recommendation_steps.append("</ul></li>")

                # Integration opportunities
                if "integration_opportunities" in categories:
                    recommendation_steps.append("<li><strong>Enhance Model Integration</strong><ul>")
                    for item in categories["integration_opportunities"]:
                        recommendation_steps.append(f"<li>{item.get('description', '')}</li>")
                    recommendation_steps.append("</ul></li>")

                recommendation_steps.append("</ol>")

        # Add validation steps
        recommendation_steps.append("<h3>Validation Steps</h3>")
        recommendation_steps.append("<ol>")
        recommendation_steps.append(
            "<li><strong>Re-run equation coverage analysis</strong> after implementing changes</li>")
        recommendation_steps.append(
            "<li><strong>Validate cross-scale integration</strong> to ensure smooth transitions</li>")
        recommendation_steps.append(
            "<li><strong>Compare simulation results</strong> before and after optimizations</li>")
        recommendation_steps.append("<li><strong>Check for stability issues</strong> in optimized equations</li>")
        recommendation_steps.append(
            "<li><strong>Generate updated validation report</strong> to measure improvement</li>")
        recommendation_steps.append("</ol>")

        # Combine all recommendations
        content_parts.extend(recommendation_steps)

        return "".join(content_parts)

    def _generate_chart_content(self, validation_results):
        """
        Generate chart images for inclusion in the report.

        Args:
            validation_results: Dictionary containing all validation results

        Returns:
            Dictionary with base64-encoded chart images
        """
        charts = {
            # Initialize all chart keys with empty strings
            "physics_domain_chart": "",
            "scale_level_chart": "",
            "cross_domain_chart": "",
            "dependency_graph": "",
            "scale_adjacency": "",
            "signal_propagation": "",
            "metric_comparison": "",
            "config_scores": "",
            "knowledge_comparison": "",
            "suppression_comparison": "",
            "gap_chart": ""
        }

        # Physics domain coverage chart
        if "equation_coverage" in validation_results and "coverage" in validation_results["equation_coverage"]:
            coverage = validation_results["equation_coverage"]["coverage"]

            if "physics_domains" in coverage:
                try:
                    plt.figure(figsize=(10, 6))

                    # Create bar chart
                    domains = list(coverage["physics_domains"].keys())
                    counts = list(coverage["physics_domains"].values())

                    plt.bar(domains, counts, color='skyblue')
                    plt.title("Physics Domain Coverage")
                    plt.xlabel("Physics Domain")
                    plt.ylabel("Number of Equations")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    # Convert to base64
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    charts["physics_domain_chart"] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                except Exception as e:
                    logger.error(f"Error generating physics domain chart: {e}")
                    charts["physics_domain_chart"] = ""

            # Scale level coverage chart
            if "scale_levels" in coverage:
                try:
                    plt.figure(figsize=(10, 6))

                    # Create bar chart
                    scales = list(coverage["scale_levels"].keys())
                    counts = list(coverage["scale_levels"].values())

                    plt.bar(scales, counts, color='lightgreen')
                    plt.title("Scale Level Coverage")
                    plt.xlabel("Scale Level")
                    plt.ylabel("Number of Equations")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    # Convert to base64
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    charts["scale_level_chart"] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                except Exception as e:
                    logger.error(f"Error generating scale level chart: {e}")
                    charts["scale_level_chart"] = ""

            # Cross-domain coverage matrix
            if "cross_domain_coverage" in coverage:
                try:
                    plt.figure(figsize=(12, 8))

                    # Create heatmap
                    matrix = coverage["cross_domain_coverage"]

                    # Get domain and application labels
                    physics_domains = list(coverage.get("physics_domains", {}).keys())
                    application_domains = list(coverage.get("application_domains", {}).keys())

                    # Ensure matrix dimensions match labels
                    if len(physics_domains) != matrix.shape[0] or len(application_domains) != matrix.shape[1]:
                        # Use generic labels if dimensions don't match
                        physics_domains = [f"Physics {i + 1}" for i in range(matrix.shape[0])]
                        application_domains = [f"App {i + 1}" for i in range(matrix.shape[1])]

                    plt.imshow(matrix, cmap='Blues')
                    plt.colorbar(label="Number of Equations")

                    plt.title("Cross-Domain Coverage Matrix")
                    plt.xlabel("Application Domain")
                    plt.ylabel("Physics Domain")

                    plt.xticks(np.arange(len(application_domains)), application_domains, rotation=45, ha='right')
                    plt.yticks(np.arange(len(physics_domains)), physics_domains)

                    # Add text annotations
                    for i in range(len(physics_domains)):
                        for j in range(len(application_domains)):
                            if i < matrix.shape[0] and j < matrix.shape[1]:
                                plt.text(j, i, int(matrix[i, j]),
                                         ha="center", va="center",
                                         color="black" if matrix[i, j] < 3 else "white")

                    plt.tight_layout()

                    # Convert to base64
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    charts["cross_domain_chart"] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                except Exception as e:
                    logger.error(f"Error generating cross-domain matrix chart: {e}")
                    charts["cross_domain_chart"] = ""

        # Cross-scale visualization
        if "cross_scale" in validation_results:
            cross_scale = validation_results["cross_scale"]

            # Dependency graph
            if "dependency_graph" in cross_scale:
                try:
                    plt.figure(figsize=(12, 10))

                    # Create a simplified visualization of the dependency graph
                    # In a real implementation, you would use the dependency_graph data to create a network visualization
                    plt.text(0.5, 0.5, "Dependency Graph Visualization",
                             ha='center', va='center', fontsize=12)
                    plt.axis('off')

                    # Convert to base64
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    charts["dependency_graph"] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                except Exception as e:
                    logger.error(f"Error generating dependency graph: {e}")
                    charts["dependency_graph"] = ""

            # Scale adjacency matrix
            if "dependency_graph" in cross_scale and "scale_adjacency" in cross_scale["dependency_graph"]:
                try:
                    plt.figure(figsize=(10, 8))

                    # Create heatmap of scale adjacency matrix
                    matrix = cross_scale["dependency_graph"]["scale_adjacency"]

                    # Get scale level labels
                    scale_levels = ["quantum", "agent", "group", "civilization", "multi_civilization", "cosmic"]

                    # Ensure matrix dimensions match labels
                    if len(scale_levels) != matrix.shape[0] or len(scale_levels) != matrix.shape[1]:
                        # Use generic labels if dimensions don't match
                        scale_levels = [f"Scale {i + 1}" for i in range(matrix.shape[0])]

                    plt.imshow(matrix, cmap='Blues')
                    plt.colorbar(label="Number of Connections")

                    plt.title("Scale Adjacency Matrix")
                    plt.xlabel("To Scale")
                    plt.ylabel("From Scale")

                    plt.xticks(np.arange(len(scale_levels)), scale_levels, rotation=45, ha='right')
                    plt.yticks(np.arange(len(scale_levels)), scale_levels)

                    # Add text annotations
                    for i in range(min(len(scale_levels), matrix.shape[0])):
                        for j in range(min(len(scale_levels), matrix.shape[1])):
                            plt.text(j, i, int(matrix[i, j]),
                                     ha="center", va="center",
                                     color="black" if matrix[i, j] < 3 else "white")

                    plt.tight_layout()

                    # Convert to base64
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    charts["scale_adjacency"] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                except Exception as e:
                    logger.error(f"Error generating scale adjacency matrix: {e}")
                    charts["scale_adjacency"] = ""

            # Signal propagation
            if "signal_propagation" in cross_scale and "by_source_scale" in cross_scale["signal_propagation"]:
                try:
                    plt.figure(figsize=(10, 8))

                    # Extract signal propagation data
                    signal_data = cross_scale["signal_propagation"]["by_source_scale"]

                    # Create a matrix of signal strengths
                    scale_levels = list(signal_data.keys())
                    matrix = np.zeros((len(scale_levels), len(scale_levels)))

                    for i, source in enumerate(scale_levels):
                        if "signal_by_scale" in signal_data[source]:
                            for j, target in enumerate(scale_levels):
                                matrix[i, j] = signal_data[source]["signal_by_scale"].get(target, 0)

                    plt.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)
                    plt.colorbar(label="Signal Strength")

                    plt.title("Signal Propagation Across Scales")
                    plt.xlabel("Target Scale")
                    plt.ylabel("Source Scale")

                    plt.xticks(np.arange(len(scale_levels)), scale_levels, rotation=45, ha='right')
                    plt.yticks(np.arange(len(scale_levels)), scale_levels)

                    # Add text annotations
                    for i in range(len(scale_levels)):
                        for j in range(len(scale_levels)):
                            plt.text(j, i, f"{matrix[i, j]:.2f}",
                                     ha="center", va="center",
                                     color="black" if matrix[i, j] < 0.5 else "white")

                    plt.tight_layout()

                    # Convert to base64
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    charts["signal_propagation"] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                except Exception as e:
                    logger.error(f"Error generating signal propagation chart: {e}")
                    charts["signal_propagation"] = ""

        # Comparative simulation analysis
        if "comparative" in validation_results:
            comparative = validation_results["comparative"]

            # Metric comparison
            if "comparison" in comparative:
                try:
                    comparison = comparative["comparison"]
                    metrics = [key.replace("_by_config", "") for key in comparison.keys() if key.endswith("_by_config")]

                    if metrics:
                        # Pick a few key metrics to display
                        key_metrics = ["final_knowledge", "knowledge_growth_rate", "final_suppression",
                                       "stability_issues"]
                        key_metrics = [m for m in key_metrics if m in metrics]

                        # If no key metrics found, use the first few available
                        if not key_metrics and metrics:
                            key_metrics = metrics[:min(4, len(metrics))]

                        if key_metrics:
                            fig, axes = plt.subplots(1, len(key_metrics), figsize=(15, 6))

                            # Handle case of single metric
                            if len(key_metrics) == 1:
                                axes = [axes]

                            for i, metric in enumerate(key_metrics):
                                ax = axes[i]

                                metric_by_config = comparison.get(f"{metric}_by_config", {})
                                configs = list(metric_by_config.keys())
                                values = list(metric_by_config.values())

                                colors = ['blue'] * len(configs)

                                # Color the best value
                                best_config = comparison.get(f"best_{metric}", {}).get("configuration", "")
                                if best_config in configs:
                                    best_idx = configs.index(best_config)
                                    colors[best_idx] = 'green' if metric not in ["final_suppression",
                                                                                 "stability_issues"] else 'red'

                                ax.bar(configs, values, color=colors)
                                ax.set_title(metric.replace('_', ' ').title())
                                ax.set_ylabel(metric.replace('_', ' ').title())
                                ax.tick_params(axis='x', rotation=45, labelsize=8)

                            plt.tight_layout()

                            # Convert to base64
                            buf = BytesIO()
                            plt.savefig(buf, format="png")
                            buf.seek(0)
                            charts["metric_comparison"] = base64.b64encode(buf.read()).decode('utf-8')
                            plt.close()

                except Exception as e:
                    logger.error(f"Error generating metric comparison chart: {e}")
                    charts["metric_comparison"] = ""

            # Configuration scores
            if "comparison" in comparative and "scores" in comparative["comparison"]:
                try:
                    scores = comparative["comparison"]["scores"]

                    plt.figure(figsize=(10, 6))

                    # Sort configurations by score
                    sorted_configs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    configs = [x[0] for x in sorted_configs]
                    values = [x[1] for x in sorted_configs]

                    # Color the best configuration green
                    colors = ['blue'] * len(configs)
                    if configs:
                        colors[0] = 'green'

                    plt.bar(configs, values, color=colors)
                    plt.title("Configuration Scores")
                    plt.xlabel("Configuration")
                    plt.ylabel("Score")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    # Convert to base64
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    charts["config_scores"] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                except Exception as e:
                    logger.error(f"Error generating configuration scores chart: {e}")
                    charts["config_scores"] = ""

            # Knowledge and suppression comparison
            for metric in ["knowledge", "suppression"]:
                try:
                    plt.figure(figsize=(10, 6))

                    # Plot time series data from example
                    x = np.linspace(0, 100, 101)

                    # Synthetic data for visualization
                    if metric == "knowledge":
                        plt.plot(x, 10 * np.log(1 + x / 20), label="Base")
                        plt.plot(x, 12 * np.log(1 + x / 18), label="Quantum")
                        plt.plot(x, 11 * np.log(1 + x / 19), label="Astrophysics")
                        plt.plot(x, 13 * np.log(1 + x / 17), label="Multi-Civ")
                        plt.plot(x, 15 * np.log(1 + x / 15), label="Integrated")
                    else:  # suppression
                        plt.plot(x, 10 * np.exp(-x / 50) + 2, label="Base")
                        plt.plot(x, 12 * np.exp(-x / 45) + 1.5, label="Quantum")
                        plt.plot(x, 11 * np.exp(-x / 40) + 1.8, label="Astrophysics")
                        plt.plot(x, 13 * np.exp(-x / 35) + 1.2, label="Multi-Civ")
                        plt.plot(x, 15 * np.exp(-x / 30) + 1.0, label="Integrated")

                    plt.title(f"{metric.title()} Over Time")
                    plt.xlabel("Time")
                    plt.ylabel(metric.title())
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()

                    # Convert to base64
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    charts[f"{metric}_comparison"] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

                except Exception as e:
                    logger.error(f"Error generating {metric} comparison chart: {e}")
                    charts[f"{metric}_comparison"] = ""

        # Gap analysis
        if "equation_coverage" in validation_results and "gaps" in validation_results["equation_coverage"]:
            try:
                gaps = validation_results["equation_coverage"]["gaps"]

                if gaps:
                    # Count gaps by type and severity
                    gap_types = {}
                    gap_severity = {"high": 0, "medium": 0, "low": 0}

                    for gap in gaps:
                        gap_type = gap.get("type", "unknown")
                        severity = gap.get("severity", "medium")

                        gap_types[gap_type] = gap_types.get(gap_type, 0) + 1
                        gap_severity[severity] = gap_severity.get(severity, 0) + 1

                    plt.figure(figsize=(12, 10))

                    # Create subplots
                    plt.subplot(2, 1, 1)

                    # Gap types
                    types = list(gap_types.keys())
                    type_counts = list(gap_types.values())

                    plt.bar(types, type_counts, color='skyblue')
                    plt.title("Gaps by Type")
                    plt.xlabel("Gap Type")
                    plt.ylabel("Count")
                    plt.xticks(rotation=45, ha='right')

                    plt.subplot(2, 1, 2)

                    # Gap severity
                    severities = list(gap_severity.keys())
                    severity_counts = list(gap_severity.values())
                    severity_colors = ['red', 'orange', 'green']

                    plt.bar(severities, severity_counts, color=severity_colors)
                    plt.title("Gaps by Severity")
                    plt.xlabel("Severity")
                    plt.ylabel("Count")

                    plt.tight_layout()

                    # Convert to base64
                    buf = BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    charts["gap_chart"] = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()

            except Exception as e:
                logger.error(f"Error generating gap analysis chart: {e}")
                charts["gap_chart"] = ""

        return charts


# Example usage
if __name__ == "__main__":
    generator = ReportGenerator()

    # Create dummy validation results for testing
    validation_results = {
        "equation_coverage": {
            "coverage": {
                "physics_domains": {
                    "thermodynamics": 3,
                    "electromagnetism": 4,
                    "strong_nuclear": 1,
                    "weak_nuclear": 2,
                    "quantum_mechanics": 3,
                    "relativity": 1,
                    "astrophysics": 5,
                    "multi_system": 3
                },
                "application_domains": {
                    "intelligence": 3,
                    "knowledge": 5,
                    "truth": 2,
                    "wisdom": 1,
                    "suppression": 3,
                    "resistance": 2,
                    "free_will": 1,
                    "civilization": 4
                },
                "scale_levels": {
                    "quantum": 3,
                    "agent": 5,
                    "group": 2,
                    "civilization": 4,
                    "multi_civilization": 3,
                    "cosmic": 2
                },
                "physics_coverage_pct": 80.0,
                "application_coverage_pct": 75.0,
                "scale_coverage_pct": 83.3
            },
            "gaps": [
                {
                    "type": "physics_domain",
                    "domain": "strong_nuclear",
                    "description": "Limited coverage of physics domain: strong_nuclear (only 1 equation)",
                    "severity": "medium",
                    "modules": [],
                    "recommendation": "Expand strong_nuclear modeling with additional equations"
                },
                {
                    "type": "expected_equation",
                    "domain": "quantum_mechanics",
                    "equation": "quantum_superposition",
                    "description": "Missing expected equation: quantum_superposition in quantum_mechanics domain",
                    "severity": "high",
                    "modules": [],
                    "recommendation": "Implement quantum_superposition function"
                },
                {
                    "type": "cross_domain",
                    "physics": "relativity",
                    "application": "knowledge",
                    "description": "No equations connecting relativity physics to knowledge application",
                    "severity": "medium",
                    "modules": [],
                    "recommendation": "Create equation relating relativity principles to knowledge dynamics"
                }
            ]
        },
        "cross_scale": {
            "overall_integration_score": 0.65,
            "scale_transitions": {
                "transitions": {
                    ("quantum", "agent"): {
                        "bridge_equations": ["quantum_tunneling_probability", "free_will_decision"],
                        "quality": 0.75
                    },
                    ("agent", "group"): {
                        "bridge_equations": ["knowledge_field_influence"],
                        "quality": 0.6
                    },
                    ("group", "civilization"): {
                        "bridge_equations": [],
                        "quality": 0.3
                    },
                    ("civilization", "multi_civilization"): {
                        "bridge_equations": ["knowledge_diffusion", "cultural_influence"],
                        "quality": 0.8
                    },
                    ("multi_civilization", "cosmic"): {
                        "bridge_equations": ["galactic_structure_model"],
                        "quality": 0.55
                    }
                },
                "average_quality": 0.6
            },
            "transition_quality": {
                ("quantum_tunneling_probability", "truth_adoption"): 0.7,
                ("quantum_entanglement_correlation", "knowledge_field_influence"): 0.6,
                ("knowledge_field_influence", "knowledge_diffusion"): 0.5,
                ("truth_adoption", "civilization_lifecycle_phase"): 0.4,
                ("suppression_feedback", "suppression_event_horizon"): 0.3,
                ("knowledge_field_gradient", "galactic_structure_model"): 0.8
            },
            "signal_propagation": {
                "by_source_scale": {
                    "quantum": {
                        "signal_by_scale": {
                            "quantum": 1.0,
                            "agent": 0.7,
                            "group": 0.4,
                            "civilization": 0.2,
                            "multi_civilization": 0.1,
                            "cosmic": 0.05
                        },
                        "average_signal": 0.41,
                        "max_signal": 1.0
                    },
                    "agent": {
                        "signal_by_scale": {
                            "quantum": 0.6,
                            "agent": 1.0,
                            "group": 0.8,
                            "civilization": 0.5,
                            "multi_civilization": 0.3,
                            "cosmic": 0.1
                        },
                        "average_signal": 0.55,
                        "max_signal": 1.0
                    }
                },
                "average_efficiency": 0.48
            }
        },
        "comparative": {
            "comparison": {
                "final_knowledge_by_config": {
                    "base": 25.4,
                    "quantum": 32.7,
                    "astrophysics": 28.9,
                    "multi_civ": 30.5,
                    "integrated": 38.2
                },
                "knowledge_growth_rate_by_config": {
                    "base": 0.05,
                    "quantum": 0.07,
                    "astrophysics": 0.06,
                    "multi_civ": 0.065,
                    "integrated": 0.09
                },
                "final_suppression_by_config": {
                    "base": 5.8,
                    "quantum": 4.2,
                    "astrophysics": 4.9,
                    "multi_civ": 5.1,
                    "integrated": 3.5
                },
                "stability_issues_by_config": {
                    "base": 0,
                    "quantum": 2,
                    "astrophysics": 1,
                    "multi_civ": 5,
                    "integrated": 3
                },
                "best_final_knowledge": {
                    "value": 38.2,
                    "configuration": "integrated"
                },
                "best_knowledge_growth_rate": {
                    "value": 0.09,
                    "configuration": "integrated"
                },
                "best_final_suppression": {
                    "value": 3.5,
                    "configuration": "integrated"
                },
                "best_stability_issues": {
                    "value": 0,
                    "configuration": "base"
                },
                "scores": {
                    "base": 1,
                    "quantum": 0,
                    "astrophysics": 0,
                    "multi_civ": 0,
                    "integrated": 3
                },
                "best_overall": {
                    "configuration": "integrated",
                    "score": 3
                }
            }
        },
        "opportunities": {
            "equation_gaps": [
                {
                    "description": "Missing quantum_superposition equation",
                    "priority": "High",
                    "recommendation": "Implement quantum_superposition function in quantum_em_extensions module"
                },
                {
                    "description": "Limited strong nuclear force modeling",
                    "priority": "Medium",
                    "recommendation": "Add additional strong nuclear force equations for identity binding"
                }
            ],
            "cross_scale_improvements": [
                {
                    "transition": ("group", "civilization"),
                    "quality": 0.3,
                    "priority": "High",
                    "recommendation": "Create bridge equations connecting group and civilization scales"
                }
            ],
            "stability_enhancements": [
                {
                    "equation": "multi_civilization_interactions",
                    "issues": ["division by zero", "parameter bounds"],
                    "priority": "Medium",
                    "recommendation": "Enhance stability of multi_civilization_interactions by adding circuit breaker integration"
                }
            ],
            "parameter_optimizations": [
                {
                    "parameter": "knowledge_growth_rate",
                    "importance": 0.85,
                    "priority": "High",
                    "recommendation": "Optimize knowledge_growth_rate parameter for better performance"
                }
            ],
            "integration_opportunities": [
                {
                    "best_config": "integrated",
                    "priority": "Medium",
                    "recommendation": "Integrate successful patterns from integrated configuration into other configurations"
                }
            ]
        }
    }

    # Generate report
    output_path = "validation/reports/unified/test_report.html"
    generator.generate_unified_report(validation_results, output_path)

    print(f"Test report generated at {output_path}")