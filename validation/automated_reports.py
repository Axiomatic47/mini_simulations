"""
Automated validation reporting system that compiles findings across all components.
This script handles report generation and scheduling for continuous validation.
"""

import os
import json
import time
import schedule
import datetime
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from jinja2 import Environment, FileSystemLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("validation/logs/validation_reports.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("validation_reports")


class ValidationReporter:
    """
    A class to manage automated validation reporting across all components.
    """

    def __init__(self, output_dir="validation/reports"):
        """
        Initialize the reporter.

        Parameters:
            output_dir: Root directory for validation reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        Path("validation/logs").mkdir(parents=True, exist_ok=True)

        # Create directories for each component
        self.component_dirs = {
            "dimension": self.output_dir / "dimension",
            "sensitivity": self.output_dir / "sensitivity",
            "edge_case": self.output_dir / "edge_case",
            "cross_level": self.output_dir / "cross_level",
            "dimensional": self.output_dir / "dimensional",
            "historical": self.output_dir / "historical"
        }

        for dir_path in self.component_dirs.values():
            dir_path.mkdir(exist_ok=True)

        # Initialize component results
        self.results = {}

        # Initialize history tracking
        self.history_file = self.output_dir / "validation_history.csv"
        if not self.history_file.exists():
            # Create with header
            pd.DataFrame(columns=[
                "timestamp", "overall_status", "dimension_status",
                "sensitivity_status", "edge_case_status", "cross_level_status",
                "dimensional_status", "historical_status", "report_path"
            ]).to_csv(self.history_file, index=False)

        # Load history
        self.history = pd.read_csv(self.history_file)

    def set_component_result(self, component, result):
        """
        Set results for a specific validation component.

        Parameters:
            component: Component name
            result: Validation result object
        """
        self.results[component] = result

        # Save result to JSON file
        result_file = self.component_dirs[component] / "latest_result.json"

        # Convert to serializable format if needed
        if isinstance(result, dict):
            # Keep as is
            serializable_result = result
        else:
            # Try to convert to dict
            try:
                serializable_result = result.to_dict()
            except:
                # Fall back to simplified representation
                serializable_result = {"summary": str(result)}

        with open(result_file, "w") as f:
            json.dump(serializable_result, f, default=str, indent=2)

        logger.info(f"Saved {component} results to {result_file}")

    def get_component_status(self, component):
        """
        Determine the status of a component.

        Parameters:
            component: Component name

        Returns:
            Status string: "Success", "Warning", "Error", or "Unknown"
        """
        result = self.results.get(component)

        if result is None:
            return "Unknown"

        # Component-specific status determination
        if component == "dimension":
            # Check if any dimensions were fixed
            fixed_count = result.get("fixed_count", 0)
            return "Success" if fixed_count == 0 else "Warning"

        elif component == "sensitivity":
            # Always success if completed
            return "Success"

        elif component == "edge_case":
            # Check for high severity issues
            high_severity_count = 0
            for func_result in result.values():
                recommendations = func_result.get("recommendations", [])
                high_severity_count += sum(1 for r in recommendations if r.get("severity") in ["high", "critical"])
            return "Success" if high_severity_count == 0 else "Error"

        elif component == "cross_level":
            # Check for dependency violations
            violations = result.get("level_dependencies", {}).get("violations", [])
            return "Success" if not violations else "Error"

        elif component == "dimensional":
            # Check for inconsistent dimensions
            inconsistent = sum(1 for r in result.values() if r.get("status") == "INCONSISTENT")
            return "Success" if inconsistent == 0 else "Error"

        elif component == "historical":
            # Check overall error threshold
            overall_rmse = result.get("overall", {}).get("rmse", float("inf"))
            return "Success" if overall_rmse < 10.0 else "Warning"

        return "Unknown"

    def get_overall_status(self):
        """
        Determine the overall validation status.

        Returns:
            Status string: "Success", "Warning", "Error", or "Unknown"
        """
        statuses = [self.get_component_status(c) for c in self.results.keys()]

        if "Error" in statuses:
            return "Error"
        elif "Warning" in statuses:
            return "Warning"
        elif "Success" in statuses and len(statuses) == len(self.component_dirs):
            return "Success"
        else:
            return "Incomplete"

    def generate_summary_visualizations(self):
        """
        Generate summary visualizations of validation status history.
        """
        if len(self.history) == 0:
            logger.warning("No validation history available for visualization")
            return

        # Ensure the history has a proper timestamp column
        if "timestamp" not in self.history.columns:
            logger.warning("Timestamp column missing from history")
            return

        # Convert timestamp to datetime
        self.history["datetime"] = pd.to_datetime(self.history["timestamp"])

        # 1. Status history timeline
        plt.figure(figsize=(15, 8))

        # Define status colors
        status_colors = {
            "Success": "green",
            "Warning": "orange",
            "Error": "red",
            "Unknown": "gray",
            "Incomplete": "lightblue"
        }

        # Plot status over time for each component
        components = ["overall", "dimension", "sensitivity", "edge_case",
                      "cross_level", "dimensional", "historical"]

        # Create status numeric mapping for plotting
        status_map = {
            "Success": 3,
            "Warning": 2,
            "Error": 1,
            "Unknown": 0,
            "Incomplete": 0
        }

        # Plot each component's status
        for i, component in enumerate(components):
            status_col = f"{component}_status"
            if status_col in self.history.columns:
                # Convert status to numeric for plotting
                numeric_status = self.history[status_col].map(status_map)

                # Get colors
                colors = self.history[status_col].map(status_colors)

                # Plot
                plt.scatter(
                    self.history["datetime"],
                    [i] * len(self.history),
                    c=colors,
                    s=100,
                    marker="s"
                )

        # Set y-ticks to component names
        plt.yticks(range(len(components)), components)

        # Format x-axis
        plt.gcf().autofmt_xdate()

        # Add legend
        for status, color in status_colors.items():
            plt.plot([], [], "s", color=color, label=status)

        plt.legend(loc="upper right")
        plt.title("Validation Status History")
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / "status_history.png", dpi=300)
        plt.close()

        # 2. Component status distribution
        plt.figure(figsize=(15, 8))

        # Create subplots for each component
        for i, component in enumerate(components):
            status_col = f"{component}_status"
            if status_col in self.history.columns:
                # Count status occurrences
                status_counts = self.history[status_col].value_counts()

                # Plot
                ax = plt.subplot(2, 4, i + 1)
                wedges, texts, autotexts = ax.pie(
                    status_counts,
                    labels=status_counts.index,
                    autopct="%1.1f%%",
                    colors=[status_colors.get(s, "gray") for s in status_counts.index]
                )

                # Set title
                ax.set_title(component)

        plt.suptitle("Validation Status Distribution")
        plt.tight_layout()

        # Save
        plt.savefig(self.output_dir / "status_distribution.png", dpi=300)
        plt.close()

        # 3. Recent trend analysis
        # Use the most recent 10 validation runs
        recent_history = self.history.tail(10).copy()
        if len(recent_history) > 1:
            plt.figure(figsize=(15, 8))

            # Create line plot of overall status
            overall_status_numeric = recent_history["overall_status"].map(status_map)

            plt.plot(
                recent_history["datetime"],
                overall_status_numeric,
                marker="o",
                linestyle="-",
                color="blue"
            )

            # Add status regions
            plt.axhspan(2.5, 3.5, color="green", alpha=0.2, label="Success")
            plt.axhspan(1.5, 2.5, color="orange", alpha=0.2, label="Warning")
            plt.axhspan(0.5, 1.5, color="red", alpha=0.2, label="Error")
            plt.axhspan(-0.5, 0.5, color="gray", alpha=0.2, label="Unknown")

            # Add labels
            plt.yticks([0, 1, 2, 3], ["Unknown", "Error", "Warning", "Success"])

            # Format x-axis
            plt.gcf().autofmt_xdate()

            plt.title("Recent Overall Validation Status Trend")
            plt.legend()
            plt.tight_layout()

            # Save
            plt.savefig(self.output_dir / "recent_trend.png", dpi=300)
            plt.close()

    def generate_report(self):
        """
        Generate a comprehensive validation report.

        Returns:
            Path to the generated report
        """
        # Create timestamp for report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment for templating
        try:
            template_dir = Path("validation/templates")
            if not template_dir.exists():
                # Create template directory
                template_dir.mkdir(parents=True, exist_ok=True)

                # Create basic template
                with open(template_dir / "report_template.html", "w") as f:
                    f.write(self._get_default_template())

            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template("report_template.html")
        except Exception as e:
            logger.error(f"Error loading template: {e}")
            # Fall back to simple approach
            template = None

        # Get overall and component statuses
        overall_status = self.get_overall_status()
        component_statuses = {
            component: self.get_component_status(component)
            for component in self.results.keys()
        }

        # Generate summary visualizations
        self.generate_summary_visualizations()

        # Copy summary visualizations to report directory
        for viz_file in ["status_history.png", "status_distribution.png", "recent_trend.png"]:
            src_path = self.output_dir / viz_file
            if src_path.exists():
                import shutil
                shutil.copy(src_path, report_dir / viz_file)

        # Prepare template context
        context = {
            "timestamp": timestamp,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": overall_status,
            "component_statuses": component_statuses,
            "results": self.results
        }

        # Generate HTML report
        if template:
            # Use Jinja2 template
            html_content = template.render(**context)
        else:
            # Fallback to simple HTML
            html_content = self._generate_simple_html_report(context)

        # Write HTML report
        report_path = report_dir / "validation_report.html"
        with open(report_path, "w") as f:
            f.write(html_content)

        logger.info(f"Generated validation report at {report_path}")

        # Update history
        history_entry = {
            "timestamp": timestamp,
            "overall_status": overall_status,
            "report_path": str(report_path)
        }

        # Add component statuses
        for component, status in component_statuses.items():
            history_entry[f"{component}_status"] = status

        # Append to history
        self.history = pd.concat([
            self.history,
            pd.DataFrame([history_entry])
        ])

        # Save updated history
        self.history.to_csv(self.history_file, index=False)

        return report_path

    def _get_default_template(self):
        """
        Get default HTML template for reports.

        Returns:
            Template string
        """
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        .report-section {
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            background-color: #f9f9f9;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        .summary-table th, .summary-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .summary-table th {
            background-color: #f2f2f2;
        }
        .status-success {
            color: green;
            font-weight: bold;
        }
        .status-warning {
            color: orange;
            font-weight: bold;
        }
        .status-error {
            color: red;
            font-weight: bold;
        }
        .status-unknown {
            color: gray;
            font-weight: bold;
        }
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        .visualization img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <h1>Validation Report</h1>
    <p>Generated on: {{ date }}</p>

    <div class="report-section">
        <h2>Executive Summary</h2>
        <p>Overall Status: <span class="status-{{ overall_status.lower() }}">{{ overall_status }}</span></p>

        <table class="summary-table">
            <tr>
                <th>Component</th>
                <th>Status</th>
            </tr>
            {% for component, status in component_statuses.items() %}
            <tr>
                <td>{{ component }}</td>
                <td class="status-{{ status.lower() }}">{{ status }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="report-section">
        <h2>Historical Trends</h2>

        <div class="visualization">
            <h3>Validation Status History</h3>
            <img src="status_history.png" alt="Validation Status History">
        </div>

        <div class="visualization">
            <h3>Component Status Distribution</h3>
            <img src="status_distribution.png" alt="Status Distribution">
        </div>

        <div class="visualization">
            <h3>Recent Trend</h3>
            <img src="recent_trend.png" alt="Recent Trend">
        </div>
    </div>

    {% for component, result in results.items() %}
    <div class="report-section">
        <h2>{{ component }} Validation</h2>

        <p>Status: <span class="status-{{ component_statuses[component].lower() }}">{{ component_statuses[component] }}</span></p>

        <div class="component-details">
            <pre>{{ result }}</pre>
        </div>
    </div>
    {% endfor %}
</body>
</html>
"""

    def _generate_simple_html_report(self, context):
        """
        Generate a simple HTML report without templates.

        Parameters:
            context: Report context

        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report {context['timestamp']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                .unknown {{ color: gray; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Validation Report</h1>
            <p>Generated on: {context['date']}</p>

            <h2>Status Summary</h2>
            <p>Overall Status: <span class="{context['overall_status'].lower()}">{context['overall_status']}</span></p>

            <table>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                </tr>
        """

        # Add component statuses
        for component, status in context['component_statuses'].items():
            html += f"""
                <tr>
                    <td>{component}</td>
                    <td class="{status.lower()}">{status}</td>
                </tr>
            """

        html += """
            </table>

            <h2>Historical Trends</h2>
        """

        # Add visualizations
        for viz_file in ["status_history.png", "status_distribution.png", "recent_trend.png"]:
            if (Path("validation/reports") / viz_file).exists():
                html += f"""
                <h3>{viz_file.replace('.png', '').replace('_', ' ').title()}</h3>
                <img src="{viz_file}" alt="{viz_file}">
                """

        # Add component details
        for component, result in context['results'].items():
            html += f"""
            <h2>{component} Validation</h2>
            <p>Status: <span class="{context['component_statuses'][component].lower()}">{context['component_statuses'][component]}</span></p>
            <pre>{json.dumps(result, indent=2, default=str)}</pre>
            """

        html += """
        </body>
        </html>
        """

        return html

    def schedule_report_generation(self, interval_hours=24):
        """
        Schedule automatic report generation.

        Parameters:
            interval_hours: Interval in hours
        """

        # Define job
        def job():
            logger.info(f"Running scheduled validation report generation")
            try:
                report_path = self.generate_report()
                logger.info(f"Scheduled report generated: {report_path}")
            except Exception as e:
                logger.error(f"Error generating scheduled report: {e}")

        # Schedule job
        schedule.every(interval_hours).hours.do(job)

        logger.info(f"Scheduled validation reports every {interval_hours} hours")

    def run_scheduled_jobs(self):
        """
        Run the scheduled jobs continuously.
        """
        logger.info("Starting scheduled job runner")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduled job runner stopped by user")
        except Exception as e:
            logger.error(f"Error in scheduled job runner: {e}")


def setup_automated_reports(validation_results=None):
    """
    Set up automated validation reporting.

    Parameters:
        validation_results: Optional initial validation results

    Returns:
        ValidationReporter object
    """
    reporter = ValidationReporter()

    # Set initial results if provided
    if validation_results:
        for component, result in validation_results.items():
            reporter.set_component_result(component, result)

    # Generate initial report
    reporter.generate_report()

    # Schedule periodic reports
    reporter.schedule_report_generation(interval_hours=24)

    return reporter


# Example usage
if __name__ == "__main__":
    # Sample validation results
    validation_results = {
        "dimension": {
            "fixed_count": 2,
            "warning_count": 3
        },
        "sensitivity": {
            "top_parameters": ["param1", "param2"],
            "interactions": [("param1", "param2", 0.8)]
        },
        "edge_case": {
            "func1": {
                "patterns_found": {
                    "division_by_zero": ["line 10"],
                    "log_of_non_positive": ["line 15"]
                },
                "recommendations": [
                    {"severity": "high", "issue": "Division by zero"}
                ]
            }
        },
        "cross_level": {
            "level_dependencies": {
                "violations": []
            }
        },
        "dimensional": {
            "eq1": {"status": "CONSISTENT"},
            "eq2": {"status": "INCONSISTENT"}
        },
        "historical": {
            "overall": {"rmse": 12.5}
        }
    }

    # Set up reporter
    reporter = setup_automated_reports(validation_results)

    # Generate report
    report_path = reporter.generate_report()
    print(f"Report generated: {report_path}")

    # Run scheduled jobs (comment out to avoid blocking)
    # reporter.run_scheduled_jobs()