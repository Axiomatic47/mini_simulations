"""
Multi-Civilization Simulation Dashboard Server
Provides API endpoints and serves the dashboard frontend
"""
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from flask import Flask, send_from_directory, jsonify, request, Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dashboard.log')
    ]
)
logger = logging.getLogger('dashboard')

# Create argument parser
parser = argparse.ArgumentParser(description='Launch interactive dashboard for multi-civilization simulations')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the dashboard server on')
parser.add_argument('--port', type=int, default=5000, help='Port to run the dashboard server on')
parser.add_argument('--data-dir', type=str, default=None, help='Directory containing simulation data')
parser.add_argument('--auto-run', action='store_true', help='Automatically run a simulation if no data is found')
parser.add_argument('--debug', action='store_true', help='Run in debug mode')
args = parser.parse_args()

# Ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent
data_dir = Path(args.data_dir) if args.data_dir else BASE_DIR / 'outputs' / 'data'
dashboard_dir = BASE_DIR / 'outputs' / 'dashboard'
data_dir.mkdir(parents=True, exist_ok=True)
dashboard_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Data directory: {data_dir}")
logger.info(f"Dashboard directory: {dashboard_dir}")

# Create Flask app for the dashboard API
app = Flask(__name__, static_folder=str(dashboard_dir))


# Data preparation functions
def prepare_simulation_data():
    """
    Prepare simulation data for the dashboard.
    """
    try:
        # Try to read data from statistics CSV
        stats_file = data_dir / "multi_civilization_statistics.csv"

        logger.info(f"Reading simulation data from: {stats_file}")

        if not stats_file.exists():
            logger.warning(f"Statistics file not found: {stats_file}")
            if args.auto_run:
                logger.info("Auto-run enabled, running a new simulation...")
                run_new_simulation()
                # Check again after running
                if not stats_file.exists():
                    logger.error("Failed to create statistics file even after auto-run")
                    return []
            else:
                logger.info("Creating placeholder data...")
                create_placeholder_data()
                # Check again after creating placeholder data
                if not stats_file.exists():
                    logger.error("Failed to create statistics file with placeholder data")
                    return []

        # Read the statistics CSV
        df = pd.read_csv(stats_file)
        logger.info(f"Successfully read statistics data: {len(df)} rows")

        # Replace NaN with None (which becomes null in JSON)
        json_data = df.replace({np.nan: None}).to_dict(orient='records')
        return json_data

    except Exception as e:
        logger.error(f"Error preparing simulation data: {e}", exc_info=True)
        return []


def prepare_event_data():
    """
    Prepare event data for the dashboard.
    """
    try:
        # Try to read data from events CSV
        events_file = data_dir / "multi_civilization_events.csv"

        logger.info(f"Reading event data from: {events_file}")

        if not events_file.exists():
            logger.warning(f"Events file not found: {events_file}")
            # If events file is missing but we've already created placeholder data for statistics,
            # we might need to create it for events too
            if (data_dir / "multi_civilization_statistics.csv").exists():
                logger.info("Creating placeholder event data...")
                create_placeholder_data()

            # Check again after potential creation
            if not events_file.exists():
                logger.error("Events file still not found after placeholder creation attempt")
                return []

        # Read the events CSV
        df = pd.read_csv(events_file)
        logger.info(f"Successfully read events data: {len(df)} rows")

        # Replace NaN with None (which becomes null in JSON)
        json_data = df.replace({np.nan: None}).to_dict(orient='records')
        return json_data

    except Exception as e:
        logger.error(f"Error preparing event data: {e}", exc_info=True)
        return []


def prepare_stability_data():
    """
    Prepare stability metrics for the dashboard.
    """
    try:
        # Try to read data from stability CSV
        stability_file = data_dir / "multi_civilization_stability.csv"

        logger.info(f"Reading stability data from: {stability_file}")

        if not stability_file.exists():
            logger.warning(f"Stability file not found: {stability_file}")
            # If stability file is missing but we've already created other placeholder data,
            # create it as well
            if (data_dir / "multi_civilization_statistics.csv").exists():
                logger.info("Creating placeholder stability data...")
                create_placeholder_data()

            # Check again after potential creation
            if not stability_file.exists():
                logger.error("Stability file still not found after placeholder creation attempt")
                return {}

        # Read the stability CSV
        df = pd.read_csv(stability_file)
        logger.info(f"Successfully read stability data")

        # Replace NaN with None (which becomes null in JSON)
        if len(df) > 0:
            json_data = df.iloc[0].replace({np.nan: None}).to_dict()
            return json_data
        else:
            logger.warning("Stability file is empty")
            return {}

    except Exception as e:
        logger.error(f"Error preparing stability data: {e}", exc_info=True)
        return {}


def run_new_simulation():
    """
    Run a new multi-civilization simulation if no data is found.
    """
    try:
        logger.info("Running a new multi-civilization simulation...")
        # This would import and run the simulation script
        try:
            from simulations.multi_civilization_simulation import run_simulation
            run_simulation(output_dir=data_dir)
            logger.info("Simulation completed.")
        except ImportError as e:
            logger.error(f"Failed to import simulation module: {e}")
            raise
    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        logger.info("Creating placeholder data instead...")
        create_placeholder_data()


def create_placeholder_data():
    """
    Create placeholder data for testing the dashboard when no simulation data is available.
    """
    logger.info("Creating placeholder data...")
    try:
        # Create mock simulation statistics
        timesteps = 150
        simulation_data = []

        for t in range(timesteps):
            # Add some random fluctuations to make it look realistic
            civilization_count = max(1, round(5 + 2 * np.sin(t / 20) + np.random.rand() * 2))
            knowledge_mean = min(50, 1 + t * 0.3 + np.random.rand() * 3)
            suppression_mean = max(0.5, 7 - t * 0.04 + np.random.rand() * 2)
            intelligence_mean = min(50, 0.5 + t * 0.25 + np.random.rand() * 2)
            truth_mean = min(40, 0.1 + t * 0.2 + np.random.rand())
            resources_total = civilization_count * (50 + t * 0.5 + np.random.rand() * 10)
            stability_issues = round(t * 0.05 + np.random.rand() * 3)

            simulation_data.append({
                'Time': t,
                'Civilization_Count': civilization_count,
                'knowledge_mean': knowledge_mean,
                'knowledge_max': knowledge_mean + 5 + np.random.rand() * 10,
                'knowledge_min': max(0, knowledge_mean - 5 - np.random.rand() * 5),
                'suppression_mean': suppression_mean,
                'suppression_max': suppression_mean + 3 + np.random.rand() * 5,
                'suppression_min': max(0.1, suppression_mean - 3 - np.random.rand() * 2),
                'intelligence_mean': intelligence_mean,
                'truth_mean': truth_mean,
                'resources_total': resources_total,
                'Stability_Issues': stability_issues,
                'Timestep': 0.2 + 0.8 * min(1, 1 - (stability_issues / 20))
            })

        # Save to CSV
        stats_file = data_dir / "multi_civilization_statistics.csv"
        pd.DataFrame(simulation_data).to_csv(stats_file, index=False)
        logger.info(f"Created placeholder statistics data at {stats_file}")

        # Create mock event data
        event_types = ['collision', 'merger', 'collapse', 'spawn', 'new_civilization']
        event_data = []

        for i in range(50):
            event_type = event_types[np.random.randint(0, len(event_types))]
            time = np.random.randint(0, timesteps)

            event_data.append({
                'id': i,
                'type': event_type,
                'time': time,
                'civ_id1': np.random.randint(0, 10),
                'civ_id2': np.random.randint(0, 10),
                'position_x': np.random.rand() * 10,
                'position_y': np.random.rand() * 10,
                'description': f"{event_type} event at time {time}"
            })

        # Save to CSV
        events_file = data_dir / "multi_civilization_events.csv"
        pd.DataFrame(event_data).to_csv(events_file, index=False)
        logger.info(f"Created placeholder events data at {events_file}")

        # Create mock stability data
        stability_data = {
            'Total_Stability_Issues': 87,
            'Circuit_Breaker_Triggers': 52,
            'Max_Knowledge': 48.7,
            'Max_Suppression': 15.3,
            'Max_Intelligence': 42.1,
            'Max_Truth': 37.9,
            'Total_Collisions': 12,
            'Total_Mergers': 8,
            'Total_Collapses': 6,
            'Total_Spawns': 15,
            'Total_New_Civilizations': 9,
            'Used_Dimensional_Analysis': True
        }

        # Save to CSV
        stability_file = data_dir / "multi_civilization_stability.csv"
        pd.DataFrame([stability_data]).to_csv(stability_file, index=False)
        logger.info(f"Created placeholder stability data at {stability_file}")

        logger.info("Placeholder data created successfully.")

    except Exception as e:
        logger.error(f"Error creating placeholder data: {e}", exc_info=True)
        raise


def convert_numpy_types(obj):
    """
    Convert NumPy types to Python native types recursively.
    Works with dictionaries, lists, and scalar values.

    This helper ensures all JSON responses are properly serializable.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# Update your error handling to ensure consistent error format:

def json_response(data, status=200):
    """Create a Flask JSON response with proper error format."""
    if status >= 400 and isinstance(data, dict) and 'error' not in data:
        data = {'error': 'An error occurred'} if not data else data
    converted_data = convert_numpy_types(data)
    return jsonify(converted_data), status

def interpret_correlation(correlation):
    """Provide a simple interpretation of the correlation value."""
    if correlation > 0.8:
        return "Strong positive correlation: As knowledge increases, suppression tends to strongly increase."
    elif correlation > 0.5:
        return "Moderate positive correlation: As knowledge increases, suppression tends to moderately increase."
    elif correlation > 0.2:
        return "Weak positive correlation: As knowledge increases, suppression tends to slightly increase."
    elif correlation > -0.2:
        return "No significant correlation: Knowledge and suppression appear to be independent."
    elif correlation > -0.5:
        return "Weak negative correlation: As knowledge increases, suppression tends to slightly decrease."
    elif correlation > -0.8:
        return "Moderate negative correlation: As knowledge increases, suppression tends to moderately decrease."
    else:
        return "Strong negative correlation: As knowledge increases, suppression tends to strongly decrease."


# Define API routes
@app.route('/')
def serve_dashboard():
    """Serve the React dashboard HTML page."""
    try:
        logger.info("Serving dashboard HTML")
        return send_from_directory(dashboard_dir, 'index.html')
    except Exception as e:
        logger.error(f"Error serving dashboard HTML: {e}", exc_info=True)
        return f"Error serving dashboard: {str(e)}", 500


@app.route('/dashboard.js')
def serve_dashboard_js():
    """Serve the React dashboard JavaScript file."""
    try:
        logger.info("Serving dashboard.js")
        return send_from_directory(dashboard_dir, 'dashboard.js')
    except Exception as e:
        logger.error(f"Error serving dashboard.js: {e}", exc_info=True)
        return f"Error serving dashboard.js: {str(e)}", 500


@app.route('/dashboard-component.js')
def serve_dashboard_component_js():
    """DEPRECATED: Support for legacy component naming (redirects to dashboard.js)."""
    logger.info("Redirecting legacy dashboard-component.js request to dashboard.js")
    return send_from_directory(dashboard_dir, 'dashboard.js')


@app.route('/chart-dashboard.js')
def serve_chart_dashboard_js():
    """DEPRECATED: Serve the Chart.js dashboard file (legacy implementation)."""
    logger.info(f"Attempting to serve deprecated chart-dashboard.js (keeping for compatibility)")
    return send_from_directory(dashboard_dir, 'chart-dashboard.js')


@app.route('/api/data/multi_civilization_statistics.csv')
def get_simulation_data():
    """API endpoint for simulation statistics data."""
    try:
        logger.info("API request for simulation statistics")
        data = prepare_simulation_data()
        if not data:
            logger.warning("No simulation data available")
            return json_response([])

        logger.info(f"Returning simulation data: {len(data)} records")
        return json_response(data)
    except Exception as e:
        logger.error(f"Error in simulation data API: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)


@app.route('/api/data/multi_civilization_events.csv')
def get_event_data():
    """API endpoint for event data."""
    try:
        logger.info("API request for event data")
        data = prepare_event_data()
        if not data:
            logger.warning("No event data available")
            return json_response([])

        logger.info(f"Returning event data: {len(data)} records")
        return json_response(data)
    except Exception as e:
        logger.error(f"Error in event data API: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)


@app.route('/api/data/multi_civilization_stability.csv')
def get_stability_data():
    """API endpoint for stability metrics."""
    try:
        logger.info("API request for stability metrics")
        data = prepare_stability_data()
        if not data:
            logger.warning("No stability data available")
            return json_response({})

        logger.info("Returning stability data")
        return json_response(data)
    except Exception as e:
        logger.error(f"Error in stability data API: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)

# In your multi_civilization_dashboard.py, add a simple test endpoint:
@app.route('/api/test/numpy-conversion')
def test_numpy_conversion():
    """Test endpoint for NumPy type conversion."""
    import numpy as np
    test_data = {
        'int64': np.int64(42),
        'float64': np.float64(3.14),
        'array': np.array([1, 2, 3]),
        'nested': {'array': np.array([4, 5, 6])}
    }
    return json_response(test_data)

@app.route('/api/analysis/knowledge_suppression_correlation')
def get_knowledge_suppression_correlation():
    """
    API endpoint for correlation analysis between knowledge and suppression.

    Returns the correlation coefficient and trend data over time.
    """
    try:
        df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")

        # Calculate correlation between knowledge and suppression means
        correlation = df['knowledge_mean'].corr(df['suppression_mean'])

        # Generate trend data
        trend_data = []
        for i in range(len(df) - 1):
            if i % 10 == 0:  # sample every 10 points for efficiency
                trend_data.append({
                    'time': int(df.iloc[i]['Time']),
                    'knowledge_mean': float(df.iloc[i]['knowledge_mean']),
                    'suppression_mean': float(df.iloc[i]['suppression_mean'])
                })

        result = {
            'correlation_coefficient': float(correlation),
            'trend_data': trend_data,
            'interpretation': interpret_correlation(correlation)
        }

        return json_response(result)
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return json_response({"error": str(e)}, 500)


@app.route('/api/data/civilization_comparison')
def get_civilization_comparison():
    """
    Compare top vs bottom civilizations based on specified metric.

    Query parameters:
    - metric: Metric to compare (default: knowledge)
    - time: Time point for comparison (default: latest available)
    - count: Number of civilizations to compare from each group (default: 3)
    """
    try:
        # Get comparison parameters
        metric = request.args.get('metric', 'knowledge')
        count = int(request.args.get('count', 3))

        # Load statistics data to get the latest time point
        stats_df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")

        # Determine time point for comparison
        if request.args.get('time'):
            time_point = int(request.args.get('time'))
        else:
            time_point = int(stats_df['Time'].max())  # Use latest available time

        # Now we need to get data for individual civilizations
        # This is a placeholder - in a real implementation, you would load
        # detailed data about each civilization at the specified time point

        # For this example, we'll create simulated data
        # In a real implementation, you would load this from a file or database
        civs_data = []
        civ_count = stats_df.loc[stats_df['Time'] == time_point, 'Civilization_Count'].iloc[0]

        # Generate simulated data for individual civilizations
        for i in range(int(civ_count)):
            knowledge = np.random.normal(
                stats_df.loc[stats_df['Time'] == time_point, 'knowledge_mean'].iloc[0],
                stats_df.loc[stats_df['Time'] == time_point, 'knowledge_max'].iloc[0] / 5
            )
            suppression = np.random.normal(
                stats_df.loc[stats_df['Time'] == time_point, 'suppression_mean'].iloc[0],
                stats_df.loc[stats_df['Time'] == time_point, 'suppression_max'].iloc[0] / 5
            )

            civs_data.append({
                'id': int(i),  # Convert NumPy types to Python native types
                'knowledge': float(max(0, knowledge)),
                'suppression': float(max(0, suppression)),
                'intelligence': float(max(0, np.random.normal(10, 3))),
                'resources': float(max(0, np.random.normal(50, 15))),
                'influence': float(max(0, np.random.normal(5, 1.5))),
            })

        # Sort by the specified metric
        sorted_civs = sorted(civs_data, key=lambda x: x[metric], reverse=True)

        # Get top and bottom civilizations
        top_civs = sorted_civs[:count]
        bottom_civs = sorted_civs[-count:]

        # Prepare comparison data
        result = {
            'time_point': int(time_point),
            'comparison_metric': metric,
            'top_civilizations': top_civs,
            'bottom_civilizations': bottom_civs,
            'metadata': {
                'total_civilizations': int(civ_count),
                'mean_value': float(np.mean([civ[metric] for civ in civs_data])),
                'max_value': float(np.max([civ[metric] for civ in civs_data])),
                'min_value': float(np.min([civ[metric] for civ in civs_data]))
            }
        }

        return json_response(result)
    except Exception as e:
        logger.error(f"Error generating civilization comparison: {e}")
        return json_response({"error": str(e)}, 500)


@app.route('/api/meta/available_metrics')
def get_available_metrics():
    """
    Return metadata about available metrics for the dashboard.

    This helps the frontend dynamically adjust to available data.
    """
    try:
        from datetime import datetime
        # Read statistics file to get column names
        stats_path = data_dir / "multi_civilization_statistics.csv"
        if not stats_path.exists():
            return json_response({"error": "Statistics file not found"}, 404)

        stats_df = pd.read_csv(stats_path)

        # Group metrics into categories
        metric_categories = {
            "basic": ["Time", "Civilization_Count", "Timestep"],
            "knowledge": [col for col in stats_df.columns if "knowledge" in col.lower()],
            "suppression": [col for col in stats_df.columns if "suppression" in col.lower()],
            "intelligence": [col for col in stats_df.columns if "intelligence" in col.lower()],
            "truth": [col for col in stats_df.columns if "truth" in col.lower()],
            "resources": [col for col in stats_df.columns if "resources" in col.lower()],
            "influence": [col for col in stats_df.columns if "influence" in col.lower()],
            "stability": [col for col in stats_df.columns if "stability" in col.lower()],
            "other": []
        }

        # Categorize any columns not already in a category
        for col in stats_df.columns:
            if not any(col in category for category in metric_categories.values()):
                metric_categories["other"].append(col)

        # Get event types
        event_types = []
        events_path = data_dir / "multi_civilization_events.csv"
        if events_path.exists():
            events_df = pd.read_csv(events_path)
            if 'type' in events_df.columns:
                event_types = events_df['type'].unique().tolist()

        result = {
            "metric_categories": metric_categories,
            "event_types": event_types,
            "time_range": {
                "min": int(stats_df['Time'].min()) if 'Time' in stats_df.columns else 0,
                "max": int(stats_df['Time'].max()) if 'Time' in stats_df.columns else 0
            },
            "last_updated": datetime.now().isoformat()
        }

        return json_response(result)
    except Exception as e:
        logger.error(f"Error getting available metrics: {e}")
        return json_response({"error": str(e)}, 500)


@app.route('/api/export/csv')
def export_csv():
    """
    Export simulation data as CSV file for download.

    Query parameters:
    - dataset: Dataset to export (statistics, events, stability)
    """
    try:
        # Determine which dataset to export
        dataset = request.args.get('dataset', 'statistics')

        if dataset == 'statistics':
            filename = "multi_civilization_statistics.csv"
            content_filename = "simulation_statistics.csv"
        elif dataset == 'events':
            filename = "multi_civilization_events.csv"
            content_filename = "simulation_events.csv"
        elif dataset == 'stability':
            filename = "multi_civilization_stability.csv"
            content_filename = "simulation_stability.csv"
        else:
            return json_response({"error": "Invalid dataset specified"}, 400)

        # Read the file
        file_path = data_dir / filename
        if not file_path.exists():
            return json_response({"error": f"File not found: {filename}"}, 404)

        # Read the file content
        with open(file_path, 'r') as file:
            csv_content = file.read()

        # Create response with CSV content
        response = Response(csv_content, mimetype='text/csv')
        response.headers['Content-Disposition'] = f'attachment; filename={content_filename}'

        return response
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        return json_response({"error": str(e)}, 500)


if __name__ == '__main__':
    logger.info(f"Starting dashboard server on http://{args.host}:{args.port}")

    # Check if data files exist
    stats_file = data_dir / "multi_civilization_statistics.csv"
    events_file = data_dir / "multi_civilization_events.csv"
    stability_file = data_dir / "multi_civilization_stability.csv"

    if not (stats_file.exists() and events_file.exists() and stability_file.exists()):
        logger.warning("One or more data files not found.")
        if args.auto_run:
            logger.info("Auto-run enabled, running a new simulation...")
            run_new_simulation()
        else:
            logger.info("Creating placeholder data...")
            create_placeholder_data()

    # Debug info
    logger.info(f"Stats file exists: {stats_file.exists()}")
    logger.info(f"Events file exists: {events_file.exists()}")
    logger.info(f"Stability file exists: {stability_file.exists()}")

    # Start the server
    app.run(host=args.host, port=args.port, debug=args.debug)