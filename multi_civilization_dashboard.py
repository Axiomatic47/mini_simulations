import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Create argument parser
parser = argparse.ArgumentParser(description='Launch interactive dashboard for multi-civilization simulations')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the dashboard server on')
parser.add_argument('--port', type=int, default=5000, help='Port to run the dashboard server on')
parser.add_argument('--data-dir', type=str, default=None, help='Directory containing simulation data')
parser.add_argument('--auto-run', action='store_true', help='Automatically run a simulation if no data is found')
args = parser.parse_args()

# Ensure output directories exist - adjusted to use correct paths
BASE_DIR = Path(__file__).resolve().parent
data_dir = Path(args.data_dir) if args.data_dir else BASE_DIR / 'outputs' / 'data'
dashboard_dir = BASE_DIR / 'outputs' / 'dashboard'
data_dir.mkdir(parents=True, exist_ok=True)
dashboard_dir.mkdir(parents=True, exist_ok=True)

# Create Flask app for the dashboard API
app = Flask(__name__, static_folder=str(dashboard_dir))
CORS(app)  # Allow cross-origin requests


# Data preparation functions
def prepare_simulation_data(data_file=None):
    """
    Prepare simulation data for the dashboard.
    If data_file is provided, read from it, otherwise use default location.
    """
    try:
        # Try to read data from statistics CSV
        if data_file:
            stats_file = Path(data_file)
        else:
            stats_file = data_dir / "multi_civilization_statistics.csv"

        if not stats_file.exists():
            print(f"Statistics file not found: {stats_file}")
            if args.auto_run:
                print("Auto-run enabled, running a new simulation...")
                run_new_simulation()
                # Check again after running
                if not stats_file.exists():
                    return []
            else:
                return []

        # Read the statistics CSV
        df = pd.read_csv(stats_file)

        # Convert to list of dictionaries for JSON
        data = df.to_dict('records')
        return data

    except Exception as e:
        print(f"Error preparing simulation data: {e}")
        return []


def prepare_event_data(data_file=None):
    """
    Prepare event data for the dashboard.
    """
    try:
        # Try to read data from events CSV
        if data_file:
            events_file = Path(data_file)
        else:
            events_file = data_dir / "multi_civilization_events.csv"

        if not events_file.exists():
            print(f"Events file not found: {events_file}")
            return []

        # Read the events CSV
        df = pd.read_csv(events_file)

        # Convert to list of dictionaries for JSON
        data = df.to_dict('records')
        return data

    except Exception as e:
        print(f"Error preparing event data: {e}")
        return []


def prepare_stability_data(data_file=None):
    """
    Prepare stability metrics for the dashboard.
    """
    try:
        # Try to read data from stability CSV
        if data_file:
            stability_file = Path(data_file)
        else:
            stability_file = data_dir / "multi_civilization_stability.csv"

        if not stability_file.exists():
            print(f"Stability file not found: {stability_file}")
            return {}

        # Read the stability CSV
        df = pd.read_csv(stability_file)

        # Convert to dictionary for JSON
        data = df.iloc[0].to_dict()
        return data

    except Exception as e:
        print(f"Error preparing stability data: {e}")
        return {}


def run_new_simulation():
    """
    Run a new multi-civilization simulation if no data is found.
    """
    try:
        print("Running a new multi-civilization simulation...")
        # This would import and run the simulation script
        from simulations.multi_civilization_simulation import run_simulation
        run_simulation(output_dir=data_dir)
        print("Simulation completed.")
    except Exception as e:
        print(f"Error running simulation: {e}")
        print("Creating placeholder simulation data...")
        create_placeholder_data()


def create_placeholder_data():
    """
    Create placeholder data for testing the dashboard when no simulation data is available.
    """
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
    pd.DataFrame(simulation_data).to_csv(data_dir / "multi_civilization_statistics.csv", index=False)

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
    pd.DataFrame(event_data).to_csv(data_dir / "multi_civilization_events.csv", index=False)

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
    pd.DataFrame([stability_data]).to_csv(data_dir / "multi_civilization_stability.csv", index=False)

    print("Placeholder data created.")


# Define API routes
@app.route('/')
def serve_dashboard():
    """Serve the dashboard HTML page."""
    return send_from_directory(dashboard_dir, 'index.html')


@app.route('/dashboard.js')
def serve_dashboard_js():
    """Serve the dashboard JavaScript file."""
    return send_from_directory(dashboard_dir, 'dashboard.js')


@app.route('/api/data/multi_civilization_statistics.csv')
def get_simulation_data():
    """API endpoint for simulation statistics data."""
    data = prepare_simulation_data()
    return jsonify(data)


@app.route('/api/data/multi_civilization_events.csv')
def get_event_data():
    """API endpoint for event data."""
    data = prepare_event_data()
    return jsonify(data)


@app.route('/api/data/multi_civilization_stability.csv')
def get_stability_data():
    """API endpoint for stability metrics."""
    data = prepare_stability_data()
    return jsonify(data)


# Now the script just serves the existing files without trying to create new ones
if __name__ == '__main__':
    print(f"Data directory: {data_dir}")
    print(f"Dashboard directory: {dashboard_dir}")

    # Check if data files exist
    stats_file = data_dir / "multi_civilization_statistics.csv"
    events_file = data_dir / "multi_civilization_events.csv"
    stability_file = data_dir / "multi_civilization_stability.csv"

    if not (stats_file.exists() and events_file.exists() and stability_file.exists()):
        print("One or more data files not found.")
        if args.auto_run:
            print("Auto-run enabled, running a new simulation...")
            run_new_simulation()
        else:
            print("Creating placeholder data...")
            create_placeholder_data()

    print(f"Starting dashboard server on http://{args.host}:{args.port}")
    print("Dashboard will serve files from:", dashboard_dir)
    print("Data will be read from:", data_dir)
    app.run(host=args.host, port=args.port, debug=True)