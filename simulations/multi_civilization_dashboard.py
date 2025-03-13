import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import os
import argparse
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Create argument parser
parser = argparse.ArgumentParser(description='Launch interactive dashboard for multi-civilization simulations')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the dashboard server on')
parser.add_argument('--port', type=int, default=5000, help='Port to run the dashboard server on')
parser.add_argument('--data-dir', type=str, default=None, help='Directory containing simulation data')
parser.add_argument('--auto-run', action='store_true', help='Automatically run a simulation if no data is found')
args = parser.parse_args()

# Ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = Path(args.data_dir) if args.data_dir else BASE_DIR / 'outputs' / 'data'
dashboard_dir = BASE_DIR / 'outputs' / 'dashboard'
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


def create_dashboard_html():
    """
    Create the HTML file for the dashboard.
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Civilization Simulation Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/react@17.0.2/umd/react.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/react-dom@17.0.2/umd/react-dom.production.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/recharts@2.1.9/umd/Recharts.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.16.4/babel.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f0f2f5;
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel" src="dashboard.js"></script>
</body>
</html>
    """

    with open(dashboard_dir / 'index.html', 'w') as f:
        f.write(html_content)

    print(f"Dashboard HTML file created at {dashboard_dir / 'index.html'}")


def create_dashboard_js():
    """
    Create the JavaScript file for the dashboard with the React component.
    """
    # This is a placeholder - in a real implementation, you would
    # use the full React component code from the artifact
    js_content = """
const { useState, useEffect } = React;
const { 
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    BarChart, Bar, ScatterChart, Scatter, ZAxis, Cell, PieChart, Pie
} = Recharts;

const MultiCivilizationDashboard = () => {
    const [simulationData, setSimulationData] = useState(null);
    const [eventData, setEventData] = useState(null);
    const [stabilityData, setStabilityData] = useState(null);
    const [selectedTimeStep, setSelectedTimeStep] = useState(0);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [view, setView] = useState('overview');

    // Load data on component mount
    useEffect(() => {
        const loadData = async () => {
            try {
                // Fetch simulation data
                const statsResponse = await fetch('/api/data/multi_civilization_statistics.csv');
                const statsData = await statsResponse.json();
                setSimulationData(statsData);

                // Fetch event data
                const eventsResponse = await fetch('/api/data/multi_civilization_events.csv');
                const eventsData = await eventsResponse.json();
                setEventData(eventsData);

                // Fetch stability data
                const stabilityResponse = await fetch('/api/data/multi_civilization_stability.csv');
                const stabilityData = await stabilityResponse.json();
                setStabilityData(stabilityData);

                setLoading(false);
            } catch (err) {
                console.error('Error loading data:', err);
                setError('Failed to load simulation data');
                setLoading(false);
            }
        };

        loadData();
    }, []);

    if (loading) {
        return <div className="flex justify-center items-center h-screen">Loading simulation data...</div>;
    }

    if (error) {
        return <div className="flex justify-center items-center h-screen text-red-600">{error}</div>;
    }

    // Main return statement with UI structure
    return (
        <div className="container mx-auto p-4">
            <h1 className="text-2xl font-bold mb-4">Multi-Civilization Simulation Dashboard</h1>
            <p>Data loaded successfully. Replace this component with the full dashboard implementation.</p>
            <pre className="bg-gray-100 p-4 mt-4 rounded max-h-96 overflow-auto">
                {JSON.stringify({simulationData: simulationData?.slice(0, 2), eventData: eventData?.slice(0, 2)}, null, 2)}
            </pre>
        </div>
    );
};

ReactDOM.render(<MultiCivilizationDashboard />, document.getElementById('root'));
    """

    with open(dashboard_dir / 'dashboard.js', 'w') as f:
        f.write(js_content)

    print(f"Dashboard JS file created at {dashboard_dir / 'dashboard.js'}")
    print("NOTE: This is a placeholder. Replace with the full React component code.")


if __name__ == '__main__':
    print(f"Data directory: {data_dir}")

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

    # Create dashboard files
    create_dashboard_html()
    create_dashboard_js()

    print(f"Starting dashboard server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)