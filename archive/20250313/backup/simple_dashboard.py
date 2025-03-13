import os
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, send_from_directory, jsonify

# Ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / 'outputs' / 'data'
dashboard_dir = BASE_DIR / 'outputs' / 'dashboard'
data_dir.mkdir(parents=True, exist_ok=True)
dashboard_dir.mkdir(parents=True, exist_ok=True)

# Create Flask app
app = Flask(__name__)


# Create placeholder data if needed
def create_placeholder_data():
    """Create mock data for testing."""
    print("Creating placeholder data...")

    # Create mock simulation statistics
    timesteps = 150
    simulation_data = []

    for t in range(timesteps):
        # Add random fluctuations
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
    print("Placeholder data created successfully.")


# Create static files
def create_static_files():
    """Create HTML and JS files."""

    # Create HTML file
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Civilization Simulation Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/recharts@2.1.16/umd/Recharts.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
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

    print(f"HTML file created at {dashboard_dir / 'index.html'}")

    # The JS content is too large to include here
    # We'll assume it's saved in a file


# Define API routes
@app.route('/')
def serve_index():
    """Serve the dashboard HTML page."""
    return send_from_directory(dashboard_dir, 'index.html')


@app.route('/dashboard.js')
def serve_dashboard_js():
    """Serve the dashboard JavaScript file."""
    return send_from_directory(dashboard_dir, 'dashboard.js')


@app.route('/api/data/multi_civilization_statistics.csv')
def get_simulation_data():
    """API endpoint for simulation statistics data."""
    try:
        df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")
        return jsonify(df.to_dict('records'))
    except Exception as e:
        print(f"Error serving statistics data: {e}")
        return jsonify([])


@app.route('/api/data/multi_civilization_events.csv')
def get_event_data():
    """API endpoint for event data."""
    try:
        df = pd.read_csv(data_dir / "multi_civilization_events.csv")
        return jsonify(df.to_dict('records'))
    except Exception as e:
        print(f"Error serving events data: {e}")
        return jsonify([])


@app.route('/api/data/multi_civilization_stability.csv')
def get_stability_data():
    """API endpoint for stability metrics."""
    try:
        df = pd.read_csv(data_dir / "multi_civilization_stability.csv")
        return jsonify(df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error serving stability data: {e}")
        return jsonify({})


if __name__ == '__main__':
    # Check if data files exist
    stats_file = data_dir / "multi_civilization_statistics.csv"
    events_file = data_dir / "multi_civilization_events.csv"
    stability_file = data_dir / "multi_civilization_stability.csv"

    if not (stats_file.exists() and events_file.exists() and stability_file.exists()):
        print("Data files not found. Creating placeholder data...")
        create_placeholder_data()

    # Create static files
    create_static_files()

    # Copy dashboard.js from this directory if it exists
    js_file = Path("dashboard.js")
    if js_file.exists():
        print(f"Copying dashboard.js to {dashboard_dir}")
        with open(js_file, 'r') as src:
            with open(dashboard_dir / 'dashboard.js', 'w') as dst:
                dst.write(src.read())
    else:
        print("dashboard.js not found in current directory.")
        print("Please make sure dashboard.js is in the same directory as this script.")

    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True)