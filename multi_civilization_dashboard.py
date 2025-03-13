import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from flask import Flask, send_from_directory, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dashboard')

# Create argument parser
parser = argparse.ArgumentParser(description='Launch dashboard for multi-civilization simulations')
parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run on')
parser.add_argument('--port', type=int, default=5000, help='Port to run on')
parser.add_argument('--data-dir', type=str, default=None, help='Directory with simulation data')
args = parser.parse_args()

# Ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent
data_dir = Path(args.data_dir) if args.data_dir else BASE_DIR / 'outputs' / 'data'
dashboard_dir = BASE_DIR / 'outputs' / 'dashboard'
data_dir.mkdir(parents=True, exist_ok=True)
dashboard_dir.mkdir(parents=True, exist_ok=True)

# Create Flask app
app = Flask(__name__)

# Define API routes
@app.route('/')
def serve_dashboard():
    # Serve the dashboard HTML page
    return send_from_directory(dashboard_dir, 'index.html')

@app.route('/dashboard.js')
def serve_dashboard_js():
    # Serve the dashboard JavaScript file
    return send_from_directory(dashboard_dir, 'dashboard.js')

@app.route('/chart-dashboard.js')
def serve_chart_dashboard_js():
    # Serve the Chart.js dashboard file
    return send_from_directory(dashboard_dir, 'chart-dashboard.js')

@app.route('/api/data/multi_civilization_statistics.csv')
def get_simulation_data():
    # API endpoint for simulation statistics data
    try:
        df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")
        # Replace NaN with None (becomes null in JSON)
        json_data = df.replace({np.nan: None}).to_dict(orient='records')
        return jsonify(json_data)
    except Exception as e:
        logger.error(f"Error serving statistics data: {e}")
        return jsonify([])

@app.route('/api/data/multi_civilization_events.csv')
def get_event_data():
    # API endpoint for event data
    try:
        df = pd.read_csv(data_dir / "multi_civilization_events.csv")
        # Replace NaN with None (becomes null in JSON)
        json_data = df.replace({np.nan: None}).to_dict(orient='records')
        return jsonify(json_data)
    except Exception as e:
        logger.error(f"Error serving events data: {e}")
        return jsonify([])

@app.route('/api/data/multi_civilization_stability.csv')
def get_stability_data():
    # API endpoint for stability metrics
    try:
        df = pd.read_csv(data_dir / "multi_civilization_stability.csv")
        # Replace NaN with None (becomes null in JSON)
        json_data = df.iloc[0].replace({np.nan: None}).to_dict()
        return jsonify(json_data)
    except Exception as e:
        logger.error(f"Error serving stability data: {e}")
        return jsonify({})

if __name__ == '__main__':
    print(f"Starting dashboard server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
