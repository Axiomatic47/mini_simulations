# Updated README.md for Multi-Civilization Simulation Dashboard

```markdown
# Multi-Civilization Simulation Dashboard

An interactive React-based visualization dashboard for exploring multi-civilization simulation data from the Axiomatic Intelligence Growth Simulation Framework.

## Overview

This dashboard provides real-time interactive visualization capabilities for exploring simulation data. It enables researchers and analysts to:

- Track civilization dynamics over time
- Analyze knowledge and suppression patterns
- Monitor key events (collisions, mergers, collapses)
- Assess simulation stability metrics
- Explore correlations between different variables

## File Structure

```
/
├── multi_civilization_dashboard.py  (Flask server)
├── outputs/
│   ├── data/                        (CSV data files)
│   │   ├── multi_civilization_statistics.csv
│   │   ├── multi_civilization_events.csv
│   │   └── multi_civilization_stability.csv
│   └── dashboard/                   (Frontend files)
│       ├── index.html               (HTML with React setup)
│       └── dashboard.js             (React/Recharts component)
```

## Technology Stack

- **Backend**: Flask server for API endpoints and serving static files
- **Frontend**:
  - **React**: Component-based UI framework
  - **Recharts**: React charting library for data visualization
  - **Tailwind CSS**: Utility-first CSS framework

## Installation

### Prerequisites

- Python 3.7+
- Flask
- Pandas
- NumPy
- Modern web browser with JavaScript enabled

### Setup

1. Make sure your simulation data is located in `outputs/data/` or specify a custom location with the `--data-dir` parameter
2. Run the initialization script to set up the dashboard files:
   ```bash
   python react_fix.py
   ```

## Usage

### Starting the Dashboard

Run the dashboard server:

```bash
python multi_civilization_dashboard.py
```

Then open your browser and navigate to: http://127.0.0.1:5000

### Command-Line Options

- `--host`: Specify the host to run the server on (default: 127.0.0.1)
- `--port`: Specify the port to run the server on (default: 5000)
- `--data-dir`: Specify a custom directory for simulation data
- `--debug`: Run in debug mode for development

Example:
```bash
python multi_civilization_dashboard.py --port 8080 --debug
```

## Features

### Interactive Time Navigation

- **Time Slider**: Navigate through simulation timesteps
- **Current Stats Panel**: Shows statistics for the selected time point
- **Multiple Visualization Views**: Overview, Civilizations, Events, Stability, and Analysis

### Advanced Analysis

- **Knowledge-Suppression Correlation**: Visualize and analyze correlations between key metrics
- **Civilization Comparison**: Compare top and bottom civilizations based on knowledge, suppression, etc.
- **Event Timeline Analysis**: Track important events throughout simulation history

### Data Filtering

- **Time Range Selection**: Focus on specific periods in the simulation
- **Data Downsampling**: Improve performance for large datasets
- **Metric Selection**: Choose which metrics to display on charts

## Development

To modify or extend the dashboard:

1. Edit `dashboard.js` to change the visualization components
2. Run the Flask server to see your changes
3. The React component will automatically reload when you refresh the page

## Data Format

The dashboard expects three CSV files in your data directory:

1. **multi_civilization_statistics.csv**: Time-series data for civilization statistics
2. **multi_civilization_events.csv**: Event data from the simulation
3. **multi_civilization_stability.csv**: Overall stability metrics

## Troubleshooting

### Dashboard Shows No Data

**Possible causes**:
- CSV files don't exist in the expected location
- API endpoints returning errors
- JSON parsing issues with NaN values

**Solutions**:
- Check browser console for error messages
- Verify CSV files exist in the data directory
- Make sure the server is properly replacing NaN values with null

### Charts Not Rendering

**Possible causes**:
- React or Recharts libraries not loading
- JavaScript errors in the dashboard component

**Solutions**:
- Check browser console for JavaScript errors
- Verify all required libraries are loaded in index.html
- Use the react_fix.py script to restore the dashboard setup

### API Endpoint Errors (500 Internal Server Error)

**Possible causes**:
- Missing Flask dependencies
- Errors in server-side code

**Solutions**:
- Check server logs for error details
- Verify all necessary imports are present (including `request` from Flask)
- Run in debug mode with `--debug` flag for more detailed error information
```

I also notice you're getting a 500 error with the civilization comparison endpoint. Let me provide a quick fix script (`react_fix.py`) to address both the frontend React setup and the backend Flask issue:

```python
#!/usr/bin/env python
"""
React Dashboard Fix Script for Multi-Civilization Simulation
This script fixes common issues with the React dashboard implementation and
ensures the Flask backend has all necessary imports.
"""
import os
import shutil
from pathlib import Path

# Ensure directory structure
BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BASE_DIR / 'outputs' / 'dashboard'
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Create the fixed index.html
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Civilization Simulation Dashboard</title>
    
    <!-- React Core -->
    <script crossorigin src="https://unpkg.com/react@17/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
    
    <!-- Required dependency for Recharts -->
    <script crossorigin src="https://unpkg.com/prop-types@15.7.2/prop-types.min.js"></script>
    <!-- D3 (dependency for Recharts) -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    
    <!-- Recharts -->
    <script src="https://unpkg.com/recharts@2.1.16/umd/Recharts.js"></script>
    
    <!-- Babel -->
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    
    <!-- Basic styles -->
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 20px;
        }
        .chart-container {
            height: 300px;
            width: 100%;
        }
    </style>
</head>
<body>
    <div id="root"></div>
    
    <!-- Dashboard React Component -->
    <script type="text/babel" src="/dashboard.js"></script>
</body>
</html>
"""

# Write the index.html file
with open(DASHBOARD_DIR / 'index.html', 'w') as f:
    f.write(INDEX_HTML)
print(f"✅ Fixed index.html written to {DASHBOARD_DIR / 'index.html'}")

# Fix Flask backend imports
FLASK_BACKEND = BASE_DIR / 'multi_civilization_dashboard.py'
if FLASK_BACKEND.exists():
    with open(FLASK_BACKEND, 'r') as f:
        content = f.read()
    
    # Check if 'request' is imported from Flask
    if 'from flask import' in content and 'request' not in content:
        # Add request to imports
        content = content.replace(
            'from flask import Flask, send_from_directory, jsonify',
            'from flask import Flask, send_from_directory, jsonify, request'
        )
        
        # Backup original file
        shutil.copy(FLASK_BACKEND, FLASK_BACKEND.with_suffix('.py.bak'))
        
        # Write updated content
        with open(FLASK_BACKEND, 'w') as f:
            f.write(content)
        print(f"✅ Added missing 'request' import to Flask backend")
    else:
        print("ℹ️ Flask backend imports look good")
else:
    print("⚠️ Could not find Flask backend file")

print("\nReact dashboard setup complete! Start the server with:")
print("python multi_civilization_dashboard.py --debug")
```

This script will fix the React setup and ensure the Flask backend has the necessary imports. It looks like the error you're seeing with the civilization comparison endpoint might be due to the missing `request` import.