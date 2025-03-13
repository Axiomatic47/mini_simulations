# Multi-Civilization Simulation Dashboard

An interactive visualization dashboard for exploring multi-civilization simulation data from the Axiomatic Intelligence Growth Simulation Framework.

![Dashboard Preview](https://via.placeholder.com/800x450?text=Multi-Civilization+Dashboard)

## Overview

This dashboard provides real-time interactive visualization capabilities for exploring simulation data. It enables researchers and analysts to:

- Track civilization dynamics over time
- Analyze knowledge and suppression patterns
- Monitor key events (collisions, mergers, collapses)
- Assess simulation stability metrics
- Explore correlations between different variables

## Installation

### Prerequisites

- Python 3.7+
- Flask
- Pandas
- NumPy
- Modern web browser with JavaScript enabled

### Setup

1. Clone this repository or download the source code
2. Install required Python packages:

```bash
pip install flask flask-cors pandas numpy
```

3. Make sure your simulation data is in the expected location (`outputs/data/`) or specify a custom location with the `--data-dir` parameter

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
- `--auto-run`: Automatically run a new simulation if no data is found

Example:
```bash
python multi_civilization_dashboard.py --port 8080 --auto-run
```

## Features

### Interactive Time Navigation

- **Time Slider**: Navigate through simulation timesteps
- **Current Stats Panel**: Shows statistics for the selected time point
- **Timeline Charts**: Display trends across all timesteps

### Multiple Views

1. **Overview**:
   - Civilization count over time
   - Knowledge & suppression metrics
   - Event distribution
   - Civilization positions

2. **Civilizations**:
   - Knowledge range (min, mean, max)
   - Suppression range (min, mean, max)
   - Intelligence & Truth growth
   - Resource allocation

3. **Events**:
   - Event type distribution
   - Event timeline
   - Time-period analysis
   - Recent events table

4. **Stability**:
   - Stability issues over time
   - Adaptive timestep behavior
   - Key stability metrics

## Data Format

The dashboard expects three CSV files in your data directory:

1. **multi_civilization_statistics.csv**: Time-series data for civilization statistics
   - Required columns: Time, Civilization_Count, knowledge_mean, knowledge_max, knowledge_min, suppression_mean, suppression_max, suppression_min, intelligence_mean, truth_mean, resources_total, Stability_Issues, Timestep

2. **multi_civilization_events.csv**: Event data from the simulation
   - Required columns: id, type, time, civ_id1, civ_id2, position_x, position_y, description

3. **multi_civilization_stability.csv**: Overall stability metrics
   - Expected fields: Total_Stability_Issues, Circuit_Breaker_Triggers, Max_Knowledge, Max_Suppression, Max_Intelligence, Max_Truth, Total_Collisions, Total_Mergers, Total_Collapses, Total_Spawns, Total_New_Civilizations, Used_Dimensional_Analysis

## System Architecture

The dashboard follows a client-server architecture:

### Backend (Flask)

- Reads CSV simulation data and serves it via REST API endpoints
- Handles data processing and transformation
- Provides HTTP endpoints for the frontend

### Frontend (Chart.js)

- Fetches data from the API endpoints
- Renders interactive visualizations
- Handles user interaction and timeline navigation

## Folder Structure

```
/
├── multi_civilization_dashboard.py   # Main dashboard server
├── outputs/
│   ├── data/                         # Simulation data (CSV files)
│   │   ├── multi_civilization_statistics.csv
│   │   ├── multi_civilization_events.csv
│   │   └── multi_civilization_stability.csv
│   └── dashboard/                    # Frontend files
│       ├── index.html
│       └── dashboard.js
├── simulations/                      # Simulation code
│   └── multi_civilization_simulation.py
└── README.md
```

## Customization

### Modifying the Dashboard

The dashboard files are in the `outputs/dashboard/` directory:

- `index.html`: HTML structure and layout
- `dashboard.js`: Visualization logic and data handling

### Changing Colors

You can modify the color scheme by editing the `colorMap` and `eventColorMap` objects in `dashboard.js`.

### Adding New Charts

To add a new chart:

1. Add an HTML container in `index.html`
2. Create the chart initialization in `initializeCharts()` in `dashboard.js`
3. Add any necessary data transformation
4. Update the chart in the `updateCharts()` function

## Troubleshooting

### Dashboard Shows No Data

**Possible causes**:
- CSV files don't exist in the expected location
- CSV files have incorrect format or column names
- API endpoints returning errors

**Solutions**:
- Check console for error messages
- Verify CSV files exist in the data directory
- Validate CSV format with pandas

### Charts Not Rendering

**Possible causes**:
- JavaScript errors
- Chart.js library not loading
- Data format incompatible with chart configuration

**Solutions**:
- Check browser console for JavaScript errors
- Verify Chart.js is properly loaded
- Test with simplified data to isolate the issue

## Development Notes

### Backend Technology

The backend uses Flask to serve:
- Static files (HTML, JS)
- Data via JSON API endpoints

### Frontend Technology

The frontend uses:
- Chart.js for data visualization
- Bootstrap for responsive layout
- Vanilla JavaScript for interactivity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Chart.js - For the visualization library
- Bootstrap - For the responsive UI framework
- Flask - For the lightweight Python server