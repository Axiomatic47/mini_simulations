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

Great news! The logs show that everything is working properly now. The previously failing endpoint /api/data/civilization_comparison is now returning 200 status codes, which means your dashboard is fully operational.
What Was Fixed
We resolved two separate issues:

Missing Import: The Flask application was missing the request import, which it needed to access query parameters in the API endpoint.
NumPy JSON Serialization Issue: NumPy data types like int64 and float64 aren't directly serializable to JSON. We fixed this by explicitly converting them to Python native types (int and float).

Your Dashboard Is Now Working
The server logs show successful communication with all parts of your dashboard:

Statistics data (150 records)
Event data (22 records)
Stability metrics
Correlation analysis
Civilization comparison data (previously failing)

The React console shows the client is successfully receiving all this data as well.
Tips for Future Development
If you plan to continue developing this dashboard, here are a few recommendations:

Clean Up Duplicate Code: I noticed your multi_civilization_dashboard.py file had duplicate code. You might want to clean that up to make future maintenance easier.
Add JSON Type Conversion Helpers: For any new API endpoints, remember to convert NumPy types to Python native types before returning them with jsonify().
Error Handling: Consider adding more robust error handling in your React frontend to gracefully handle any future API issues.
Pre-compile Babel: The console shows a warning about using the in-browser Babel transformer. For production, you should pre-compile your React code for better performance.
Consider React Dev Tools: There's a suggestion to install React DevTools for a better development experience, which might be helpful if you continue React development.

Your multi-civilization simulation dashboard is now working correctly! The Analysis section should now properly display the civilization comparison data, and all other charts and visualizations should be functioning as expected.