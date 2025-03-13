# Multi-Civilization Simulation Dashboard

An interactive React-based visualization dashboard for exploring multi-civilization simulation data.

## File Structure

- `index.html`: The main HTML file that loads the React-based dashboard
- `dashboard.js`: The React/Recharts component that renders the visualization

## Technology Stack

- **React**: Frontend library for building the user interface
- **Recharts**: Charting library for visualization
- **Tailwind CSS**: Utility-first CSS framework for styling

## Data Sources

The dashboard fetches data from these endpoints:

- `/api/data/multi_civilization_statistics.csv`: Time-series statistics data
- `/api/data/multi_civilization_events.csv`: Events that occurred during the simulation
- `/api/data/multi_civilization_stability.csv`: Overall stability metrics

## Handling NaN Values

The server automatically converts NaN values to null in the JSON response to ensure proper parsing.
