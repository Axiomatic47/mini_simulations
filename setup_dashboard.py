"""
Dashboard Setup Script
Standardizes the dashboard files structure and ensures consistent naming.
"""
import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard-setup')

# Define directories
BASE_DIR = Path(__file__).resolve().parent
dashboard_dir = BASE_DIR / 'outputs' / 'dashboard'
data_dir = BASE_DIR / 'outputs' / 'data'

# Ensure directories exist
dashboard_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Backup existing files
backup_dir = dashboard_dir / 'backup'
backup_dir.mkdir(exist_ok=True)

logger.info("Backing up existing dashboard files...")
for file in dashboard_dir.glob('*.js'):
    if file.name != 'dashboard.js':  # Don't backup the target file
        backup_path = backup_dir / file.name
        shutil.copy2(file, backup_path)
        logger.info(f"Backed up {file.name} to {backup_path}")

for file in dashboard_dir.glob('*.html'):
    if file.name != 'index.html':  # Don't backup the main HTML file
        backup_path = backup_dir / file.name
        shutil.copy2(file, backup_path)
        logger.info(f"Backed up {file.name} to {backup_path}")

# Step 1: Rename dashboard-component.js to dashboard.js
component_path = dashboard_dir / 'dashboard-component.js'
dashboard_path = dashboard_dir / 'dashboard.js'

if component_path.exists():
    # Backup existing dashboard.js if it exists
    if dashboard_path.exists():
        backup_path = backup_dir / 'dashboard.js.bak'
        shutil.copy2(dashboard_path, backup_path)
        logger.info(f"Backed up existing dashboard.js to {backup_path}")

    # Copy the content (using copy instead of rename to preserve the original)
    shutil.copy2(component_path, dashboard_path)
    logger.info(f"Copied dashboard-component.js to dashboard.js")
else:
    logger.warning("dashboard-component.js not found. Please make sure it exists.")

# Step 2: Create a standardized index.html
logger.info("Creating standardized index.html")
index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Civilization Simulation Dashboard</title>
    <!-- Tailwind CSS for styling -->
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">

    <!-- React dependencies -->
    <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>

    <!-- Recharts library -->
    <script src="https://unpkg.com/recharts@2.5.0/umd/Recharts.js"></script>

    <!-- Babel for JSX support -->
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f0f2f5;
        }

        /* Loading indicator */
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-size: 2rem;
            color: #4a5568;
        }

        /* Error message */
        .error-message {
            background-color: #fed7d7;
            color: #9b2c2c;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 2rem;
        }
    </style>
</head>
<body>
    <!-- Root element where React will render the dashboard -->
    <div id="root">
        <div class="loading">Loading dashboard...</div>
    </div>

    <!-- Make Recharts available globally for the dashboard component -->
    <script>
        // Make sure Recharts is globally available
        window.Recharts = Recharts;

        // Add error handling for script loading
        window.addEventListener('error', function(e) {
            if (e.target.tagName === 'SCRIPT') {
                console.error('Error loading script:', e.target.src);
                document.getElementById('root').innerHTML = `
                    <div class="error-message">
                        <h2 class="text-xl font-bold mb-2">Error Loading Dashboard</h2>
                        <p>Failed to load required script: ${e.target.src}</p>
                        <p class="mt-2">Please check the console for more details.</p>
                    </div>
                `;
            }
        }, true);
    </script>

    <!-- Load the dashboard component -->
    <script type="text/babel" src="/dashboard.js"></script>
</body>
</html>
"""

with open(dashboard_dir / 'index.html', 'w') as f:
    f.write(index_html)
    logger.info("Updated index.html with standardized content")

# Step 3: Create a README.md file in the dashboard directory
readme_content = """# Multi-Civilization Simulation Dashboard

This directory contains the frontend files for the multi-civilization simulation dashboard.

## File Structure

- `index.html`: The main HTML file that loads the React-based dashboard
- `dashboard.js`: The React/Recharts component that renders the visualization

## Technology Stack

- **React**: Frontend library for building the user interface
- **Recharts**: Charting library for visualization
- **Tailwind CSS**: Utility-first CSS framework for styling

## Development

To modify or extend the dashboard:

1. Edit `dashboard.js` to change the visualization components
2. Run the Flask server using `python multi_civilization_dashboard.py` from the parent directory
3. View the dashboard at http://127.0.0.1:5000

## Data Sources

The dashboard fetches data from these endpoints:

- `/api/data/multi_civilization_statistics.csv`: Time-series statistics data
- `/api/data/multi_civilization_events.csv`: Events that occurred during the simulation
- `/api/data/multi_civilization_stability.csv`: Overall stability metrics

## Handling NaN Values

The server automatically converts NaN values to null in the JSON response to ensure proper parsing.
"""

with open(dashboard_dir / 'README.md', 'w') as f:
    f.write(readme_content)
    logger.info("Created README.md with documentation")

logger.info("Dashboard setup complete!")
logger.info(f"Dashboard files are now standardized in {dashboard_dir}")
logger.info("Next step: Update the server code with multi_civilization_dashboard.py")