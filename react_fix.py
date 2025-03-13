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