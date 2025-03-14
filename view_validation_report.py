#!/usr/bin/env python
"""
Script to set up and run the multi-civilization dashboard server.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard-setup')


def setup_dashboard():
    """Set up the dashboard files."""
    logger.info("Setting up dashboard files...")

    # Paths
    base_dir = Path.cwd()
    dashboard_dir = base_dir / "outputs" / "dashboard"
    os.makedirs(dashboard_dir, exist_ok=True)

    # Copy the necessary files to the dashboard directory
    sources = [
        ("index.html", base_dir / "index.html"),
        ("dashboard.js", base_dir / "dashboard.js"),
        ("ErrorBoundary.js", base_dir / "ErrorBoundary.js")
    ]

    for filename, source_path in sources:
        if source_path.exists():
            target_path = dashboard_dir / filename
            logger.info(f"Copying {source_path} to {target_path}")
            shutil.copy2(source_path, target_path)
        else:
            logger.warning(f"Source file {source_path} does not exist")

    logger.info("Dashboard files setup complete")
    return True


def run_dashboard_server():
    """Run the multi-civilization dashboard server."""
    try:
        logger.info("Starting dashboard server...")

        # Check if Python module exists
        dashboard_script = Path("multi_civilization_dashboard.py")
        if not dashboard_script.exists():
            logger.error(f"Dashboard script not found: {dashboard_script}")
            return False

        # Run the dashboard server
        cmd = [sys.executable, "multi_civilization_dashboard.py", "--host", "127.0.0.1", "--port", "5000", "--debug"]
        logger.info(f"Running command: {' '.join(cmd)}")

        # Using subprocess.Popen to run the server in the background
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Capture initial output to check for startup errors
        import time
        time.sleep(2)  # Wait for server to start

        # Check if the server is running
        if server_process.poll() is not None:
            # Server has already terminated
            stdout, stderr = server_process.communicate()
            logger.error(f"Dashboard server failed to start.\nStdout: {stdout}\nStderr: {stderr}")
            return False

        logger.info("Dashboard server started successfully")

        # Open the dashboard in a web browser
        import webbrowser
        dashboard_url = "http://127.0.0.1:5000"
        logger.info(f"Opening dashboard at {dashboard_url}")
        webbrowser.open(dashboard_url)

        logger.info("Dashboard is now accessible in your web browser")

        # Keep script running to keep the server alive
        print("\nDashboard server is running.")
        print("Press Ctrl+C to stop the server and exit.")

        # Wait for the server process to complete
        try:
            stdout, stderr = server_process.communicate()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping server...")
            server_process.terminate()
            server_process.wait()
            logger.info("Server stopped")

        return True

    except Exception as e:
        logger.error(f"Error running dashboard server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    print("Setting up and running the Multi-Civilization Simulation Dashboard...")
    setup_success = setup_dashboard()

    if setup_success:
        run_dashboard_server()
    else:
        logger.error("Failed to set up dashboard files")
        sys.exit(1)