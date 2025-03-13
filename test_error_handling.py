#!/usr/bin/env python
"""
Error Handling Test Script for Multi-Civilization Dashboard

This script automatically tests the error handling capabilities of both
the Flask backend and React frontend, then restores the system to its
original state afterward.

Usage:
    python test_error_handling.py
"""

import os
import sys
import time
import shutil
import logging
import subprocess
import requests
import signal
import json
import re
import threading
import tempfile
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'error_handling_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('error_test')

# Configuration
BASE_DIR = Path.cwd()
FLASK_APP = BASE_DIR / "multi_civilization_dashboard.py"
DATA_DIR = BASE_DIR / "outputs" / "data"
DASHBOARD_DIR = BASE_DIR / "outputs" / "dashboard"
DASHBOARD_JS = DASHBOARD_DIR / "dashboard.js"
BACKUP_DIR = BASE_DIR / "test_backups"
SERVER_URL = "http://127.0.0.1:5000"
SERVER_PROCESS = None

# Ensure backup directory exists
BACKUP_DIR.mkdir(exist_ok=True, parents=True)

# Test results storage
test_results = {
    "backend_tests": {},
    "frontend_tests": {},
    "passed": 0,
    "failed": 0,
    "total": 0
}


# ==================== Helper Functions ====================

def create_backup(file_path):
    """Create a backup of a file."""
    backup_path = BACKUP_DIR / file_path.name
    logger.info(f"Backing up {file_path} to {backup_path}")
    shutil.copy2(file_path, backup_path)
    return backup_path


def restore_from_backup(original_path, backup_path):
    """Restore a file from its backup."""
    logger.info(f"Restoring {original_path} from {backup_path}")
    shutil.copy2(backup_path, original_path)


def start_flask_server():
    """Start the Flask server in debug mode."""
    global SERVER_PROCESS
    logger.info("Starting Flask server...")

    # Start server as a subprocess
    SERVER_PROCESS = subprocess.Popen(
        [sys.executable, str(FLASK_APP), "--debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for server to start
    time.sleep(3)

    # Check if server is running
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            logger.info("Flask server started successfully")
            return True
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Flask server")
        return False


def stop_flask_server():
    """Stop the Flask server."""
    global SERVER_PROCESS
    if SERVER_PROCESS:
        logger.info("Stopping Flask server...")
        try:
            # Send SIGTERM signal on Unix
            if os.name != 'nt':  # Not Windows
                SERVER_PROCESS.send_signal(signal.SIGTERM)
            else:
                # On Windows
                import ctypes
                ctypes.windll.kernel32.TerminateProcess(
                    int(SERVER_PROCESS._handle), 1
                )
        except:
            # Fallback to terminate
            SERVER_PROCESS.terminate()

        SERVER_PROCESS.wait(timeout=5)
        logger.info("Flask server stopped")


def modify_flask_app(modification, restore=False):
    """Modify the Flask app with a specific test case."""
    if restore:
        # Restore from backup
        backup_path = BACKUP_DIR / FLASK_APP.name
        restore_from_backup(FLASK_APP, backup_path)
        return

    # Backup original file
    create_backup(FLASK_APP)

    # Read the file content
    with open(FLASK_APP, 'r') as f:
        content = f.read()

    # Apply modification
    modified_content = modification(content)

    # Write modified content
    with open(FLASK_APP, 'w') as f:
        f.write(modified_content)


def modify_dashboard_js(modification, restore=False):
    """Modify the dashboard.js file with a specific test case."""
    if restore:
        # Restore from backup
        backup_path = BACKUP_DIR / DASHBOARD_JS.name
        restore_from_backup(DASHBOARD_JS, backup_path)
        return

    # Backup original file
    create_backup(DASHBOARD_JS)

    # Read the file content
    with open(DASHBOARD_JS, 'r') as f:
        content = f.read()

    # Apply modification
    modified_content = modification(content)

    # Write modified content
    with open(DASHBOARD_JS, 'w') as f:
        f.write(modified_content)


def make_api_request(endpoint, expected_status=200):
    """Make a request to the API and check the status code."""
    try:
        url = f"{SERVER_URL}{endpoint}"
        logger.info(f"Making API request to {url}")
        response = requests.get(url)
        if response.status_code == expected_status:
            logger.info(f"API request successful: {response.status_code}")
            return True, response
        else:
            logger.error(f"API request failed: {response.status_code} (expected {expected_status})")
            return False, response
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {e}")
        return False, None


def rename_file(file_path, new_name):
    """Rename a file temporarily."""
    new_path = file_path.parent / new_name
    logger.info(f"Renaming {file_path} to {new_path}")
    os.rename(file_path, new_path)
    return new_path


def record_test_result(category, test_name, passed, message):
    """Record a test result."""
    test_results[category][test_name] = {
        "passed": passed,
        "message": message
    }
    if passed:
        test_results["passed"] += 1
    else:
        test_results["failed"] += 1
    test_results["total"] += 1


# ==================== Test Functions ====================

def test_missing_files():
    """Test handling of missing files."""
    test_name = "missing_files"
    try:
        # Backup original file
        stats_file = DATA_DIR / "multi_civilization_statistics.csv"
        backup_path = create_backup(stats_file)

        # Rename file to simulate missing file
        temp_path = rename_file(stats_file, "multi_civilization_statistics.csv.temp")

        # Restart the server and check if it generates placeholder data
        stop_flask_server()
        start_flask_server()

        # Wait for placeholder data to be generated
        time.sleep(2)

        # Check if the file was recreated
        file_recreated = Path(stats_file).exists()

        if file_recreated:
            record_test_result("backend_tests", test_name, True,
                               "Successfully handled missing file by creating placeholder data")
        else:
            record_test_result("backend_tests", test_name, False,
                               "Failed to handle missing file: placeholder data not created")

        # Restore original file
        os.rename(temp_path, stats_file)

    except Exception as e:
        logger.error(f"Error in {test_name} test: {e}")
        record_test_result("backend_tests", test_name, False, f"Test error: {str(e)}")

        # Ensure file is restored
        try:
            if 'temp_path' in locals():
                os.rename(temp_path, stats_file)
        except:
            if 'backup_path' in locals():
                restore_from_backup(stats_file, backup_path)


def test_json_type_conversion():
    """Test NumPy JSON type conversion."""
    test_name = "json_type_conversion"

    def add_test_code(content):
        # Find get_civilization_comparison function
        match = re.search(
            r'@app\.route\(\'/api/data/civilization_comparison\'\)\s*def get_civilization_comparison\(\):', content)
        if not match:
            logger.error("Could not find civilization_comparison function")
            return content

        # Add test code after function definition
        insert_pos = match.end()
        test_code = """
    # Test code for JSON type conversion
    try:
        import numpy as np
        logger.info("=== JSON TYPE CONVERSION TEST ===")
        test_array = np.array([1, 2, 3, 4, 5])
        test_float = np.float64(3.14159)
        test_int = np.int64(42)
        test_result = {
            'test_array': test_array,
            'test_float': test_float,
            'test_int': test_int,
            'nested': {'array': test_array}
        }

        # Log original types
        logger.info(f"Original types: {type(test_array)}, {type(test_float)}, {type(test_int)}")

        # Convert and log converted types
        converted = convert_numpy_types(test_result)
        logger.info(f"Converted types: {type(converted['test_array'])}, {type(converted['test_float'])}, {type(converted['test_int'])}")

        # Test if conversion was successful
        if (isinstance(converted['test_array'], list) and
            isinstance(converted['test_float'], float) and
            isinstance(converted['test_int'], int) and
            isinstance(converted['nested']['array'], list)):
            logger.info("JSON type conversion successful!")
            with open(f"{DATA_DIR}/type_conversion_test_result.txt", "w") as f:
                f.write("SUCCESS")
        else:
            logger.error("JSON type conversion failed!")
            with open(f"{DATA_DIR}/type_conversion_test_result.txt", "w") as f:
                f.write("FAILURE")
    except Exception as e:
        logger.error(f"Error in JSON type conversion test: {e}")
        with open(f"{DATA_DIR}/type_conversion_test_result.txt", "w") as f:
            f.write(f"ERROR: {str(e)}")
"""
        modified_content = content[:insert_pos] + test_code + content[insert_pos:]
        return modified_content

    try:
        # Delete test result file if it exists
        test_result_file = DATA_DIR / "type_conversion_test_result.txt"
        if test_result_file.exists():
            os.remove(test_result_file)

        # Modify Flask app to add test code
        modify_flask_app(add_test_code)

        # Restart the server
        stop_flask_server()
        start_flask_server()

        # Trigger the test by making a request
        success, _ = make_api_request("/api/data/civilization_comparison")

        # Wait for test to complete
        time.sleep(2)

        # Check test result
        if test_result_file.exists():
            with open(test_result_file, 'r') as f:
                result_content = f.read()

            if "SUCCESS" in result_content:
                record_test_result("backend_tests", test_name, True,
                                   "NumPy types were successfully converted to JSON-serializable types")
            else:
                record_test_result("backend_tests", test_name, False,
                                   f"NumPy type conversion failed: {result_content}")
        else:
            record_test_result("backend_tests", test_name, False,
                               "Test result file not found")

    except Exception as e:
        logger.error(f"Error in {test_name} test: {e}")
        record_test_result("backend_tests", test_name, False, f"Test error: {str(e)}")

    finally:
        # Restore Flask app
        modify_flask_app(None, restore=True)
        # Clean up test result file
        if test_result_file.exists():
            os.remove(test_result_file)


def test_api_error_response():
    """Test API error response handling."""
    test_name = "api_error_response"

    def add_error_to_endpoint(content):
        # Find get_simulation_data function
        match = re.search(
            r'@app\.route\(\'/api/data/multi_civilization_statistics.csv\'\)\s*def get_simulation_data\(\):', content)
        if not match:
            logger.error("Could not find get_simulation_data function")
            return content

        # Add error code after function definition
        insert_pos = match.end()
        error_code = """
    # Test code for API error response
    logger.info("=== API ERROR RESPONSE TEST ===")
    return json_response({"error": "This is a test error"}, 500)
"""
        modified_content = content[:insert_pos] + error_code + content[insert_pos:]
        return modified_content

    try:
        # Modify Flask app to add error code
        modify_flask_app(add_error_to_endpoint)

        # Restart the server
        stop_flask_server()
        start_flask_server()

        # Make request and expect 500 error
        success, response = make_api_request("/api/data/multi_civilization_statistics.csv", expected_status=500)

        # Check if response contains error message
        response_data = response.json() if response else {}
        error_present = "error" in response_data and response_data["error"] == "This is a test error"

        if success and error_present:
            record_test_result("backend_tests", test_name, True,
                               "API correctly returned error response with status 500")
        else:
            record_test_result("backend_tests", test_name, False,
                               f"API error response test failed: {response_data}")

    except Exception as e:
        logger.error(f"Error in {test_name} test: {e}")
        record_test_result("backend_tests", test_name, False, f"Test error: {str(e)}")

    finally:
        # Restore Flask app
        modify_flask_app(None, restore=True)


def test_error_boundary():
    """Test React ErrorBoundary component."""
    test_name = "error_boundary"

    def add_error_to_component(content):
        # Find the renderOverview function
        match = re.search(r'const renderOverview = \(\) => {', content)
        if not match:
            logger.error("Could not find renderOverview function")
            return content

        # Add error code at the beginning of the function
        insert_pos = match.end()
        error_code = """
    // ErrorBoundary test code
    console.log("=== ERRORBOUNDARY TEST ===");
    // Create a deliberate error
    const causeTestError = null;
    causeTestError.nonExistentMethod(); // This will cause a TypeError to test ErrorBoundary
"""
        modified_content = content[:insert_pos] + error_code + content[insert_pos:]
        return modified_content

    try:
        # Modify dashboard.js to add error code
        modify_dashboard_js(add_error_to_component)

        # Restart the server to serve modified JS
        stop_flask_server()
        start_flask_server()

        # Log test instructions for manual verification
        message = (
            "ErrorBoundary test requires manual verification:\n"
            "1. Open the dashboard in a browser\n"
            "2. Check if the Overview section shows the error boundary UI\n"
            "3. Other sections should still work\n"
            "This test has been marked as 'MANUAL VERIFICATION REQUIRED'"
        )
        logger.info(message)

        record_test_result("frontend_tests", test_name, None,
                           "Manual verification required - check if ErrorBoundary UI appears in Overview section")

    except Exception as e:
        logger.error(f"Error in {test_name} test: {e}")
        record_test_result("frontend_tests", test_name, False, f"Test error: {str(e)}")

    finally:
        # Restore dashboard.js
        modify_dashboard_js(None, restore=True)


def test_api_error_handling_frontend():
    """Test frontend handling of API errors."""
    test_name = "api_error_handling_frontend"

    def modify_api_utility(content):
        # Find the api object definition
        match = re.search(r'const api = {', content)
        if not match:
            logger.error("Could not find api object")
            return content

        # Find the getSimulationData method
        method_match = re.search(r'async getSimulationData\([^)]*\) {[^}]*}', content)
        if not method_match:
            logger.error("Could not find getSimulationData method")
            return content

        # Replace the method with one that always throws an error
        old_method = method_match.group(0)
        new_method = """async getSimulationData(params = {}) {
    console.log("=== API ERROR HANDLING TEST ===");
    throw new Error("This is a test API error from the frontend");
  }"""

        modified_content = content.replace(old_method, new_method)
        return modified_content

    try:
        # Modify dashboard.js to make API throw error
        modify_dashboard_js(modify_api_utility)

        # Restart the server to serve modified JS
        stop_flask_server()
        start_flask_server()

        # Log test instructions for manual verification
        message = (
            "Frontend API error handling test requires manual verification:\n"
            "1. Open the dashboard in a browser\n"
            "2. Check if an error message is displayed instead of simulation data\n"
            "3. The error message should include 'This is a test API error from the frontend'\n"
            "4. There should be a retry button\n"
            "This test has been marked as 'MANUAL VERIFICATION REQUIRED'"
        )
        logger.info(message)

        record_test_result("frontend_tests", test_name, None,
                           "Manual verification required - check if API error message is displayed")

    except Exception as e:
        logger.error(f"Error in {test_name} test: {e}")
        record_test_result("frontend_tests", test_name, False, f"Test error: {str(e)}")

    finally:
        # Restore dashboard.js
        modify_dashboard_js(None, restore=True)


def test_retry_functionality():
    """Test the retry functionality in the frontend."""
    test_name = "retry_functionality"

    def modify_api_counter(content):
        # Add a counter for API calls that will succeed after a certain number of attempts
        counter_code = """
// Test retry functionality
let apiCallCounter = 0;

"""

        # Find the api object definition
        match = re.search(r'const api = {', content)
        if not match:
            logger.error("Could not find api object")
            return content

        # Add counter before api object
        insert_pos = match.start()
        content_with_counter = content[:insert_pos] + counter_code + content[insert_pos:]

        # Find the getSimulationData method
        method_match = re.search(r'async getSimulationData\([^)]*\) {[^}]*}', content_with_counter)
        if not method_match:
            logger.error("Could not find getSimulationData method")
            return content_with_counter

        # Replace the method with one that fails first, then succeeds
        old_method = method_match.group(0)
        new_method = """async getSimulationData(params = {}) {
    console.log("=== RETRY FUNCTIONALITY TEST ===");
    apiCallCounter++;
    console.log(`API call attempt #${apiCallCounter}`);

    if (apiCallCounter <= 2) {
      // Fail on first two attempts
      console.log("Simulating API failure for retry test");
      throw new Error(`Test API error - attempt #${apiCallCounter}`);
    }

    // Succeed on third attempt
    console.log("API call succeeding on attempt #" + apiCallCounter);
    const url = this.buildApiUrl('/api/data/multi_civilization_statistics.csv', params);
    return this.fetchWithErrorHandling(url);
  }"""

        modified_content = content_with_counter.replace(old_method, new_method)
        return modified_content

    try:
        # Modify dashboard.js to implement retry test
        modify_dashboard_js(modify_api_counter)

        # Restart the server to serve modified JS
        stop_flask_server()
        start_flask_server()

        # Log test instructions for manual verification
        message = (
            "Retry functionality test requires manual verification:\n"
            "1. Open the dashboard in a browser\n"
            "2. The first load will fail and show an error message\n"
            "3. Click the retry button\n"
            "4. The second attempt will also fail\n"
            "5. Click retry again, and the third attempt should succeed\n"
            "6. Check the browser console to see the API call counter\n"
            "This test has been marked as 'MANUAL VERIFICATION REQUIRED'"
        )
        logger.info(message)

        record_test_result("frontend_tests", test_name, None,
                           "Manual verification required - check if retry functionality works after multiple attempts")

    except Exception as e:
        logger.error(f"Error in {test_name} test: {e}")
        record_test_result("frontend_tests", test_name, False, f"Test error: {str(e)}")

    finally:
        # Restore dashboard.js
        modify_dashboard_js(None, restore=True)


def test_non_critical_api_failures():
    """Test handling of non-critical API failures (correlation and comparison data)."""
    test_name = "non_critical_api_failures"

    def add_error_to_correlation_endpoint(content):
        # Find get_knowledge_suppression_correlation function
        match = re.search(
            r'@app\.route\(\'/api/analysis/knowledge_suppression_correlation\'\)\s*def get_knowledge_suppression_correlation\(\):',
            content)
        if not match:
            logger.error("Could not find get_knowledge_suppression_correlation function")
            return content

        # Add error code after function definition
        insert_pos = match.end()
        error_code = """
    # Test code for non-critical API error
    logger.info("=== NON-CRITICAL API ERROR TEST ===")
    return json_response({"error": "This is a test error for non-critical endpoint"}, 500)
"""
        modified_content = content[:insert_pos] + error_code + content[insert_pos:]
        return modified_content

    try:
        # Modify Flask app to add error code to correlation endpoint
        modify_flask_app(add_error_to_correlation_endpoint)

        # Restart the server
        stop_flask_server()
        start_flask_server()

        # Log test instructions for manual verification
        message = (
            "Non-critical API failure test requires manual verification:\n"
            "1. Open the dashboard in a browser\n"
            "2. Navigate to the Analysis section\n"
            "3. The Knowledge-Suppression Correlation chart should show an error message\n"
            "4. However, the rest of the dashboard should still work\n"
            "This test has been marked as 'MANUAL VERIFICATION REQUIRED'"
        )
        logger.info(message)

        record_test_result("frontend_tests", test_name, None,
                           "Manual verification required - check if dashboard works despite correlation API failure")

    except Exception as e:
        logger.error(f"Error in {test_name} test: {e}")
        record_test_result("frontend_tests", test_name, False, f"Test error: {str(e)}")

    finally:
        # Restore Flask app
        modify_flask_app(None, restore=True)


# ==================== Main Function ====================

def run_tests():
    """Run all error handling tests."""
    logger.info("===== Starting Error Handling Tests =====")

    try:
        # Backup all files first
        create_backup(FLASK_APP)
        create_backup(DASHBOARD_JS)
        for data_file in DATA_DIR.glob("*.csv"):
            create_backup(data_file)

        # Start the server for initial tests
        start_flask_server()

        # Run backend tests
        logger.info("--- Running Backend Tests ---")
        test_missing_files()
        test_json_type_conversion()
        test_api_error_response()

        # Run frontend tests
        logger.info("--- Running Frontend Tests ---")
        test_error_boundary()
        test_api_error_handling_frontend()
        test_retry_functionality()
        test_non_critical_api_failures()

    except Exception as e:
        logger.error(f"Error running tests: {e}")

    finally:
        # Stop the server
        stop_flask_server()

        # Restore all files
        logger.info("Restoring all files to original state...")
        backup_files = list(BACKUP_DIR.glob("*"))
        for backup_file_path in backup_files:
            original_path = BASE_DIR
            if backup_file_path.name.endswith(".csv"):
                original_path = DATA_DIR
            elif backup_file_path.name == "dashboard.js":
                original_path = DASHBOARD_DIR

            original_path = original_path / backup_file_path.name
            restore_from_backup(original_path, backup_file_path)

        # Restart the server with original files
        start_flask_server()

        # Generate test report
        logger.info("===== Error Handling Test Report =====")
        logger.info(f"Total tests: {test_results['total']}")
        logger.info(f"Passed: {test_results['passed']}")
        logger.info(f"Failed: {test_results['failed']}")
        logger.info(
            f"Manual verification required: {test_results['total'] - test_results['passed'] - test_results['failed']}")

        logger.info("--- Backend Tests ---")
        for test_name, result in test_results["backend_tests"].items():
            status = "PASSED" if result["passed"] else "FAILED" if result["passed"] is False else "MANUAL"
            logger.info(f"{test_name}: {status} - {result['message']}")

        logger.info("--- Frontend Tests ---")
        for test_name, result in test_results["frontend_tests"].items():
            status = "PASSED" if result["passed"] else "FAILED" if result["passed"] is False else "MANUAL"
            logger.info(f"{test_name}: {status} - {result['message']}")

        # Save test report to file
        report_path = BASE_DIR / f"error_handling_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"Test report saved to {report_path}")

        # Final cleanup
        logger.info("Cleaning up...")
        for backup_file_path in BACKUP_DIR.glob("*"):
            backup_file_path.unlink()

        try:
            BACKUP_DIR.rmdir()
            logger.info("Backup directory removed")
        except:
            logger.warning("Could not remove backup directory - it may not be empty")

        # Final message
        logger.info("===== Error Handling Tests Completed =====")
        logger.info("System has been restored to its original state")
        logger.info("Check the test report for detailed results")

        # Stop server
        stop_flask_server()


if __name__ == "__main__":
    run_tests()