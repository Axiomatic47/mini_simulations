"""
Fix that uses a simpler approach with Bootstrap instead of Tailwind and inline chart.js
"""
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bundle-fix')

# Define paths
BASE_DIR = Path(__file__).resolve().parent
dashboard_dir = BASE_DIR / 'outputs' / 'dashboard'

logger.info(f"Dashboard directory: {dashboard_dir}")

# Create an index.html file with standard, reliable libraries
index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Civilization Simulation Dashboard</title>
    <!-- Bootstrap CSS (very reliable CDN) -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
        }
        .nav-pills .nav-link.active {
            background-color: #0d6efd;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Multi-Civilization Simulation Dashboard</h1>

        <!-- Navigation Tabs -->
        <ul class="nav nav-pills mb-3" id="dashboard-tabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="overview-tab" data-bs-toggle="pill" data-bs-target="#overview" type="button" role="tab" aria-controls="overview" aria-selected="true">Overview</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="civilizations-tab" data-bs-toggle="pill" data-bs-target="#civilizations" type="button" role="tab" aria-controls="civilizations" aria-selected="false">Civilizations</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="events-tab" data-bs-toggle="pill" data-bs-target="#events" type="button" role="tab" aria-controls="events" aria-selected="false">Events</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="stability-tab" data-bs-toggle="pill" data-bs-target="#stability" type="button" role="tab" aria-controls="stability" aria-selected="false">Stability</button>
            </li>
        </ul>

        <!-- Time Slider -->
        <div class="card mb-4">
            <div class="card-body">
                <h5 class="card-title">Time Step: <span id="time-step-value">0</span></h5>
                <input type="range" class="form-range" id="time-slider" min="0" max="149" value="0">

                <div class="row mt-3" id="current-stats">
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body">
                                <h6 class="card-subtitle mb-2 text-muted">Civilizations</h6>
                                <h3 id="civ-count">0</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body">
                                <h6 class="card-subtitle mb-2 text-muted">Avg. Knowledge</h6>
                                <h3 id="knowledge-value">0</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body">
                                <h6 class="card-subtitle mb-2 text-muted">Avg. Suppression</h6>
                                <h3 id="suppression-value">0</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body">
                                <h6 class="card-subtitle mb-2 text-muted">Events</h6>
                                <h3 id="events-value">0</h3>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab Content -->
        <div class="tab-content" id="dashboard-tab-content">
            <!-- Overview Tab -->
            <div class="tab-pane fade show active" id="overview" role="tabpanel" aria-labelledby="overview-tab">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Civilization Count</h5>
                                <div class="chart-container">
                                    <canvas id="civilization-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Knowledge & Suppression</h5>
                                <div class="chart-container">
                                    <canvas id="knowledge-suppression-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Event Distribution</h5>
                                <div class="chart-container">
                                    <canvas id="event-distribution-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Intelligence & Truth</h5>
                                <div class="chart-container">
                                    <canvas id="intelligence-truth-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Civilizations Tab -->
            <div class="tab-pane fade" id="civilizations" role="tabpanel" aria-labelledby="civilizations-tab">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Knowledge Range</h5>
                                <div class="chart-container">
                                    <canvas id="knowledge-range-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Suppression Range</h5>
                                <div class="chart-container">
                                    <canvas id="suppression-range-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Events Tab -->
            <div class="tab-pane fade" id="events" role="tabpanel" aria-labelledby="events-tab">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Event Types</h5>
                                <div class="chart-container">
                                    <canvas id="event-types-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Recent Events</h5>
                                <div class="table-responsive">
                                    <table class="table table-striped" id="recent-events-table">
                                        <thead>
                                            <tr>
                                                <th>Time</th>
                                                <th>Type</th>
                                                <th>Description</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <!-- Will be populated by JavaScript -->
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Stability Tab -->
            <div class="tab-pane fade" id="stability" role="tabpanel" aria-labelledby="stability-tab">
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Stability Issues Over Time</h5>
                                <div class="chart-container">
                                    <canvas id="stability-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Stability Metrics</h5>
                                <div class="chart-container">
                                    <canvas id="stability-metrics-chart"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>

    <!-- Helper script to handle NaN values in JSON -->
    <script>
    // Add safe JSON parsing function to handle NaN values
    function safeParseJSON(text) {
        // Replace NaN with null before parsing
        let processedText = text;

        // Replace "NaN" with null, maintaining valid JSON syntax
        processedText = processedText.replace(/"([^"]+)":\s*NaN/g, '"$1": null');

        // Replace any standalone NaN values
        processedText = processedText.replace(/:\s*NaN\s*([,}])/g, ': null$1');

        try {
            return JSON.parse(processedText);
        } catch (e) {
            console.error('JSON parse error even after NaN handling:', e);
            console.error('Processed text (first 500 chars):', processedText.substring(0, 500));
            throw e;
        }
    }

    // Override fetch to handle NaN values
    const originalFetch = window.fetch;
    window.fetch = function() {
        return originalFetch.apply(this, arguments)
            .then(response => {
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return response.text().then(text => {
                        try {
                            const json = safeParseJSON(text);
                            return {
                                ok: response.ok,
                                status: response.status,
                                statusText: response.statusText,
                                headers: response.headers,
                                json: () => Promise.resolve(json),
                                text: () => Promise.resolve(text)
                            };
                        } catch (e) {
                            console.error('Error handling NaN values:', e);
                            throw e;
                        }
                    });
                }
                return response;
            });
    };
    </script>

    <!-- ChartJS Dashboard JavaScript -->
    <script src="/chart-dashboard.js"></script>
</body>
</html>"""

with open(dashboard_dir / 'index.html', 'w') as f:
    f.write(index_html)
    logger.info("Created Bootstrap-based index.html")

# Now create the chart-dashboard.js file
chart_dashboard_js = """// State variables
let simulationData = [];
let eventData = [];
let stabilityData = {};
let selectedTimeStep = 0;
let charts = {};

// Colors for visualization
const colorMap = {
    'knowledge': 'rgba(76, 175, 80, 0.7)',
    'suppression': 'rgba(244, 67, 54, 0.7)',
    'intelligence': 'rgba(33, 150, 243, 0.7)',
    'truth': 'rgba(156, 39, 176, 0.7)',
    'resources': 'rgba(255, 235, 59, 0.7)'
};

const eventColorMap = {
    'collision': 'rgba(244, 67, 54, 0.7)',
    'merger': 'rgba(156, 39, 176, 0.7)',
    'collapse': 'rgba(0, 0, 0, 0.7)',
    'spawn': 'rgba(76, 175, 80, 0.7)',
    'new_civilization': 'rgba(33, 150, 243, 0.7)'
};

// Initialize the dashboard once the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initializing...');
    initializeDashboard();
});

// Initialize the dashboard
async function initializeDashboard() {
    try {
        // Fetch data
        console.log("Fetching simulation data...");
        const statsPromise = fetch('/api/data/multi_civilization_statistics.csv')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch statistics: ${response.status}`);
                }
                return response.json();
            });

        const eventsPromise = fetch('/api/data/multi_civilization_events.csv')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch events: ${response.status}`);
                }
                return response.json();
            });

        const stabilityPromise = fetch('/api/data/multi_civilization_stability.csv')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch stability data: ${response.status}`);
                }
                return response.json();
            });

        // Wait for all data to be fetched
        console.log("Waiting for all data promises to resolve...");
        const results = await Promise.all([
            statsPromise.catch(error => {
                console.error("Error in statistics promise:", error);
                return [];
            }),
            eventsPromise.catch(error => {
                console.error("Error in events promise:", error);
                return [];
            }),
            stabilityPromise.catch(error => {
                console.error("Error in stability promise:", error);
                return {};
            })
        ]);

        simulationData = results[0];
        eventData = results[1];
        stabilityData = results[2];

        console.log('Data loaded:', {
            simulationCount: simulationData.length,
            eventCount: eventData.length,
            stabilityData: Object.keys(stabilityData).length > 0 ? "loaded" : "empty"
        });

        // Set up the time slider
        const timeSlider = document.getElementById('time-slider');
        if (!timeSlider) {
            console.error("Time slider element not found!");
        } else {
            timeSlider.max = simulationData.length - 1;
            timeSlider.addEventListener('input', handleTimeSliderChange);
        }

        // Initialize all charts
        if (simulationData.length > 0) {
            initializeCharts();
            // Update current time step display
            updateCurrentTimeStep();
        } else {
            console.error("No simulation data available to initialize charts");
            alert("Error: No simulation data available. Please check the server logs.");
        }

    } catch (error) {
        console.error('Error initializing dashboard:', error);
        alert(`Failed to load dashboard data: ${error.message}`);
    }
}

// Handle time slider change
function handleTimeSliderChange(event) {
    selectedTimeStep = parseInt(event.target.value, 10);
    updateCurrentTimeStep();
    updateCharts();
}

// Update the current time step display
function updateCurrentTimeStep() {
    const timeStepElement = document.getElementById('time-step-value');
    if (timeStepElement) {
        timeStepElement.textContent = selectedTimeStep;
    }

    const currentState = getCurrentState();
    if (currentState) {
        const civCountElement = document.getElementById('civ-count');
        const knowledgeValueElement = document.getElementById('knowledge-value');
        const suppressionValueElement = document.getElementById('suppression-value');
        const eventsValueElement = document.getElementById('events-value');

        if (civCountElement) civCountElement.textContent = currentState.Civilization_Count;
        if (knowledgeValueElement) knowledgeValueElement.textContent = currentState.knowledge_mean.toFixed(1);
        if (suppressionValueElement) suppressionValueElement.textContent = currentState.suppression_mean.toFixed(1);
        if (eventsValueElement) eventsValueElement.textContent = getEventsUpToTime().length;
    }
}

// Get current state at the selected time
function getCurrentState() {
    if (!simulationData || simulationData.length === 0) return null;
    return simulationData[Math.min(selectedTimeStep, simulationData.length - 1)];
}

// Get events up to current time
function getEventsUpToTime() {
    if (!eventData) return [];
    return eventData.filter(event => event.time <= selectedTimeStep);
}

// Count event types
function getEventTypeCounts() {
    const events = getEventsUpToTime();
    const counts = {};

    events.forEach(event => {
        counts[event.type] = (counts[event.type] || 0) + 1;
    });

    return counts;
}

// Initialize all charts
function initializeCharts() {
    try {
        console.log("Initializing charts...");

        // Check if Chart library is available
        if (typeof Chart === 'undefined') {
            console.error("Chart.js library not found!");
            return;
        }

        // Overview tab charts
        charts.civilization = createLineChart(
            'civilization-chart',
            simulationData.map(d => d.Time),
            [{ 
                label: 'Civilization Count',
                data: simulationData.map(d => d.Civilization_Count),
                borderColor: 'rgba(75, 192, 192, 1)',
                tension: 0.1
            }]
        );

        charts.knowledgeSuppression = createLineChart(
            'knowledge-suppression-chart',
            simulationData.map(d => d.Time),
            [
                {
                    label: 'Knowledge',
                    data: simulationData.map(d => d.knowledge_mean),
                    borderColor: colorMap.knowledge,
                    tension: 0.1
                },
                {
                    label: 'Suppression',
                    data: simulationData.map(d => d.suppression_mean),
                    borderColor: colorMap.suppression,
                    tension: 0.1
                }
            ]
        );

        // Event distribution pie chart
        const eventCounts = getEventTypeCounts();
        const eventTypes = Object.keys(eventCounts);

        const eventDistributionElement = document.getElementById('event-distribution-chart');
        if (!eventDistributionElement) {
            console.error("Event distribution chart element not found!");
        } else {
            charts.eventDistribution = new Chart(
                eventDistributionElement,
                {
                    type: 'pie',
                    data: {
                        labels: eventTypes,
                        datasets: [{
                            data: Object.values(eventCounts),
                            backgroundColor: eventTypes.map(type => eventColorMap[type] || 'rgba(0, 0, 0, 0.1)')
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                }
            );
        }

        // Intelligence & Truth chart
        charts.intelligenceTruth = createLineChart(
            'intelligence-truth-chart',
            simulationData.map(d => d.Time),
            [
                {
                    label: 'Intelligence',
                    data: simulationData.map(d => d.intelligence_mean),
                    borderColor: colorMap.intelligence,
                    tension: 0.1
                },
                {
                    label: 'Truth',
                    data: simulationData.map(d => d.truth_mean),
                    borderColor: colorMap.truth,
                    tension: 0.1
                }
            ]
        );

        // Civilizations tab charts
        charts.knowledgeRange = createLineChart(
            'knowledge-range-chart',
            simulationData.map(d => d.Time),
            [
                {
                    label: 'Max',
                    data: simulationData.map(d => d.knowledge_max),
                    borderColor: 'rgba(76, 175, 80, 0.3)',
                    fill: '+1',
                    tension: 0.1
                },
                {
                    label: 'Mean',
                    data: simulationData.map(d => d.knowledge_mean),
                    borderColor: colorMap.knowledge,
                    tension: 0.1
                },
                {
                    label: 'Min',
                    data: simulationData.map(d => d.knowledge_min),
                    borderColor: 'rgba(76, 175, 80, 0.3)',
                    fill: false,
                    tension: 0.1
                }
            ]
        );

        charts.suppressionRange = createLineChart(
            'suppression-range-chart',
            simulationData.map(d => d.Time),
            [
                {
                    label: 'Max',
                    data: simulationData.map(d => d.suppression_max),
                    borderColor: 'rgba(244, 67, 54, 0.3)',
                    fill: '+1',
                    tension: 0.1
                },
                {
                    label: 'Mean',
                    data: simulationData.map(d => d.suppression_mean),
                    borderColor: colorMap.suppression,
                    tension: 0.1
                },
                {
                    label: 'Min',
                    data: simulationData.map(d => d.suppression_min),
                    borderColor: 'rgba(244, 67, 54, 0.3)',
                    fill: false,
                    tension: 0.1
                }
            ]
        );

        // Events tab charts
        const eventTypeLabels = Object.keys(eventColorMap);
        const eventTypeData = eventTypeLabels.map(type => eventCounts[type] || 0);

        const eventTypesElement = document.getElementById('event-types-chart');
        if (!eventTypesElement) {
            console.error("Event types chart element not found!");
        } else {
            charts.eventTypes = new Chart(
                eventTypesElement,
                {
                    type: 'bar',
                    data: {
                        labels: eventTypeLabels,
                        datasets: [{
                            label: 'Count',
                            data: eventTypeData,
                            backgroundColor: eventTypeLabels.map(type => eventColorMap[type])
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                }
            );
        }

        // Populate recent events table
        updateEventsTable();

        // Stability tab charts
        charts.stability = createLineChart(
            'stability-chart',
            simulationData.map(d => d.Time),
            [{
                label: 'Stability Issues',
                data: simulationData.map(d => d.Stability_Issues),
                borderColor: 'rgba(211, 47, 47, 0.7)',
                tension: 0.1
            }]
        );

        // Stability metrics bar chart
        if (stabilityData && Object.keys(stabilityData).length > 0) {
            const metricsLabels = ['Stability Issues', 'Circuit Breaker', 'Collisions', 'Mergers', 'Collapses'];
            const metricsValues = [
                stabilityData.Total_Stability_Issues || 0,
                stabilityData.Circuit_Breaker_Triggers || 0,
                stabilityData.Total_Collisions || 0,
                stabilityData.Total_Mergers || 0,
                stabilityData.Total_Collapses || 0
            ];

            const stabilityMetricsElement = document.getElementById('stability-metrics-chart');
            if (!stabilityMetricsElement) {
                console.error("Stability metrics chart element not found!");
            } else {
                charts.stabilityMetrics = new Chart(
                    stabilityMetricsElement,
                    {
                        type: 'bar',
                        data: {
                            labels: metricsLabels,
                            datasets: [{
                                label: 'Count',
                                data: metricsValues,
                                backgroundColor: 'rgba(136, 132, 216, 0.7)'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    }
                );
            }
        }

        console.log("Charts initialized successfully.");
    } catch (error) {
        console.error("Error initializing charts:", error);
    }
}

// Helper to create line charts
function createLineChart(elementId, labels, datasets) {
    try {
        const element = document.getElementById(elementId);
        if (!element) {
            console.error(`Chart element not found: #${elementId}`);
            return null;
        }

        return new Chart(
            element,
            {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    elements: {
                        point: {
                            radius: 0
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: {
                                precision: 1
                            }
                        }
                    }
                }
            }
        );
    } catch (error) {
        console.error(`Error creating chart ${elementId}:`, error);
        return null;
    }
}

// Update events table
function updateEventsTable() {
    try {
        const events = getEventsUpToTime();
        const tableBody = document.querySelector('#recent-events-table tbody');
        if (!tableBody) {
            console.error("Events table body not found!");
            return;
        }

        tableBody.innerHTML = '';

        // Sort events by time (most recent first) and take the last 10
        const recentEvents = [...events]
            .sort((a, b) => b.time - a.time)
            .slice(0, 10);

        // Create table rows
        recentEvents.forEach(event => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${event.time}</td>
                <td><span class="badge bg-secondary">${event.type}</span></td>
                <td>${event.description}</td>
            `;
            tableBody.appendChild(row);
        });

        // If no events, show a message
        if (recentEvents.length === 0) {
            const row = document.createElement('tr');
            row.innerHTML = '<td colspan="3" class="text-center">No events to display</td>';
            tableBody.appendChild(row);
        }
    } catch (error) {
        console.error("Error updating events table:", error);
    }
}

// Update charts with current time step
function updateCharts() {
    try {
        // Update events distribution pie chart
        const eventCounts = getEventTypeCounts();
        const eventTypes = Object.keys(eventCounts);

        if (charts.eventDistribution) {
            charts.eventDistribution.data.labels = eventTypes;
            charts.eventDistribution.data.datasets[0].data = Object.values(eventCounts);
            charts.eventDistribution.data.datasets[0].backgroundColor = eventTypes.map(type => eventColorMap[type] || 'rgba(0, 0, 0, 0.1)');
            charts.eventDistribution.update();
        }

        // Update events table
        updateEventsTable();
    } catch (error) {
        console.error("Error updating charts:", error);
    }
}"""

with open(dashboard_dir / 'chart-dashboard.js', 'w') as f:
    f.write(chart_dashboard_js)
    logger.info("Created Chart.js dashboard implementation")

# Create a simplified server file
server_py = """import logging
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
"""

with open(BASE_DIR / 'multi_civilization_dashboard.py', 'w') as f:
    f.write(server_py)
    logger.info("Created simplified server file")

logger.info("Fix complete! Now run: python multi_civilization_dashboard.py")