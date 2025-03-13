import os
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, send_from_directory, jsonify

# Ensure output directories exist
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / 'outputs' / 'data'
dashboard_dir = BASE_DIR / 'outputs' / 'dashboard'
data_dir.mkdir(parents=True, exist_ok=True)
dashboard_dir.mkdir(parents=True, exist_ok=True)

# Create Flask app
app = Flask(__name__)


# Create placeholder data if needed
def create_placeholder_data():
    """Create mock data for testing."""
    print("Creating placeholder data...")

    # Create mock simulation statistics
    timesteps = 150
    simulation_data = []

    for t in range(timesteps):
        # Add random fluctuations
        civilization_count = max(1, round(5 + 2 * np.sin(t / 20) + np.random.rand() * 2))
        knowledge_mean = min(50, 1 + t * 0.3 + np.random.rand() * 3)
        suppression_mean = max(0.5, 7 - t * 0.04 + np.random.rand() * 2)
        intelligence_mean = min(50, 0.5 + t * 0.25 + np.random.rand() * 2)
        truth_mean = min(40, 0.1 + t * 0.2 + np.random.rand())
        resources_total = civilization_count * (50 + t * 0.5 + np.random.rand() * 10)
        stability_issues = round(t * 0.05 + np.random.rand() * 3)

        simulation_data.append({
            'Time': t,
            'Civilization_Count': civilization_count,
            'knowledge_mean': knowledge_mean,
            'knowledge_max': knowledge_mean + 5 + np.random.rand() * 10,
            'knowledge_min': max(0, knowledge_mean - 5 - np.random.rand() * 5),
            'suppression_mean': suppression_mean,
            'suppression_max': suppression_mean + 3 + np.random.rand() * 5,
            'suppression_min': max(0.1, suppression_mean - 3 - np.random.rand() * 2),
            'intelligence_mean': intelligence_mean,
            'truth_mean': truth_mean,
            'resources_total': resources_total,
            'Stability_Issues': stability_issues,
            'Timestep': 0.2 + 0.8 * min(1, 1 - (stability_issues / 20))
        })

    # Save to CSV
    pd.DataFrame(simulation_data).to_csv(data_dir / "multi_civilization_statistics.csv", index=False)

    # Create mock event data
    event_types = ['collision', 'merger', 'collapse', 'spawn', 'new_civilization']
    event_data = []

    for i in range(50):
        event_type = event_types[np.random.randint(0, len(event_types))]
        time = np.random.randint(0, timesteps)

        event_data.append({
            'id': i,
            'type': event_type,
            'time': time,
            'civ_id1': np.random.randint(0, 10),
            'civ_id2': np.random.randint(0, 10),
            'position_x': np.random.rand() * 10,
            'position_y': np.random.rand() * 10,
            'description': f"{event_type} event at time {time}"
        })

    # Save to CSV
    pd.DataFrame(event_data).to_csv(data_dir / "multi_civilization_events.csv", index=False)

    # Create mock stability data
    stability_data = {
        'Total_Stability_Issues': 87,
        'Circuit_Breaker_Triggers': 52,
        'Max_Knowledge': 48.7,
        'Max_Suppression': 15.3,
        'Max_Intelligence': 42.1,
        'Max_Truth': 37.9,
        'Total_Collisions': 12,
        'Total_Mergers': 8,
        'Total_Collapses': 6,
        'Total_Spawns': 15,
        'Total_New_Civilizations': 9,
        'Used_Dimensional_Analysis': True
    }

    # Save to CSV
    pd.DataFrame([stability_data]).to_csv(data_dir / "multi_civilization_stability.csv", index=False)
    print("Placeholder data created successfully.")


# Create the single HTML file with Chart.js instead of Recharts
def create_minimal_dashboard():
    with open(dashboard_dir / 'index.html', 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Civilization Simulation Dashboard</title>
    <!-- Use a lightweight CSS framework -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">

    <!-- Use Chart.js instead of Recharts for more reliable loading -->
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

    <!-- Custom Dashboard Script -->
    <script>
        // State variables
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
                [simulationData, eventData, stabilityData] = await Promise.all([
                    statsPromise, eventsPromise, stabilityPromise
                ]);

                console.log('Data loaded:', {
                    simulationCount: simulationData.length,
                    eventCount: eventData.length,
                    stabilityData
                });

                // Set up the time slider
                const timeSlider = document.getElementById('time-slider');
                timeSlider.max = simulationData.length - 1;
                timeSlider.addEventListener('input', handleTimeSliderChange);

                // Initialize all charts
                initializeCharts();

                // Update current time step display
                updateCurrentTimeStep();

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
            document.getElementById('time-step-value').textContent = selectedTimeStep;

            const currentState = getCurrentState();
            if (currentState) {
                document.getElementById('civ-count').textContent = currentState.Civilization_Count;
                document.getElementById('knowledge-value').textContent = currentState.knowledge_mean.toFixed(1);
                document.getElementById('suppression-value').textContent = currentState.suppression_mean.toFixed(1);
                document.getElementById('events-value').textContent = getEventsUpToTime().length;
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

            charts.eventDistribution = new Chart(
                document.getElementById('event-distribution-chart'),
                {
                    type: 'pie',
                    data: {
                        labels: eventTypes,
                        datasets: [{
                            data: eventTypes.map(type => eventCounts[type]),
                            backgroundColor: eventTypes.map(type => eventColorMap[type] || 'rgba(0, 0, 0, 0.1)')
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false
                    }
                }
            );

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

            charts.eventTypes = new Chart(
                document.getElementById('event-types-chart'),
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
            if (stabilityData) {
                const metricsLabels = ['Stability Issues', 'Circuit Breaker', 'Collisions', 'Mergers', 'Collapses'];
                const metricsValues = [
                    stabilityData.Total_Stability_Issues || 0,
                    stabilityData.Circuit_Breaker_Triggers || 0,
                    stabilityData.Total_Collisions || 0,
                    stabilityData.Total_Mergers || 0,
                    stabilityData.Total_Collapses || 0
                ];

                charts.stabilityMetrics = new Chart(
                    document.getElementById('stability-metrics-chart'),
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

        // Helper to create line charts
        function createLineChart(elementId, labels, datasets) {
            return new Chart(
                document.getElementById(elementId),
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
                        },
                        plugins: {
                            annotation: {
                                annotations: {
                                    line1: {
                                        type: 'line',
                                        yMin: 0,
                                        yMax: 0,
                                        borderColor: 'rgb(255, 99, 132)'
                                    }
                                }
                            }
                        }
                    }
                }
            );
        }

        // Update events table
        function updateEventsTable() {
            const events = getEventsUpToTime();
            const tableBody = document.querySelector('#recent-events-table tbody');
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
        }

        // Update charts with current time step
        function updateCharts() {
            // Update vertical line annotation in charts to show current time
            Object.values(charts).forEach(chart => {
                if (chart.config && chart.config.type === 'line') {
                    chart.options.plugins.annotation = {
                        annotations: {
                            line1: {
                                type: 'line',
                                xMin: selectedTimeStep,
                                xMax: selectedTimeStep,
                                borderColor: 'rgba(255, 99, 132, 0.7)',
                                borderWidth: 2
                            }
                        }
                    };
                    chart.update();
                }
            });

            // Update events distribution pie chart
            const eventCounts = getEventTypeCounts();
            const eventTypes = Object.keys(eventCounts);

            if (charts.eventDistribution) {
                charts.eventDistribution.data.labels = eventTypes;
                charts.eventDistribution.data.datasets[0].data = eventTypes.map(type => eventCounts[type]);
                charts.eventDistribution.data.datasets[0].backgroundColor = eventTypes.map(type => eventColorMap[type] || 'rgba(0, 0, 0, 0.1)');
                charts.eventDistribution.update();
            }

            // Update events table
            updateEventsTable();
        }
    </script>
</body>
</html>""")

    print(f"Minimal dashboard created at {dashboard_dir / 'index.html'}")


# Define API routes
@app.route('/')
def serve_index():
    """Serve the dashboard HTML page."""
    return send_from_directory(dashboard_dir, 'index.html')


@app.route('/api/data/multi_civilization_statistics.csv')
def get_simulation_data():
    """API endpoint for simulation statistics data."""
    try:
        df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")
        return jsonify(df.to_dict('records'))
    except Exception as e:
        print(f"Error serving statistics data: {e}")
        return jsonify([])


@app.route('/api/data/multi_civilization_events.csv')
def get_event_data():
    """API endpoint for event data."""
    try:
        df = pd.read_csv(data_dir / "multi_civilization_events.csv")
        return jsonify(df.to_dict('records'))
    except Exception as e:
        print(f"Error serving events data: {e}")
        return jsonify([])


@app.route('/api/data/multi_civilization_stability.csv')
def get_stability_data():
    """API endpoint for stability metrics."""
    try:
        df = pd.read_csv(data_dir / "multi_civilization_stability.csv")
        return jsonify(df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error serving stability data: {e}")
        return jsonify({})


if __name__ == '__main__':
    # Check if data files exist
    stats_file = data_dir / "multi_civilization_statistics.csv"
    events_file = data_dir / "multi_civilization_events.csv"
    stability_file = data_dir / "multi_civilization_stability.csv"

    if not (stats_file.exists() and events_file.exists() and stability_file.exists()):
        print("Data files not found. Creating placeholder data...")
        create_placeholder_data()

    # Create the minimal dashboard
    create_minimal_dashboard()

    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True)