// State variables
let simulationData = [];
let eventData = [];
let stabilityData = {};
let selectedTimeStep = 0;
let charts = {};
let timeRange = { start: 0, end: 150 };
let downSampleFactor = 1;

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

    // Add filter controls to the page
    addFilterControls();
});

// Helper function to build API URLs with query parameters
function buildApiUrl(endpoint, params = {}) {
    const url = new URL(endpoint, window.location.origin);
    Object.keys(params).forEach(key => {
        if (params[key] !== null && params[key] !== undefined) {
            url.searchParams.append(key, params[key]);
        }
    });
    return url.toString();
}

// Add filter controls to the page
function addFilterControls() {
    const timeSliderContainer = document.querySelector('.card-body');
    if (!timeSliderContainer) return;

    // Create filter controls
    const filterControls = document.createElement('div');
    filterControls.className = 'mt-4 border-top pt-3';
    filterControls.innerHTML = `
        <h6 class="mb-3">Data Filters</h6>
        <div class="row">
            <div class="col-md-6 mb-3">
                <label class="form-label">Time Range</label>
                <div class="d-flex align-items-center">
                    <input type="number" class="form-control form-control-sm me-2" id="time-start"
                        value="${timeRange.start}" min="0" style="width: 80px;">
                    <span class="me-2">to</span>
                    <input type="number" class="form-control form-control-sm me-2" id="time-end"
                        value="${timeRange.end}" min="1" style="width: 80px;">
                    <button class="btn btn-sm btn-outline-secondary" id="apply-time-range">Apply</button>
                </div>
            </div>
            <div class="col-md-6 mb-3">
                <label class="form-label">Data Sampling</label>
                <select class="form-select form-select-sm" id="downsampling" style="width: auto;">
                    <option value="1">No sampling</option>
                    <option value="2">Every 2nd point</option>
                    <option value="5">Every 5th point</option>
                    <option value="10">Every 10th point</option>
                </select>
            </div>
        </div>
        <div class="row">
            <div class="col-12 mb-3">
                <div class="d-flex justify-content-between">
                    <button class="btn btn-sm btn-primary" id="export-csv">Export Data</button>
                    <div class="form-text text-muted">
                        <span id="data-points-count"></span> data points loaded
                    </div>
                </div>
            </div>
        </div>
    `;

    timeSliderContainer.appendChild(filterControls);

    // Add event listeners
    document.getElementById('apply-time-range').addEventListener('click', function() {
        const start = parseInt(document.getElementById('time-start').value);
        const end = parseInt(document.getElementById('time-end').value);

        if (start >= 0 && end > start) {
            timeRange = { start, end };
            loadDashboardData();
        } else {
            alert('Please enter a valid time range (start must be less than end)');
        }
    });

    document.getElementById('downsampling').addEventListener('change', function() {
        downSampleFactor = parseInt(this.value);
        loadDashboardData();
    });

    document.getElementById('export-csv').addEventListener('click', function() {
        window.location.href = buildApiUrl('/api/export/csv', { dataset: 'statistics' });
    });
}

// Initialize the dashboard
async function initializeDashboard() {
    try {
        // Load initial metadata to get time range
        console.log("Fetching available metrics...");
        const metadataResponse = await fetch('/api/meta/available_metrics')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch metrics metadata: ${response.status}`);
                }
                return response.json();
            })
            .catch(error => {
                console.error("Error fetching metrics:", error);
                return null;
            });

        if (metadataResponse && metadataResponse.time_range) {
            timeRange = {
                start: metadataResponse.time_range.min,
                end: metadataResponse.time_range.max
            };

            // Update time slider
            const timeSlider = document.getElementById('time-slider');
            if (timeSlider) {
                timeSlider.min = timeRange.start;
                timeSlider.max = timeRange.end;
                timeSlider.value = Math.floor((timeRange.start + timeRange.end) / 2);
                selectedTimeStep = Math.floor((timeRange.start + timeRange.end) / 2);
            }

            // Update input fields
            if (document.getElementById('time-start')) {
                document.getElementById('time-start').value = timeRange.start;
            }
            if (document.getElementById('time-end')) {
                document.getElementById('time-end').value = timeRange.end;
            }
        }

        // Load the data with the established time range
        await loadDashboardData();

    } catch (error) {
        console.error('Error initializing dashboard:', error);
        alert(`Failed to load dashboard data: ${error.message}`);
    }
}

// Load dashboard data with current filters
async function loadDashboardData() {
    try {
        // Fetch data with filters
        console.log("Fetching simulation data with filters...");

        // Prepare API parameters
        const statsParams = {
            time_start: timeRange.start,
            time_end: timeRange.end,
            down_sample: downSampleFactor > 1 ? downSampleFactor : null
        };

        // Fetch filtered statistics data
        const statsUrl = buildApiUrl('/api/data/multi_civilization_statistics.csv', statsParams);
        const statsResponse = await fetch(statsUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch statistics: ${response.status}`);
                }
                return response.json();
            })
            .catch(error => {
                console.error("Error in statistics fetch:", error);
                return [];
            });

        // Fetch filtered events data
        const eventsUrl = buildApiUrl('/api/data/multi_civilization_events.csv', {
            time_start: timeRange.start,
            time_end: timeRange.end
        });
        const eventsResponse = await fetch(eventsUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch events: ${response.status}`);
                }
                return response.json();
            })
            .catch(error => {
                console.error("Error in events fetch:", error);
                return [];
            });

        // Fetch stability data
        const stabilityResponse = await fetch('/api/data/multi_civilization_stability.csv')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to fetch stability data: ${response.status}`);
                }
                return response.json();
            })
            .catch(error => {
                console.error("Error in stability fetch:", error);
                return {};
            });

        simulationData = statsResponse;
        eventData = eventsResponse;
        stabilityData = stabilityResponse;

        console.log('Data loaded:', {
            simulationCount: simulationData.length,
            eventCount: eventData.length,
            stabilityData: Object.keys(stabilityData).length > 0 ? "loaded" : "empty"
        });

        // Update data points count display
        if (document.getElementById('data-points-count')) {
            document.getElementById('data-points-count').textContent = simulationData.length;
        }

        // Set up the time slider
        const timeSlider = document.getElementById('time-slider');
        if (!timeSlider) {
            console.error("Time slider element not found!");
        } else {
            timeSlider.min = timeRange.start;
            timeSlider.max = timeRange.end;
            if (selectedTimeStep < timeRange.start || selectedTimeStep > timeRange.end) {
                selectedTimeStep = Math.floor((timeRange.start + timeRange.end) / 2);
                timeSlider.value = selectedTimeStep;
            }
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
        console.error('Error loading dashboard data:', error);
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

    // Find the closest time point
    let closestState = simulationData[0];
    let closestDiff = Math.abs(simulationData[0].Time - selectedTimeStep);

    for (let i = 1; i < simulationData.length; i++) {
        const diff = Math.abs(simulationData[i].Time - selectedTimeStep);
        if (diff < closestDiff) {
            closestDiff = diff;
            closestState = simulationData[i];
        }
    }

    return closestState;
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

        // Clear any existing charts to avoid duplicates
        Object.values(charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        charts = {};

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
        const eventData = eventTypes.map(type => eventCounts[type]);

        charts.eventDistribution = new Chart(
            document.getElementById('event-distribution-chart'),
            {
                type: 'pie',
                data: {
                    labels: eventTypes,
                    datasets: [{
                        data: eventData,
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
        if (stabilityData && Object.keys(stabilityData).length > 0) {
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
                <td>${event.description || `${event.type} event at time ${event.time}`}</td>
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
        const eventData = eventTypes.map(type => eventCounts[type]);

        if (charts.eventDistribution) {
            charts.eventDistribution.data.labels = eventTypes;
            charts.eventDistribution.data.datasets[0].data = eventData;
            charts.eventDistribution.data.datasets[0].backgroundColor =
                eventTypes.map(type => eventColorMap[type] || 'rgba(0, 0, 0, 0.1)');
            charts.eventDistribution.update();
        }

        // Update events table
        updateEventsTable();

        // Add vertical line at current time step to all line charts
        const timeAnnotation = {
            type: 'line',
            mode: 'vertical',
            scaleID: 'x',
            value: selectedTimeStep,
            borderColor: 'rgba(255, 0, 0, 0.7)',
            borderWidth: 2,
            label: {
                content: 'Current',
                enabled: true,
                position: 'top'
            }
        };

        // This would need a plugin to work, which we're not implementing here
        // Charts would need to be recreated with proper annotations
    } catch (error) {
        console.error("Error updating charts:", error);
    }
}