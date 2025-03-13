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
}