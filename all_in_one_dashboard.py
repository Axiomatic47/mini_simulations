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


# Create the single HTML file with all JS embedded
def create_single_file_dashboard():
    with open(dashboard_dir / 'index.html', 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Civilization Simulation Dashboard</title>
    <!-- Use Tailwind directly from CDN -->
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">

    <!-- Load React -->
    <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>

    <!-- Load a STABLE version of Recharts -->
    <script src="https://unpkg.com/recharts@2.1.16/umd/Recharts.min.js"></script>

    <!-- Load Babel -->
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f0f2f5;
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <!-- Make absolutely sure Recharts is available -->
    <script>
        // Wait for window.Recharts to be available
        function checkRecharts() {
            if (typeof Recharts !== 'undefined') {
                console.log("Recharts loaded successfully!", Recharts);
                window.RechartsReady = true;
                window.RechartsObject = Recharts;
            } else {
                console.log("Recharts not yet available, trying again in 100ms");
                setTimeout(checkRecharts, 100);
            }
        }

        // Start checking
        checkRecharts();
    </script>

    <!-- Dashboard script -->
    <script type="text/babel">
        // React hooks
        const { useState, useEffect } = React;

        // Wait for Recharts to be available
        function waitForRecharts(callback) {
            if (window.RechartsReady) {
                callback(window.RechartsObject);
            } else {
                console.log("Waiting for Recharts to load...");
                setTimeout(() => waitForRecharts(callback), 100);
            }
        }

        // Main dashboard component
        const MultiCivilizationDashboard = () => {
            const [simulationData, setSimulationData] = useState(null);
            const [eventData, setEventData] = useState(null);
            const [stabilityData, setStabilityData] = useState(null);
            const [selectedTimeStep, setSelectedTimeStep] = useState(0);
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);
            const [recharts, setRecharts] = useState(null);
            const [view, setView] = useState('overview'); // overview, civilizations, events, stability

            // Load Recharts first
            useEffect(() => {
                waitForRecharts(rechartsObj => {
                    console.log("Recharts loaded in React component", rechartsObj);
                    setRecharts(rechartsObj);
                });
            }, []);

            const colorMap = {
                'knowledge': '#4caf50',
                'suppression': '#f44336',
                'intelligence': '#2196f3',
                'truth': '#9c27b0',
                'influence': '#ff9800',
                'resources': '#ffeb3b',
                'size': '#795548'
            };

            const eventColorMap = {
                'collision': '#f44336',
                'merger': '#9c27b0',
                'collapse': '#000000',
                'spawn': '#4caf50',
                'new_civilization': '#2196f3'
            };

            // Load simulation data once Recharts is available
            useEffect(() => {
                if (!recharts) return; // Wait for Recharts to be available

                const loadData = async () => {
                    try {
                        setLoading(true);
                        console.log("Fetching simulation data...");

                        // Fetch data from API endpoints
                        const statsResponse = await fetch('/api/data/multi_civilization_statistics.csv');
                        if (!statsResponse.ok) {
                            throw new Error(`Failed to fetch statistics data: ${statsResponse.status} ${statsResponse.statusText}`);
                        }
                        const statsData = await statsResponse.json();
                        console.log("Received stats data:", statsData ? statsData.length : 0, "records");

                        const eventsResponse = await fetch('/api/data/multi_civilization_events.csv');
                        if (!eventsResponse.ok) {
                            throw new Error(`Failed to fetch events data: ${eventsResponse.status} ${eventsResponse.statusText}`);
                        }
                        const eventsData = await eventsResponse.json();
                        console.log("Received events data:", eventsData ? eventsData.length : 0, "records");

                        const stabilityResponse = await fetch('/api/data/multi_civilization_stability.csv');
                        if (!stabilityResponse.ok) {
                            throw new Error(`Failed to fetch stability data: ${stabilityResponse.status} ${stabilityResponse.statusText}`);
                        }
                        const stabilityData = await stabilityResponse.json();
                        console.log("Received stability data");

                        setSimulationData(statsData);
                        setEventData(eventsData);
                        setStabilityData(stabilityData);

                        setLoading(false);
                    } catch (err) {
                        console.error('Error loading simulation data:', err);
                        setError(`Failed to load simulation data: ${err.message}`);
                        setLoading(false);
                    }
                };

                loadData();
            }, [recharts]);

            // Time slider handler
            const handleTimeSliderChange = (event) => {
                setSelectedTimeStep(parseInt(event.target.value, 10));
            };

            // Filter events up to the selected time
            const getEventsUpToTime = () => {
                if (!eventData) return [];
                return eventData.filter(event => event.time <= selectedTimeStep);
            };

            // Get current state at the selected time
            const getCurrentState = () => {
                if (!simulationData) return null;
                return simulationData[Math.min(selectedTimeStep, simulationData.length - 1)];
            };

            // Count event types
            const getEventTypeCounts = () => {
                const events = getEventsUpToTime();
                const counts = {};

                events.forEach(event => {
                    counts[event.type] = (counts[event.type] || 0) + 1;
                });

                return Object.entries(counts).map(([type, count]) => ({
                    type,
                    count,
                    color: eventColorMap[type] || '#999'
                }));
            };

            // Navigation buttons
            const renderNavButtons = () => (
                <div className="flex space-x-4 mb-4">
                    <button
                        className={`px-4 py-2 rounded ${view === 'overview' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
                        onClick={() => setView('overview')}
                    >
                        Overview
                    </button>
                    <button
                        className={`px-4 py-2 rounded ${view === 'civilizations' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
                        onClick={() => setView('civilizations')}
                    >
                        Civilizations
                    </button>
                    <button
                        className={`px-4 py-2 rounded ${view === 'events' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
                        onClick={() => setView('events')}
                    >
                        Events
                    </button>
                    <button
                        className={`px-4 py-2 rounded ${view === 'stability' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
                        onClick={() => setView('stability')}
                    >
                        Stability
                    </button>
                </div>
            );

            // If Recharts isn't loaded yet or there's no data, show loading state
            if (!recharts || loading) {
                return (
                    <div className="flex justify-center items-center h-screen">
                        <div className="text-xl">
                            {!recharts ? "Loading Recharts library..." : "Loading simulation data..."}
                        </div>
                    </div>
                );
            }

            // Extract components from Recharts
            const {
                LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
                BarChart, Bar, ScatterChart, Scatter, ZAxis, Cell, PieChart, Pie
            } = recharts;

            // Handle errors
            if (error) {
                return (
                    <div className="flex justify-center items-center h-screen">
                        <div className="text-xl text-red-600">{error}</div>
                    </div>
                );
            }

            // Overview view
            const renderOverview = () => {
                const currentState = getCurrentState();

                if (!currentState) return <div>No data available</div>;

                // Prepare data for the civilization positions
                const civPositions = Array.from({ length: currentState.Civilization_Count || 5 }, (_, i) => ({
                    x: Math.random() * 10,  // In real implementation, would use actual positions
                    y: Math.random() * 10,
                    z: Math.random() * (currentState.knowledge_max || 50),  // Size based on knowledge
                    id: i
                }));

                return (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-lg font-semibold mb-3">Civilization Count</h3>
                            <ResponsiveContainer width="100%" height={200}>
                                <LineChart data={simulationData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="Time" />
                                    <YAxis />
                                    <Tooltip />
                                    <Line
                                        type="monotone"
                                        dataKey="Civilization_Count"
                                        stroke="#8884d8"
                                        dot={false}
                                        strokeWidth={2}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-lg font-semibold mb-3">Knowledge & Suppression</h3>
                            <ResponsiveContainer width="100%" height={200}>
                                <LineChart data={simulationData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="Time" />
                                    <YAxis />
                                    <Tooltip />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="knowledge_mean"
                                        stroke={colorMap.knowledge}
                                        name="Knowledge"
                                        dot={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="suppression_mean"
                                        stroke={colorMap.suppression}
                                        name="Suppression"
                                        dot={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-lg font-semibold mb-3">Event Distribution</h3>
                            <ResponsiveContainer width="100%" height={200}>
                                <PieChart>
                                    <Pie
                                        data={getEventTypeCounts()}
                                        dataKey="count"
                                        nameKey="type"
                                        cx="50%"
                                        cy="50%"
                                        outerRadius={80}
                                        fill="#8884d8"
                                        label={entry => entry.type}
                                    >
                                        {getEventTypeCounts().map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-lg font-semibold mb-3">Civilization Positions</h3>
                            <ResponsiveContainer width="100%" height={200}>
                                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid />
                                    <XAxis type="number" dataKey="x" name="X Position" domain={[0, 10]} />
                                    <YAxis type="number" dataKey="y" name="Y Position" domain={[0, 10]} />
                                    <ZAxis type="number" dataKey="z" range={[50, 400]} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                    <Scatter name="Civilizations" data={civPositions} fill="#8884d8">
                                        {civPositions.map((entry, index) => (
                                            <Cell
                                                key={`cell-${index}`}
                                                fill={`rgb(${Math.round(25 + 230 * (entry.z / (currentState.knowledge_max || 50)))}, 100, 50)`}
                                            />
                                        ))}
                                    </Scatter>
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                );
            };

            // Civilizations view (similar structure to overview but focused on civilization data)
            const renderCivilizations = () => {
                const currentState = getCurrentState();
                if (!currentState) return <div>No data available</div>;

                return (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-lg font-semibold mb-3">Knowledge Range</h3>
                            <ResponsiveContainer width="100%" height={250}>
                                <LineChart data={simulationData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="Time" />
                                    <YAxis />
                                    <Tooltip />
                                    <Line
                                        type="monotone"
                                        dataKey="knowledge_max"
                                        stroke={colorMap.knowledge}
                                        strokeOpacity={0.8}
                                        dot={false}
                                        activeDot={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="knowledge_mean"
                                        stroke={colorMap.knowledge}
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="knowledge_min"
                                        stroke={colorMap.knowledge}
                                        strokeOpacity={0.8}
                                        dot={false}
                                        activeDot={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-lg font-semibold mb-3">Suppression Range</h3>
                            <ResponsiveContainer width="100%" height={250}>
                                <LineChart data={simulationData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="Time" />
                                    <YAxis />
                                    <Tooltip />
                                    <Line
                                        type="monotone"
                                        dataKey="suppression_max"
                                        stroke={colorMap.suppression}
                                        strokeOpacity={0.8}
                                        dot={false}
                                        activeDot={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="suppression_mean"
                                        stroke={colorMap.suppression}
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="suppression_min"
                                        stroke={colorMap.suppression}
                                        strokeOpacity={0.8}
                                        dot={false}
                                        activeDot={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                );
            };

            // Events view
            const renderEvents = () => {
                const events = getEventsUpToTime();
                const eventCounts = getEventTypeCounts();

                // Simple version that just shows event types
                return (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-lg font-semibold mb-3">Event Types</h3>
                            <ResponsiveContainer width="100%" height={250}>
                                <BarChart data={eventCounts}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="type" />
                                    <YAxis />
                                    <Tooltip />
                                    <Bar dataKey="count">
                                        {eventCounts.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                );
            };

            // Stability view
            const renderStability = () => {
                if (!simulationData || !stabilityData) return <div>No stability data available</div>;

                // Prepare data for the stability metrics bar chart
                const stabilityMetrics = [
                    { name: 'Stability Issues', value: stabilityData.Total_Stability_Issues },
                    { name: 'Circuit Breaker', value: stabilityData.Circuit_Breaker_Triggers }
                ];

                return (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="bg-white p-4 rounded shadow">
                            <h3 className="text-lg font-semibold mb-3">Stability Metrics</h3>
                            <ResponsiveContainer width="100%" height={250}>
                                <BarChart data={stabilityMetrics}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="name" />
                                    <YAxis />
                                    <Tooltip />
                                    <Bar dataKey="value" fill="#8884d8" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                );
            };

            return (
                <div className="container mx-auto p-4">
                    <h1 className="text-2xl font-bold mb-4">Multi-Civilization Simulation Dashboard</h1>

                    {/* Navigation buttons */}
                    {renderNavButtons()}

                    {/* Time slider */}
                    <div className="mb-6 bg-white p-4 rounded shadow">
                        <h2 className="text-lg font-semibold mb-2">Time Step: {selectedTimeStep}</h2>
                        <input
                            type="range"
                            min="0"
                            max={simulationData ? simulationData.length - 1 : 100}
                            value={selectedTimeStep}
                            onChange={handleTimeSliderChange}
                            className="w-full"
                        />

                        {/* Current state summary */}
                        {getCurrentState() && (
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                                <div className="bg-gray-100 p-3 rounded">
                                    <div className="text-sm text-gray-600">Civilizations</div>
                                    <div className="text-2xl font-bold">{getCurrentState().Civilization_Count || 0}</div>
                                </div>
                                <div className="bg-gray-100 p-3 rounded">
                                    <div className="text-sm text-gray-600">Avg. Knowledge</div>
                                    <div className="text-2xl font-bold">{(getCurrentState().knowledge_mean || 0).toFixed(1)}</div>
                                </div>
                                <div className="bg-gray-100 p-3 rounded">
                                    <div className="text-sm text-gray-600">Avg. Suppression</div>
                                    <div className="text-2xl font-bold">{(getCurrentState().suppression_mean || 0).toFixed(1)}</div>
                                </div>
                                <div className="bg-gray-100 p-3 rounded">
                                    <div className="text-sm text-gray-600">Events</div>
                                    <div className="text-2xl font-bold">{getEventsUpToTime().length}</div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* View specific content */}
                    {view === 'overview' && renderOverview()}
                    {view === 'civilizations' && renderCivilizations()}
                    {view === 'events' && renderEvents()}
                    {view === 'stability' && renderStability()}
                </div>
            );
        };

        // Render the dashboard
        ReactDOM.render(<MultiCivilizationDashboard />, document.getElementById('root'));
    </script>
</body>
</html>""")

    print(f"All-in-one HTML dashboard created at {dashboard_dir / 'index.html'}")


# Define API routes
@app.route('/')
def serve_index():
    """Serve the dashboard HTML page."""
    return send_from_directory(dashboard_dir, 'index.html')


@app.route('/dashboard.js')
def serve_dashboard_js():
    """Serve the dashboard JavaScript file."""
    return send_from_directory(dashboard_dir, 'dashboard.js')


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

    # Create all-in-one HTML file
    create_single_file_dashboard()

    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True)