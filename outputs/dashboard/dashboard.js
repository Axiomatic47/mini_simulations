// Access React from globals
const { useState, useEffect } = React;

// Debug logging
console.log('Dashboard script loading...');
console.log('React available:', React ? 'Yes' : 'No');
console.log('ReactDOM available:', ReactDOM ? 'Yes' : 'No');
console.log('Recharts available:', window.Recharts ? 'Yes' : 'No');

// Explicitly load Recharts components from global Recharts object
const LineChart = window.Recharts.LineChart;
const Line = window.Recharts.Line;
const XAxis = window.Recharts.XAxis;
const YAxis = window.Recharts.YAxis;
const CartesianGrid = window.Recharts.CartesianGrid;
const Tooltip = window.Recharts.Tooltip;
const Legend = window.Recharts.Legend;
const ResponsiveContainer = window.Recharts.ResponsiveContainer;
const BarChart = window.Recharts.BarChart;
const Bar = window.Recharts.Bar;
const ScatterChart = window.Recharts.ScatterChart;
const Scatter = window.Recharts.Scatter;
const ZAxis = window.Recharts.ZAxis;
const Cell = window.Recharts.Cell;
const PieChart = window.Recharts.PieChart;
const Pie = window.Recharts.Pie;

const MultiCivilizationDashboard = () => {
  const [simulationData, setSimulationData] = useState(null);
  const [eventData, setEventData] = useState(null);
  const [stabilityData, setStabilityData] = useState(null);
  const [selectedTimeStep, setSelectedTimeStep] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [view, setView] = useState('overview'); // overview, civilizations, events, stability

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

  // Load simulation data
  useEffect(() => {
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
  }, []);

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
                label={(entry) => entry.type}
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

  // Civilizations view
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

        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold mb-3">Intelligence & Truth Growth</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={simulationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="intelligence_mean"
                stroke={colorMap.intelligence}
                name="Intelligence"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="truth_mean"
                stroke={colorMap.truth}
                name="Truth"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold mb-3">Total Resources</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={simulationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Time" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="resources_total"
                stroke={colorMap.resources}
                strokeWidth={2}
                dot={false}
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

    // Count events by type for the bar chart
    const eventCounts = getEventTypeCounts();

    // Group events by time period (every 10 time steps)
    const eventsByPeriod = {};
    events.forEach(event => {
      const period = Math.floor(event.time / 10) * 10;
      eventsByPeriod[period] = eventsByPeriod[period] || { period };
      eventsByPeriod[period][event.type] = (eventsByPeriod[period][event.type] || 0) + 1;
    });

    const eventTimeData = Object.values(eventsByPeriod).sort((a, b) => a.period - b.period);

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

        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold mb-3">Event Timeline</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={simulationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Time" />
              <YAxis />
              <Tooltip />
              {/* Mark events on the time axis */}
              {events.slice(0, 20).map((event, index) => (
                <Line
                  key={`event-${index}`}
                  type="monotone"
                  dataKey="Civilization_Count"
                  stroke={eventColorMap[event.type]}
                  activeDot={false}
                  dot={[{ x: event.time, y: event.time % 5 + 1 }]}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-4 rounded shadow col-span-1 md:col-span-2">
          <h3 className="text-lg font-semibold mb-3">Events by Time Period</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={eventTimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" />
              <YAxis />
              <Tooltip />
              <Legend />
              {Object.keys(eventColorMap).map(eventType => (
                <Bar
                  key={eventType}
                  dataKey={eventType}
                  stackId="a"
                  fill={eventColorMap[eventType]}
                />
              ))}
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
      { name: 'Circuit Breaker', value: stabilityData.Circuit_Breaker_Triggers },
      { name: 'Collisions', value: stabilityData.Total_Collisions },
      { name: 'Mergers', value: stabilityData.Total_Mergers },
      { name: 'Collapses', value: stabilityData.Total_Collapses }
    ];

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold mb-3">Stability Issues Over Time</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={simulationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Time" />
              <YAxis />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="Stability_Issues"
                stroke="#d32f2f"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold mb-3">Adaptive Timestep</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={simulationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Time" />
              <YAxis domain={[0, 'dataMax']} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="Timestep"
                stroke="#2196f3"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-4 rounded shadow col-span-1 md:col-span-2">
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

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-xl">Loading simulation data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-xl text-red-600">{error}</div>
      </div>
    );
  }

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

      <div className="mt-6 text-sm text-gray-600">
        {stabilityData?.Used_Dimensional_Analysis && (
          <div className="bg-gray-200 p-2 rounded inline-block">
            Using dimensional analysis
          </div>
        )}
      </div>
    </div>
  );
};

// Render the dashboard component to the DOM
ReactDOM.render(<MultiCivilizationDashboard />, document.getElementById('root'));