/**
 * Multi-Civilization Simulation Dashboard (React Implementation)
 *
 * This is the primary dashboard visualization for the multi-civilization simulation.
 * Uses React and Recharts for rendering.
 */

// Use React's hooks directly (they're already defined in the global React object)
const { useState, useEffect } = React;

// Get Recharts components from the global Recharts object
const {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ScatterChart, Scatter, ZAxis, Cell, PieChart, Pie,
  AreaChart, Area, ComposedChart
} = Recharts;  // Direct reference to global Recharts object

/**
 * Error Boundary Component
 * Catches JavaScript errors anywhere in child component tree
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: 0
    };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to console
    console.error("Error caught by ErrorBoundary:", error, errorInfo);
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  handleRetry = () => {
    this.setState(prevState => ({
      hasError: false,
      retryCount: prevState.retryCount + 1
    }));
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="bg-white p-6 rounded shadow text-center">
          <h3 className="text-lg font-semibold mb-4">Something went wrong</h3>
          <p className="text-gray-600 mb-4">We encountered an error while loading this component.</p>
          <details className="mb-4 text-left">
            <summary className="cursor-pointer text-blue-600">Error Details</summary>
            <pre className="mt-2 p-3 bg-gray-100 rounded text-red-600 text-sm overflow-auto">
              {this.state.error && this.state.error.toString()}
            </pre>
          </details>
          <button
            onClick={this.handleRetry}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      );
    }

    // Normally, just render children
    return this.props.children;
  }
}

/**
 * API Utility Object
 * Handles all API requests with error handling
 */
const api = {
  /**
   * Generic fetch wrapper with error handling
   */
  async fetchWithErrorHandling(url, options = {}) {
    try {
      const response = await fetch(url, options);

      if (!response.ok) {
        // Try to parse error message if available
        let errorMessage;
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || `Error: ${response.status} ${response.statusText}`;
        } catch (e) {
          errorMessage = `Error: ${response.status} ${response.statusText}`;
        }

        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error; // Re-throw to let components handle it
    }
  },

  /**
   * Helper to build URLs with query parameters
   */
  buildApiUrl(endpoint, params = {}) {
    const url = new URL(endpoint, window.location.origin);
    Object.keys(params).forEach(key => {
      if (params[key] !== null && params[key] !== undefined) {
        url.searchParams.append(key, params[key]);
      }
    });
    return url.toString();
  },

  /**
   * Specific API methods
   */
  async getMetrics() {
    return this.fetchWithErrorHandling('/api/meta/available_metrics');
  },

  async getSimulationData(params = {}) {
    const url = this.buildApiUrl('/api/data/multi_civilization_statistics.csv', params);
    return this.fetchWithErrorHandling(url);
  },

  async getEventData(params = {}) {
    const url = this.buildApiUrl('/api/data/multi_civilization_events.csv', params);
    return this.fetchWithErrorHandling(url);
  },

  async getStabilityData() {
    return this.fetchWithErrorHandling('/api/data/multi_civilization_stability.csv');
  },

  async getCorrelationData() {
    return this.fetchWithErrorHandling('/api/analysis/knowledge_suppression_correlation');
  },

  async getCivilizationComparison(params = {}) {
    const url = this.buildApiUrl('/api/data/civilization_comparison', params);
    return this.fetchWithErrorHandling(url);
  },

  async exportData(dataset) {
    window.location.href = this.buildApiUrl('/api/export/csv', { dataset });
  }
};

const MultiCivilizationDashboard = () => {
  // State variables
  const [simulationData, setSimulationData] = useState(null);
  const [eventData, setEventData] = useState(null);
  const [stabilityData, setStabilityData] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [correlationData, setCorrelationData] = useState(null);
  const [availableMetrics, setAvailableMetrics] = useState(null);
  const [selectedTimeStep, setSelectedTimeStep] = useState(0);
  const [timeRange, setTimeRange] = useState({ start: 0, end: 150 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [view, setView] = useState('overview'); // overview, civilizations, events, stability, analysis
  const [selectedMetrics, setSelectedMetrics] = useState(['knowledge_mean', 'suppression_mean']);
  const [downSampleFactor, setDownSampleFactor] = useState(1); // Default no downsampling
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [dataRefreshKey, setDataRefreshKey] = useState(0); // Used to force data refresh

  // Color configurations
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

  // Load available metrics to understand what data is available
  useEffect(() => {
    const loadMetadata = async () => {
      try {
        setLoading(true);
        const data = await api.getMetrics();
        setAvailableMetrics(data);

        // Update time range based on available data
        if (data.time_range) {
          setTimeRange({
            start: data.time_range.min,
            end: data.time_range.max
          });
          // Set selected time step to the middle by default
          setSelectedTimeStep(Math.floor((data.time_range.min + data.time_range.max) / 2));
        }
        setError(null);
      } catch (err) {
        console.error('Error loading metrics metadata:', err);
        setError(`Failed to load metrics information: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    loadMetadata();
  }, [dataRefreshKey]);

  // Load simulation data
  useEffect(() => {
    const loadData = async () => {
      if (!availableMetrics) return;

      try {
        setLoading(true);
        console.log("Fetching simulation data...");

        // Prepare API parameters
        const statsParams = {
          time_start: timeRange.start,
          time_end: timeRange.end,
          metrics: selectedMetrics.join(','),
          down_sample: downSampleFactor > 1 ? downSampleFactor : null
        };

        // Fetch data in parallel for better performance
        const [statsData, eventsData, stabilityData, correlationData] = await Promise.all([
          // Get statistics data
          api.getSimulationData(statsParams),

          // Get events data
          api.getEventData({
            time_start: timeRange.start,
            time_end: timeRange.end
          }),

          // Get stability data
          api.getStabilityData(),

          // Get correlation data
          api.getCorrelationData().catch(err => {
            console.warn("Correlation data not available:", err);
            return null;
          })
        ]);

        // Log received data
        console.log("Received stats data:", statsData ? statsData.length : 0, "records");
        console.log("Received events data:", eventsData ? eventsData.length : 0, "records");
        console.log("Received stability data");

        // Update state with fetched data
        setSimulationData(statsData);
        setEventData(eventsData);
        setStabilityData(stabilityData);
        if (correlationData) {
          setCorrelationData(correlationData);
        }
        setError(null);
      } catch (err) {
        console.error('Error loading simulation data:', err);
        setError(`Failed to load simulation data: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    if (availableMetrics) {
      loadData();
    }
  }, [timeRange, selectedMetrics, downSampleFactor, availableMetrics, dataRefreshKey]);

  // Load civilization comparison data when changing time step
  useEffect(() => {
    const loadComparisonData = async () => {
      if (!simulationData) return;

      try {
        // Fetch civilization comparison data
        const comparisonData = await api.getCivilizationComparison({
          time: selectedTimeStep,
          metric: 'knowledge',
          count: 3
        });

        setComparisonData(comparisonData);
      } catch (err) {
        console.warn('Civilization comparison data not available:', err);
        // We don't set the error state here as this is not critical
        // Just leave the comparison data as null
      }
    };

    loadComparisonData();
  }, [selectedTimeStep, simulationData]);

  // Function to retry data loading
  const handleRetryDataLoad = () => {
    setDataRefreshKey(prev => prev + 1); // This will trigger useEffect to run again
  };

  // Time slider handler
  const handleTimeSliderChange = (event) => {
    setSelectedTimeStep(parseInt(event.target.value, 10));
  };

  // Handle time range change
  const handleTimeRangeChange = (start, end) => {
    setTimeRange({ start, end });
  };

  // Handle metric selection
  const handleMetricSelection = (metric) => {
    if (selectedMetrics.includes(metric)) {
      // Remove metric if already selected
      setSelectedMetrics(selectedMetrics.filter(m => m !== metric));
    } else {
      // Add metric if not already selected
      setSelectedMetrics([...selectedMetrics, metric]);
    }
  };

  // Handle downsampling change
  const handleDownSampleChange = (factor) => {
    setDownSampleFactor(factor);
  };

  // Export data handler
  const handleExportData = async (dataset) => {
    setIsExporting(true);
    try {
      await api.exportData(dataset);
      // setTimeout to give the browser time to start the download
      setTimeout(() => setIsExporting(false), 1000);
    } catch (err) {
      console.error('Error exporting data:', err);
      setIsExporting(false);
    }
  };

  // Toggle fullscreen mode
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
    setIsFullscreen(!isFullscreen);
  };

  // Filter events up to the selected time
  const getEventsUpToTime = () => {
    if (!eventData) return [];
    return eventData.filter(event => event.time <= selectedTimeStep);
  };

  // Get current state at the selected time
  const getCurrentState = () => {
    if (!simulationData) return null;
    const closeTimePoints = simulationData.filter(d => d.Time <= selectedTimeStep);
    if (closeTimePoints.length === 0) return null;

    // Find the closest time point
    return closeTimePoints.reduce((prev, curr) => {
      return (Math.abs(curr.Time - selectedTimeStep) < Math.abs(prev.Time - selectedTimeStep) ? curr : prev);
    });
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
      <button
        className={`px-4 py-2 rounded ${view === 'analysis' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
        onClick={() => setView('analysis')}
      >
        Analysis
      </button>
    </div>
  );

  // Controls panel
  const renderControls = () => (
    <div className="bg-white p-4 rounded shadow mb-4">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-lg font-semibold">Simulation Controls</h2>
        <div className="flex space-x-2">
          <button
            className="bg-blue-600 text-white px-4 py-1 rounded text-sm"
            onClick={() => handleExportData('statistics')}
            disabled={isExporting}
          >
            {isExporting ? 'Exporting...' : 'Export Data'}
          </button>
          <button
            className="bg-gray-200 px-4 py-1 rounded text-sm"
            onClick={toggleFullscreen}
          >
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h3 className="text-sm font-medium mb-1">Time Range</h3>
          <div className="flex space-x-2 items-center">
            <input
              type="number"
              className="w-16 px-2 py-1 border rounded"
              value={timeRange.start}
              onChange={(e) => handleTimeRangeChange(parseInt(e.target.value), timeRange.end)}
              min={availableMetrics?.time_range?.min || 0}
              max={timeRange.end - 1}
            />
            <span>to</span>
            <input
              type="number"
              className="w-16 px-2 py-1 border rounded"
              value={timeRange.end}
              onChange={(e) => handleTimeRangeChange(timeRange.start, parseInt(e.target.value))}
              min={timeRange.start + 1}
              max={availableMetrics?.time_range?.max || 150}
            />
            <button
              className="bg-gray-200 px-2 py-1 rounded text-sm"
              onClick={() => handleTimeRangeChange(
                availableMetrics?.time_range?.min || 0,
                availableMetrics?.time_range?.max || 150
              )}
            >
              Reset
            </button>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-medium mb-1">Data Sampling</h3>
          <div className="flex space-x-2">
            <select
              className="px-2 py-1 border rounded"
              value={downSampleFactor}
              onChange={(e) => handleDownSampleChange(parseInt(e.target.value))}
            >
              <option value="1">No sampling</option>
              <option value="2">Every 2nd point</option>
              <option value="5">Every 5th point</option>
              <option value="10">Every 10th point</option>
            </select>
            <span className="text-sm text-gray-600 self-center">
              {simulationData ? `(${simulationData.length} points)` : ''}
            </span>
          </div>
        </div>
      </div>

      <div className="mt-4">
        <h3 className="text-sm font-medium mb-1">Metrics Selection</h3>
        <div className="flex flex-wrap gap-2">
          {availableMetrics?.metric_categories?.knowledge?.map(metric => (
            <button
              key={metric}
              className={`px-2 py-1 text-xs rounded ${
                selectedMetrics.includes(metric)
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-200'
              }`}
              onClick={() => handleMetricSelection(metric)}
            >
              {metric}
            </button>
          ))}
          {availableMetrics?.metric_categories?.suppression?.map(metric => (
            <button
              key={metric}
              className={`px-2 py-1 text-xs rounded ${
                selectedMetrics.includes(metric)
                  ? 'bg-red-600 text-white'
                  : 'bg-gray-200'
              }`}
              onClick={() => handleMetricSelection(metric)}
            >
              {metric}
            </button>
          ))}
          {availableMetrics?.metric_categories?.intelligence?.map(metric => (
            <button
              key={metric}
              className={`px-2 py-1 text-xs rounded ${
                selectedMetrics.includes(metric)
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-200'
              }`}
              onClick={() => handleMetricSelection(metric)}
            >
              {metric}
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  // Time slider and current stats
  const renderTimeSlider = () => {
    const currentState = getCurrentState();

    return (
      <div className="mb-6 bg-white p-4 rounded shadow">
        <h2 className="text-lg font-semibold mb-2">Time Step: {selectedTimeStep}</h2>
        <input
          type="range"
          min={timeRange.start}
          max={timeRange.end}
          value={selectedTimeStep}
          onChange={handleTimeSliderChange}
          className="w-full"
        />

        {currentState && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div className="bg-gray-100 p-3 rounded">
              <div className="text-sm text-gray-600">Civilizations</div>
              <div className="text-2xl font-bold">{currentState.Civilization_Count || 0}</div>
            </div>
            <div className="bg-gray-100 p-3 rounded">
              <div className="text-sm text-gray-600">Avg. Knowledge</div>
              <div className="text-2xl font-bold">{(currentState.knowledge_mean || 0).toFixed(1)}</div>
            </div>
            <div className="bg-gray-100 p-3 rounded">
              <div className="text-sm text-gray-600">Avg. Suppression</div>
              <div className="text-2xl font-bold">{(currentState.suppression_mean || 0).toFixed(1)}</div>
            </div>
            <div className="bg-gray-100 p-3 rounded">
              <div className="text-sm text-gray-600">Events</div>
              <div className="text-2xl font-bold">{getEventsUpToTime().length}</div>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Overview view
  const renderOverview = () => {
    const currentState = getCurrentState();

    if (!currentState) return <div>No data available</div>;

    // Prepare data for the civilization positions
    const civPositions = comparisonData
      ? [...comparisonData.top_civilizations, ...comparisonData.bottom_civilizations].map(civ => ({
          x: Math.random() * 10,  // In real implementation, would use actual positions
          y: Math.random() * 10,
          z: civ.knowledge,  // Size based on knowledge
          influence: civ.influence,
          id: civ.id
        }))
      : Array.from({ length: currentState.Civilization_Count || 5 }, (_, i) => ({
          x: Math.random() * 10,
          y: Math.random() * 10,
          z: Math.random() * (currentState.knowledge_max || 50),
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
            <AreaChart data={simulationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Time" />
              <YAxis />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="resources_total"
                stroke={colorMap.resources}
                fill={colorMap.resources}
                fillOpacity={0.3}
              />
            </AreaChart>
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
            <ComposedChart data={simulationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Time" />
              <YAxis />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="Civilization_Count"
                fill="#8884d8"
                stroke="#8884d8"
                fillOpacity={0.2}
              />
              {/* Mark events on the chart */}
              {Object.entries(eventColorMap).map(([eventType, color]) => {
                // Only show event types that exist in the data
                const relevantEvents = events.filter(e => e.type === eventType);
                if (relevantEvents.length === 0) return null;

                // Create scatter points for each event of this type
                const scatterData = relevantEvents.map(event => ({
                  Time: event.time,
                  value: 2 // Fixed value for visibility
                }));

                return (
                  <Scatter
                    key={eventType}
                    name={eventType}
                    data={scatterData}
                    fill={color}
                    shape="star"
                  />
                );
              })}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-4 rounded shadow col-span-1 md:col-span-2">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-lg font-semibold">Events by Time Period</h3>
            <div className="text-sm text-gray-600">
              {events.length} total events
            </div>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={eventTimeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" label="Time Period" />
              <YAxis label={{ value: 'Event Count', angle: -90, position: 'insideLeft' }} />
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

        <div className="bg-white p-4 rounded shadow col-span-1 md:col-span-2 mt-4">
          <h3 className="text-lg font-semibold mb-3">Recent Events Log</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Description
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Details
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {events.slice(0, 10).map((event, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {event.time}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full"
                        style={{
                          backgroundColor: eventColorMap[event.type] || '#999',
                          color: 'white'
                        }}
                      >
                        {event.type}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {event.description || `${event.type} event at time ${event.time}`}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {event.type === 'merger' &&
                        `Civ ${event.absorber} absorbed Civ ${event.absorbed}`}
                      {event.type === 'collapse' &&
                        `Civ ${event.civilization} collapsed`}
                      {event.type === 'spawn' &&
                        `Spawned from Civ ${event.parent}`}
                    </td>
                  </tr>
                ))}
                {events.length === 0 && (
                  <tr>
                    <td colSpan="4" className="px-6 py-4 text-center text-sm text-gray-500">
                      No events to display
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
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

        <div className="bg-white p-4 rounded shadow col-span-1 md:col-span-2">
          <h3 className="text-lg font-semibold mb-3">Stability Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-100 p-3 rounded">
              <div className="text-sm text-gray-600">Total Stability Issues</div>
              <div className="text-xl font-bold">{stabilityData.Total_Stability_Issues || 0}</div>
            </div>
            <div className="bg-gray-100 p-3 rounded">
              <div className="text-sm text-gray-600">Max Knowledge</div>
              <div className="text-xl font-bold">{(stabilityData.Max_Knowledge || 0).toFixed(1)}</div>
            </div>
            <div className="bg-gray-100 p-3 rounded">
              <div className="text-sm text-gray-600">Max Suppression</div>
              <div className="text-xl font-bold">{(stabilityData.Max_Suppression || 0).toFixed(1)}</div>
            </div>
            <div className="bg-gray-100 p-3 rounded">
              <div className="text-sm text-gray-600">Using Dimensional Analysis</div>
              <div className="text-xl font-bold">{stabilityData.Used_Dimensional_Analysis ? 'Yes' : 'No'}</div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Analysis view
  const renderAnalysis = () => {
    if (!correlationData) {
      return (
        <div className="bg-white p-6 rounded shadow text-center">
          <h3 className="text-lg font-semibold mb-4">Advanced Analysis</h3>
          <p className="text-gray-600">Correlation data is not available. Please make sure the API endpoint is properly implemented.</p>
        </div>
      );
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold mb-3">Knowledge-Suppression Correlation</h3>
          <div className="mb-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Correlation Coefficient:</span>
              <span className="text-lg font-bold">{correlationData.correlation_coefficient.toFixed(3)}</span>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              {correlationData.interpretation}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid />
              <XAxis
                type="number"
                dataKey="knowledge_mean"
                name="Knowledge"
                label={{ value: 'Knowledge', position: 'bottom' }}
              />
              <YAxis
                type="number"
                dataKey="suppression_mean"
                name="Suppression"
                label={{ value: 'Suppression', angle: -90, position: 'left' }}
              />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Knowledge vs Suppression" data={correlationData.trend_data} fill="#8884d8" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold mb-3">Civilization Comparison</h3>
          {comparisonData ? (
            <>
              <div className="mb-4">
                <div className="text-sm text-gray-600">
                  Comparing top and bottom 3 civilizations at time {comparisonData.time_point}
                </div>
                <div className="mt-2 grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm font-medium text-gray-500">Top Civilizations</div>
                    <ul className="mt-1">
                      {comparisonData.top_civilizations.map((civ, idx) => (
                        <li key={idx} className="text-sm">
                          Civ {civ.id}: Knowledge {civ.knowledge.toFixed(1)}, Influence {civ.influence.toFixed(1)}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-500">Bottom Civilizations</div>
                    <ul className="mt-1">
                      {comparisonData.bottom_civilizations.map((civ, idx) => (
                        <li key={idx} className="text-sm">
                          Civ {civ.id}: Knowledge {civ.knowledge.toFixed(1)}, Influence {civ.influence.toFixed(1)}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart
                  data={[...comparisonData.top_civilizations, ...comparisonData.bottom_civilizations]}
                  layout="vertical"
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis
                    dataKey="id"
                    type="category"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => `Civ ${value}`}
                  />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="knowledge" fill={colorMap.knowledge} name="Knowledge" />
                  <Bar dataKey="influence" fill={colorMap.influence} name="Influence" />
                </BarChart>
              </ResponsiveContainer>
            </>
          ) : (
            <div className="text-center p-8 text-gray-500">
              Select a time point to see civilization comparison
            </div>
          )}
        </div>

        <div className="bg-white p-4 rounded shadow col-span-1 md:col-span-2">
          <h3 className="text-lg font-semibold mb-3">Knowledge Growth Analysis</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={simulationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="Time" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Area
                yAxisId="left"
                type="monotone"
                dataKey="knowledge_mean"
                fill={colorMap.knowledge}
                stroke={colorMap.knowledge}
                fillOpacity={0.3}
                name="Knowledge (Mean)"
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="knowledge_max"
                stroke={colorMap.knowledge}
                strokeDasharray="5 5"
                name="Knowledge (Max)"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="Civilization_Count"
                stroke="#8884d8"
                name="Civilization Count"
              />
              {/* Mark major events if available */}
              {eventData && Object.entries(eventColorMap).map(([eventType, color]) => {
                if (eventType === 'new_civilization' || eventType === 'collapse') {
                  const relevantEvents = eventData.filter(e => e.type === eventType);
                  if (relevantEvents.length === 0) return null;

                  // Create scatter points for each event of this type
                  const scatterData = relevantEvents.map(event => ({
                    Time: event.time,
                    value: eventType === 'new_civilization' ? 1 : -1 // Up for new, down for collapse
                  }));

                  return (
                    <Scatter
                      key={eventType}
                      yAxisId="right"
                      name={eventType}
                      data={scatterData}
                      fill={color}
                      shape="triangle"
                    />
                  );
                }
                return null;
              })}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  // Main rendering logic with loading and error states
  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-xl">Loading simulation data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col justify-center items-center h-screen bg-gray-50 p-6">
        <div className="text-xl text-red-600 mb-4">{error}</div>
        <button
          onClick={handleRetryDataLoad}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Retry Loading Data
        </button>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Multi-Civilization Simulation Dashboard</h1>

      {/* Navigation buttons */}
      {renderNavButtons()}

      {/* Wrap key components with ErrorBoundary */}
      <ErrorBoundary>
        {/* Controls panel */}
        {renderControls()}
      </ErrorBoundary>

      <ErrorBoundary>
        {/* Time slider */}
        {renderTimeSlider()}
      </ErrorBoundary>

      {/* View specific content - each wrapped in ErrorBoundary */}
      <ErrorBoundary>
        {view === 'overview' && renderOverview()}
      </ErrorBoundary>

      <ErrorBoundary>
        {view === 'civilizations' && renderCivilizations()}
      </ErrorBoundary>

      <ErrorBoundary>
        {view === 'events' && renderEvents()}
      </ErrorBoundary>

      <ErrorBoundary>
        {view === 'stability' && renderStability()}
      </ErrorBoundary>

      <ErrorBoundary>
        {view === 'analysis' && renderAnalysis()}
      </ErrorBoundary>

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

ReactDOM.render(
  <ErrorBoundary>
    <MultiCivilizationDashboard />
  </ErrorBoundary>,
  document.getElementById('root')
);