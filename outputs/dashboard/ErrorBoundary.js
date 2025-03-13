// ErrorBoundary.js - Copy this to a separate file in your project
// This component catches errors in React components and displays a fallback UI

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

    // You could also log to an error reporting service here
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

// api.js - API utility functions with error handling
// Copy this to a separate file or incorporate into your React components

// API utility functions
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
   * Specific API functions
   */
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

  async getCivilizationComparison(params = {}) {
    const url = this.buildApiUrl('/api/data/civilization_comparison', params);
    return this.fetchWithErrorHandling(url);
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
  }
};

// Usage example in a component:
/*
const DataDisplay = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const result = await api.getSimulationData({
          time_start: 0,
          time_end: 150
        });
        setData(result);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  if (loading) return <p>Loading...</p>;
  if (error) return <p className="text-red-600">Error: {error}</p>;

  return (
    <div>
      {/* Display your data here */}
    </div>
  );
}

// Then wrap components with ErrorBoundary:
<ErrorBoundary>
  <DataDisplay />
</ErrorBoundary>
*/