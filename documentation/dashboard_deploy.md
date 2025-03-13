# Multi-Civilization Dashboard Production Deployment Guide

This guide provides detailed instructions for deploying the multi-civilization dashboard to a production environment. It covers setting up a proper production server, security considerations, performance optimizations, and monitoring strategies.

## Table of Contents

1. [Production Server Setup](#production-server-setup)
2. [Database Integration](#database-integration)
3. [Security Implementation](#security-implementation)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Backup and Recovery](#backup-and-recovery)
7. [Scaling Strategies](#scaling-strategies)
8. [Continuous Deployment](#continuous-deployment)

## Production Server Setup

### WSGI Server Configuration

Flask's built-in development server is not suitable for production. Use a production-grade WSGI server instead:

#### Gunicorn (Linux/Mac)

1. **Install Gunicorn**:
   ```bash
   pip install gunicorn
   ```

2. **Create a WSGI Entry Point** (`wsgi.py`):
   ```python
   from dashboard.app import app

   if __name__ == "__main__":
       app.run()
   ```

3. **Run with Gunicorn**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app
   ```

4. **Create a Systemd Service** (`/etc/systemd/system/dashboard.service`):
   ```ini
   [Unit]
   Description=Multi-Civilization Dashboard
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/path/to/dashboard
   ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

5. **Enable and Start the Service**:
   ```bash
   sudo systemctl enable dashboard
   sudo systemctl start dashboard
   ```

#### Waitress (Windows)

1. **Install Waitress**:
   ```bash
   pip install waitress
   ```

2. **Create a WSGI Entry Point** (`wsgi.py`):
   ```python
   from waitress import serve
   from dashboard.app import app

   if __name__ == "__main__":
       serve(app, host='0.0.0.0', port=8000)
   ```

3. **Run with Waitress**:
   ```bash
   python wsgi.py
   ```

4. **Create a Windows Service** (using NSSM):
   ```bash
   nssm install DashboardService "C:\path\to\python.exe" "C:\path\to\wsgi.py"
   nssm set DashboardService AppDirectory "C:\path\to\dashboard"
   nssm start DashboardService
   ```

### Nginx as a Reverse Proxy

Using Nginx as a reverse proxy provides benefits like SSL termination, static file serving, and load balancing:

1. **Install Nginx**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install nginx

   # CentOS/RHEL
   sudo yum install epel-release
   sudo yum install nginx
   ```

2. **Create Nginx Configuration** (`/etc/nginx/sites-available/dashboard`):
   ```nginx
   server {
       listen 80;
       server_name dashboard.yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }

       location /static {
           alias /path/to/dashboard/app/static;
           expires 30d;
       }
   }
   ```

3. **Enable the Site**:
   ```bash
   sudo ln -s /etc/nginx/sites-available/dashboard /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

### SSL Configuration with Let's Encrypt

1. **Install Certbot**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install certbot python3-certbot-nginx

   # CentOS/RHEL
   sudo yum install certbot python3-certbot-nginx
   ```

2. **Obtain SSL Certificate**:
   ```bash
   sudo certbot --nginx -d dashboard.yourdomain.com
   ```

3. **Auto-renewal Setup**:
   ```bash
   sudo certbot renew --dry-run
   ```

### Docker Deployment

For containerized deployment:

1. **Create a Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   RUN mkdir -p /app/outputs/data /app/outputs/dashboard

   EXPOSE 8000

   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "wsgi:app"]
   ```

2. **Create Docker Compose** (`docker-compose.yml`):
   ```yaml
   version: '3'

   services:
     dashboard:
       build: .
       ports:
         - "8000:8000"
       volumes:
         - ./outputs/data:/app/outputs/data
       restart: unless-stopped
       environment:
         - FLASK_ENV=production
         - SECRET_KEY=your-secret-key
       depends_on:
         - postgres   # If using a database

     postgres:        # Optional: for database persistence
       image: postgres:13
       environment:
         - POSTGRES_PASSWORD=securepassword
         - POSTGRES_USER=dashboard
         - POSTGRES_DB=simulation_data
       volumes:
         - postgres_data:/var/lib/postgresql/data
       restart: unless-stopped

   volumes:
     postgres_data:
   ```

3. **Deploy with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

## Database Integration

### SQLite for Small Deployments

For simpler deployments with moderate data size:

```python
import sqlite3
import json
from datetime import datetime
import pandas as pd

def init_db():
    conn = sqlite3.connect('dashboard.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''
    CREATE TABLE IF NOT EXISTS simulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        parameters TEXT
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS simulation_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        simulation_id INTEGER,
        time INTEGER,
        data TEXT,
        FOREIGN KEY (simulation_id) REFERENCES simulations (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def save_simulation(name, parameters, data_df):
    conn = sqlite3.connect('dashboard.db')
    c = conn.cursor()
    
    # Insert simulation record
    c.execute(
        "INSERT INTO simulations (name, timestamp, parameters) VALUES (?, ?, ?)",
        (name, datetime.now().isoformat(), json.dumps(parameters))
    )
    simulation_id = c.lastrowid
    
    # Insert each timestep
    for _, row in data_df.iterrows():
        time = row['Time']
        row_data = row.to_json()
        c.execute(
            "INSERT INTO simulation_data (simulation_id, time, data) VALUES (?, ?, ?)",
            (simulation_id, time, row_data)
        )
    
    conn.commit()
    conn.close()
    return simulation_id

def get_simulation_list():
    conn = sqlite3.connect('dashboard.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT id, name, timestamp FROM simulations ORDER BY timestamp DESC")
    rows = c.fetchall()
    
    conn.close()
    return [dict(row) for row in rows]

def get_simulation_data(simulation_id):
    conn = sqlite3.connect('dashboard.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT time, data FROM simulation_data WHERE simulation_id = ? ORDER BY time", (simulation_id,))
    rows = c.fetchall()
    
    conn.close()
    
    # Convert to DataFrame-like structure
    data = []
    for row in rows:
        entry = json.loads(row['data'])
        data.append(entry)
    
    return data
```

### PostgreSQL for Production

For larger deployments with significant data volume:

1. **Install Required Packages**:
   ```bash
   pip install psycopg2-binary sqlalchemy
   ```

2. **Database Connection and Models**:
   ```python
   from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
   from sqlalchemy.ext.declarative import declarative_base
   from sqlalchemy.orm import sessionmaker, relationship
   import datetime
   import json

   Base = declarative_base()

   class Simulation(Base):
       __tablename__ = 'simulations'
       
       id = Column(Integer, primary_key=True)
       name = Column(String, nullable=False)
       timestamp = Column(DateTime, default=datetime.datetime.utcnow)
       parameters = Column(String)
       
       timesteps = relationship("SimulationTimestep", back_populates="simulation")
       
   class SimulationTimestep(Base):
       __tablename__ = 'simulation_timesteps'
       
       id = Column(Integer, primary_key=True)
       simulation_id = Column(Integer, ForeignKey('simulations.id'))
       time = Column(Integer)
       civilization_count = Column(Integer)
       knowledge_mean = Column(Float)
       suppression_mean = Column(Float)
       intelligence_mean = Column(Float)
       truth_mean = Column(Float)
       
       simulation = relationship("Simulation", back_populates="timesteps")

   # Database connection
   DB_URI = "postgresql://username:password@localhost/dashboard"
   engine = create_engine(DB_URI)
   Base.metadata.create_all(engine)
   Session = sessionmaker(bind=engine)

   def save_simulation(name, parameters, data_df):
       session = Session()
       
       # Create simulation record
       sim = Simulation(name=name, parameters=json.dumps(parameters))
       session.add(sim)
       session.flush()  # Get ID without committing
       
       # Create timestep records
       for _, row in data_df.iterrows():
           timestep = SimulationTimestep(
               simulation_id=sim.id,
               time=row['Time'],
               civilization_count=row['Civilization_Count'],
               knowledge_mean=row['knowledge_mean'],
               suppression_mean=row['suppression_mean'],
               intelligence_mean=row['intelligence_mean'],
               truth_mean=row['truth_mean']
           )
           session.add(timestep)
       
       session.commit()
       session.close()
       return sim.id

   def get_simulation_list():
       session = Session()
       simulations = session.query(Simulation).order_by(Simulation.timestamp.desc()).all()
       
       result = [
           {"id": sim.id, "name": sim.name, "timestamp": sim.timestamp.isoformat()}
           for sim in simulations
       ]
       
       session.close()
       return result

   def get_simulation_data(simulation_id):
       session = Session()
       
       timesteps = session.query(SimulationTimestep).filter_by(
           simulation_id=simulation_id
       ).order_by(SimulationTimestep.time).all()
       
       result = [
           {
               "Time": ts.time,
               "Civilization_Count": ts.civilization_count,
               "knowledge_mean": ts.knowledge_mean,
               "suppression_mean": ts.suppression_mean,
               "intelligence_mean": ts.intelligence_mean,
               "truth_mean": ts.truth_mean
           }
           for ts in timesteps
       ]
       
       session.close()
       return result
   ```

3. **Add New Endpoints**:
   ```python
   @app.route('/api/simulations')
   def api_simulations():
       simulations = get_simulation_list()
       return jsonify(simulations)

   @app.route('/api/simulations/<int:simulation_id>')
   def api_simulation_data(simulation_id):
       data = get_simulation_data(simulation_id)
       return jsonify(data)

   @app.route('/api/simulations/<int:simulation_id>/charts')
   def api_simulation_charts(simulation_id):
       data = get_simulation_data(simulation_id)
       
       # Extract key metrics for charts
       chart_data = {
           "knowledge": [{"time": item["Time"], "value": item["knowledge_mean"]} for item in data],
           "suppression": [{"time": item["Time"], "value": item["suppression_mean"]} for item in data],
           "civilizations": [{"time": item["Time"], "value": item["Civilization_Count"]} for item in data]
       }
       
       return jsonify(chart_data)
   ```

## Security Implementation

### Authentication with Flask-Login

1. **Install Flask-Login**:
   ```bash
   pip install flask-login
   ```

2. **Setup Basic Authentication**:
   ```python
   from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
   from werkzeug.security import generate_password_hash, check_password_hash

   app = Flask(__name__)
   app.config['SECRET_KEY'] = 'your-secure-secret-key'

   login_manager = LoginManager()
   login_manager.init_app(app)
   login_manager.login_view = 'login'

   class User(UserMixin):
       def __init__(self, id, username, password_hash):
           self.id = id
           self.username = username
           self.password_hash = password_hash
           
       def check_password(self, password):
           return check_password_hash(self.password_hash, password)

   # Store users (replace with database in production)
   users = {
       'admin': User(1, 'admin', generate_password_hash('secure-admin-password'))
   }

   @login_manager.user_loader
   def load_user(user_id):
       return users.get(user_id)

   @app.route('/login', methods=['GET', 'POST'])
   def login():
       if request.method == 'POST':
           username = request.form.get('username')
           password = request.form.get('password')
           
           user = users.get(username)
           
           if user and user.check_password(password):
               login_user(user)
               next_page = request.args.get('next')
               return redirect(next_page or url_for('index'))
               
           flash('Invalid username or password')
           
       return render_template('login.html')

   @app.route('/logout')
   @login_required
   def logout():
       logout_user()
       return redirect(url_for('index'))

   @app.route('/')
   @login_required
   def index():
       return render_template('index.html')
   ```

3. **Create Login Template** (`templates/login.html`):
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <title>Login - Multi-Civilization Dashboard</title>
       <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
   </head>
   <body class="bg-light">
       <div class="container py-5">
           <div class="row justify-content-center">
               <div class="col-md-6">
                   <div class="card">
                       <div class="card-header">
                           <h4 class="mb-0">Login</h4>
                       </div>
                       <div class="card-body">
                           {% with messages = get_flashed_messages() %}
                           {% if messages %}
                           <div class="alert alert-danger">
                               {{ messages[0] }}
                           </div>
                           {% endif %}
                           {% endwith %}
                           
                           <form method="post">
                               <div class="mb-3">
                                   <label for="username" class="form-label">Username</label>
                                   <input type="text" class="form-control" id="username" name="username" required>
                               </div>
                               <div class="mb-3">
                                   <label for="password" class="form-label">Password</label>
                                   <input type="password" class="form-control" id="password" name="password" required>
                               </div>
                               <button type="submit" class="btn btn-primary">Login</button>
                           </form>
                       </div>
                   </div>
               </div>
           </div>
       </div>
   </body>
   </html>
   ```

### CSRF Protection

1. **Install Flask-WTF**:
   ```bash
   pip install flask-wtf
   ```

2. **Implement CSRF Protection**:
   ```python
   from flask_wtf.csrf import CSRFProtect

   csrf = CSRFProtect(app)

   # In your login form:
   from flask_wtf import FlaskForm
   from wtforms import StringField, PasswordField, SubmitField
   from wtforms.validators import DataRequired

   class LoginForm(FlaskForm):
       username = StringField('Username', validators=[DataRequired()])
       password = PasswordField('Password', validators=[DataRequired()])
       submit = SubmitField('Login')

   @app.route('/login', methods=['GET', 'POST'])
   def login():
       form = LoginForm()
       if form.validate_on_submit():
           # Process login
           username = form.username.data
           password = form.password.data
           # ...
       return render_template('login.html', form=form)
   ```

3. **Update Login Template**:
   ```html
   <form method="post">
       {{ form.csrf_token }}
       <div class="mb-3">
           {{ form.username.label(class_="form-label") }}
           {{ form.username(class_="form-control") }}
       </div>
       <div class="mb-3">
           {{ form.password.label(class_="form-label") }}
           {{ form.password(class_="form-control") }}
       </div>
       {{ form.submit(class_="btn btn-primary") }}
   </form>
   ```

### API Security

1. **API Authentication with Tokens**:
   ```python
   import secrets
   
   # Generate token for API access
   def generate_api_token():
       return secrets.token_hex(16)
   
   # API tokens (store in database in production)
   api_tokens = {
       'abcdef1234567890': {'user': 'api_user', 'scope': 'read'}
   }
   
   # API authentication decorator
   def api_auth_required(f):
       @wraps(f)
       def decorated(*args, **kwargs):
           token = request.headers.get('X-API-Token')
           if not token or token not in api_tokens:
               return jsonify({'error': 'Unauthorized'}), 401
           return f(*args, **kwargs)
       return decorated
   
   @app.route('/api/data', methods=['GET'])
   @api_auth_required
   def get_data():
       # Your protected API endpoint
       return jsonify({'data': 'Protected data'})
   ```

2. **Rate Limiting**:
   ```python
   from flask_limiter import Limiter
   from flask_limiter.util import get_remote_address

   limiter = Limiter(
       app,
       key_func=get_remote_address,
       default_limits=["200 per day", "50 per hour"]
   )

   @app.route('/api/data')
   @limiter.limit("10 per minute")
   def get_data():
       # Rate-limited endpoint
       return jsonify({'data': 'Limited access data'})
   ```

## Performance Optimization

### Data Caching

1. **Install Flask-Caching**:
   ```bash
   pip install Flask-Caching
   ```

2. **Implement Cache**:
   ```python
   from flask_caching import Cache

   cache_config = {
       "CACHE_TYPE": "SimpleCache",  # Use Redis or Memcached in production
       "CACHE_DEFAULT_TIMEOUT": 300
   }
   cache = Cache(app, config=cache_config)

   @app.route('/api/data/multi_civilization_statistics.csv')
   @cache.cached(timeout=60)  # Cache for 60 seconds
   def get_simulation_data():
       try:
           df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")
           return jsonify(df.to_dict('records'))
       except Exception as e:
           print(f"Error serving statistics data: {e}")
           return jsonify([])
   ```

3. **Redis Caching** (for production):
   ```python
   cache_config = {
       "CACHE_TYPE": "RedisCache",
       "CACHE_REDIS_HOST": "localhost",
       "CACHE_REDIS_PORT": 6379,
       "CACHE_REDIS_DB": 0,
       "CACHE_DEFAULT_TIMEOUT": 300
   }
   ```

### Data Processing Optimization

1. **Downsampling for Large Datasets**:
   ```python
   def downsample_timeseries(data, max_points=1000):
       """Downsample time series data to a maximum number of points."""
       if len(data) <= max_points:
           return data
           
       # Calculate step size
       step = len(data) // max_points
       
       # Sample at regular intervals
       return data[::step]
   
   @app.route('/api/data/downsampled')
   def get_downsampled_data():
       """Return downsampled data for efficient visualization."""
       try:
           df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")
           data = df.to_dict('records')
           
           # Downsample based on the visualization needs
           downsampled = downsample_timeseries(data, max_points=500)
           
           return jsonify(downsampled)
       except Exception as e:
           print(f"Error serving downsampled data: {e}")
           return jsonify([])
   ```

2. **Pre-aggregation for Dashboard Metrics**:
   ```python
   def precompute_aggregations():
       """Precompute statistical aggregations for dashboard metrics."""
       try:
           df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")
           
           # Compute aggregations
           aggregations = {
               "knowledge": {
                   "max": df['knowledge_max'].max(),
                   "min": df['knowledge_min'].min(),
                   "avg": df['knowledge_mean'].mean(),
                   "final": df.iloc[-1]['knowledge_mean']
               },
               "suppression": {
                   "max": df['suppression_max'].max(),
                   "min": df['suppression_min'].min(),
                   "avg": df['suppression_mean'].mean(),
                   "final": df.iloc[-1]['suppression_mean']
               },
               "stability": {
                   "total_issues": df['Stability_Issues'].sum(),
                   "max_issues": df['Stability_Issues'].max()
               }
           }
           
           # Cache the results
           cache.set("dashboard_aggregations", aggregations)
           return aggregations
           
       except Exception as e:
           print(f"Error precomputing aggregations: {e}")
           return {}
   
   @app.route('/api/metrics/aggregations')
   @cache.cached(timeout=300)
   def get_aggregations():
       """Return precomputed aggregations."""
       aggs = cache.get("dashboard_aggregations")
       if not aggs:
           aggs = precompute_aggregations()
       return jsonify(aggs)
   ```

### Frontend Performance

1. **Lazy Loading Charts**:
   ```javascript
   // Only initialize charts when tab is activated
   document.querySelectorAll('button[data-bs-toggle="pill"]').forEach(button => {
       button.addEventListener('shown.bs.tab', function (event) {
           const targetId = event.target.getAttribute('data-bs-target').substring(1);
           initializeChartsForTab(targetId);
       });
   });

   const initializedCharts = {};

   function initializeChartsForTab(tabId) {
       if (initializedCharts[tabId]) return;

       const chartContainers = document.querySelectorAll(`#${tabId} .chart-container`);
       chartContainers.forEach(container => {
           const chartId = container.querySelector('canvas').id;
           createChartById(chartId);
       });

       initializedCharts[tabId] = true;
   }
   ```

2. **Page Size Optimization**:
   - Minify CSS and JavaScript files
   - Use compression for HTTP responses
   - Implement appropriate caching headers

3. **HTTP/2 Support**:
   Update Nginx configuration to enable HTTP/2:
   ```nginx
   server {
       listen 443 ssl http2;
       # ...
   }
   ```

## Monitoring and Logging

### Application Logging

1. **Configure Flask Logging**:
   ```python
   import logging
   from logging.handlers import RotatingFileHandler
   import os

   def setup_logging(app):
       if not os.path.exists('logs'):
           os.mkdir('logs')
           
       file_handler = RotatingFileHandler(
           'logs/dashboard.log', 
           maxBytes=10240, 
           backupCount=10
       )
       file_handler.setFormatter(logging.Formatter(
           '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
       ))
       file_handler.setLevel(logging.INFO)
       
       app.logger.addHandler(file_handler)
       app.logger.setLevel(logging.INFO)
       app.logger.info('Dashboard startup')
   ```

2. **Log Key Events**:
   ```python
   @app.route('/api/data/multi_civilization_statistics.csv')
   def get_simulation_data():
       try:
           app.logger.info('Data request for statistics')
           df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")
           data = df.to_dict('records')
           app.logger.info(f'Returned {len(data)} data points')
           return jsonify(data)
       except Exception as e:
           app.logger.error(f'Error serving statistics data: {e}')
           return jsonify([])
   ```

### Health Checks

1. **Basic Health Check Endpoint**:
   ```python
   @app.route('/health')
   def health_check():
       try:
           # Check if data files exist
           stats_file = data_dir / "multi_civilization_statistics.csv"
           events_file = data_dir / "multi_civilization_events.csv"
           stability_file = data_dir / "multi_civilization_stability.csv"
           
           files_exist = (
               stats_file.exists() and 
               events_file.exists() and 
               stability_file.exists()
           )
           
           if not files_exist:
               return jsonify({
                   "status": "warning",
                   "message": "One or more data files missing"
               }), 200
           
           # Check database if used
           if has_database:
               check_database_connection()
           
           return jsonify({
               "status": "healthy",
               "timestamp": datetime.datetime.now().isoformat(),
               "version": "1.0.0"
           }), 200
           
       except Exception as e:
           app.logger.error(f"Health check failed: {e}")
           return jsonify({
               "status": "unhealthy",
               "message": str(e)
           }), 500
   ```

2. **Extended Health Check with Metrics**:
   ```python
   @app.route('/health/extended')
   def extended_health_check():
       try:
           # Basic health status
           health = {
               "status": "healthy",
               "timestamp": datetime.datetime.now().isoformat(),
               "version": "1.0.0"
           }
           
           # System metrics
           import psutil
           
           health["metrics"] = {
               "cpu_percent": psutil.cpu_percent(),
               "memory_percent": psutil.virtual_memory().percent,
               "disk_percent": psutil.disk_usage('/').percent,
           }
           
           # Application metrics
           health["app_metrics"] = {
               "uptime_seconds": (datetime.datetime.now() - app.start_time).total_seconds(),
               "request_count": app.request_count,
               "error_count": app.error_count
           }
           
           # Data metrics
           if os.path.exists(data_dir / "multi_civilization_statistics.csv"):
               df = pd.read_csv(data_dir / "multi_civilization_statistics.csv")
               health["data_metrics"] = {
                   "timesteps": len(df),
                   "last_updated": datetime.datetime.fromtimestamp(
                       os.path.getmtime(data_dir / "multi_civilization_statistics.csv")
                   ).isoformat()
               }
           
           return jsonify(health), 200
           
       except Exception as e:
           app.logger.error(f"Extended health check failed: {e}")
           return jsonify({
               "status": "unhealthy",
               "message": str(e)
           }), 500
   ```

### Monitoring Setup

1. **Prometheus Integration**:
   ```bash
   pip install prometheus-flask-exporter
   ```

   ```python
   from prometheus_flask_exporter import PrometheusMetrics

   metrics = PrometheusMetrics(app)

   # Static information as metric
   metrics.info('app_info', 'Application info', version='1.0.0')

   # Request count by endpoint
   @app.route('/api/data/<path:filename>')
   @metrics.counter('api_requests_count', 'Number of API requests by endpoint',
                    labels={'endpoint': lambda: request.path})
   def api_data(filename):
       # ...
   ```

2. **Grafana Dashboard Setup**:
   - Install Grafana on your server
   - Add Prometheus as a data source
   - Create a dashboard with panels for:
     - API request rate
     - Error rate
     - Response time by endpoint
     - System resource utilization
     - Data metrics (number of simulations, etc.)

## Backup and Recovery

### Data Backup Strategy

1. **Automated Database Backup**:
   ```python
   import subprocess
   import datetime
   
   def backup_database():
       """Create a backup of the PostgreSQL database."""
       timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
       backup_file = f'backups/dashboard_db_{timestamp}.sql'
       
       try:
           # Ensure backup directory exists
           os.makedirs('backups', exist_ok=True)
           
           # Run pg_dump command
           subprocess.run([
               'pg_dump',
               '-h', 'localhost',
               '-U', 'postgres',
               '-d', 'dashboard',
               '-f', backup_file
           ], check=True)
           
           app.logger.info(f'Database backup created: {backup_file}')
           return True
           
       except Exception as e:
           app.logger.error(f'Database backup failed: {e}')
           return False
   ```

2. **Schedule Regular Backups**:
   ```python
   from apscheduler.schedulers.background import BackgroundScheduler

   scheduler = BackgroundScheduler()
   scheduler.add_job(backup_database, 'cron', hour=3)  # Run daily at 3 AM
   scheduler.start()
   ```

3. **File System Backup**:
   ```python
   import shutil

   def backup_data_files():
       """Create a backup of simulation data files."""
       timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
       backup_dir = f'backups/data_{timestamp}'
       
       try:
           # Ensure backup directory exists
           os.makedirs(backup_dir, exist_ok=True)
           
           # Copy all CSV files
           for file in data_dir.glob('*.csv'):
               shutil.copy2(file, backup_dir)
               
           app.logger.info(f'Data files backup created: {backup_dir}')
           return True
           
       except Exception as e:
           app.logger.error(f'Data files backup failed: {e}')
           return False
   ```

### Recovery Procedures

1. **Database Recovery**:
   ```python
   def restore_database(backup_file):
       """Restore the database from a backup file."""
       try:
           # Run psql command to restore
           subprocess.run([
               'psql',
               '-h', 'localhost',
               '-U', 'postgres',
               '-d', 'dashboard',
               '-f', backup_file
           ], check=True)
           
           app.logger.info(f'Database restored from: {backup_file}')
           return True
           
       except Exception as e:
           app.logger.error(f'Database restore failed: {e}')
           return False
   ```

2. **Data File Recovery**:
   ```python
   def restore_data_files(backup_dir):
       """Restore data files from a backup directory."""
       try:
           # Copy all CSV files from backup to data directory
           backup_path = Path(backup_dir)
           for file in backup_path.glob('*.csv'):
               shutil.copy2(file, data_dir)
               
           app.logger.info(f'Data files restored from: {backup_dir}')
           return True
           
       except Exception as e:
           app.logger.error(f'Data files restore failed: {e}')
           return False
   ```

## Scaling Strategies

### Horizontal Scaling

1. **Load Balancing with Nginx**:
   ```nginx
   upstream dashboard_servers {
       server 127.0.0.1:8001;
       server 127.0.0.1:8002;
       server 127.0.0.1:8003;
       server 127.0.0.1:8004;
   }

   server {
       listen 80;
       server_name dashboard.yourdomain.com;

       location / {
           proxy_pass http://dashboard_servers;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

2. **Session Management with Redis**:
   ```python
   from flask_session import Session

   app.config['SESSION_TYPE'] = 'redis'
   app.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')
   Session(app)
   ```

### Vertical Scaling

1. **Optimize Resource Usage**:
   ```python
   # Configure Gunicorn with optimal worker count
   # Number of workers = (2 * CPU cores) + 1
   import multiprocessing

   workers = (2 * multiprocessing.cpu_count()) + 1
   
   # In gunicorn command:
   # gunicorn -w {workers} -b 0.0.0.0:8000 wsgi:app
   ```

2. **Database Connection Pooling**:
   ```python
   # SQLAlchemy connection pooling
   engine = create_engine(
       DB_URI,
       pool_size=10,
       max_overflow=20,
       pool_recycle=1800
   )
   ```

### Cloud Deployment

For AWS deployment:

1. **Elastic Beanstalk Configuration** (`requirements.txt`):
   ```
   flask==2.0.1
   pandas==1.3.3
   numpy==1.21.2
   gunicorn==20.1.0
   psycopg2-binary==2.9.1
   flask-login==0.5.0
   flask-wtf==0.15.1
   flask-caching==1.10.1
   flask-session==0.4.0
   prometheus-flask-exporter==0.18.2
   apscheduler==3.8.1
   ```

2. **Dockerized Deployment** (`Dockerfile`):
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   # Set environment variables
   ENV FLASK_APP=wsgi.py
   ENV FLASK_ENV=production

   # Create necessary directories
   RUN mkdir -p outputs/data outputs/dashboard logs backups

   # Expose the application port
   EXPOSE 8000

   # Run the application with Gunicorn
   CMD gunicorn --bind 0.0.0.0:8000 wsgi:app
   ```

## Continuous Deployment

### GitHub Actions Workflow

Create a GitHub Actions workflow file (`.github/workflows/deploy.yml`):

```yaml
name: Deploy Dashboard

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Run tests
      run: |
        pytest
    
    - name: Build Docker image
      run: |
        docker build -t dashboard:${{ github.sha }} .
    
    - name: Log in to registry
      uses: docker/login-action@v1
      with:
        registry: your-registry.com
        username: ${{ secrets.REGISTRY_USERNAME }}
        password: ${{ secrets.REGISTRY_PASSWORD }}
    
    - name: Push Docker image
      run: |
        docker tag dashboard:${{ github.sha }} your-registry.com/dashboard:${{ github.sha }}
        docker tag dashboard:${{ github.sha }} your-registry.com/dashboard:latest
        docker push your-registry.com/dashboard:${{ github.sha }}
        docker push your-registry.com/dashboard:latest
    
    - name: Deploy to server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.SSH_HOST }}
        username: ${{ secrets.SSH_USERNAME }}
        key: ${{ secrets.SSH_KEY }}
        script: |
          cd /path/to/dashboard
          docker-compose pull
          docker-compose up -d
```

### Automated Testing

1. **Basic Test Setup** (`tests/test_dashboard.py`):
   ```python
   import pytest
   from app import app as flask_app

   @pytest.fixture
   def app():
       flask_app.config.update({
           "TESTING": True,
       })
       yield flask_app

   @pytest.fixture
   def client(app):
       return app.test_client()

   def test_home_page(client):
       response = client.get('/')
       assert response.status_code == 302  # Redirect to login

   def test_api_endpoints(client):
       # Test API endpoints
       response = client.get('/api/data/multi_civilization_statistics.csv')
       assert response.status_code == 200
       data = response.get_json()
       assert isinstance(data, list)
   ```

2. **Load Testing Script** (`tests/load_test.py`):
   ```python
   import requests
   import time
   import concurrent.futures
   import statistics

   def make_request(url):
       start_time = time.time()
       response = requests.get(url)
       end_time = time.time()
       return {
           'status_code': response.status_code,
           'response_time': end_time - start_time
       }

   def run_load_test(url, num_requests=100, concurrent=10):
       results = []
       
       with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
           futures = [executor.submit(make_request, url) for _ in range(num_requests)]
           
           for future in concurrent.futures.as_completed(futures):
               results.append(future.result())
       
       # Analyze results
       response_times = [r['response_time'] for r in results]
       success_count = sum(1 for r in results if r['status_code'] == 200)
       
       print(f"Total requests: {num_requests}")
       print(f"Successful requests: {success_count} ({success_count / num_requests * 100:.2f}%)")
       print(f"Average response time: {statistics.mean(response_times):.3f} seconds")
       print(f"Median response time: {statistics.median(response_times):.3f} seconds")
       print(f"Min response time: {min(response_times):.3f} seconds")
       print(f"Max response time: {max(response_times):.3f} seconds")

   if __name__ == "__main__":
       # Run load test against statistics endpoint
       run_load_test("http://localhost:5000/api/data/multi_civilization_statistics.csv")
   ```