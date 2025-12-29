# Requirements Breakdown

This document provides a clear separation of dependencies by deployment environment.

## Python Version
**Python 3.11** (specified in `.python-version`)

---

## Local Development Requirements

These are the dependencies needed for local mesh generation, GUI applications, and development tools.

### Core Mesh Generation
```
gmsh>=4.11.0
numpy>=1.24.0
scipy>=1.10.0
```

### GUI Applications & Visualization
```
PyQt5>=5.15.0          # Desktop GUI (main.py)
PyQt6>=6.4.0           # Montage GUI (montage_gui.py)
vtk>=9.2.0             # 3D visualization backend
pyvista>=0.40.0        # High-level VTK wrapper
qtrangeslider>=0.3.0   # Custom slider widget for GUI
```

### CAD File Support & Mesh I/O
```
pythonocc-core>=7.7.0  # STEP/IGES CAD file loading
meshio>=5.3.4          # Mesh format conversion (ANSYS export)
trimesh>=3.22.0        # Mesh processing utilities
coacd>=1.0.0           # Convex decomposition for hex meshing
pymesh2>=0.3.0         # Mesh repair and generation
tetgen>=0.6.0          # Alternative tetrahedral mesher
```

### Utilities
```
psutil>=5.9.0          # System resource monitoring
requests>=2.31.0       # HTTP requests
anthropic>=0.8.0       # AI Chat integration (optional)
python-dotenv>=1.0.0   # Environment variable management
```

### Development & Testing
```
pytest>=7.4.0          # Testing framework
black>=23.7.0          # Code formatting
```

### Optional: GPU Acceleration
```
# cupy-cuda12x>=12.0.0  # Uncomment if using CUDA 12
```

---

## AWS Backend Requirements (Deployed Separately)

These dependencies are used by the Flask API server running on AWS and are **NOT needed for local mesh generation**.

### Web Framework & API
```
flask>=3.0.0
flask-cors>=4.0.0
flask-jwt-extended>=4.5.3
flask-socketio>=5.3.6
werkzeug>=3.0.0
gunicorn>=21.2.0       # Production WSGI server
eventlet>=0.33.3       # Async support for SocketIO
```

### Database & ORM
```
sqlalchemy>=2.0.0
Flask-SQLAlchemy>=3.1.1
psycopg2-binary>=2.9.9  # PostgreSQL adapter
marshmallow>=3.20.1     # Serialization/validation
```

### Authentication & Security
```
bcrypt>=4.0.1
email-validator>=2.0.0
```

### Task Queue & Caching
```
celery>=5.3.4          # Distributed task queue
redis>=5.0.1           # Message broker & result backend
```

### Cloud Storage
```
boto3>=1.28.0          # AWS S3 integration
```

---

## Streamlit Web App Requirements (Optional)

If deploying the Streamlit web interface (separate from the main backend):

```
streamlit>=1.28.0      # Web app framework
plotly>=5.17.0         # Interactive 3D visualization
meshio>=5.3.4          # Already listed above
gmsh>=4.11.0           # Already listed above
numpy>=1.24.0          # Already listed above
```

---

## Installation Instructions

### For Local Development Only
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install only local requirements (subset of requirements.txt)
pip install gmsh numpy scipy PyQt5 PyQt6 vtk pyvista qtrangeslider \
            pythonocc-core meshio trimesh coacd pymesh2 tetgen \
            psutil requests anthropic python-dotenv pytest black
```

### For Full Installation (All Components)
```bash
pip install -r requirements.txt
```

### For AWS Backend Deployment
The backend uses the full `requirements.txt` but is deployed to AWS infrastructure where Flask, Celery, Redis, and PostgreSQL services are configured.

---

## System Dependencies

These are external tools that must be installed at the OS level:

- **OpenFOAM v2006+** (via WSL2 on Windows) - Required for Hex-Dominant meshing strategies
- **WSL2** (Windows users) - Required for OpenFOAM integration
- **CUDA Toolkit 12+** (optional) - Required for GPU-accelerated meshing with `cupy`
