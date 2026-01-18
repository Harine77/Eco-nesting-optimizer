# 🧵 Fabric Nesting Optimizer

An intelligent fabric cutting optimization system that uses computer vision and genetic algorithms to minimize material waste and maximize fabric utilization in textile manufacturing.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12+-blue.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Database Schema](#database-schema)
- [Algorithm Details](#algorithm-details)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)

## 🎯 Overview

Fabric Nesting Optimizer is a web application designed to help textile manufacturers and designers optimize fabric usage by automatically detecting cut patterns and arranging them efficiently on fabric sheets. The system uses advanced computer vision techniques and genetic algorithms to minimize material waste while providing real-time visualization of the optimization process.

### Key Benefits

- ✂️ Reduce fabric waste by up to 30%
- ⚡ Automatic pattern detection from images
- 🎨 Manual pattern drawing capability
- 📊 Real-time efficiency analytics
- 🔄 Intelligent nesting optimization

## ✨ Features

### Core Functionality

- **Automatic Pattern Detection**: Upload images of cut patterns and automatically extract individual parts using OpenCV
- **Manual Pattern Drawing**: Draw custom shapes directly on canvas with real-world dimension specifications
- **Genetic Algorithm Optimization**: Advanced AI-powered nesting that minimizes fabric waste through evolutionary algorithms
- **Real-time Visualization**: Interactive display of optimized layouts with gridlines and part identification
- **Multiple Input Methods**: Support for both image uploads (.jpg, .jpeg, .png) and manual shape drawing

### Analytics & Reporting

- **Efficiency Metrics**: Detailed statistics including:
  - Utilization percentage
  - Fabric usage percentage
  - Waste percentage
  - Total parts area vs used fabric area
- **Visual Feedback**: Color-coded parts with rotation indicators
- **Export Capabilities**: Download optimized layouts for production

### User Management

- **JWT Authentication**: Secure user registration and login
- **Project Organization**: Manage multiple nesting sessions
- **Session Persistence**: Save and retrieve previous optimization results

## 🛠️ Tech Stack

### Backend
```
- Python 3.8+
- Flask (Web Framework)
- OpenCV (Computer Vision & Image Processing)
- NumPy (Numerical Computing)
- PostgreSQL (Database)
- SQLAlchemy (ORM)
- Flask-JWT-Extended (Authentication)
```

### Frontend
```
- HTML5/CSS3
- JavaScript (ES6+)
- Canvas API (Drawing Interface)
```

### Algorithms
```
- Genetic Algorithm (Optimization)
- Adaptive Thresholding (Image Processing)
- Canny Edge Detection (Contour Recognition)
- Bottom-Left Heuristic (Placement Strategy)
```

## 📦 Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- PostgreSQL 12+
- pip (Python package manager)
- Git

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/Harine77/Eco-nesting-optimizer.git
cd fabric-nesting-optimizer
```

2. **Create and activate virtual environment**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up PostgreSQL database**
```bash
# Create database
createdb fabric_nesting

# Or using psql
psql -U postgres
CREATE DATABASE fabric_nesting;
\q
```

5. **Configure environment variables**

Create a `.env` file in the project root:
```env
FLASK_APP=app.py
FLASK_ENV=development
JWT_SECRET_KEY=your-super-secret-key-change-this
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/fabric_nesting
```

6. **Update database configuration in `app.py`**
```python
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://username:password@localhost:5432/fabric_nesting"
app.config["JWT_SECRET_KEY"] = "your-secret-key-here"  # Change this!
```

7. **Initialize database tables**
```bash
python
>>> from app import app, db
>>> with app.app_context():
...     db.create_all()
>>> exit()
```

8. **Run the application**
```bash
flask run
```

The application will be available at `http://localhost:5000`

## 🚀 Usage

### Method 1: Image Upload

1. **Register/Login** to create an account
2. **Upload Image**: Select a clear image of your cut patterns (PNG, JPG, JPEG)
3. **Review Detection**: System automatically detects and outlines individual parts
4. **Approve Parts**: Review and approve detected patterns
5. **Set Fabric Dimensions**: Enter fabric width and height in centimeters
6. **Set Scale**: Define the scale factor (pixels to cm conversion)
7. **Optimize**: Click optimize to generate the best nesting layout
8. **Download**: Export the optimized layout image

### Method 2: Manual Drawing

1. **Select Drawing Tool**: Choose the manual drawing option
2. **Draw Shapes**: Create shapes on the canvas
3. **Specify Dimensions**: Enter real-world dimensions for each shape
4. **Set Fabric Size**: Define fabric dimensions
5. **Optimize & Export**: Generate and download optimized layout

### Example Workflow
```
Upload Pattern Image
    ↓
Automatic Detection (OpenCV)
    ↓
Part Approval Interface
    ↓
Input Fabric Dimensions
    ↓
Genetic Algorithm Optimization
    ↓
Visualize Optimized Layout
    ↓
Download/Export Results
```

## 📊 Database Schema
```sql
-- Users Table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Projects Table
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions Table
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    session_uuid VARCHAR(100) UNIQUE NOT NULL,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    source_type VARCHAR(20),
    original_image_path TEXT,
    visualization_image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cut Parts Table
CREATE TABLE cut_parts (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    part_label VARCHAR(50),
    width FLOAT NOT NULL,
    height FLOAT NOT NULL,
    area FLOAT NOT NULL,
    real_width FLOAT,
    real_height FLOAT,
    image_path TEXT,
    approved BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Relationships
- One User → Many Projects
- One Project → Many Sessions
- One Session → Many CutParts

## 🧮 Algorithm Details

### Genetic Algorithm Configuration
```python
Population Size: 50 individuals
Generations: 100 iterations
Crossover Rate: 70%
Mutation Rate: 20%
Selection Method: Tournament Selection
Rotation Options: [0°, 90°, 180°, 270°]
```

### Fitness Function
```python
Fitness = (Utilization × 0.7) + (Fabric Efficiency × 0.3)

Where:
- Utilization = (Parts Area / Used Fabric Area) × 100
- Fabric Efficiency = 1 - (Used Fabric Area / Total Fabric Area)
```

### Image Processing Pipeline
```
1. Load Image (BGR)
   ↓
2. Convert to Grayscale
   ↓
3. Gaussian Blur (5×5 kernel)
   ↓
4. Adaptive Thresholding
   ↓
5. Canny Edge Detection
   ↓
6. Combine Results (Bitwise OR)
   ↓
7. Morphological Closing (3×3 kernel)
   ↓
8. Contour Detection
   ↓
9. Filter by Area (min 1000 pixels)
   ↓
10. Extract Individual Parts
```

## 🔌 API Endpoints

### Authentication
```http
POST /register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}
```
```http
POST /login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}

Response: { "access_token": "jwt_token_here" }
```

### Pattern Processing
```http
POST /upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <image_file>
```
```http
POST /process-drawn
Content-Type: application/json

{
  "shapes": [
    {
      "path": [{"x": 10, "y": 20}, ...],
      "label": "Part 1",
      "real_width": 50,
      "real_height": 30
    }
  ]
}
```

### Optimization
```http
POST /optimize
Content-Type: application/json

{
  "fabric_width": 200,
  "fabric_height": 150,
  "scale_factor": 0.5,
  "parts": [...],
  "session_id": "session_uuid"
}
```

### Static Files
```http
GET /uploads/<filename>
GET /processed/<filename>
```

## 📈 Performance Metrics

The system calculates and displays:

| Metric | Description | Formula |
|--------|-------------|---------|
| **Utilization** | Efficiency of part placement | (Parts Area / Used Fabric Area) × 100 |
| **Fabric Usage** | Percentage of total fabric used | (Used Fabric Area / Total Fabric Area) × 100 |
| **Efficiency** | Overall material efficiency | (Parts Area / Total Fabric Area) × 100 |
| **Waste** | Material waste percentage | 100 - Utilization |

## 🔒 Security Features

- ✅ JWT-based authentication with secure tokens
- ✅ Password hashing using Werkzeug security
- ✅ Protected API endpoints with `@jwt_required()`
- ✅ File upload validation (type and size limits)
- ✅ SQL injection prevention via SQLAlchemy ORM
- ✅ Secure filename handling with `secure_filename()`

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
```bash
   git checkout -b feature/AmazingFeature
```
3. **Commit your changes**
```bash
   git commit -m 'Add some AmazingFeature'
```
4. **Push to the branch**
```bash
   git push origin feature/AmazingFeature
```
5. **Open a Pull Request**


## 📧 Contact & Support

- **Author**: Harine B
- **Email**: harineb.gma@gmail.com
- **GitHub**: [@yHarine77](https://github.com/Harine77) 

---

**⭐ If you find this project useful, please give it a star!**

**Made with ❤️ for the textile industry**
