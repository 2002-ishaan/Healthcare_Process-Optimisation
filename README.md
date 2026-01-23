# NTH Project - Data Analysis & Dashboard

A comprehensive data analysis project with an interactive Streamlit dashboard for visualizing and exploring NTH (New Technology Horizon) metrics and insights.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Data Pipeline](#data-pipeline)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The NTH Project provides end-to-end data processing and visualization capabilities for analyzing technology trends and metrics. The project includes:

- Automated data ingestion and preprocessing
- Feature engineering and transformation pipelines
- Interactive Streamlit dashboard for data exploration
- Comprehensive visualizations and analytics

## ✨ Features

- **Data Processing Pipeline**: Automated ETL workflows for cleaning and transforming raw data
- **Interactive Dashboard**: Real-time data exploration with Streamlit
- **Advanced Analytics**: Statistical analysis and trend identification
- **Visualization Suite**: Multiple chart types including time series, distributions, and correlations
- **Export Capabilities**: Download processed data and reports
- **Responsive Design**: Works on desktop and tablet devices

## 📁 Project Structure

```
nth-project/
├── data/
│   ├── raw/                 # Raw input data files
│   ├── processed/           # Cleaned and processed data
│   └── output/              # Generated reports and exports
├── src/
│   ├── preprocessing/       # Data cleaning and transformation scripts
│   ├── analysis/            # Statistical analysis modules
│   ├── visualization/       # Plotting and chart generation
│   └── utils/               # Helper functions and utilities
├── dashboard/
│   ├── app.py               # Main Streamlit application
│   ├── components/          # Dashboard UI components
│   └── styles/              # CSS and styling files
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit and integration tests
├── config/
│   └── config.yaml          # Configuration settings
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment specification
├── .gitignore
└── README.md
```

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: Version 3.8 or higher
- **pip**: Python package installer
- **Git**: For version control
- **Virtual Environment**: Recommended (venv or conda)

### System Requirements

- OS: Windows 10+, macOS 10.14+, or Linux
- RAM: Minimum 4GB (8GB recommended)
- Disk Space: At least 2GB free

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/nth-project.git
cd nth-project
```

### Step 2: Create Virtual Environment

**Option A: Using venv (recommended)**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Option B: Using conda**

```bash
# Create conda environment
conda env create -f environment.yml

# Activate conda environment
conda activate nth-project
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check if all packages are installed correctly
python -c "import streamlit, pandas, numpy, plotly; print('All packages installed successfully!')"
```

## ⚙️ Configuration

### 1. Set Up Configuration File

Copy the example configuration file and modify it according to your needs:

```bash
cp config/config.example.yaml config/config.yaml
```

### 2. Edit Configuration Settings

Open `config/config.yaml` and update the following parameters:

```yaml
# Data paths
data:
  raw_data_path: "data/raw/"
  processed_data_path: "data/processed/"
  output_path: "data/output/"

# Dashboard settings
dashboard:
  title: "NTH Analytics Dashboard"
  port: 8501
  host: "localhost"

# Analysis parameters
analysis:
  date_column: "date"
  metrics: ["metric1", "metric2", "metric3"]
  aggregation: "daily"
```

### 3. Prepare Data Files

Place your input data files in the `data/raw/` directory. Supported formats:

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- Parquet (`.parquet`)

## 🏃 Running the Application

### Quick Start

To launch the dashboard, simply run:

```bash
streamlit run dashboard/app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

### Advanced Options

**Custom Port:**

```bash
streamlit run dashboard/app.py --server.port 8502
```

**Custom Host (for network access):**

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0
```

**Run in Background:**

```bash
nohup streamlit run dashboard/app.py &
```

### First-Time Setup

When you first run the application:

1. The app will check for required data files
2. If no processed data exists, it will automatically run the preprocessing pipeline
3. This may take 2-5 minutes depending on data size
4. Subsequent runs will be much faster

## 📊 Usage Guide

### Dashboard Navigation

1. **Data Upload Section**: Upload new data files or select from existing datasets
2. **Filter Panel**: Apply date ranges, metrics, and category filters
3. **Visualization Tabs**: 
   - Overview: Key metrics and summary statistics
   - Time Series: Trend analysis over time
   - Comparisons: Side-by-side metric comparisons
   - Distributions: Statistical distributions and histograms
   - Correlations: Relationship analysis between variables
4. **Export Options**: Download filtered data, charts, or reports

### Common Workflows

**Workflow 1: Analyze New Data**

1. Click "Upload Data" in the sidebar
2. Select your CSV/Excel file
3. Wait for preprocessing to complete
4. Explore visualizations in different tabs
5. Export results if needed

**Workflow 2: Compare Time Periods**

1. Use the date range selector in the sidebar
2. Select "Time Series" tab
3. Choose metrics to compare
4. Adjust aggregation level (daily/weekly/monthly)
5. Download comparison report

**Workflow 3: Generate Reports**

1. Apply desired filters
2. Navigate to "Export" section
3. Select report type (PDF/Excel/CSV)
4. Click "Generate Report"
5. Download from the link provided

## 🔄 Data Pipeline

The project follows a structured data pipeline:

### 1. Data Ingestion

```python
# Located in src/preprocessing/ingest.py
- Load raw data from multiple sources
- Validate data schema and types
- Handle missing or malformed records
```

### 2. Preprocessing

```python
# Located in src/preprocessing/clean.py
- Remove duplicates
- Handle missing values
- Standardize date formats
- Normalize numerical columns
- Encode categorical variables
```

### 3. Feature Engineering

```python
# Located in src/preprocessing/features.py
- Create derived metrics
- Calculate rolling averages
- Generate time-based features
- Aggregate data at different levels
```

### 4. Analysis & Visualization

```python
# Dashboard components dynamically load processed data
- Real-time metric calculations
- Interactive plotting with Plotly
- Caching for performance optimization
```

## 🛠️ Troubleshooting

### Common Issues

**Issue 1: Dashboard won't start**

```bash
# Error: Address already in use
Solution: Change the port number
streamlit run dashboard/app.py --server.port 8502
```

**Issue 2: Import errors**

```bash
# Error: ModuleNotFoundError
Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Issue 3: Data file not found**

```bash
# Error: FileNotFoundError
Solution: Check your config.yaml paths and ensure data files exist
- Verify data/raw/ directory contains input files
- Update paths in config/config.yaml if needed
```

**Issue 4: Memory errors with large datasets**

```bash
Solution: Process data in chunks
- Modify config.yaml: chunk_size: 10000
- Or filter data by date range before processing
```

**Issue 5: Slow dashboard performance**

```bash
Solution: Clear Streamlit cache
streamlit cache clear
```

### Getting Help

If you encounter issues not covered here:

1. Check the [Issues](https://github.com/your-username/nth-project/issues) page
2. Review error logs in `logs/` directory
3. Run diagnostic script: `python src/utils/diagnose.py`
4. Contact the team via email or Slack

## 👥 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing

Before submitting a PR:

```bash
# Run unit tests
pytest tests/

# Run linting
flake8 src/ dashboard/

# Check code formatting
black --check src/ dashboard/
```
