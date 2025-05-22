# Synthetic Financial Data Generator

## Overview

This is a Streamlit-based web application designed to generate high-quality synthetic financial data while preserving statistical distributions and relationships from Excel input files. The application provides a complete pipeline for data processing, synthesis, validation, and visualization while maintaining privacy standards.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular Python architecture with clear separation of concerns:

### Frontend Architecture
- **Streamlit Framework**: Web-based interface providing file upload, configuration options, and interactive visualizations
- **Session State Management**: Maintains application state across user interactions
- **Responsive Layout**: Wide layout with expandable sidebar for configuration

### Backend Architecture
- **Modular Design**: Separate modules for data processing, synthesis, validation, and visualization
- **Object-Oriented Components**: Each major functionality encapsulated in dedicated classes
- **Error Handling**: Comprehensive exception handling throughout the pipeline

## Key Components

### 1. Data Processor (`src/data_processor.py`)
- **Purpose**: Handles Excel file loading and automatic schema detection
- **Key Features**: 
  - Automatic column type detection (numerical, categorical, datetime, boolean)
  - Data cleaning and validation
  - Statistical analysis of input data
- **Input Validation**: Requires minimum 3 rows for meaningful analysis

### 2. Data Synthesizer (`src/synthesizer.py`)
- **Purpose**: Generates synthetic data using advanced algorithms
- **Synthesis Methods**: 
  - GaussianCopula (default)
  - CTGAN
  - CopulaGAN
- **Privacy Levels**: Standard, High, Maximum
- **Dependencies**: SDV (Synthetic Data Vault) library with fallback handling

### 3. Data Validator (`src/validator.py`)
- **Purpose**: Validates quality and privacy of generated synthetic data
- **Privacy Metrics**: 
  - Leakage score calculation
  - Similarity threshold analysis
  - Privacy level assessment
- **Quality Checks**: Statistical distribution comparisons

### 4. Data Visualizer (`src/visualizer.py`)
- **Purpose**: Creates comparative visualizations between original and synthetic data
- **Visualization Types**: 
  - Distribution histograms
  - Statistical comparisons
  - Interactive Plotly charts

### 5. Utility Functions (`utils/helpers.py`)
- **Purpose**: Common utility functions for data handling
- **Key Features**: 
  - Download button generation
  - Data format conversion
  - Number formatting

## Data Flow

1. **File Upload**: User uploads Excel file through Streamlit interface
2. **Data Processing**: DataProcessor analyzes file structure and column types
3. **Configuration**: User configures synthesis parameters (method, privacy level, output size)
4. **Synthesis**: DataSynthesizer generates synthetic data based on original patterns
5. **Validation**: DataValidator checks privacy and quality metrics
6. **Visualization**: DataVisualizer creates comparison charts
7. **Export**: User can download synthetic data in various formats

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework (v1.45.1+)
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **OpenPyXL**: Excel file handling

### Optional Dependencies
- **SDV (Synthetic Data Vault)**: Advanced synthesis algorithms
- **SciPy**: Statistical functions for validation

### System Dependencies
- **Python 3.11+**: Runtime environment
- **Nix Package Manager**: Development environment management

## Deployment Strategy

### Development Environment
- **Replit Platform**: Cloud-based development with Nix package management
- **Port Configuration**: Application runs on port 5000
- **Auto-scaling**: Configured for autoscale deployment target

### Runtime Configuration
- **Streamlit Server**: Headless mode with custom port binding
- **Theme**: Light theme for better readability
- **Memory Management**: Session state for maintaining user data

### Workflow Automation
- **Parallel Execution**: Multiple workflow tasks can run simultaneously
- **Port Monitoring**: Automatic port detection and health checks
- **Process Management**: Shell execution with proper process handling

## Privacy and Security Considerations

### Data Protection
- **No Persistent Storage**: Application doesn't store user data permanently
- **Privacy Validation**: Built-in privacy metrics to ensure synthetic data doesn't leak original information
- **Configurable Privacy Levels**: Multiple privacy settings for different use cases

### Statistical Preservation
- **Distribution Matching**: Maintains statistical properties of original data
- **Correlation Preservation**: Keeps relationships between variables intact
- **Quality Metrics**: Validates synthetic data quality against original patterns