# Radigenius Inference API

A FastAPI-powered inference API for the Radigenius model, designed to run in a dockerized environment on RunPod.

## Overview

This API serves as an inference endpoint for the Radigenius model, allowing clients to make requests and receive predictions. The project is structured for deployment in containerized environments, particularly optimized for RunPod.

## Features

- Robust model inference via FastAPI
- Utility scripts for managing the service lifecycle
- Modular architecture for maintainability

## Getting Started

### Prerequisites

- Docker
- RunPod account (for production deployment)
- Git

### Setup and Deployment

The project includes several utility scripts to simplify deployment:

1. **Initialize the environment**:
   ```
   sh ./scripts/init.sh
   ```
   This script:
   - Updates the operating system
   - Installs virtualenv
   - Sets up a Python virtual environment
   - Runs the application

2. **Start or restart the service**:
   ```
   sh ./scripts/run.sh
   ```
   This script:
   - Pulls the latest changes from the main branch
   - Activates the virtual environment
   - Installs dependencies
   - Starts the server

3. **Stop the service**:
   ```
   sh ./scripts/kill.sh
   ```
   This script terminates all running server instances.

## API Documentation

Once running, you can access:
- API: http://[your-host]:8000
- Interactive API docs: http://[your-host]:8000/docs
- Alternative API docs: http://[your-host]:8000/redoc

## Project Structure

- `app/`: Main application package
  - `main.py`: FastAPI application entry point
  - `api/`: API endpoints
    - `routes/`: Route modules including inference endpoints
  - `services/`: Service modules
    - `radigenius/`: Radigenius model implementation
  - `core/`: Core application modules

## Production Considerations

- **CORS Configuration**: For production use cases, ensure proper CORS settings are configured to restrict access to your API from unauthorized domains.
- **Debug Mode**: Always set `DEBUG=False` in environment variables for production deployments to prevent exposing sensitive information.

## Extending the Project

- Add new routes in `app/api/routes/`
- Modify model behavior in `app/services/radigenius/`
- Configure application settings in `app/core/config.py`

## Troubleshooting

If you encounter issues:
1. Check the logs for error messages
2. Verify that the model files are correctly placed
3. Ensure all dependencies are installed
4. Make sure proper permissions are set for the scripts