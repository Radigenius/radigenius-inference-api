# FastAPI Base Project

A clean, well-structured FastAPI project template to kickstart your API development.

## Features

- Modular project structure
- Database integration with SQLAlchemy
- Configuration management
- Ready-to-use API routes example

## Getting Started

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the development server:
   ```
   python -m app.main
   ```
   
   Alternatively:
   ```
   uvicorn app.main:app --reload
   ```

5. Open your browser and navigate to:
   - API: http://localhost:8000
   - Interactive API docs: http://localhost:8000/docs
   - Alternative API docs: http://localhost:8000/redoc

## Project Structure

- `app/`: Main application package
  - `main.py`: FastAPI application entry point
  - `api/`: API endpoints
    - `routes/`: Route modules for different resources
  - `core/`: Core application modules
    - `config.py`: Configuration settings
  - `db/`: Database related modules
    - `database.py`: Database connection setup
  - `models/`: SQLAlchemy models

## Extending the Project

- Add new routes in `app/api/routes/`
- Define new models in `app/models/`
- Configure application settings in `app/core/config.py` 