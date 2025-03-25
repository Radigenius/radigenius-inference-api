import os
import typer
import asyncio
import logging
import pathlib
import datetime
from logging.handlers import TimedRotatingFileHandler
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import inference_route
from app.core.config import settings
from app.services.radigenius.inference import RadiGenius

# Configure logging
def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory if it doesn't exist
    log_dir = os.getenv("LOG_DIR", "logs")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate today's log filename directly
    date_str = datetime.datetime.now().strftime("%m-%d-%Y")
    log_file = os.path.join(log_dir, f"radigenius-log-{date_str}.txt")
    
    # Create formatters and handlers separately
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler (simple FileHandler using today's date)
    file_handler = logging.FileHandler(filename=log_file, encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove any existing handlers and add our handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set levels for specific loggers if needed
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Log at startup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level")
    logger.info(f"Log file: {os.path.abspath(log_file)}")
    return logger

# Initialize logger
logger = setup_logging()

# Create CLI app
cli = typer.Typer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialization before startup (previously in @app.on_event("startup"))
    logger.info("Initializing RadiGenius model...")
    instance = RadiGenius()
    logger.info("RadiGenius model initialized successfully")
    yield
    # Cleanup on shutdown
    logger.info("Shutting down RadiGenius model...")
    instance.kill_model()
    logger.info("RadiGenius model shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inference_route.router, prefix="/api")

@app.get("/")
async def root():
    logger.debug("Root endpoint accessed")
    return {"message": "Welcome to RadiGenius API"}

# CLI Commands
@cli.command()
def init_model():
    """Initialize the model without starting the server"""
    
    logger.info("Running model initialization...")
    typer.echo("Running model initialization...")
    
    RadiGenius()
    
    logger.info("Model initialization complete.")
    typer.echo("Model initialization complete. You can now run the server with --skip-init option.")

@cli.command()
def run_server(
    host: str = "0.0.0.0", 
    port: int = 8000, 
    skip_init: bool = False
):
    """Run the FastAPI server"""
    import uvicorn
    
    if skip_init:
        logger.info("Starting server with pre-initialized model")
        typer.echo("Starting server with pre-initialized model")
    else:
        logger.info("Starting server (model will be initialized during startup)")
        typer.echo("Starting server (model will be initialized during startup)")
    
    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    uvicorn.run(
        "app.main:app", 
        host=host, 
        port=port, 
        reload=os.getenv("DEBUG", "False").lower() == "true",
        log_config=log_config
    )

if __name__ == "__main__":
    cli() 