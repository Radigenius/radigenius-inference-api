import os
import typer
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import inference_route
from app.core.config import settings
from app.services.radigenius.inference import RadiGenius

logger = logging.getLogger(__name__)

# Create CLI app
cli = typer.Typer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialization before startup (previously in @app.on_event("startup"))
    instance = RadiGenius()
    yield
    # Cleanup on shutdown
    instance.kill_model()

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
        # You may want to modify the app startup event if skip_init is True
    else:
        logger.info("Starting server (model will be initialized during startup)")
        typer.echo("Starting server (model will be initialized during startup)")
    
    uvicorn.run("app.main:app", host=host, port=port, reload=os.getenv("DEBUG", False))

if __name__ == "__main__":
    cli() 