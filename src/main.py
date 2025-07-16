from src.api.config import LOGGING_CONFIG

import logging
import logging.config
logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from starlette.requests import Request
from starlette.responses import JSONResponse

from src.api.endpoints import generate, page

# Initialize FastAPI app
app = FastAPI(title="Ficbot API", version="1.1")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail} (status {exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Ensure the instance folder exists
instance_path = os.path.join(os.getcwd(), "instance")
os.makedirs(instance_path, exist_ok=True)

# Include endpoints
app.include_router(page.router, prefix="", tags=["page"])
app.include_router(generate.router, prefix="/generate", tags=["generate"])

current_dir = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/health", status_code=200)
def health_check():
    """Root endpoint for API health check."""
    return {"status": "ok"}
