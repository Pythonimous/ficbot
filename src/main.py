import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.endpoints import generate

# Initialize FastAPI app
app = FastAPI(title="Ficbot API", version="1.1")

# Ensure the instance folder exists
instance_path = os.path.join(os.getcwd(), "instance")
os.makedirs(instance_path, exist_ok=True)

# Include endpoints
app.include_router(generate.router, prefix="", tags=["generate"])

current_dir = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/health", status_code=200)
def health_check():
    """Root endpoint for API health check."""
    return {"status": "ok"}
