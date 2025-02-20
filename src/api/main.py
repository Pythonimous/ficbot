import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.api.endpoints import generate

# Initialize FastAPI app
app = FastAPI(title="Ficbot API", version="1.1")

# Configurations
UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB limit

# Ensure the instance folder exists
instance_path = os.path.join(os.getcwd(), "instance")
os.makedirs(instance_path, exist_ok=True)

# Include endpoints
app.include_router(generate.router, prefix="", tags=["generate"])

current_dir = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(current_dir, "../frontend/static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/health")
def health():
    """Root endpoint for API health check."""
    return {"message": "Welcome to the Ficbot API!"}
