import os
from fastapi import FastAPI
from src.api.endpoints import generate

# Initialize FastAPI app
app = FastAPI(title="Ficbot API", version="1.0")

# Configurations
UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB limit

# Ensure the instance folder exists
instance_path = os.path.join(os.getcwd(), "instance")
os.makedirs(instance_path, exist_ok=True)

# Include endpoints
app.include_router(generate.router, prefix="", tags=["Generation"])

@app.get("/")
def root():
    """Root endpoint for API health check."""
    return {"message": "Welcome to the Ficbot API!"}
