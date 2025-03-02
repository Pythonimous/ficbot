import os
import uuid 
import base64
import requests
import dotenv

from fastapi import Request, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from src.api.models.generate import NameRequest
from src.api.utils import validate_image, clean_old_images
from src.api.config import TEMPLATE_DIR, UPLOAD_DIR, UPLOAD_EXTENSIONS, MAX_CONTENT_LENGTH

router = APIRouter()

templates = Jinja2Templates(directory=TEMPLATE_DIR)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/")
@router.post("/")
async def render(request: Request):
    """Render the generation.html template."""
    return templates.TemplateResponse("generation.html", {"request": request})


@router.get("/upload_image")
async def upload_image_page(request: Request):
    """Renders the image upload page."""
    return templates.TemplateResponse("generation.html", {"request": request})


@router.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Saves the uploaded image and returns it to the client."""
    # Secure filename & validate extension
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Wrong extension: only .jpg, .png, .gif files are allowed")
    
    filename = f"{uuid.uuid4().hex}{file_ext}"

    # Validate image integrity
    image_bytes = await file.read()  # Read image content

    # Check file size (assuming 2MB limit)
    if len(image_bytes) > MAX_CONTENT_LENGTH:
        raise HTTPException(status_code=413, detail="File is too large. Only .jpg, .png, .gif up to 2MB are allowed.")

    if validate_image(image_bytes) not in UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Broken file: only valid .jpg, .png, .gif files are allowed. Please check your image and try again.")

    # Delete old images (except example.jpg)
    clean_old_images(exclude=["example.jpg"])

    # Save the file
    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    # Generate URL (assuming FastAPI serves static files from /static/)
    image_url = f"/static/images/{filename}"

    # Debug print
    print(f"Returning JSON: {{'success': True, 'imgUrl': '{image_url}'}}")

    return JSONResponse(content={"success": True, "imgUrl": image_url})
