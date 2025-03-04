import os
import uuid 
import base64
import requests
import dotenv

from urllib.parse import urljoin

from fastapi import Request, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from src.api.models.generate import ImageRequest
from src.api.utils import validate_image, clean_old_images, get_local_image_path
from src.api.config import settings, TEMPLATE_DIR, UPLOAD_DIR, UPLOAD_EXTENSIONS, MAX_CONTENT_LENGTH, ENV_DIR

router = APIRouter()

templates = Jinja2Templates(directory=TEMPLATE_DIR)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if ENV_DIR.exists():
    dotenv.load_dotenv(ENV_DIR)
    VPS_URL = os.getenv("VPS_URL")
    if not VPS_URL:
        raise RuntimeError("VPS_URL is not set. Please configure your .env file.")


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


@router.post("/convert_to_anime")
async def convert_to_anime(request_data: ImageRequest):
    """Receives a Base64 image, processes it with AnimeGAN2, saves the output, and returns the new URL."""

    # Decode Base64 input to raw image bytes
    try:
        image_bytes = base64.b64decode(request_data.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 image: {str(e)}")

    # Generate a unique filename for the input image
    original_filename = f"original_{uuid.uuid4().hex}.png"
    original_image_path = UPLOAD_DIR / original_filename

    # Save the uploaded image
    try:
        with open(original_image_path, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded image: {str(e)}")

    if settings.testing:
        return JSONResponse(content={"success": True, "animeImgUrl": f"static/images/{original_filename}"})

    # Encode the saved image as Base64 to send to inference
    with open(original_image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()

    # Send request to Inference container
    payload = {
        "image": encoded_image
    }

    # AnimeGAN2 PyTorch implementation sourced from:
    # https://github.com/bryandlee/animegan2-pytorch

    response = requests.post(
        urljoin(VPS_URL, "convert_to_anime"),
        json=payload
    )

    # Check response
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Anime conversion failed")

    result = response.json()
    anime_image_base64 = result.get("anime_image", None)
    
    if not anime_image_base64:
        raise HTTPException(status_code=500, detail="Anime conversion failed")

    # Decode Base64 back into image bytes
    try:
        anime_image_bytes = base64.b64decode(anime_image_base64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decoding anime image: {str(e)}")

    # Generate a unique filename for the anime output image
    anime_filename = f"anime_{uuid.uuid4().hex}.png"
    anime_image_path = UPLOAD_DIR / anime_filename

    # Save the anime image
    try:
        with open(anime_image_path, "wb") as f:
            f.write(anime_image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving anime image: {str(e)}")

    # Clean up old images
    clean_old_images(exclude=["example.jpg", original_filename, anime_filename])

    # Generate public URL for the anime image
    anime_image_url = f"static/images/{anime_filename}"

    return JSONResponse(content={"success": True, "animeImgUrl": anime_image_url})