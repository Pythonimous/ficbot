import os
import uuid 
import base64
import requests
import dotenv

from fastapi import Request, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from src.api.models.generate import NameRequest, BioRequest
from src.api.utils import get_local_image_path
from src.api.config import settings, ROOT_DIR, TEMPLATE_DIR, UPLOAD_DIR

env_dir = ROOT_DIR.parent / '.env'

if env_dir.exists():
    dotenv.load_dotenv(env_dir)
    VPS_URL = os.getenv("VPS_URL")
    if not VPS_URL:
        raise RuntimeError("VPS_URL is not set. Please configure your .env file.")

router = APIRouter()

templates = Jinja2Templates(directory=TEMPLATE_DIR)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/name")
async def render_name_page(request: Request):
    """Renders the name generation page."""
    return templates.TemplateResponse("generation.html", {"request": request})


@router.post("/name")
async def generate_character_name(request_data: NameRequest):
    """Generates a name based on the request image."""

    # Construct file paths
    img_path = get_local_image_path(request_data.imageSrc)

    # Ensure the image file exists
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        with open(img_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error decoding an image: {str(e)}")

    if settings.testing:
        return JSONResponse(content={"success": True, "name": "Test Name"})
    
    # Send request to Inference container
    payload = {
        "type": "name",
        "image": encoded_image,
        "diversity": request_data.diversity,
        "min_name_length": request_data.min_name_length
    }
    response = requests.post(
        VPS_URL,
        json=payload
    )

    # Check response
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Inference function failed")

    result = response.json()

    name = result.get("name", None)
    if name:
        return {"success": True, "name": name}
    else:
        raise HTTPException(status_code=500, detail="Name generation failed")



@router.post("/bio")
async def generate_character_bio(request_data: BioRequest):
    """Generates a bio based on the request name."""

    if settings.testing:
        return JSONResponse(content={"success": True, "bio": "Test Bio"})
    
    # Send request to Inference container
    payload = {
        "type": "bio",
        "name": request_data.name,
        "diversity": request_data.diversity,
        "max_bio_length": request_data.max_bio_length
    }
    response = requests.post(
        VPS_URL,
        json=payload
    )

    # Check response
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Inference function failed")

    result = response.json()

    bio = result.get("bio", None)
    if bio:
        return {"success": True, "bio": bio}
    else:
        raise HTTPException(status_code=500, detail="Bio generation failed")