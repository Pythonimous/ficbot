import os
import uuid 
import base64
import requests

from dotenv import load_dotenv

from fastapi import Request, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from src.api.models.name import NameRequest
from src.api.utils import validate_image, get_local_image_path, clean_old_images, MODEL_DIR, UPLOAD_DIR, TEMPLATE_DIR

from src.core.inference import generate_name

load_dotenv()

LAMBDA_URL = os.getenv("LAMBDA_URL")
if not LAMBDA_URL:
    raise RuntimeError("LAMBDA_URL is not set. Please configure your .env file.")

router = APIRouter()

templates = Jinja2Templates(directory=TEMPLATE_DIR)

UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.get("/")
@router.post("/")
async def render(request: Request):
    """Render the generation.html template."""
    return templates.TemplateResponse("generation.html", {"request": request})


@router.get("/upload_image/")
async def upload_image_page(request: Request):
    """Renders the image upload page."""
    return templates.TemplateResponse("generation.html", {"request": request})


@router.post("/upload_image/")
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
    if len(image_bytes) > 2 * 1024 * 1024:
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


@router.get("/name/")
async def render_name_page(request: Request):
    """Renders the name generation page."""
    return templates.TemplateResponse("generation.html", {"request": request})


@router.post("/name/")
async def generate_character_name(request_data: NameRequest):
    """Generates a name from Lambda based on the request image."""

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

    if os.environ.get("TESTING"):
        return JSONResponse(content={"success": True, "name": "Test Name"})
    
    # Send request to AWS Lambda
    response = requests.post(
        LAMBDA_URL,
        json={
            "image": encoded_image,
            "diversity": request_data.diversity,
            "min_name_length": request_data.min_name_length
        }
    )

    # Check response
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Lambda function failed")

    result = response.json()

    return {"success": True, "name": result.get("name", "Name generation failed")}
