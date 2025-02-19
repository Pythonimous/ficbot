import os

from werkzeug.utils import secure_filename

from fastapi import Request, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.models.name import NameRequest
from src.api.utils import validate_image, get_local_image_path, PROJECT_DIR, FRONTEND_DIR

from src.core.inference import generate_name

router = APIRouter()

TEMPLATE_DIR = FRONTEND_DIR / 'templates'

templates = Jinja2Templates(directory=TEMPLATE_DIR)

current_dir = os.path.dirname(os.path.abspath(__file__))

UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
UPLOAD_DIR = FRONTEND_DIR / 'static/images'
MODEL_DIR = PROJECT_DIR / 'models/img_name/tf'

@router.get("/")
@router.post("/")
async def render(request: Request):
    """Render the generation.html template."""
    return templates.TemplateResponse("generation.html", {"request": request})


'''
@router.exception_handler(StarletteHTTPException)
async def too_large_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handles file size limit exceeded (413 Payload Too Large)."""
    if exc.status_code == 413:
        return PlainTextResponse(
            "File is too large. Only .jpg, .png, .gif up to 2MB are allowed.", status_code=413
        )
    return await request.app.default_exception_handler(request, exc)
'''

@router.get("/upload_image/")
async def upload_image_page(request: Request):
    """Renders the image upload page."""
    return templates.TemplateResponse("generation.html", {"request": request})


@router.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    """Saves the uploaded image and returns it to the client."""
    # Secure filename & validate extension
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Wrong extension: only .jpg, .png, .gif files are allowed")

    # Validate image integrity
    image_bytes = await file.read()  # Read image content

    if validate_image(image_bytes) not in UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Broken file: only valid .jpg, .png, .gif files are allowed. Please check your image and try again.")

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
    """Generates a name based on the request image."""
    # Construct file paths
    img_path = get_local_image_path(request_data.imageSrc)
    model_path = MODEL_DIR / 'img2name.keras'
    maps_path = MODEL_DIR / 'maps.pkl'

    # Ensure the image file exists
    if not os.path.exists(img_path):
        raise HTTPException(status_code=400, detail="Image file not found")

    # Generate name using model inference
    name = generate_name(
        img_path,
        model_path,
        maps_path,
        diversity=request_data.diversity,
        min_name_length=request_data.min_name_length
    )

    return {"success": True, "name": name}
