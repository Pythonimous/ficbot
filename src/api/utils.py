import io
import os
from datetime import datetime, timezone
from pathlib import Path

from urllib.parse import urlparse
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "core/models"

FRONTEND_DIR = ROOT_DIR / "frontend"
TEMPLATE_DIR = FRONTEND_DIR / 'templates'
UPLOAD_DIR = FRONTEND_DIR / 'static/images'


def get_local_image_path(imageSrc: str) -> str:
    """Extracts the local file path from the provided imageSrc URL."""
    parsed_url = urlparse(imageSrc)
    file_path = parsed_url.path.lstrip("/")  # Remove leading slash to avoid absolute path issues
    return FRONTEND_DIR / file_path


def validate_image(image_bytes: bytes) -> str:
    """
    Validates the uploaded image file.

    Args:
        image_bytes (bytes): The raw image data.

    Returns:
        str: The detected image format (or None if not detected).
    """
    try:
        # Load the image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check if image is valid
        image.verify()  # This detects broken/corrupt images

        # Get detected format (Pillow uses uppercase, e.g., 'JPEG', 'PNG')
        detected_format = image.format.lower()
        detected_ext = f".{detected_format}"
        return detected_ext

    except Exception as e:
        return None
    
def clean_old_images(exclude: list, max_age_seconds: int = 300):

    """Removes all images older than 'max_age_seconds' from the upload folder, except for excluded ones.

    :param exclude: List of filenames to exclude from deletion.
    :param max_age_seconds: Maximum age of files to keep (default: 5 minutes)

    """
    if os.environ.get("TESTING"):
        max_age_seconds = 2  # Set a shorter timeout for testing

    now = datetime.now(timezone.utc).timestamp()
    
    for file in UPLOAD_DIR.glob("*"):
        if file.name not in exclude:
            file_age = now - file.stat().st_mtime  # Calculate file age
            
            if file_age > max_age_seconds:  # Only delete old files
                file.unlink()  # Delete the file


if __name__ == '__main__':
    print(get_local_image_path("http://127.0.0.1:8000/static/images/example.jpg"))
