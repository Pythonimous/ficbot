import io
from pathlib import Path

from urllib.parse import urlparse
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = ROOT_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

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


if __name__ == '__main__':
    print(get_local_image_path("http://127.0.0.1:8000/static/images/example.jpg"))
