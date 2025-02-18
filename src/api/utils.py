from PIL import Image
import io

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
