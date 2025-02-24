from pathlib import Path

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    testing: bool = False

settings = Settings()

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT_DIR / 'templates'
UPLOAD_DIR = ROOT_DIR / 'static/images'

UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB limit