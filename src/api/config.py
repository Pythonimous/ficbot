from pathlib import Path

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    testing: bool = False

settings = Settings()

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT_DIR / 'templates'
UPLOAD_DIR = ROOT_DIR / 'static/images'

ENV_DIR = ROOT_DIR.parent / '.env'

UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB limit

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "INFO",
        },
        "rotating_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "ficbot.log",
            "maxBytes": 10*1024*1024,  # 10 MB
            "backupCount": 5,
            "formatter": "json",
            "level": "INFO",
        }
    },
    "root": {
        "handlers": ["console", "rotating_file"],
        "level": "INFO",
    },
}