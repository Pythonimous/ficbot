from pydantic import BaseModel

class ImageRequest(BaseModel):
    image: str  # base64 encoded image

class NameRequest(BaseModel):
    imageSrc: str
    diversity: float
    min_name_length: int


class BioRequest(BaseModel):
    name: str
    diversity: float
    max_bio_length: int