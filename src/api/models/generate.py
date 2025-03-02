from pydantic import BaseModel

class NameRequest(BaseModel):
    imageSrc: str
    diversity: float
    min_name_length: int


class BioRequest(BaseModel):
    name: str
    diversity: float
    max_bio_length: int