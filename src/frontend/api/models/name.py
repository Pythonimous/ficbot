from pydantic import BaseModel

class NameRequest(BaseModel):
    imageSrc: str
    diversity: float
    min_name_length: int