from pydantic import BaseModel

class ImageDTO(BaseModel):
    url: str
    local_path: str