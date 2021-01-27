from pydantic import BaseModel
from typing import List

class PredictionResponseDto(BaseModel):
    filename: str
    contentype: str    
    label: str
