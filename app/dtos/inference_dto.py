from typing import List, Union, Literal
from pydantic import BaseModel, Field

from app.enums.inference_enum import ModelTypes

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    image: str  # Base64 encoded image or URL

class Template(BaseModel):
    role: str
    content: List[Union[TextContent, ImageContent]]

class InferenceRequest(BaseModel):
    max_new_tokens: int
    temperature: float = Field(..., description="Temperature for sampling")
    min_p: float = Field(..., description="Min probability for sampling")
    model: ModelTypes
    prompt: str
    attachments: List[str] # list of urls of images
    stream: bool = True