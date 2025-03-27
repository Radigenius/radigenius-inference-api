from typing import List, Union, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from app.enums.inference_enum import ModelTypes

class ContentDto(BaseModel):
    type: Literal["text", "image"]
    text: Optional[str] = None
    image: Optional[str] = None

    @field_validator('type')
    def validate_content_type(cls, v, info):
        return v

    @field_validator('text')
    def validate_text(cls, v, info):
        values = info.data
        if values.get('type') == 'text' and not v:
            raise ValueError("text field is required when type is 'text'")
        return v

    @field_validator('image')
    def validate_image(cls, v, info):
        values = info.data
        if values.get('type') == 'image' and not v:
            raise ValueError("image field is required when type is 'image'")
        return v

class MessageDto(BaseModel):
    role: Literal["user", "assistant"]
    content: List[ContentDto]

class ConfigsDto(BaseModel):
    max_new_tokens: int
    temperature: float = Field(..., description="Temperature for sampling")
    min_p: float = Field(..., description="Min probability for sampling")
    model: ModelTypes
    stream: bool = True

class InferenceRequest(BaseModel):
    
    """
    Request body for inference
    configs: ConfigsDto -> configs for the inference
    conversation_history: ConversationHistoryDto -> list of messages between user and assistant
    message: MessageDto -> new message from user
    """
    
    configs: ConfigsDto
    conversation_history: List[MessageDto] = []
    message: MessageDto

class HealthCheckResponse(BaseModel):
    is_healthy: bool
    reason: str