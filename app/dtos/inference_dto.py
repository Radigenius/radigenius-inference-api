from typing import List, Union, Literal
from pydantic import BaseModel, Field

from app.enums.inference_enum import ModelTypes

class ContentDto(BaseModel):
    type: Literal["text", "image"]
    text: str
    image: str 

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