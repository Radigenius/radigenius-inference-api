from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from enum import Enum

from app.dtos.inference_dto import InferenceRequest
from app.services.radigenius.inference import RadiGenius
router = APIRouter(
    prefix="/inference",
    tags=["inference"],
    responses={404: {"description": "Not found"}},
)

class ModelTypes(str, Enum):
    RadiGenius = "RadiGenius"



@router.post("/", response_model=str)
def generate(request: InferenceRequest):
    """
    Generate a response using the provided parameters and template.
    
    - max_new_tokens: Maximum number of tokens to generate
    - temperature: Controls randomness (higher = more random)
    - min_p: Minimum probability threshold for token selection
    - model: Model to use for generation
    - template: Conversation template with text and/or images
    """

    radi_genius = RadiGenius()

    return radi_genius.send_message(request)