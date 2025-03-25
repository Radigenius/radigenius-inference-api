from enum import Enum

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.dtos.inference_dto import InferenceRequest
from app.services.radigenius.inference import RadiGenius
router = APIRouter(
    prefix="/inference",
    tags=["inference"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", response_model=str)
def generate(request: InferenceRequest):
    """
    Generate a response using the provided parameters and template.
    
    - max_new_tokens: Maximum number of tokens to generate
    - temperature: Controls randomness (higher = more random)
    - min_p: Minimum probability threshold for token selection
    - model: Model to use for generation
    - prompt: Prompt to generate a response from
    - attachments: List of urls of images
    - stream: Whether to stream the response
    """

    radi_genius = RadiGenius()

    if request.stream:
        return StreamingResponse(
            radi_genius.send_message(request),
            media_type="text/event-stream"
        )

    return radi_genius.send_message(request)


@router.get("/healthy", response_model=bool)
def healthy():
    return RadiGenius.is_healthy()