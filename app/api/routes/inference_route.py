from enum import Enum

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.dtos.inference_dto import InferenceRequest, HealthCheckResponse
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
    configs: ConfigsDto -> configs for the inference
    conversation_history: ConversationHistoryDto -> list of messages between user and assistant
    message: MessageDto -> new message from user
    """

    radi_genius = RadiGenius()

    if request.configs.stream:
        return StreamingResponse(
            radi_genius.send_message(request),
            media_type="text/event-stream"
        )

    return radi_genius.send_message(request)


@router.get("/healthy", response_model=HealthCheckResponse)
def healthy():
    is_healthy, reason = RadiGenius.is_healthy()
    return {"is_healthy": is_healthy, "reason": reason}