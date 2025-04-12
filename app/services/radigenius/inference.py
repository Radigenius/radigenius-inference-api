from typing import List
import logging
import torch
import os
from huggingface_hub import snapshot_download
from unsloth import FastVisionModel
from PIL import Image
import copy
import requests
import re

from app.exceptions.model_exceptions import ModelInferenceException, ModelDownloadException
from app.decorators.model_initialized_guard import model_initialized_guard
from app.dtos.inference_dto import InferenceRequest, MessageDto, ContentDto

from .config import get_config

logger = logging.getLogger(__name__)

class RadiGenius:
    model = None
    tokenizer = None
    device = None
    is_mock = False

    def __init__(self) -> None:
        # Check if we should run in mock mode based on DEBUG env
        # detect development mode in fastapi
        RadiGenius.is_mock = os.environ.get('DEBUG', False)
        
        if not RadiGenius.is_mock:
            if RadiGenius.model is None or RadiGenius.tokenizer is None:
                self.initialize_model()
            
            if RadiGenius.device is None:
                self.initialize_device()
        else:
            logger.info("Running RadiGenius in mock mode (DEBUG=True)")

    @classmethod
    def initialize_model(cls):
        """Initialize and return the base model and tokenizer."""
        config = get_config()
        
        logger.info(f'initializing model with config: {config}')
        model, tokenizer = FastVisionModel.from_pretrained(**config)
        
        logger.info('initializing model for inference')
        FastVisionModel.for_inference(model)

        cls.model = model
        cls.tokenizer = tokenizer

        logger.info('model initialized')

    @classmethod
    def initialize_device(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def download_model(cls):
        """
        Downloads the model files from Hugging Face Hub to the local cache directory.
        This method can be called independently to pre-download the model files.
        
        Returns:
            str: Path to the downloaded model directory
        """
        if cls.is_mock:
            logger.info("Mock mode: Skipping model download")
            return "/mock/model/path"
            
        try:
            # Get model configuration from config.py
            config = get_config()
            model_name = config["model_name"]
            cache_dir = config["cache_dir"]
            
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            logger.info(f"Downloading model '{model_name}' to {cache_dir}...")
            
            # Download the model files
            local_dir = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_dir=os.path.join(cache_dir, model_name.split("/")[-1]),
                local_dir_use_symlinks=False,
                resume_download=True,
                revision="main"
            )
            
            logger.info(f"Model successfully downloaded to {local_dir}")
                
            return local_dir
        except Exception as e:
            raise ModelDownloadException(errors=[str(e)])


    @staticmethod
    def _create_template(request: InferenceRequest) -> tuple[List[MessageDto], List[str]]:
        """
        Creates a template from the conversation history and current message.
        Removes image URLs from the template but keeps the image type.
        
        Returns:
            tuple: (template, image_urls)
                - template: The conversation template with image URLs removed
                - image_urls: List of all image URLs in order
        """
        image_urls = []

        template: List[MessageDto] = []
        
        # Process conversation history
        conversation_history = request.conversation_history
        template = copy.deepcopy(conversation_history)
        template.append(request.message)
        
        # Extract image URLs from conversation history
        for message in template:
            for i, content_item in enumerate(message.content):
                if content_item.type == "image":
                    image_urls.append(content_item.image)
                    message.content[i] = {"type": "image"}
        
        return template, image_urls

    def _send_message_mock(self, request: InferenceRequest):

        prompt = request.prompt

        # Create a mock response using a portion of the input content
        mock_response = f"[MOCK RESPONSE] Echo of your prompt: '{prompt[:100]}...'"
        
        # Log the mock operation
        logger.info(f"Mock RadiGenius used. Input length: {len(prompt)}")
        
        if request.config.stream:
                # For mock streaming, split response into words and yield them one by one
                def mock_stream():
                    for word in mock_response.split():
                        yield word + " "
                return mock_stream()
        
        return mock_response

    @classmethod
    def _stream_output(cls, generation_kwargs: dict):
        """
        Handles streaming output from the model using a TextIteratorStreamer.
        Yields chunks of generated text as they become available.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        # Create streamer and attach to generation kwargs
        streamer = TextIteratorStreamer(
            tokenizer=cls.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generation_kwargs["streamer"] = streamer

        # Start generation in a separate thread
        thread = Thread(target=cls.model.generate, kwargs=generation_kwargs)
        thread.start()

        def process_streaming_output():
            logger.info('Streaming output started')
            assistant_output = ""

            try:
                for new_text in streamer:
                    if new_text.strip():
                        assistant_output += new_text
                        yield f"{new_text}\n\n"
            finally:
                logger.info(f'Streaming completed. Response: {assistant_output}')
                thread.join()
                generation_kwargs.pop("streamer", None)
                del streamer
                del thread
                torch.cuda.empty_cache()
                logger.info('Streaming output cleanup completed')

        return process_streaming_output()

    @classmethod
    @model_initialized_guard
    def send_message(cls, request: InferenceRequest):
        """
        Send a message to the model and get a response.
        Supports streaming and non-streaming inference.
        """
        logger.info(f'sending message with request: {request}')

        if cls.is_mock:
            return cls._send_message_mock(request)

        try:
            # Create the chat template and extract image URLs
            template, image_urls = cls._create_template(request)

            # Load images in correct order
            images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

            # Format the input prompt
            input_text = cls.tokenizer.apply_chat_template(template, add_generation_prompt=True)

            # Tokenize both images and text
            inputs = cls.tokenizer(
                images,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(cls.device)

            # Common generation parameters
            common_kwargs = dict(
                max_new_tokens=request.configs.max_new_tokens,
                temperature=request.configs.temperature,
                min_p=request.configs.min_p,
            )

            # If streaming is enabled
            if request.configs.stream:
                return cls._stream_output({**inputs, **common_kwargs})

            # Generate output (non-streaming)
            output_ids = cls.model.generate(**inputs, **common_kwargs)

            # Extract newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][input_length:]
            generated_text = cls.tokenizer.decode(new_tokens, skip_special_tokens=True)

            logger.info(f'generated text: {generated_text}')
            return generated_text

        except Exception as e:
            raise ModelInferenceException(errors=[str(e)])

        finally:
            for var in [
                "inputs", "output_ids", "new_tokens",
                "generated_text", "template", "image_urls", "images"
            ]:
                if var in locals():
                    del locals()[var]
            torch.cuda.empty_cache()
            logger.info('Inference cleanup completed')
    @staticmethod
    def _prepare_response(generated_text: str):
        parts = generated_text.split("assistant\n\n")

        if len(parts) < 1:
            raise ModelInferenceException(errors=["No assistant response found"])

        return parts[-1]

    @classmethod
    def kill_model(cls):
        cls.model = None
        cls.tokenizer = None
        cls.device = None
        cls.is_mock = False

    @classmethod
    def is_healthy(cls):
        reason = ""
        if cls.model is None:
            reason = "Model is not initialized"
        elif cls.tokenizer is None:
            reason = "Tokenizer is not initialized"
        elif cls.device is None:
            reason = "Device is not initialized"

        if reason:
            logger.error(f'RadiGenius is not healthy: {reason}')
            return False, reason

        logger.info("RadiGenius is healthy")
        return True, "RadiGenius is healthy"
