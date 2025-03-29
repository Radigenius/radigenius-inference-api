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

from app.exceptions.model_exceptions import ModelInferenceException, ModelDownloadException, ModelNotInitializedException
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
            raise ModelDownloadException(str(e))


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
        from transformers import TextIteratorStreamer
        from threading import Thread
            
        # Create a streamer
        streamer = TextIteratorStreamer(cls.tokenizer, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        
        # Start generation in a separate thread
        thread = Thread(target=cls.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        def process_streaming_output():
            """
            Process the streaming output from the model:
            1. Discard everything until the First occurrence of "assistant\n\n" is found
            2. Stream all content after that marker
            3. Collect the complete output for logging
            """
            logger.info('Streaming output started')
            
            buffer = ""
            assistant_output = ""
            marker = "assistant"
            marker_found = False
            
            for token in streamer:
                buffer += token
                
                # Check if we have the marker in our buffer
                if marker in buffer and not marker_found:
                    # Found the marker for the first time
                    marker_found = True
                    # Use split to find the first occurrence
                    _, after_marker = buffer.split(marker, 1)
                    assistant_output = after_marker
                    yield after_marker
                elif marker_found:
                    # Marker already found, directly stream the new token
                    assistant_output += token
                    yield token
                
            logger.info(f'Streaming completed. Response length: {len(assistant_output)}')
        
        return process_streaming_output()

    @classmethod
    @model_initialized_guard
    def send_message(cls, request: InferenceRequest):
        """
        Send a message to the model and get a response.
        If in mock mode, returns a simplified response based on the input.
        """
        logger.info(f'sending message with request: {request}')

        if cls.is_mock:
            return cls._send_message_mock(request)
        
        try:
            # Get template and image URLs
            template, image_urls = cls._create_template(request)
            
            # Load all images in the correct order
            images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
            
            # Apply the chat template
            input_text = cls.tokenizer.apply_chat_template(template, add_generation_prompt=True)
            
            # Encode both images and text
            inputs = cls.tokenizer(images, input_text, add_special_tokens=False, return_tensors="pt").to(cls.device)
            
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=request.configs.max_new_tokens,
                temperature=request.configs.temperature,
                min_p=request.configs.min_p,
            )

            logger.debug(f'generation kwargs: {generation_kwargs}')

            if request.configs.stream:
                return cls._stream_output(generation_kwargs)

            # Non-streaming mode
            output_ids = cls.model.generate(
                **inputs,
                max_new_tokens=request.configs.max_new_tokens,
                temperature=request.configs.temperature,
                min_p=request.configs.min_p
            )
            generated_text = cls.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            logger.info(f'generated text: {generated_text}')
            
            return cls._prepare_response(generated_text)

        except ModelNotInitializedException:
            raise
        except Exception as e:
            raise ModelInferenceException(str(e))
        
    @staticmethod
    def _prepare_response(generated_text: str):
        """Extract only the most recent assistant response from the generated text and format it properly."""
        # Split by 'assistant' marker and take the last part
        parts = generated_text.split("assistant")
        
        if len(parts) <= 1:
            raise ModelInferenceException("No assistant response found")
        
        # Get the last assistant response
        last_response = parts[-1].strip()
        
        # Remove any leading newlines or formatting markers
        last_response = last_response.lstrip('\n: ')
        
        # If there's a "user" marker after this, truncate to only include content before it
        if "user" in last_response:
            last_response = last_response.split("user")[0].strip()
        
        # Clean up the text formatting
        # Replace multiple newlines with double newlines for paragraph breaks
        last_response = re.sub(r'\n{3,}', '\n\n', last_response)
        
        # Fix markdown formatting issues
        # Ensure proper spacing for headers
        last_response = re.sub(r'(#+)([^ #])', r'\1 \2', last_response)
        
        # Properly format lists if broken
        last_response = re.sub(r'(\n[*-]) ([^\n]+)(?=\n[*-])', r'\1 \2', last_response)
        
        formatted_output = last_response.strip()

        logger.info(f'formatted output: {formatted_output}')

        return formatted_output

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
