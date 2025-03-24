import torch
import os

# ======================================================================
# INFERENCE-OPTIMIZED CONFIGURATION
# ======================================================================
# Model settings optimized for inference

DEVELOPMENT_CONFIG = {
    "model_name": "0llheaven/Llama-3.2-11B-Vision-Radiology-mini",
    
    # CPU-friendly configuration
    "load_in_8bit": False,
    
    "dtype": torch.float32,
    
    # Disable for inference
    "use_gradient_checkpointing": False,
    
    # Reduced sequence length for CPU memory constraints
    "max_seq_length": 2048,
    
    # Same cache directory
    "cache_dir": f"app/model_cache",
}

PRODUCTION_CONFIG = {
    "model_name": "0llheaven/Llama-3.2-11B-Vision-Radiology-mini",
    
    # For highest quality inference on RTX 4090 (24GB VRAM):
    # - False: Uses full precision for maximum quality
    # - True: More memory efficient but slightly lower quality
    "load_in_4bit": False,  # Set to False for highest quality on RTX 4090
    
    # Best precision for medical imaging on RTX 4090
    "dtype": torch.bfloat16,
    
    # Disable for inference (not needed, improves speed)
    "use_gradient_checkpointing": False,
    
    # Can use longer sequences for inference (fits in 24GB)
    "max_seq_length": 4096,
    
    # Same cache directory
    "cache_dir": f"app/model_cache",
}

# ======================================================================
# HELPER FUNCTION TO SELECT CONFIGURATION
# ======================================================================

def get_config():
    """
    Returns the appropriate configuration based on DEBUG setting.
    
    Returns:
        Dict containing the configuration
    """
    if os.environ.get('DEBUG', False):
        return DEVELOPMENT_CONFIG
    else:
        return PRODUCTION_CONFIG