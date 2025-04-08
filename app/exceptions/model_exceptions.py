from app.exceptions.base import BaseException

class ModelNotInitializedException(BaseException):
    """
    Exception raised when the model is not initialized.
    """
    def __init__(self, errors=[]):
        self.message = "Model is not initialized"
        self.code = 500
        self.errors = errors
        self.key = "model_not_initialized_exception"

class ModelInferenceException(BaseException):
    """
    Exception raised when the model inference fails.
    """
    def __init__(self, errors=[]):
        self.message = "Model inference failed"
        self.code = 500
        self.errors = errors
        self.key = "model_inference_exception"


class ModelDownloadException(BaseException):
    """
    Exception raised when the model download fails.
    """
    def __init__(self, errors=[]):
        self.message = "Model download failed"
        self.code = 500
        self.errors = errors
        self.key = "model_download_exception"