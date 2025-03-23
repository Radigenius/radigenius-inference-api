
class ModelNotInitializedException(Exception):
    """
    Exception raised when the model is not initialized.
    """
    def __init__(self, message="Model is not initialized"):
        self.message = message
        super().__init__(self.message)

class ModelInferenceException(Exception):
    """
    Exception raised when the model inference fails.
    """
    def __init__(self, message="Model inference failed"):
        self.message = message
        super().__init__(self.message)

class ModelDownloadException(Exception):
    """
    Exception raised when the model download fails.
    """
    def __init__(self, message="Model download failed"):
        self.message = message
        super().__init__(self.message)