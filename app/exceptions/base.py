class BaseException(Exception):
    def __init__(self, message: str, code: int, errors: list, key: str):
        self.message = message
        self.code = code
        self.errors = errors
        self.key = key

    def __str__(self):
        return f"{self.key}: {self.message}"
