from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Radigenius Inference API"
    PROJECT_DESCRIPTION: str = "A tiny project used as a gateway to the Radigenius model"
    VERSION: str = "0.1.0"
    
    class Config:
        env_file = ".env"

settings = Settings() 