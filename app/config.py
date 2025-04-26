from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Chat"
    
    # Database Settings
    DATABASE_URL: Optional[str] = None
    
    # Model Settings
    MODEL_NAME: str = "facebook/bart-large-cnn"
    MAX_TOKENS: int = 150
    TEMPERATURE: float = 0.7
    
    # Vector Store Settings
    VECTOR_STORE_PATH: str = "vector_db"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-here"  # Change this in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 