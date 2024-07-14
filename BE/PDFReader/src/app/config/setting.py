from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "PDF Processing Service"
    DEBUG: bool = False

    class Config:
        env_file = ".env"


settings = Settings()