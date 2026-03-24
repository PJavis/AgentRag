# src/pam/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from urllib.parse import quote_plus  # để escape password an toàn

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path.cwd() / ".env",  # dùng thư mục hiện tại (gốc dự án) khi chạy lệnh
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str

    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5433

    HF_TOKEN: str | None = None

    NEO4J_PASSWORD: str = "neo4j123456"   # default từ docker-compose
    NEO4J_USER: str = "neo4j"

    @property
    def DATABASE_URL(self) -> str:
        from urllib.parse import quote_plus
        password = quote_plus(self.POSTGRES_PASSWORD)
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{password}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()