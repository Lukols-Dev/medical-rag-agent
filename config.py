import logging
import os
from pathlib import Path
from typing import Self

from pydantic import SecretStr, field_validator, model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

# HTTP clients log every request at INFO - too noisy for the terminal chat
for noisy_logger in ("httpx", "httpx2"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    """
    Application settings. Field name = environment variable name (case-insensitive).
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_ignore_empty=True,
        str_strip_whitespace=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    # LangSmith
    langsmith_api_key:SecretStr | None = None
    langsmith_tracing:bool = False
    langsmith_project:str = Field(min_length=1)
    langsmith_endpoint: str = "https://api.smith.langchain.com"


    # OpenAI
    openai_api_key: SecretStr

    # OpenRouter (chat model for the agent)
    openrouter_api_key: SecretStr

    # Postgres (pgvector)
    database_url: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_langsmith(self) -> Self:
        """
        Tracing needs an API key and a project name - fail at startup instead of on the first trace.
        """

        if not self.langsmith_tracing:
            return self #tracing is off - nothing to check

        missing = []
        if not self.langsmith_api_key:
            missing.append("LANGSMITH_API_KEY")
        if not self.langsmith_project:
            missing.append("LANGSMITH_PROJECT")

        if missing:
            raise ValueError(
                f"LANGSMITH_TRACING=true, but these are missing: {', '.join(missing)} "
                "- add them to .env or set LANGSMITH_TRACING=false"
            )
        return self
    
    @field_validator("openai_api_key")
    @classmethod
    def _check_openai_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith("sk-"):
            raise ValueError("OPENAI_API_KEY should start with 'sk-'")
        return value

    @field_validator("database_url")
    @classmethod
    def _check_database_url(cls, value: str) -> str:
        if not value.startswith("postgresql+psycopg://"):
            raise ValueError("DATABASE_URL must start with 'postgresql+psycopg://'")
        return value


def _configure_langsmith(settings: Settings) -> None:
    """
    Copy LangSmith settings to os.environ`.
    """
    # LangSmith enables tracing only for the exact lowercase string "true".
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()

    if settings.langsmith_tracing:
        logger.info("✅ LangSmith is enabled and the API key is set (project: %s)", settings.langsmith_project)
    else:
        logger.info("⚠️ LangSmith is disabled")

settings = Settings()
_configure_langsmith(settings)
