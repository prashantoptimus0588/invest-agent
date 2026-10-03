from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: str = ""
    langsmith_api_key: str = ""
    database_url: str = "sqlite:///./dev.db"
    paper_mode: bool = True
    trading_enabled: bool = True
    max_trade_inr: int = 5000
    daily_cap_inr: int = 15000
    approval_secret: str = "change-me"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
