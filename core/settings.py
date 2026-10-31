"""Central configuration. Every other module imports `settings` and `company`."""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Postgres
    postgres_user: str = "rag"
    postgres_password: str = "rag_dev_password"
    postgres_db: str = "ragdb"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "support_kb"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # S3-compatible storage
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin123"
    s3_bucket: str = "support-kb"
    s3_region: str = "us-east-1"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    model_cache_dir: str = ""      # empty -> ~/.cache/rag-models (kept out of OneDrive/project)
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"   # 3B: usable on a laptop CPU

    # Where git sources are cloned. Empty -> ~/.cache/rag-sources. Keep this OUT of OneDrive:
    # OneDrive locks files inside .git and corrupts clones. Absolute or project-relative.
    clone_dir: str = ""

    # "production" makes the API refuse to start with a weak/default JWT_SECRET
    app_env: str = "dev"

    # Prometheus scrape protection for GET /metrics. Empty = open (fine on localhost).
    metrics_token: str = ""

    # Where the browser reaches Grafana, for the admin page. Empty = no embedded dashboard.
    grafana_url: str = "http://localhost:3000"

    # Auth
    jwt_secret: str = "change-me-in-real-use"
    jwt_expire_minutes: int = 60

    @property
    def resolved_clone_dir(self) -> str:
        return self.clone_dir or str(Path.home() / ".cache" / "rag-sources")

    @property
    def resolved_model_cache(self) -> str:
        return self.model_cache_dir or str(Path.home() / ".cache" / "rag-models")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_company() -> dict:
    with open(ROOT / "config" / "company.yaml") as f:
        return yaml.safe_load(f)


settings = get_settings()
company = get_company()
