# app/config.py
import os
import logging
import sys
from enum import Enum
from typing import Optional

from fastapi import Request
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic_settings import BaseSettings


class VectorDBType(Enum):
    PGVECTOR = "pgvector"
    ATLAS = "atlas"


class RAGSettings(BaseSettings):
    debug_mode: bool = False
    rag_host: str = "0.0.0.0"
    rag_port: int = 8080
    chunk_size: int = 1000
    chunk_overlap: int = 200
    pdf_extract_images: bool = False
    vector_db_type: VectorDBType = VectorDBType.PGVECTOR
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = RAGSettings()

# Global configuration variables
debug_mode = settings.debug_mode
RAG_HOST = settings.rag_host
RAG_PORT = settings.rag_port
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
PDF_EXTRACT_IMAGES = settings.pdf_extract_images
VECTOR_DB_TYPE = settings.vector_db_type

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def get_env_variable(var_name: str, default_value: str = "") -> str:
    """Get environment variable with default value."""
    return os.getenv(var_name, default_value)


class LogMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next):
        # Log request
        logger.info(f"{request.method} {request.url.path}")
        
        response = await call_next(request)
        
        # Log response
        logger.info(f"Response status: {response.status_code}")
        
        return response