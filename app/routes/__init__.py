# app/routes/__init__.py
from .document_routes import router as document_routes
from .pgvector_routes import router as pgvector_routes

__all__ = ["document_routes", "pgvector_routes"]