import os
import logging
from dotenv import load_dotenv
from typing import List, Optional, Dict
from enum import Enum

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

class VectorDBType(Enum):
    PGVECTOR = "pgvector"
    ATLAS_MONGO = "atlas-mongo"

def get_env_variable(name: str, default: Optional[str] = None, required: bool = True) -> Optional[str]:
    """Get environment variable with proper error handling."""
    value = os.getenv(name, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {name} is not set")
    return value

# Core Configuration
OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
PORT = int(get_env_variable("PORT", "8000", required=False))
HOST = get_env_variable("HOST", "0.0.0.0", required=False)

# Document processing configuration
RAG_UPLOAD_DIR = get_env_variable("RAG_UPLOAD_DIR", "uploads", required=False)
CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1000", required=False))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "200", required=False))
MAX_UPLOAD_SIZE_MB = int(get_env_variable("MAX_UPLOAD_SIZE_MB", "50", required=False))

# Embeddings Configuration
EMBEDDINGS_PROVIDER = get_env_variable("EMBEDDINGS_PROVIDER", "openai", required=False)
EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "text-embedding-ada-002", required=False)

# Vector Database Configuration
vector_db_type_str = get_env_variable("VECTOR_DB_TYPE", "pgvector", required=False)
VECTOR_DB_TYPE = VectorDBType(vector_db_type_str.lower())

# Database Configuration
DATABASE_URL = get_env_variable("DATABASE_URL", required=False)
PGUSER = get_env_variable("PGUSER", required=False)
PGPASSWORD = get_env_variable("PGPASSWORD", required=False)
PGHOST = get_env_variable("PGHOST", required=False)
PGPORT = int(get_env_variable("PGPORT", "5432", required=False))
PGDATABASE = get_env_variable("PGDATABASE", required=False)

# Connection string prioritization
if DATABASE_URL:
    CONNECTION_STRING = DATABASE_URL
elif all([PGUSER, PGPASSWORD, PGHOST, PGDATABASE]):
    CONNECTION_STRING = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
else:
    CONNECTION_STRING = None
    logger.warning("No database connection string available")

# MongoDB Atlas Configuration (for atlas-mongo vector store)
ATLAS_MONGO_DB_URI = get_env_variable("ATLAS_MONGO_DB_URI", required=False)
ATLAS_SEARCH_INDEX = get_env_variable("ATLAS_SEARCH_INDEX", required=False)
MONGO_VECTOR_COLLECTION = get_env_variable("MONGO_VECTOR_COLLECTION", required=False)

# Collection Configuration
COLLECTION_NAME = get_env_variable("COLLECTION_NAME", "vector_index", required=False)

# JWT Configuration
JWT_SECRET_KEY = get_env_variable("JWT_SECRET_KEY", "your-secret-key-change-this-in-production", required=False)
JWT_ALGORITHM = get_env_variable("JWT_ALGORITHM", "HS256", required=False)
JWT_EXPIRATION_HOURS = int(get_env_variable("JWT_EXPIRATION_HOURS", "24", required=False))

# CORS Configuration
CORS_ORIGINS = get_env_variable("CORS_ORIGINS", "*", required=False).split(",")

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS = int(get_env_variable("RATE_LIMIT_REQUESTS", "100", required=False))
RATE_LIMIT_PERIOD = int(get_env_variable("RATE_LIMIT_PERIOD", "60", required=False))

# Mem0 Integration Configuration
MEM0_API_KEY = get_env_variable("MEM0_API_KEY", required=False)
MEM0_BASE_URL = get_env_variable("MEM0_BASE_URL", "https://api.mem0.ai", required=False)

# DragonflyDB Configuration
DRAGONFLY_URL = get_env_variable("DRAGONFLY_URL", required=False)
DRAGONFLY_HOST = get_env_variable("DRAGONFLY_HOST", "localhost", required=False)
DRAGONFLY_PORT = int(get_env_variable("DRAGONFLY_PORT", "6379", required=False))
DRAGONFLY_PASSWORD = get_env_variable("DRAGONFLY_PASSWORD", required=False)
CACHE_TTL = int(get_env_variable("CACHE_TTL", "3600", required=False))  # 1 hour default

# Build connection string for DragonflyDB if not provided
if not DRAGONFLY_URL and DRAGONFLY_HOST and DRAGONFLY_PORT:
    if DRAGONFLY_PASSWORD:
        DRAGONFLY_URL = f"redis://:{DRAGONFLY_PASSWORD}@{DRAGONFLY_HOST}:{DRAGONFLY_PORT}"
    else:
        DRAGONFLY_URL = f"redis://{DRAGONFLY_HOST}:{DRAGONFLY_PORT}"

# Default user for hybrid memory system
DEFAULT_HYBRID_USER = get_env_variable("DEFAULT_HYBRID_USER", "quinn_may", required=False)

# Create uploads directory if it doesn't exist
os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)

logger.info("Configuration loaded successfully")
logger.info(f"Vector DB Type: {VECTOR_DB_TYPE}")
logger.info(f"Embeddings Provider: {EMBEDDINGS_PROVIDER}")
logger.info(f"Collection Name: {COLLECTION_NAME}")
logger.info(f"Upload Directory: {RAG_UPLOAD_DIR}")
logger.info(f"Max Upload Size: {MAX_UPLOAD_SIZE_MB}MB")
logger.info(f"Mem0 Integration: {'Enabled' if MEM0_API_KEY else 'Disabled'}")
logger.info(f"DragonflyDB Caching: {'Enabled' if DRAGONFLY_URL else 'Disabled'}")

# Import vector store implementations
try:
    from app.vector_stores.pgvector_store import PGVectorStore
    from app.vector_stores.atlas_mongo_store import AtlasMongoStore
    logger.info("Vector store implementations imported successfully")
except ImportError as e:
    logger.warning(f"Could not import vector store implementations: {e}")

def get_vector_store(connection_string: str, embeddings, collection_name: str, mode: str, **kwargs):
    """Factory function to create vector store instances."""
    if mode == "async":
        # PGVector store for async operations
        return PGVectorStore(
            connection_string=connection_string,
            embeddings=embeddings,
            collection_name=collection_name,
        )
    elif mode == "atlas-mongo":
        # MongoDB Atlas Vector Search store
        search_index = kwargs.get("search_index", collection_name)
        return AtlasMongoStore(
            connection_string=connection_string,
            embeddings=embeddings,
            collection_name=collection_name,
            search_index=search_index,
        )
    else:
        raise ValueError(f"Unsupported vector store mode: {mode}")

# Initialize embeddings based on provider
def init_embeddings(provider: str, model: str):
    """Initialize embeddings based on the provider."""
    if provider.lower() == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=OPENAI_API_KEY,
            )
        except ImportError:
            raise ImportError("OpenAI embeddings not available. Install langchain-openai.")
    elif provider.lower() == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model)
        except ImportError:
            raise ImportError("HuggingFace embeddings not available. Install langchain-huggingface.")
    elif provider.lower() == "sentence_transformers":
        try:
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            return SentenceTransformerEmbeddings(model_name=model)
        except ImportError:
            raise ImportError("SentenceTransformers embeddings not available. Install sentence-transformers.")
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")

try:
    embeddings = init_embeddings(EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL)
    logger.info(f"Initialized embeddings of type: {type(embeddings)}")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    raise ValueError(f"Unsupported embeddings provider: {EMBEDDINGS_PROVIDER}")

embeddings = init_embeddings(EMBEDDINGS_PROVIDER, EMBEDDINGS_MODEL)

logger.info(f"Initialized embeddings of type: {type(embeddings)}")

# Vector store initialization
vector_store = None
retriever = None

def initialize_vector_store():
    global vector_store, retriever
    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        vector_store = get_vector_store(
            connection_string=CONNECTION_STRING,
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            mode="async",
        )
    elif VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
        # Backward compatability check
        if MONGO_VECTOR_COLLECTION:
            logger.info(
                f"DEPRECATED: Please remove env var MONGO_VECTOR_COLLECTION and instead use COLLECTION_NAME and ATLAS_SEARCH_INDEX. You can set both as same, but not neccessary. See README for more information."
            )
            ATLAS_SEARCH_INDEX = MONGO_VECTOR_COLLECTION
            COLLECTION_NAME = MONGO_VECTOR_COLLECTION
        vector_store = get_vector_store(
            connection_string=ATLAS_MONGO_DB_URI,
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            mode="atlas-mongo",
            search_index=ATLAS_SEARCH_INDEX,
        )
    else:
        raise ValueError(f"Unsupported vector store type: {VECTOR_DB_TYPE}")
    
    retriever = vector_store.as_retriever()
    return vector_store, retriever

# Initialize immediately for imports
try:
    vector_store, retriever = initialize_vector_store()
except Exception as e:
    logger.warning(f"Failed to initialize vector store during import: {e}")
    # Keep vector_store and retriever as None for now

known_source_ext = [
    "go",
    "py",
    "java",
    "sh",
    "bat",
    "ps1",
    "js",
    "ts",
    "html",
    "css",
    "cpp",
    "hpp",
    "h",
    "c",
    "cs",
    "php",
    "rb",
    "swift",
    "kt",
    "scala",
    "clj",
    "hs",
    "elm",
    "rs",
    "dart",
    "lua",
    "pl",
    "r",
    "m",
    "sql",
    "json",
    "xml",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "conf",
    "properties",
    "env",
    "dockerfile",
    "makefile",
    "cmake",
    "gradle",
    "maven",
    "sbt",
    "cabal",
    "cargo",
    "poetry",
    "pipfile",
    "requirements",
    "gemfile",
    "package",
    "bower",
    "composer",
]

logger.info("Configuration initialization completed")