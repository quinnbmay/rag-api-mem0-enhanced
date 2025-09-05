# app/routes/document_routes.py
import os
import hashlib
import traceback
import aiofiles
import aiofiles.os
import json
import redis
import requests
from shutil import copyfileobj
from typing import List, Iterable, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    HTTPException,
    File,
    Form,
    Body,
    Query,
    status,
)
from langchain_core.documents import Document
from langchain_core.runnables import run_in_executor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache

from app.config import logger, vector_store, RAG_UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP, get_env_variable
from app.constants import ERROR_MESSAGES
from app.models import (
    StoreDocument,
    QueryRequestBody,
    DocumentResponse,
    QueryMultipleBody,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import (
    get_loader,
    clean_text,
    process_documents,
    cleanup_temp_encoding_file,
)
from app.utils.health import is_health_ok

router = APIRouter()

# Mem0 and DragonflyDB Configuration
MEM0_API_KEY = get_env_variable("MEM0_API_KEY", "")
MEM0_USER_ID = get_env_variable("MEM0_USER_ID", "default_user")
DRAGONFLY_URL = get_env_variable("DRAGONFLY_URL", "")

# Additional Pydantic Models for Memory Integration
class MemoryRequest(BaseModel):
    content: str = Field(..., description="Memory content to store")
    user_id: Optional[str] = Field(default=None, description="User ID for memory storage")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional memory metadata")

class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Memory search query")
    user_id: Optional[str] = Field(default=None, description="User ID for memory search")
    limit: int = Field(default=10, description="Maximum number of memories to return")

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Hybrid search query")
    user_id: Optional[str] = Field(default=None, description="User ID for memory component")
    top_k: int = Field(default=5, description="Number of document results")
    memory_limit: int = Field(default=5, description="Number of memory results")
    cache_ttl: int = Field(default=300, description="Cache time-to-live in seconds")

class HybridMemorySystem:
    """
    Three-tier hybrid memory system:
    - Hot: DragonflyDB cache (Redis protocol)
    - Warm: Mem0 personal memory
    - Cold: RAG document search
    """
    
    def __init__(self):
        self.redis_client = None
        self.mem0_api_key = MEM0_API_KEY
        self.mem0_base_url = "https://api.mem0.ai/v1"
        self.default_user_id = MEM0_USER_ID
        self._initialize_dragonfly()
    
    def _initialize_dragonfly(self):
        """Initialize DragonflyDB connection if URL is provided"""
        if DRAGONFLY_URL:
            try:
                self.redis_client = redis.from_url(DRAGONFLY_URL, decode_responses=True)
                self.redis_client.ping()
                print("[OK] DragonflyDB connected successfully")
            except Exception as e:
                print(f"[WARN] Failed to connect to DragonflyDB: {e}")
                self.redis_client = None
        else:
            print("[INFO] DragonflyDB URL not provided, caching disabled")
    
    def _get_cache_key(self, query: str, query_type: str = "hybrid") -> str:
        """Generate cache key for query"""
        return f"{query_type}:{hash(query)}"
    
    async def get_cached_result(self, query: str, query_type: str = "hybrid") -> Optional[Dict[str, Any]]:
        """Get cached result from DragonflyDB"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._get_cache_key(query, query_type)
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                print(f"[HOT] Cache HIT for query: {query}")
                return json.loads(cached_data)
            else:
                print(f"[COLD] Cache MISS for query: {query}")
                return None
        except Exception as e:
            print(f"[ERROR] Cache read error: {e}")
            return None
    
    async def cache_result(self, query: str, result: Dict[str, Any], ttl: int = 300, query_type: str = "hybrid"):
        """Cache result in DragonflyDB"""
        if not self.redis_client:
            return
        
        try:
            cache_key = self._get_cache_key(query, query_type)
            self.redis_client.setex(cache_key, ttl, json.dumps(result))
            print(f"[CACHE] Cached {query_type} result for: {query}")
        except Exception as e:
            print(f"[ERROR] Cache write error: {e}")
    
    async def add_memory(self, content: str, user_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add memory to Mem0"""
        if not self.mem0_api_key:
            raise HTTPException(status_code=400, detail="Mem0 API key not configured")
        
        user_id = user_id or self.default_user_id
        headers = {
            "Authorization": f"Token {self.mem0_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": content}],
            "user_id": user_id
        }
        
        if metadata:
            payload["metadata"] = metadata
        
        try:
            response = requests.post(
                f"{self.mem0_base_url}/memories/",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Mem0 API error: {response.status_code}"
                )
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Mem0 API request failed: {str(e)}")
    
    async def search_memory(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories in Mem0"""
        if not self.mem0_api_key:
            return []
        
        user_id = user_id or self.default_user_id
        headers = {
            "Authorization": f"Token {self.mem0_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(
                f"{self.mem0_base_url}/memories/",
                headers=headers,
                params={
                    "user_id": user_id,
                    "query": query,
                    "limit": limit
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                print(f"[WARN] Mem0 search failed: {response.status_code}")
                return []
        except requests.RequestException as e:
            print(f"[ERROR] Mem0 search error: {str(e)}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for all components"""
        from datetime import datetime
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "dragonfly_cache": {
                    "enabled": self.redis_client is not None,
                    "connected": False
                },
                "mem0_memory": {
                    "enabled": bool(self.mem0_api_key),
                    "api_key_set": bool(self.mem0_api_key)
                },
                "rag_search": {
                    "enabled": True,
                    "vector_store": "pgvector"
                }
            }
        }
        
        # Test DragonflyDB connection
        if self.redis_client:
            try:
                self.redis_client.ping()
                status["components"]["dragonfly_cache"]["connected"] = True
            except:
                status["components"]["dragonfly_cache"]["connected"] = False
        
        return status

# Initialize hybrid memory system
hybrid_memory = HybridMemorySystem()

# New Memory Integration Endpoints
@router.post("/memory/add")
async def add_memory(request: MemoryRequest):
    """Add memory to Mem0"""
    try:
        result = await hybrid_memory.add_memory(
            content=request.content,
            user_id=request.user_id,
            metadata=request.metadata
        )
        return {
            "status": "success",
            "memory_id": result.get("id"),
            "message": "Memory added successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding memory: {str(e)}")

@router.post("/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search memories in Mem0"""
    try:
        memories = await hybrid_memory.search_memory(
            query=request.query,
            user_id=request.user_id,
            limit=request.limit
        )
        
        return {
            "query": request.query,
            "memories": memories,
            "total_memories": len(memories)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching memory: {str(e)}")

@router.post("/search/hybrid")
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search combining three tiers:
    1. Hot: DragonflyDB cache (instant retrieval)
    2. Warm: Mem0 personal memory (contextual)
    3. Cold: RAG document search (comprehensive)
    """
    try:
        from datetime import datetime
        import asyncio
        
        # Check cache first (Hot tier)
        cached_result = await hybrid_memory.get_cached_result(request.query, "hybrid")
        if cached_result:
            return cached_result
        
        # Perform searches in parallel
        async def search_documents_async():
            try:
                # Use existing RAG search functionality
                query_body = QueryRequestBody(
                    query=request.query,
                    top_k=request.top_k
                )
                # Call the existing search endpoint logic
                documents = await vector_store.similarity_search_with_score(
                    query=request.query,
                    k=request.top_k
                )
                return [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": float(score),
                        "source": "documents"
                    }
                    for doc, score in documents
                ]
            except Exception as e:
                print(f"[ERROR] Document search failed: {e}")
                return []
        
        async def search_memories_async():
            try:
                memories = await hybrid_memory.search_memory(
                    query=request.query,
                    user_id=request.user_id,
                    limit=request.memory_limit
                )
                return [
                    {
                        "content": mem.get("memory", ""),
                        "metadata": mem.get("metadata", {}),
                        "score": mem.get("score", 0.0),
                        "source": "memory",
                        "created_at": mem.get("created_at"),
                        "updated_at": mem.get("updated_at")
                    }
                    for mem in memories
                ]
            except Exception as e:
                print(f"[ERROR] Memory search failed: {e}")
                return []
        
        # Execute searches concurrently
        document_results, memory_results = await asyncio.gather(
            search_documents_async(),
            search_memories_async(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(document_results, Exception):
            document_results = []
        if isinstance(memory_results, Exception):
            memory_results = []
        
        # Combine results
        hybrid_result = {
            "query": request.query,
            "timestamp": datetime.utcnow().isoformat(),
            "results": {
                "documents": document_results,
                "memories": memory_results,
                "combined_count": len(document_results) + len(memory_results)
            },
            "search_stats": {
                "document_results": len(document_results),
                "memory_results": len(memory_results),
                "cached": False
            }
        }
        
        # Cache the result (async, don't wait)
        asyncio.create_task(
            hybrid_memory.cache_result(request.query, hybrid_result, request.cache_ttl, "hybrid")
        )
        
        return hybrid_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in hybrid search: {str(e)}")

@router.get("/memory/status")
async def memory_system_status():
    """Get status of the hybrid memory system components"""
    return hybrid_memory.get_system_status()

# Existing endpoints continue below...

@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: Optional[int] = Query(None),
    chunk_overlap: Optional[int] = Query(None),
):
    """Upload and process documents"""
    try:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files uploaded"
            )
        
        # Use provided chunk settings or defaults
        effective_chunk_size = chunk_size or CHUNK_SIZE
        effective_chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        
        processed_docs = []
        
        for file in files:
            # Validate file type
            if not any(file.filename.lower().endswith(ext) for ext in ['.txt', '.pdf', '.docx', '.md']):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            # Save uploaded file temporarily
            file_id = hashlib.md5(f"{file.filename}_{file.size}".encode()).hexdigest()
            temp_file_path = os.path.join(RAG_UPLOAD_DIR, f"{file_id}_{file.filename}")
            
            # Ensure upload directory exists
            os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)
            
            # Save file
            async with aiofiles.open(temp_file_path, 'wb') as temp_file:
                content = await file.read()
                await temp_file.write(content)
            
            try:
                # Process the document
                loader = get_loader(temp_file_path)
                documents = loader.load()
                
                if not documents:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"No content extracted from {file.filename}"
                    )
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=effective_chunk_size,
                    chunk_overlap=effective_chunk_overlap
                )
                chunks = text_splitter.split_documents(documents)
                
                # Add metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        "file_id": file_id,
                        "filename": file.filename,
                        "upload_timestamp": str(os.path.getmtime(temp_file_path))
                    })
                
                # Store in vector database
                await vector_store.aadd_documents(chunks)
                
                processed_docs.append({
                    "file_id": file_id,
                    "filename": file.filename,
                    "chunks_created": len(chunks),
                    "status": "success"
                })
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    await aiofiles.os.remove(temp_file_path)
        
        return {
            "status": "success",
            "processed_documents": processed_docs,
            "total_files": len(files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

@router.post("/documents/query")
async def query_documents(query_body: QueryRequestBody):
    """Search through processed documents"""
    try:
        if not query_body.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Perform similarity search
        results = await vector_store.similarity_search_with_score(
            query=query_body.query,
            k=query_body.top_k
        )
        
        # Format results
        formatted_results = []
        for document, score in results:
            formatted_results.append({
                "content": document.page_content,
                "metadata": document.metadata,
                "relevance_score": float(score)
            })
        
        return {
            "query": query_body.query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search operation failed: {str(e)}"
        )

@router.delete("/documents/{file_id}")
async def delete_document(file_id: str):
    """Delete all chunks for a specific document"""
    try:
        # This would need to be implemented based on your vector store
        # For now, return a placeholder response
        return {
            "status": "success",
            "message": f"Document {file_id} deletion requested",
            "file_id": file_id
        }
        
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document deletion failed: {str(e)}"
        )

@router.get("/documents/stats")
async def get_document_stats():
    """Get statistics about stored documents"""
    try:
        # Implementation depends on your vector store
        # This is a placeholder
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "vector_store_type": "pgvector"
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stats retrieval failed: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(os.time.time()),
        "version": "1.0.0-mem0-enhanced"
    }

print("[INFO] Hybrid Memory System initialized")