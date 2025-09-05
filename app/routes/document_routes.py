import json
import os
from typing import List, Optional, Dict, Any
from uuid import uuid4
import asyncio
import redis
import requests
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from app.config import get_env_variable, logger
from app.models import (
    Document,
    ProcessDocumentRequest,
    SearchRequest,
    SearchResult,
    DocumentDeleteRequest,
    DocumentListResponse,
    DocumentFilter,
    ProcessingQueueResponse,
    ProcessingJobResponse,
    BatchProcessRequest
)
from app.services import (
    document_service,
    background_tasks_service
)
from app.services.database import PSQLDatabase

router = APIRouter(tags=["documents"])

# Configuration for new services
MEM0_API_KEY = get_env_variable("MEM0_API_KEY", "")
MEM0_USER_ID = get_env_variable("MEM0_USER_ID", "default_user")
DRAGONFLY_URL = get_env_variable("DRAGONFLY_URL", "")

# Pydantic models for new endpoints
class MemoryRequest(BaseModel):
    content: str = Field(..., description="Content to store in memory")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")

class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")

class HybridSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    search_type: str = Field(default="hybrid", description="Type of search: hybrid, memory_only, rag_only")

# Hybrid Memory System Implementation
class HybridMemorySystem:
    def __init__(self):
        self.dragonfly_client = None
        if DRAGONFLY_URL:
            try:
                self.dragonfly_client = redis.from_url(DRAGONFLY_URL, decode_responses=True)
                logger.info("DragonflyDB connection established")
            except Exception as e:
                logger.error(f"Failed to connect to DragonflyDB: {e}")
    
    async def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add content to Mem0 memory system"""
        if not MEM0_API_KEY:
            raise HTTPException(status_code=500, detail="Mem0 API key not configured")
        
        try:
            headers = {
                "Authorization": f"Bearer {MEM0_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [{"content": content, "role": "user"}],
                "user_id": MEM0_USER_ID
            }
            
            if metadata:
                payload["metadata"] = metadata
            
            response = requests.post(
                "https://api.mem0.ai/v1/memories/",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 201:
                result = response.json()
                
                # Cache in DragonflyDB if available
                if self.dragonfly_client:
                    try:
                        cache_key = f"memory:{MEM0_USER_ID}:{result.get('id', 'unknown')}"
                        self.dragonfly_client.setex(cache_key, 3600, json.dumps(result))
                    except Exception as e:
                        logger.warning(f"Failed to cache in DragonflyDB: {e}")
                
                return result
            else:
                logger.error(f"Mem0 API error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Failed to add memory")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Mem0 failed: {e}")
            raise HTTPException(status_code=500, detail="Memory service unavailable")
    
    async def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Mem0 memories"""
        if not MEM0_API_KEY:
            raise HTTPException(status_code=500, detail="Mem0 API key not configured")
        
        # Check cache first
        cache_key = f"search:{MEM0_USER_ID}:{hash(query)}:{limit}"
        if self.dragonfly_client:
            try:
                cached = self.dragonfly_client.get(cache_key)
                if cached:
                    logger.info("[CACHE HIT] Memory search served from DragonflyDB")
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        try:
            headers = {
                "Authorization": f"Bearer {MEM0_API_KEY}",
                "Content-Type": "application/json"
            }
            
            params = {
                "user_id": MEM0_USER_ID,
                "query": query,
                "limit": limit
            }
            
            response = requests.get(
                "https://api.mem0.ai/v1/memories/search/",
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                
                # Cache results
                if self.dragonfly_client:
                    try:
                        self.dragonfly_client.setex(cache_key, 300, json.dumps(results))
                        logger.info("[CACHE MISS] Memory search results cached")
                    except Exception as e:
                        logger.warning(f"Failed to cache search results: {e}")
                
                return results
            else:
                logger.error(f"Mem0 search error: {response.status_code} - {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Memory search failed")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Memory search request failed: {e}")
            raise HTTPException(status_code=500, detail="Memory service unavailable")
    
    async def hybrid_search(self, query: str, limit: int = 10, search_type: str = "hybrid") -> Dict[str, Any]:
        """Perform hybrid search across memory and RAG systems"""
        results = {
            "query": query,
            "search_type": search_type,
            "results": {
                "memory_results": [],
                "rag_results": [],
                "combined_results": []
            },
            "metadata": {
                "memory_count": 0,
                "rag_count": 0,
                "cache_status": "miss"
            }
        }
        
        # Check cache for hybrid search
        cache_key = f"hybrid:{MEM0_USER_ID}:{hash(query)}:{limit}:{search_type}"
        if self.dragonfly_client:
            try:
                cached = self.dragonfly_client.get(cache_key)
                if cached:
                    logger.info("[CACHE HIT] Hybrid search served from DragonflyDB")
                    cached_result = json.loads(cached)
                    cached_result["metadata"]["cache_status"] = "hit"
                    return cached_result
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        try:
            # Memory search (Tier 2: Warm)
            if search_type in ["hybrid", "memory_only"]:
                try:
                    memory_results = await self.search_memory(query, limit//2 if search_type == "hybrid" else limit)
                    results["results"]["memory_results"] = memory_results
                    results["metadata"]["memory_count"] = len(memory_results)
                    logger.info(f"Memory search returned {len(memory_results)} results")
                except Exception as e:
                    logger.warning(f"Memory search failed: {e}")
            
            # RAG search (Tier 3: Cold)
            if search_type in ["hybrid", "rag_only"]:
                try:
                    search_request = SearchRequest(query=query, limit=limit//2 if search_type == "hybrid" else limit)
                    rag_response = await document_service.search_documents(search_request)
                    results["results"]["rag_results"] = rag_response.results
                    results["metadata"]["rag_count"] = len(rag_response.results)
                    logger.info(f"RAG search returned {len(rag_response.results)} results")
                except Exception as e:
                    logger.warning(f"RAG search failed: {e}")
            
            # Combine and rank results
            combined = []
            
            # Add memory results with higher priority
            for i, mem_result in enumerate(results["results"]["memory_results"]):
                combined.append({
                    "content": mem_result.get("memory", ""),
                    "score": 0.9 - (i * 0.1),  # Higher scores for memory
                    "source": "memory",
                    "metadata": mem_result
                })
            
            # Add RAG results with lower priority
            for i, rag_result in enumerate(results["results"]["rag_results"]):
                combined.append({
                    "content": rag_result.content,
                    "score": 0.7 - (i * 0.05),  # Lower scores for RAG
                    "source": "rag",
                    "metadata": {
                        "document_id": rag_result.document_id,
                        "relevance_score": rag_result.relevance_score
                    }
                })
            
            # Sort by score and limit
            combined.sort(key=lambda x: x["score"], reverse=True)
            results["results"]["combined_results"] = combined[:limit]
            
            # Cache results
            if self.dragonfly_client:
                try:
                    self.dragonfly_client.setex(cache_key, 300, json.dumps(results))
                    logger.info("[CACHE MISS] Hybrid search results cached")
                except Exception as e:
                    logger.warning(f"Failed to cache hybrid results: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise HTTPException(status_code=500, detail="Hybrid search failed")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all memory system components"""
        status = {
            "timestamp": asyncio.get_event_loop().time(),
            "components": {
                "mem0": {
                    "configured": bool(MEM0_API_KEY),
                    "user_id": MEM0_USER_ID,
                    "status": "unknown"
                },
                "dragonfly": {
                    "configured": bool(DRAGONFLY_URL),
                    "connected": False,
                    "status": "disconnected"
                },
                "rag": {
                    "status": "available"
                }
            }
        }
        
        # Test DragonflyDB connection
        if self.dragonfly_client:
            try:
                self.dragonfly_client.ping()
                status["components"]["dragonfly"]["connected"] = True
                status["components"]["dragonfly"]["status"] = "connected"
            except Exception as e:
                status["components"]["dragonfly"]["status"] = f"error: {str(e)}"
        
        # Test Mem0 connection
        if MEM0_API_KEY:
            try:
                headers = {"Authorization": f"Bearer {MEM0_API_KEY}"}
                response = requests.get(
                    f"https://api.mem0.ai/v1/memories/?user_id={MEM0_USER_ID}&limit=1",
                    headers=headers,
                    timeout=5
                )
                if response.status_code == 200:
                    status["components"]["mem0"]["status"] = "connected"
                else:
                    status["components"]["mem0"]["status"] = f"error: {response.status_code}"
            except Exception as e:
                status["components"]["mem0"]["status"] = f"error: {str(e)}"
        
        return status

# Initialize hybrid memory system
hybrid_memory = HybridMemorySystem()

# New Memory Integration Endpoints
@router.post("/memory/add")
async def add_memory(request: MemoryRequest):
    """Add content to personal memory system"""
    try:
        result = await hybrid_memory.add_memory(request.content, request.metadata)
        return {
            "success": True,
            "message": "Memory added successfully",
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory add failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to add memory")

@router.post("/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search personal memories"""
    try:
        results = await hybrid_memory.search_memory(request.query, request.limit)
        return {
            "success": True,
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail="Memory search failed")

@router.post("/search/hybrid")
async def hybrid_search(request: HybridSearchRequest):
    """Perform hybrid search across memory and document systems"""
    try:
        results = await hybrid_memory.hybrid_search(
            request.query, 
            request.limit, 
            request.search_type
        )
        return {
            "success": True,
            "data": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail="Hybrid search failed")

@router.get("/memory/status")
async def memory_system_status():
    """Get status of all memory system components"""
    try:
        status = hybrid_memory.get_system_status()
        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

# Existing endpoints continue below...

@router.post("/upload", response_model=ProcessingJobResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """Upload one or more documents for processing."""
    
    logger.info(f"Upload request received for {len(files)} files")
    
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file extensions
    allowed_extensions = {".txt", ".pdf", ".docx", ".md"}
    for file in files:
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"File type not supported: {file.filename}. Allowed types: {', '.join(allowed_extensions)}"
            )
    
    try:
        # Create processing job
        job_id = str(uuid4())
        
        # Start background processing
        background_tasks.add_task(
            background_tasks_service.process_documents_background,
            job_id,
            files,
            request.app.state.CHUNK_SIZE,
            request.app.state.CHUNK_OVERLAP,
            request.app.state.PDF_EXTRACT_IMAGES,
            request.app.state.thread_pool
        )
        
        return ProcessingJobResponse(
            job_id=job_id,
            status="processing",
            message="Files uploaded successfully. Processing started.",
            files_count=len(files)
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload processing failed")

@router.post("/process", response_model=ProcessingJobResponse) 
async def process_document(
    request: Request,
    background_tasks: BackgroundTasks,
    process_request: ProcessDocumentRequest
):
    """Process a document from URL or text content."""
    
    logger.info(f"Process request received: {process_request.source_type}")
    
    try:
        # Create processing job
        job_id = str(uuid4())
        
        # Start background processing
        background_tasks.add_task(
            background_tasks_service.process_document_background,
            job_id,
            process_request,
            request.app.state.CHUNK_SIZE,
            request.app.state.CHUNK_OVERLAP,
            request.app.state.PDF_EXTRACT_IMAGES,
            request.app.state.thread_pool
        )
        
        return ProcessingJobResponse(
            job_id=job_id,
            status="processing",
            message="Document processing started.",
            files_count=1
        )
        
    except Exception as e:
        logger.error(f"Process request failed: {e}")
        raise HTTPException(status_code=500, detail="Document processing failed")

@router.post("/batch-process", response_model=ProcessingJobResponse)
async def batch_process_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_request: BatchProcessRequest
):
    """Process multiple documents in batch."""
    
    logger.info(f"Batch process request received for {len(batch_request.documents)} documents")
    
    if not batch_request.documents or len(batch_request.documents) == 0:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    try:
        # Create processing job
        job_id = str(uuid4())
        
        # Start background processing
        background_tasks.add_task(
            background_tasks_service.process_batch_background,
            job_id,
            batch_request,
            request.app.state.CHUNK_SIZE,
            request.app.state.CHUNK_OVERLAP,
            request.app.state.PDF_EXTRACT_IMAGES,
            request.app.state.thread_pool
        )
        
        return ProcessingJobResponse(
            job_id=job_id,
            status="processing",
            message="Batch processing started.",
            files_count=len(batch_request.documents)
        )
        
    except Exception as e:
        logger.error(f"Batch process failed: {e}")
        raise HTTPException(status_code=500, detail="Batch processing failed")

@router.get("/job/{job_id}", response_model=ProcessingJobResponse)
async def get_processing_job_status(job_id: str):
    """Get the status of a processing job."""
    
    try:
        job_status = await background_tasks_service.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
            
        return job_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job status")

@router.get("/queue", response_model=ProcessingQueueResponse)
async def get_processing_queue():
    """Get current processing queue status."""
    
    try:
        queue_status = await background_tasks_service.get_queue_status()
        return queue_status
        
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queue status")

@router.post("/search", response_model=SearchResult)
async def search_documents(search_request: SearchRequest):
    """Search through processed documents."""
    
    logger.info(f"Search request: '{search_request.query}' (limit: {search_request.limit})")
    
    try:
        results = await document_service.search_documents(search_request)
        logger.info(f"Search completed: {len(results.results)} results found")
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search operation failed")

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    document_type: Optional[str] = None
):
    """List all processed documents with optional filtering."""
    
    try:
        document_filter = DocumentFilter(
            search=search,
            document_type=document_type,
            limit=limit,
            offset=offset
        )
        
        results = await document_service.list_documents(document_filter)
        return results
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@router.get("/documents/{document_id}", response_model=Document)
async def get_document(document_id: str):
    """Get a specific document by ID."""
    
    try:
        document = await document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document")

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated chunks."""
    
    try:
        success = await document_service.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
            
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.post("/documents/delete-batch")
async def delete_documents_batch(delete_request: DocumentDeleteRequest):
    """Delete multiple documents."""
    
    try:
        results = await document_service.delete_documents_batch(delete_request.document_ids)
        return {
            "message": f"Deleted {results['deleted_count']} documents",
            "failed_deletes": results['failed_deletes']
        }
        
    except Exception as e:
        logger.error(f"Batch delete failed: {e}")
        raise HTTPException(status_code=500, detail="Batch delete operation failed")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAG API Enhanced"}