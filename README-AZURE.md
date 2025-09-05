# RAG API - Azure Deployment

This is a fork of the [RAG API](https://github.com/danny-avila/rag_api) project, customized for deployment in Azure as part of the Neo Agent infrastructure.

## Overview

The RAG API provides a FastAPI-based service for Retrieval-Augmented Generation (RAG) functionality, integrated with:
- PostgreSQL with pgvector extension for vector storage
- Azure OpenAI for embeddings
- LibreChat for the chat interface

## Architecture

The RAG API is deployed as an Azure Container App with the following components:

### Infrastructure Components
- **PostgreSQL Flexible Server**: Database with pgvector extension for vector storage
- **Container App Environment**: Isolated environment for the RAG API
- **Container App**: The RAG API service itself

### Configuration
- **Vector Database**: PostgreSQL with pgvector extension
- **Embeddings Provider**: Azure OpenAI (text-embedding-3-small)
- **Collection Name**: `librechat_rag`
- **Chunk Size**: 1500
- **Chunk Overlap**: 100

## Environment Variables

The RAG API is configured with the following environment variables:

### Server Configuration
- `RAG_HOST`: 0.0.0.0
- `RAG_PORT`: 8000
- `DEBUG_RAG_API`: true (dev) / false (prod)

### Database Configuration
- `VECTOR_DB_TYPE`: pgvector
- `POSTGRES_DB`: rag_api
- `POSTGRES_USER`: postgres
- `POSTGRES_PASSWORD`: Auto-generated
- `DB_HOST`: PostgreSQL server FQDN
- `DB_PORT`: 5432

### Vector Store Configuration
- `COLLECTION_NAME`: librechat_rag
- `CHUNK_SIZE`: 1500
- `CHUNK_OVERLAP`: 100
- `RAG_UPLOAD_DIR`: /app/uploads
- `PDF_EXTRACT_IMAGES`: false

### Embeddings Configuration
- `EMBEDDINGS_PROVIDER`: azure
- `EMBEDDINGS_MODEL`: text-embedding-3-small
- `RAG_AZURE_OPENAI_API_KEY`: From Azure OpenAI
- `RAG_AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint
- `RAG_AZURE_OPENAI_API_VERSION`: 2024-02-01
- `RAG_CHECK_EMBEDDING_CTX_LENGTH`: true

## API Endpoints

The RAG API exposes the following endpoints:

### Health Check
- `GET /health` - Health check endpoint

### Document Management
- `POST /upload` - Upload documents for processing
- `GET /documents` - List uploaded documents
- `DELETE /documents/{file_id}` - Delete a document

### Search and Retrieval
- `POST /query` - Query the vector database
- `GET /collections` - List collections

### Debug Endpoints (when DEBUG_RAG_API=true)
- `GET /debug/collections` - Debug collection information
- `GET /debug/vectors` - Debug vector information

## Integration with LibreChat

The RAG API is designed to integrate with LibreChat for enhanced chat capabilities. The integration allows:

1. **Document Upload**: Users can upload documents through LibreChat
2. **Context Retrieval**: The RAG API retrieves relevant context from uploaded documents
3. **Enhanced Responses**: LibreChat can use the retrieved context to provide more accurate responses

## Deployment

The RAG API is deployed as part of the Terraform infrastructure in the LibreChat module. The deployment includes:

1. **PostgreSQL Flexible Server** with pgvector extension
2. **Container App Environment** for isolation
3. **Container App** running the RAG API service

### Prerequisites
- Azure subscription
- Azure OpenAI service configured
- Terraform installed and configured

### Deployment Steps
1. Configure the Terraform variables
2. Run `terraform plan` to review the deployment
3. Run `terraform apply` to deploy the infrastructure

## Monitoring and Logging

The RAG API includes comprehensive logging and monitoring:

- **Application Logs**: Available through Azure Container Apps
- **Health Checks**: Automatic health monitoring
- **Debug Mode**: Enhanced logging in development environment

## Troubleshooting

### Common Issues

1. **PostgreSQL Connection Issues**
   - Verify firewall rules allow Azure services
   - Check connection string format
   - Ensure pgvector extension is installed

2. **Embeddings Issues**
   - Verify Azure OpenAI API key and endpoint
   - Check API version compatibility
   - Monitor rate limits

3. **Container App Issues**
   - Check container logs in Azure Portal
   - Verify environment variables
   - Monitor resource usage

### Debug Mode

Enable debug mode by setting `DEBUG_RAG_API=true` to get detailed logging and additional debug endpoints.

## Security Considerations

- All sensitive data is stored in Azure Key Vault
- Network access is controlled through Azure networking
- Container Apps run in isolated environments
- Database access is restricted to necessary services

## Scaling

The RAG API is configured for automatic scaling:
- **Min Replicas**: 1
- **Max Replicas**: 3
- **CPU**: 1 core
- **Memory**: 2GB

Scaling can be adjusted through the Terraform configuration. 