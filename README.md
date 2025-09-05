# RAG API Enhanced 🧠

Enhanced RAG API with Mem0 integration and DragonflyDB caching for three-tier hybrid memory architecture.

## Features

- **Three-tier Memory Architecture**:
  - Hot Tier: DragonflyDB cache (sub-millisecond access)
  - Warm Tier: Mem0 personal memory (contextual retrieval)
  - Cold Tier: RAG document search (comprehensive knowledge)

- **New Endpoints**:
  - `/memory/add` - Store personal memories
  - `/memory/search` - Search personal memories
  - `/search/hybrid` - Hybrid search across all tiers
  - `/memory/status` - System health monitoring

## Configuration

Required environment variables:
- `MEM0_API_KEY` - Mem0 API key
- `MEM0_USER_ID` - User identifier for Mem0
- `DRAGONFLY_URL` - DragonflyDB connection URL

## Deployment

Deployed on Railway with automatic environment variable detection and three-tier caching system.