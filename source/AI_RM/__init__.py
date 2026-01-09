"""
K-RSS AI Recommendation Module
==============================
Knowledge-aware recommendation engine using LLM embeddings and entity linking.

Modules:
- embedder: Generate semantic embeddings for video content
- embedding_store: ChromaDB vector store for embeddings
- entity_linking: Link video entities to DBpedia/Wikidata (TODO)
- recommender: Recommendation engine with user feedback (TODO)
- user_profile: User profile management with vector shifting (TODO)
"""

__version__ = "0.1.0"

from .config import AIRMConfig, default_config
from .embedder import Embedder, EmbeddingResult
from .embedding_store import EmbeddingStore, SearchResult, StoreStats

__all__ = [
    "AIRMConfig",
    "default_config",
    "Embedder",
    "EmbeddingResult", 
    "EmbeddingStore",
    "SearchResult",
    "StoreStats",
]
