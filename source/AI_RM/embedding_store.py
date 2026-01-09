"""
Embedding Store Module
======================
ChromaDB wrapper for storing and querying video embeddings.

Reference:
    "We employ sentence-transformers to encode demonstrations into vectors 
    and store them using ChromaDB, which facilitates ANN search during runtime."
    — Recommender AI Agent: Integrating LLMs for Interactive Recommendations
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import chromadb
import numpy as np
import pandas as pd

from .config import VectorStoreConfig, default_config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a similarity search."""
    video_id: str
    score: float  # Similarity score (0-1, higher is better)
    text: str
    metadata: Dict[str, Any]


@dataclass 
class StoreStats:
    """Statistics about the embedding store."""
    total_documents: int
    collection_name: str
    persist_path: str
    distance_metric: str


class EmbeddingStore:
    """
    ChromaDB-based vector store for video embeddings.
    
    Provides methods to add, query, and manage video embeddings with
    associated metadata. Supports filtering by metadata fields.
    
    Args:
        config: Vector store configuration. Uses default if not provided.
        
    Example:
        >>> store = EmbeddingStore()
        >>> store.add_videos(videos, embeddings)
        >>> results = store.search("machine learning", n_results=5)
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or default_config.vector_store
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        
    @property
    def client(self) -> chromadb.PersistentClient:
        """Lazy load ChromaDB client."""
        if self._client is None:
            self.config.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.config.persist_path)
            )
            logger.info(f"ChromaDB client initialized: {self.config.persist_path}")
        return self._client
    
    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the video collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            logger.info(f"Collection '{self.config.collection_name}' ready. "
                       f"Documents: {self._collection.count()}")
        return self._collection
    
    def get_stats(self) -> StoreStats:
        """Get store statistics."""
        return StoreStats(
            total_documents=self.collection.count(),
            collection_name=self.config.collection_name,
            persist_path=str(self.config.persist_path),
            distance_metric=self.config.distance_metric
        )
    
    def add_videos(
        self,
        videos: List[dict],
        embeddings: np.ndarray,
        texts: Optional[List[str]] = None
    ) -> int:
        """
        Add videos to the store.
        
        Args:
            videos: List of video metadata dictionaries.
            embeddings: Numpy array of embeddings, shape (n_videos, embedding_dim).
            texts: Optional list of texts used for embeddings.
            
        Returns:
            Number of videos added.
        """
        if len(videos) != len(embeddings):
            raise ValueError(f"Mismatch: {len(videos)} videos, {len(embeddings)} embeddings")
        
        ids = []
        metadatas = []
        documents = texts if texts else [""] * len(videos)
        
        for video in videos:
            video_id = video.get('video_id', '')
            if not video_id:
                logger.warning(f"Skipping video without ID: {video.get('title', 'unknown')}")
                continue
                
            ids.append(video_id)
            
            # Extract metadata fields
            metadata = {}
            for field in self.config.metadata_fields:
                value = video.get(field, "")
                # ChromaDB requires string, int, float, or bool
                if value is not None:
                    metadata[field] = str(value) if not isinstance(value, (int, float, bool)) else value
            metadatas.append(metadata)
        
        if ids:
            # Convert embeddings to list of lists if it's a list of arrays
            if isinstance(embeddings, list):
                embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
            else:
                embeddings_list = embeddings.tolist()
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} videos to store")
            
        return len(ids)
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar videos.
        
        Args:
            query_embedding: Query vector, shape (embedding_dim,).
            n_results: Number of results to return.
            where: Metadata filter (e.g., {"category": "Education"}).
            where_document: Document content filter.
            
        Returns:
            List of SearchResult objects, sorted by similarity.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        for i in range(len(results['ids'][0])):
            # Convert distance to similarity (for cosine: similarity = 1 - distance)
            distance = results['distances'][0][i]
            similarity = 1 - distance if self.config.distance_metric == "cosine" else 1 / (1 + distance)
            
            search_results.append(SearchResult(
                video_id=results['ids'][0][i],
                score=similarity,
                text=results['documents'][0][i] if results['documents'] else "",
                metadata=results['metadatas'][0][i] if results['metadatas'] else {}
            ))
            
        return search_results
    
    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get a single video by ID."""
        result = self.collection.get(
            ids=[video_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if not result['ids']:
            return None
            
        embedding = None
        if result.get('embeddings') is not None and len(result['embeddings']) > 0 and result['embeddings'][0] is not None:
            embedding = np.array(result['embeddings'][0])
        return {
            'video_id': result['ids'][0],
            'text': result['documents'][0] if result['documents'] else "",
            'metadata': result['metadatas'][0] if result['metadatas'] else {},
            'embedding': embedding
        }
    
    def delete_videos(self, video_ids: List[str]) -> int:
        """Delete videos by ID."""
        self.collection.delete(ids=video_ids)
        logger.info(f"Deleted {len(video_ids)} videos")
        return len(video_ids)
    
    def clear(self) -> None:
        """Delete all videos from the collection."""
        self.client.delete_collection(self.config.collection_name)
        self._collection = None
        logger.info(f"Cleared collection '{self.config.collection_name}'")
    
    def exists(self, video_id: str) -> bool:
        """Check if a video exists in the store."""
        result = self.collection.get(ids=[video_id])
        return len(result['ids']) > 0
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def to_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Export store contents to a Pandas DataFrame.
        
        Args:
            limit: Maximum number of records to export. None for all.
            
        Returns:
            DataFrame with video_id, text, and metadata columns.
        """
        # Get all data
        result = self.collection.get(
            include=["documents", "metadatas"],
            limit=limit
        )
        
        if not result['ids']:
            return pd.DataFrame()
        
        # Build DataFrame
        data = []
        for i, video_id in enumerate(result['ids']):
            row = {'video_id': video_id}
            row['text'] = result['documents'][i] if result['documents'] else ""
            
            if result['metadatas']:
                row.update(result['metadatas'][i])
            data.append(row)
            
        return pd.DataFrame(data)
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get count of videos per category."""
        df = self.to_dataframe()
        if df.empty or 'category' not in df.columns:
            return {}
        return df['category'].value_counts().to_dict()
    
    def get_channel_distribution(self) -> Dict[str, int]:
        """Get count of videos per channel."""
        df = self.to_dataframe()
        if df.empty or 'channel_name' not in df.columns:
            return {}
        return df['channel_name'].value_counts().to_dict()
    
    def sample(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get a random sample of videos."""
        result = self.collection.get(include=["documents", "metadatas"])
        if not result['ids']:
            return []
        
        indices = random.sample(range(len(result['ids'])), min(n, len(result['ids'])))
        return [
            {
                'video_id': result['ids'][i],
                'text': (result['documents'][i][:100] + "...") if result['documents'] else "",
                'metadata': result['metadatas'][i] if result['metadatas'] else {}
            }
            for i in indices
        ]
