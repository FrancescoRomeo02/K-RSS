"""
Embedder Module
===============
Generates embeddings for video content using sentence-transformers.

Reference:
    "We employ sentence-transformers to encode demonstrations into vectors..."
    — Recommender AI Agent: Integrating LLMs for Interactive Recommendations
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig, default_config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    video_id: str
    embedding: np.ndarray
    text: str
    success: bool
    error_message: Optional[str] = None


class Embedder:
    """
    Generates embeddings for video content.
    
    Uses sentence-transformers models to encode text into dense vectors
    suitable for similarity search.
    
    Args:
        config: Embedding configuration. Uses default if not provided.
        
    Example:
        >>> embedder = Embedder()
        >>> embedding = embedder.encode("Machine learning tutorial")
        >>> print(embedding.shape)  # (384,)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or default_config.embedding
        self._model: Optional[SentenceTransformer] = None
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
            logger.info(f"Model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode.
            show_progress: Show progress bar for batch encoding.
            
        Returns:
            Numpy array of shape (embedding_dim,) for single text,
            or (n_texts, embedding_dim) for multiple texts.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )
        
        if single_input:
            return embeddings[0]
        return embeddings
    
    def prepare_video_text(self, video: dict) -> str:
        """
        Prepare video metadata for embedding.
        
        Combines title and description into a single text suitable
        for embedding generation.
        
        Args:
            video: Video metadata dictionary with 'title' and 'description'.
            
        Returns:
            Combined text for embedding.
        """
        title = video.get('title', '')
        description = video.get('description', '')
        
        # Truncate description to avoid exceeding model limits
        max_desc_len = self.config.max_seq_length * 4  # Approximate chars
        if len(description) > max_desc_len:
            description = description[:max_desc_len] + "..."
            
        return f"{title}. {description}".strip()
    
    def embed_videos(
        self, 
        videos: List[dict],
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of videos.
        
        Args:
            videos: List of video metadata dictionaries.
            show_progress: Show progress bar.
            
        Returns:
            List of EmbeddingResult objects.
        """
        results = []
        texts = []
        valid_videos = []
        
        # Prepare texts
        for video in videos:
            try:
                text = self.prepare_video_text(video)
                texts.append(text)
                valid_videos.append(video)
            except Exception as e:
                results.append(EmbeddingResult(
                    video_id=video.get('video_id', 'unknown'),
                    embedding=np.array([]),
                    text='',
                    success=False,
                    error_message=str(e)
                ))
        
        # Batch encode
        if texts:
            embeddings = self.encode(texts, show_progress=show_progress)
            
            for video, text, embedding in zip(valid_videos, texts, embeddings):
                results.append(EmbeddingResult(
                    video_id=video.get('video_id', 'unknown'),
                    embedding=embedding,
                    text=text,
                    success=True
                ))
        
        return results
