"""
AI_RM Configuration
===================
Configuration settings for the AI Recommendation Module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding module."""
    # Model settings
    model_name: str = "all-MiniLM-L6-v2"  # Fast and good quality
    # Alternative: "all-mpnet-base-v2" for higher accuracy
    
    # Embedding dimensions
    embedding_dim: int = 384  # For MiniLM-L6-v2
    
    # Caching
    cache_embeddings: bool = True
    cache_path: Path = Path("data/embeddings/cache")
    
    # Processing
    batch_size: int = 32
    max_seq_length: int = 256
    normalize_embeddings: bool = True


@dataclass
class EntityLinkingConfig:
    """Configuration for entity linking module."""
    # SPARQL endpoints
    dbpedia_endpoint: str = "http://dbpedia.org/sparql"
    wikidata_endpoint: str = "https://query.wikidata.org/sparql"
    
    # NLP settings
    spacy_model: str = "en_core_web_sm"  # or "en_core_web_md" for better NER
    
    # Query settings
    max_entities_per_video: int = 5
    confidence_threshold: float = 0.5
    
    # Rate limiting
    request_delay_seconds: float = 0.5


@dataclass
class RecommenderConfig:
    """Configuration for the recommendation engine."""
    # Scoring weights
    content_weight: float = 0.6
    knowledge_weight: float = 0.2
    recency_weight: float = 0.2
    
    # Exploration vs Exploitation
    default_exploration: float = 0.3
    
    # Diversity settings
    same_channel_penalty: float = 0.5
    min_diversity_score: float = 0.3
    
    # Output
    default_top_k: int = 10
    max_candidates: int = 100


@dataclass
class UserProfileConfig:
    """Configuration for user profile management."""
    # Storage
    profiles_path: Path = Path("data/users")
    
    # Vector shifting
    learning_rate: float = 0.1
    positive_feedback_weight: float = 1.0
    negative_feedback_weight: float = 0.5
    
    # Profile initialization
    cold_start_strategy: str = "random"  # "random", "average", "zero"
    
    # History
    max_feedback_history: int = 1000


@dataclass
class AIRMConfig:
    """Main configuration for the AI Recommendation Module."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    entity_linking: EntityLinkingConfig = field(default_factory=EntityLinkingConfig)
    recommender: RecommenderConfig = field(default_factory=RecommenderConfig)
    user_profile: UserProfileConfig = field(default_factory=UserProfileConfig)
    
    # Paths
    data_path: Path = Path("data")
    models_path: Path = Path("models")
    
    # Environment
    device: str = "cpu"  # "cpu" or "cuda"
    num_workers: int = 4
    
    def __post_init__(self):
        """Ensure paths exist."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.embedding.cache_path.mkdir(parents=True, exist_ok=True)
        self.user_profile.profiles_path.mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = AIRMConfig()
