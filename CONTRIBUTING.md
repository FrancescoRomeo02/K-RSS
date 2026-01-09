# K-RSS Team Guidelines

## 👥 Team

- **Francesco Romeo** (885880)
- **Matteo Picozzi** (890228)

---

## 📐 Architettura del Sistema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           K-RSS Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   YouTube    │───▶│   Scraper    │───▶│   AI_RM      │───▶│  WebApp   │ │
│  │   RSS Feed   │    │  (XML_Scarper)│    │ (Embeddings) │    │(Streamlit)│ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                              │                   │                   │      │
│                              ▼                   ▼                   ▼      │
│                      ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│                      │  data/raw/   │    │ data/embed/  │    │data/users/│ │
│                      │  videos.json │    │  vectors.npy │    │profiles/  │ │
│                      └──────────────┘    └──────────────┘    └───────────┘ │
│                                                 │                          │
│                                                 ▼                          │
│                                         ┌──────────────┐                   │
│                                         │ Knowledge    │                   │
│                                         │ Graph (DBpedia)                  │
│                                         └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Struttura del Progetto

```
K-RSS/
├── source/
│   ├── XML_Scarper/          # Francesco - Scraping YouTube RSS
│   ├── AI_RM/                # [DA ASSEGNARE] - Recommendation Module  
│   └── webapp/               # [DA ASSEGNARE] - Streamlit Interface
├── data/
│   ├── channels/             # Input: lista canali CSV
│   ├── raw/                  # Output scraper: JSON video
│   ├── embeddings/           # Output AI_RM: vettori embeddings
│   ├── processed/            # Dati processati per training
│   └── users/                # Profili utente (vector shifting)
├── models/                   # Modelli trainati e cache HuggingFace
├── docker/                   # Dockerfile per ogni servizio
└── requirements/             # Dipendenze modulari
```

---

## 🔧 Convenzioni di Codice

### 1. Dataclasses per Strutture Dati

Usare sempre `@dataclass` con type hints per i dati strutturati:

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class VideoEmbedding:
    """Embedding vettoriale per un video."""
    video_id: str
    title_embedding: List[float]
    description_embedding: Optional[List[float]] = None
    entity_embedding: Optional[List[float]] = None
    
    # Campi calcolati in __post_init__
    combined_embedding: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        # Logica di combinazione embeddings
        pass
```

### 2. Result Objects per Error Handling

Wrappare le operazioni in oggetti risultato:

```python
@dataclass
class EmbeddingResult:
    video_id: str
    embedding: Optional[List[float]]
    success: bool
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
```

### 3. Logging Consistente

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

### 4. Docstrings

Usare Google-style docstrings:

```python
def compute_similarity(embedding_a: List[float], embedding_b: List[float]) -> float:
    """
    Calcola la similarità coseno tra due embeddings.
    
    Args:
        embedding_a: Primo vettore embedding
        embedding_b: Secondo vettore embedding
        
    Returns:
        Similarità coseno normalizzata [0, 1]
        
    Raises:
        ValueError: Se le dimensioni non corrispondono
    """
    pass
```

---

## 📦 Schema Dati JSON

### Video Scraped (`data/raw/scraped_videos.json`)

```json
{
    "metadata": {
        "scraped_at": "2024-01-01T12:00:00+00:00",
        "total_videos": 135,
        "enriched_via_api": true
    },
    "videos": [
        {
            "video_id": "abc123",
            "title": "Video Title",
            "description": "...",
            "channel_id": "UC...",
            "tags": ["tag1", "tag2"],
            "category_name": "Education",
            "duration_seconds": 600
        }
    ],
    "indices": {
        "video_by_id": {"abc123": 0},
        "videos_by_channel": {"UC...": [0, 1, 2]}
    }
}
```

### User Profile (`data/users/{user_id}.json`)

```json
{
    "user_id": "user_001",
    "created_at": "2024-01-01T12:00:00+00:00",
    "profile_vector": [0.1, 0.2, ...],
    "feedback_history": [
        {
            "video_id": "abc123",
            "feedback": "positive",
            "timestamp": "2024-01-01T12:30:00+00:00"
        }
    ],
    "preferences": {
        "exploration_weight": 0.5,
        "category_preferences": {"Education": 0.8}
    }
}
```

---

## 🔬 Componenti AI_RM

### 1. Embedding Module

Responsabilità:
- Generare embeddings con `sentence-transformers`
- Modello suggerito: `all-MiniLM-L6-v2` (veloce) o `all-mpnet-base-v2` (accurato)
- Cache embeddings per evitare ricalcoli

### 2. Entity Linking Module

Responsabilità:
- Estrarre entità dai titoli video
- Query SPARQL a DBpedia/Wikidata
- Arricchire embeddings con knowledge graph

```python
# Esempio query SPARQL
SPARQL_QUERY = """
SELECT ?entity ?label ?abstract WHERE {
    ?entity rdfs:label ?label .
    ?entity dbo:abstract ?abstract .
    FILTER(LANG(?label) = "en")
    FILTER(CONTAINS(LCASE(?label), LCASE("{entity_name}")))
}
LIMIT 5
"""
```

### 3. Recommendation Engine

Algoritmo base:

```
1. Input: user_profile_vector, candidate_videos
2. Per ogni video:
   a. Calcola similarity = cosine(user_vector, video_embedding)
   b. Applica exploration bonus = random * exploration_weight
   c. Score finale = similarity * (1 - exploration_weight) + bonus
3. Rank per score decrescente
4. Applica diversity filter (penalizza stesso canale)
5. Return top-K recommendations
```

### 4. User Profile Manager

Vector Shifting per feedback:

```python
def update_profile(user_vector: np.array, video_embedding: np.array, 
                   feedback: str, learning_rate: float = 0.1) -> np.array:
    """
    Aggiorna il profilo utente basandosi sul feedback.
    
    - Feedback positivo: avvicina il vettore all'embedding
    - Feedback negativo: allontana il vettore dall'embedding
    """
    direction = 1 if feedback == "positive" else -1
    delta = direction * learning_rate * (video_embedding - user_vector)
    return user_vector + delta
```

---

## 🐳 Comandi Docker Principali

```bash
# Build tutto
docker-compose build

# Scraping (workflow principale)
docker-compose run --rm scrape-csv

# Con enrichment API (serve YOUTUBE_API_KEY)
docker-compose run --rm scrape-enrich

# Avvia webapp
docker-compose up webapp
# → http://localhost:8501

# Jupyter per esperimenti
docker-compose up jupyter
# → http://localhost:8888

# Logs
docker-compose logs -f webapp
```

---

## 📋 Task Assignment

| Modulo               | Owner     | Status        |
| -------------------- | --------- | ------------- |
| XML_Scarper          | Francesco | ✅ Completato  |
| AI_RM/embeddings     | TBD       | 🔲 Da iniziare |
| AI_RM/entity_linking | TBD       | 🔲 Da iniziare |
| AI_RM/recommender    | TBD       | 🔲 Da iniziare |
| webapp/core          | TBD       | 🟡 Placeholder |
| webapp/xai           | TBD       | 🔲 Da iniziare |

---

## 🔀 Git Workflow

1. **Branch naming**: `feature/{module}-{description}`, es: `feature/ai_rm-embeddings`
2. **Commit messages**: prefissi `feat:`, `fix:`, `docs:`, `refactor:`
3. **Pull Requests**: review obbligatorio prima di merge in `main`

---

## 📚 Riferimenti Teorici

1. **Wu et al. (2023)** - *Recommender Systems in the Era of LLMs*
   - Embeddings post-BERT per recommendation

2. **Wang et al. (2023)** - *LLMs for Interactive Recommendation*
   - Human-in-the-loop feedback mechanisms

3. **Wang et al. (2018)** - *DKN: Deep Knowledge-Aware Network*
   - Knowledge Graph integration per news recommendation
