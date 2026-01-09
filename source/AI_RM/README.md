# AI_RM - AI Recommendation Module

Modulo di raccomandazione knowledge-aware per K-RSS.

## Struttura Pianificata

```
AI_RM/
├── __init__.py
├── embeddings/
│   ├── __init__.py
│   ├── text_embedder.py      # Sentence-transformers wrapper
│   └── video_embedder.py     # Combinazione embeddings video
├── entity_linking/
│   ├── __init__.py
│   ├── extractor.py          # Estrazione entità da testo
│   ├── sparql_client.py      # Query DBpedia/Wikidata
│   └── knowledge_graph.py    # Costruzione grafo locale
├── recommender/
│   ├── __init__.py
│   ├── engine.py             # Core recommendation logic
│   ├── scoring.py            # Similarity e ranking
│   └── diversity.py          # Filtri diversità
├── user_profile/
│   ├── __init__.py
│   ├── manager.py            # CRUD profili utente
│   └── vector_shift.py       # Aggiornamento profilo da feedback
└── config.py                 # Configurazione modulo
```

## Dipendenze

Vedi `requirements/ai_rm.txt`:
- torch, transformers, sentence-transformers (embeddings)
- SPARQLWrapper, rdflib (knowledge graph)
- spacy (NLP/entity extraction)
- numpy, pandas, scikit-learn (data processing)

## Quick Start

```python
# Placeholder - da implementare
from AI_RM.embeddings import VideoEmbedder
from AI_RM.recommender import RecommendationEngine
from AI_RM.user_profile import UserProfileManager

# Inizializza componenti
embedder = VideoEmbedder(model_name="all-MiniLM-L6-v2")
engine = RecommendationEngine(embedder)
profiles = UserProfileManager(data_path="data/users/")

# Genera raccomandazioni
user = profiles.get_or_create("user_001")
recommendations = engine.recommend(user, top_k=10)
```

## Riferimenti

- [Sentence-Transformers](https://www.sbert.net/)
- [DBpedia SPARQL](https://wiki.dbpedia.org/online-access/DBpedia-sparql)
- [Wikidata Query Service](https://query.wikidata.org/)
