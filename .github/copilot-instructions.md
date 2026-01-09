# K-RSS Copilot Instructions

## Project Overview

K-RSS is a **knowledge-aware YouTube RSS recommendation system** addressing cold start and dynamic personalization through human-in-the-loop feedback. Academic project by Francesco Romeo & Matteo Picozzi.

### Core Architecture

```
YouTube RSS → Scraper → Embeddings (LLM) → User Feedback (voting) → Recommendations
                            ↓
                    Entity Linking (DBpedia/Wikidata)
```

**Modules:**
- **XML_Scarper** (`source/XML_Scarper/`): YouTube RSS scraping, `VideoMetadata` extraction, API enrichment
- **AI_RM** (`source/AI_RM/`): Embeddings, knowledge graphs, recommendation engine, user profiles
- **webapp** (`source/webapp/`): Streamlit interface for recommendations and XAI parameter tuning

## Development Workflow

### Docker-First (Recommended)

```bash
docker-compose build                    # Build all services
docker-compose run --rm scrape-csv      # Scrape channels
docker-compose run --rm scrape-enrich   # With YouTube API (needs YOUTUBE_API_KEY)
docker-compose up webapp                # Streamlit → http://localhost:8501
docker-compose up jupyter               # Jupyter → http://localhost:8888
```

### Local Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements/all.txt
```

## Code Conventions

### Dataclasses for All Structured Data

```python
@dataclass
class VideoMetadata:
    video_id: str
    title: str
    view_count: Optional[int] = None
    tags: list = field(default_factory=list)
    
    def __post_init__(self):
        self.title_length = len(self.title)
```

### Result Objects for Error Handling

```python
@dataclass
class ScrapingResult:
    channel: ChannelMetadata
    videos: list
    success: bool
    error_message: Optional[str] = None
```

### XML Namespaces (YouTube RSS)

```python
NAMESPACES = {
    'atom': 'http://www.w3.org/2005/Atom',
    'yt': 'http://www.youtube.com/xml/schemas/2015',
    'media': 'http://search.yahoo.com/mrss/'
}
```

## Key Files & Data Flow

| Path | Purpose |
|------|---------|
| `data/channels/channels.csv` | Input channel list (category-tagged) |
| `data/raw/scraped_videos.json` | Scraped metadata + indices |
| `data/embeddings/` | LLM vector embeddings |
| `data/users/` | User profiles (vector shifting) |
| `source/AI_RM/config.py` | AI module configuration |

## Environment Variables

```bash
YOUTUBE_API_KEY       # Video enrichment (duration, tags)
TRANSFORMERS_CACHE    # HuggingFace cache (default: /app/models/transformers)
HF_HOME               # HuggingFace home (default: /app/models/huggingface)
```

## Team Guidelines

See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Task assignment table
- Git workflow
- Code patterns and docstring conventions
- AI_RM implementation details
- JSON schema specifications
