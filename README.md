# K-RSS: Knowledge-aware YouTube RSS Recommendation System

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](docker-compose.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-green?logo=python)](requirements/)

A recommendation system for YouTube RSS feeds that addresses the **cold start problem** and **dynamic personalization** through a **human-in-the-loop** approach.

## Project Overview

Unlike systems based on traditional collaborative filters, K-RSS uses a **knowledge-aware architecture**:

1. **Representation**: LLM-based embeddings for deep semantic capture of titles and entity linking (via DBpedia/Wikidata) to enrich the information context

2. **Dynamics**: Relevance feedback mechanism through positive/negative votes, dynamically updating the user profile in the latent space (vector shifting)

3. **Explainability**: Interactive Streamlit interface for manipulating critical parameters (exploration vs. exploitation), making the recommendation process transparent (XAI)

## Project Structure

```
K-RSS/
├── docker-compose.yml          # Multi-service Docker configuration
├── docker/                     # Dockerfiles for each service
│   ├── scraper.Dockerfile
│   ├── ai_rm.Dockerfile
│   ├── webapp.Dockerfile
│   ├── jupyter.Dockerfile
│   └── pipeline.Dockerfile
├── requirements/               # Python dependencies (modular)
│   ├── scraper.txt
│   ├── ai_rm.txt
│   ├── webapp.txt
│   ├── dev.txt
│   └── all.txt
├── source/                     # Source code
│   ├── XML_Scarper/           # YouTube RSS scraper
│   ├── AI_RM/                 # AI Recommendation Module
│   └── webapp/                # Streamlit web interface
│       ├── app.py             # Main application
│       └── pages/             # Multi-page Streamlit app
│           ├── 1_Analytics.py
│           └── 2_XAI_Explorer.py
├── data/                       # Data directory (mounted as volume)
│   ├── channels/              # Channel lists (CSV input)
│   ├── raw/                   # Raw scraped data (scraped_videos.json)
│   ├── processed/             # Processed data for ML
│   ├── embeddings/            # Vector embeddings
│   └── users/                 # User profiles
├── models/                     # Trained models and HuggingFace cache
└── references/                 # Research papers
```

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### 1. Build all services

```bash
docker-compose build
```

### 2. Scrape YouTube channels

```bash
# Edit channel list
nano data/channels/channels.csv

# Run scraper
docker-compose run --rm scrape-csv
```

### 3. Start the web interface

```bash
docker-compose up webapp
# Open http://localhost:8501
```

### 4. Development with Jupyter

```bash
docker-compose up jupyter
# Open http://localhost:8888
```

## Docker Services

| Service             | Description                        | Port |
| ------------------- | ---------------------------------- | ---- |
| `scraper`           | YouTube RSS Scraper (help)         | -    |
| `scrape-csv`        | Batch scraping from CSV (RSS only) | -    |
| `scrape-enrich`     | Scraping + YouTube API enrichment  | -    |
| `ai-recommendation` | ML/DL recommendation engine        | -    |
| `webapp`            | Streamlit web interface            | 8501 |
| `jupyter`           | Jupyter Lab for development        | 8888 |
| `pipeline`          | Full pipeline runner               | -    |

### Common Commands

```bash
# Build all services
docker-compose build

# Run RSS scraping only
docker-compose run --rm scrape-csv

# Run scraping with YouTube API enrichment (requires YOUTUBE_API_KEY)
YOUTUBE_API_KEY=your_key docker-compose run --rm scrape-enrich

# Start web app
docker-compose up webapp
# Open http://localhost:8501

# Start Jupyter Lab
docker-compose up jupyter
# Open http://localhost:8888

# Run scraper with custom options
docker-compose run --rm scraper python scraper.py --channel @3blue1brown --output /app/data/raw/output.json

# View logs
docker-compose logs -f webapp

# Stop all services
docker-compose down

# Clean up (remove volumes)
docker-compose down -v
```

## Embedding pipeline (Docker)

Run the embedding pipeline in Docker; the service `embedder` mounts the repository `data/` folder and reads
`/data/raw/scraped_videos.json` by default.

Build and run the embedder image:

```bash
# Build image
docker build -t krss-embedder -f docker/Dockerfile.embedder .

# Run once (mount local data/ to /data in container)
docker run --rm -v "$(pwd)/data":/data krss-embedder

# Or with docker-compose
docker compose up --build embedder
```

CLI options (local runs):

```bash
# Dry run (no write)
python -m source.AI_RM.embed_pipeline --input data/raw/scraped_videos.json --dry-run

# Default behavior: skips videos already present (resume)
python -m source.AI_RM.embed_pipeline --input data/raw/scraped_videos.json

# Force update (recompute embeddings for videos already in store)
python -m source.AI_RM.embed_pipeline --input data/raw/scraped_videos.json --update

# Disable resume (process all, will attempt to add existing ids and may fail unless --update used)
python -m source.AI_RM.embed_pipeline --input data/raw/scraped_videos.json --no-resume
```


## Data Pipeline

```
                                    K-RSS Pipeline
                                    
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  channels.csv   │────▶│   XML_Scarper   │────▶│ scraped_videos  │
│  (input)        │     │   (RSS Feed)    │     │    .json        │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                       │
                        optional │                       │
                                 ▼                       │
                        ┌─────────────────┐              │
                        │  YouTube API    │              │
                        │  (enrichment)   │              │
                        └─────────────────┘              │
                                                         │
         ┌───────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     AI_RM       │────▶│   Embeddings    │────▶│  Streamlit      │
│  (processing)   │     │   (vectors)     │     │  Web App        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  User Feedback  │
                                                │  (profiles)     │
                                                └─────────────────┘
```

### Pipeline Steps

1. **Scraping** (`scrape-csv` or `scrape-enrich`)
   - Input: `data/channels/channels.csv`
   - Output: `data/raw/scraped_videos.json`

2. **Embedding Generation** (`ai-recommendation`) - *In Development*
   - Input: `data/raw/scraped_videos.json`
   - Output: `data/embeddings/`

3. **Web Interface** (`webapp`)
   - Displays recommendations
   - Collects user feedback
   - XAI parameter exploration

## Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements/all.txt

# Run scraper (RSS only)
cd source/XML_Scarper
python scraper.py --csv ../../data/channels/channels.csv --output ../../data/raw/scraped_videos.json

# Run scraper with API enrichment
YOUTUBE_API_KEY=your_key python scraper.py --csv ../../data/channels/channels.csv --output ../../data/raw/scraped_videos.json --enrich

# Start web app
cd source/webapp
streamlit run app.py
```

## Environment Variables

| Variable             | Description                              | Required |
| -------------------- | ---------------------------------------- | -------- |
| `YOUTUBE_API_KEY`    | YouTube Data API key for enrichment      | Optional |
| `TRANSFORMERS_CACHE` | HuggingFace transformers cache directory | Optional |
| `HF_HOME`            | HuggingFace home directory               | Optional |

## References

- Wu, L., et al. (2023). *Recommender Systems in the Era of Large Language Models (LLMs): A Survey*
- Wang, X., et al. (2023). *Large Language Models for Interactive Recommendation*
- Wang, H., et al. (2018). *DKN: Deep Knowledge-Aware Network for News Recommendation*

## Team

- Francesco Romeo (885880)
- Matteo Picozzi (890228)

## License

This project is for academic purposes.
