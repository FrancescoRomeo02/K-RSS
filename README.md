# K-RSS: Knowledge-aware YouTube RSS Recommendation System

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](docker-compose.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-green?logo=python)](requirements/)

A recommendation system for YouTube RSS feeds that addresses the **cold start problem** and **dynamic personalization** through a **human-in-the-loop** approach.

## 🎯 Project Overview

Unlike systems based on traditional collaborative filters, K-RSS uses a **knowledge-aware architecture**:

1. **Representation**: LLM-based embeddings for deep semantic capture of titles and entity linking (via DBpedia/Wikidata) to enrich the information context

2. **Dynamics**: Relevance feedback mechanism through positive/negative votes, dynamically updating the user profile in the latent space (vector shifting)

3. **Explainability**: Interactive Streamlit interface for manipulating critical parameters (exploration vs. exploitation), making the recommendation process transparent (XAI)

## 📁 Project Structure

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
│   └── AI_RM/                 # AI Recommendation Module
├── data/                       # Data directory (mounted as volume)
│   ├── channels/              # Channel lists (CSV input)
│   ├── raw/                   # Raw scraped data
│   ├── processed/             # Processed data for ML
│   ├── embeddings/            # Vector embeddings
│   └── users/                 # User profiles
├── models/                     # Trained models (mounted as volume)
└── references/                 # Research papers
```

## 🚀 Quick Start

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

## 🐳 Docker Services

| Service             | Description                 | Port |
| ------------------- | --------------------------- | ---- |
| `scraper`           | YouTube RSS Scraper         | -    |
| `scrape-csv`        | Batch scraping from CSV     | -    |
| `ai-recommendation` | ML/DL recommendation engine | -    |
| `webapp`            | Streamlit web interface     | 8501 |
| `jupyter`           | Jupyter Lab for development | 8888 |
| `pipeline`          | Full pipeline runner        | -    |

### Common Commands

```bash
# Build specific service
docker-compose build scraper

# Run scraper with custom options
docker-compose run --rm scraper python scraper.py --channel CHANNEL_ID --output /app/data/raw/output.json

# Start web app in background
docker-compose up -d webapp

# View logs
docker-compose logs -f webapp

# Stop all services
docker-compose down

# Clean up (remove volumes)
docker-compose down -v
```

## 📊 Data Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  YouTube    │───▶│   Scraper   │───▶│  Embedding  │───▶│  User       │
│  RSS Feeds  │    │  (XML)      │    │  (LLM)      │    │  Feedback   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                   │                   │
                         ▼                   ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │  data/raw/  │    │  data/      │    │  data/      │
                   │  videos.json│    │  embeddings/│    │  users/     │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

## 🔧 Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements/all.txt

# Run scraper
cd source/XML_Scarper
python scraper.py --csv ../../data/channels/channels.csv --output ../../data/raw/videos.json
```

## 📚 References

- Wu, L., et al. (2023). *Recommender Systems in the Era of Large Language Models (LLMs): A Survey*
- Wang, X., et al. (2023). *Large Language Models for Interactive Recommendation*
- Wang, H., et al. (2018). *DKN: Deep Knowledge-Aware Network for News Recommendation*

## 👥 Team

- Francesco Romeo (885880)
- Matteo Picozzi (890228)

## 📄 License

This project is for academic purposes.
