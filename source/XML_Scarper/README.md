# YouTube RSS/XML Scraper for K-RSS

A comprehensive YouTube RSS feed scraper designed for building recommendation systems.

## Features

- **CSV Input**: Read channel list from CSV files with flexible column detection
- **Multiple Input Formats**: Supports channel IDs, URLs, and @handles
- **Rich Metadata Extraction**: Extracts video titles, descriptions, thumbnails, view counts, and more
- **JSON Output**: Structured JSON output optimized for recommendation systems
- **Robust Error Handling**: Retry logic and graceful error handling
- **Rate Limiting**: Configurable delays to respect server limits
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Quick Start with Docker 🐳

### Prerequisites
- Docker
- Docker Compose

### Build the image

```bash
docker-compose build
```

### Scrape channels from CSV

```bash
# Use the pre-configured service
docker-compose run --rm scrape-csv

# Or run with custom options
docker-compose run --rm scraper --csv /app/data/channels.csv --output /app/output/videos.json
```

### Scrape a single channel

```bash
# Using environment variable
CHANNEL=@Fireship docker-compose run --rm scrape-channel

# Or directly
docker-compose run --rm scraper --channel @3blue1brown --output /app/output/channel.json
```

### View help

```bash
docker-compose run --rm scraper --help
```

### Output files

I file JSON vengono salvati nella cartella `output/` che è montata come volume Docker.

---

## Installation (senza Docker)

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Scrape channels from CSV file

```bash
python scraper.py --csv data/channels.csv --output data/videos.json
```

### Scrape a single channel

```bash
python scraper.py --channel @3blue1brown --output data/single_channel.json
```

### Create a sample CSV file

```bash
python scraper.py --create-sample data/sample_channels.csv
```

### Command-line Options

| Option            | Description                        | Default                    |
| ----------------- | ---------------------------------- | -------------------------- |
| `--csv`           | Path to CSV file with channel list | -                          |
| `--channel`       | Single channel to scrape           | -                          |
| `--output, -o`    | Output JSON file path              | `data/scraped_videos.json` |
| `--column`        | CSV column name for channel IDs    | `channel`                  |
| `--delay`         | Delay between requests (seconds)   | `1.0`                      |
| `--create-sample` | Create a sample CSV file           | -                          |

## CSV Format

The input CSV should have at least one column with channel identifiers. The scraper auto-detects common column names:

```csv
channel,category
@3blue1brown,education
@Fireship,tech
UCxxxxxxxxxxxxxxxxxxxxxx,science
```

Supported channel formats:
- YouTube handles: `@channelname`
- Channel IDs: `UCxxxxxxxxxxxxxxxxxxxxxx`
- Channel URLs: `https://www.youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxxxx`
- Channel handle URLs: `https://www.youtube.com/@channelname`

## Output JSON Structure

The output JSON is optimized for recommendation systems:

```json
{
  "metadata": {
    "scraped_at": "2026-01-09T12:00:00",
    "total_channels": 10,
    "total_videos": 150,
    "failed_channels": 0,
    "source_file": "channels.csv"
  },
  "channels": [
    {
      "channel_id": "UCxxxxxx",
      "channel_name": "Channel Name",
      "channel_url": "https://youtube.com/channel/UCxxxxxx",
      "feed_url": "https://youtube.com/feeds/videos.xml?channel_id=UCxxxxxx",
      "last_updated": "2026-01-09T10:00:00",
      "video_count": 15,
      "scraped_at": "2026-01-09T12:00:00"
    }
  ],
  "videos": [
    {
      "video_id": "xxxxxxxxxxx",
      "channel_id": "UCxxxxxx",
      "channel_name": "Channel Name",
      "title": "Video Title",
      "description": "Video description...",
      "published_date": "2026-01-08T15:00:00",
      "updated_date": "2026-01-08T15:00:00",
      "thumbnail_url": "https://i.ytimg.com/vi/xxxxxxxxxxx/hqdefault.jpg",
      "video_url": "https://www.youtube.com/watch?v=xxxxxxxxxxx",
      "view_count": 100000,
      "title_length": 25,
      "description_length": 500,
      "has_description": true,
      "entities": [],
      "categories": [],
      "tags": [],
      "scraped_at": "2026-01-09T12:00:00",
      "feed_source": "youtube_rss"
    }
  ],
  "failed": [],
  "indices": {
    "video_by_id": {"xxxxxxxxxxx": 0},
    "videos_by_channel": {"UCxxxxxx": [0, 1, 2]}
  }
}
```

## Fields for Recommendation System

### Video Metadata
- `video_id`: Unique identifier for deduplication
- `title`, `description`: Text for embedding and semantic analysis
- `title_length`, `description_length`: Feature engineering
- `published_date`: For freshness/recency features
- `view_count`: Popularity signal
- `entities`, `categories`, `tags`: Placeholders for knowledge graph enrichment

### Indices
- `video_by_id`: Quick lookup by video ID
- `videos_by_channel`: Group videos by channel for channel-based recommendations

## Integration with K-RSS

This scraper is designed to work with the K-RSS recommendation system:

1. **Embedding**: Use `title` and `description` for LLM-based embeddings
2. **Entity Linking**: Populate `entities` field with DBpedia/Wikidata entities
3. **User Feedback**: Track user votes to update recommendations
4. **Freshness**: Use `published_date` for time-aware recommendations

## Example Python Usage

```python
from scraper import YouTubeRSSScraper

# Initialize scraper
scraper = YouTubeRSSScraper(request_delay=1.0)

# Scrape from CSV
result = scraper.scrape_channels_from_csv(
    csv_path='data/channels.csv',
    output_path='data/videos.json'
)

print(f"Scraped {result['metadata']['total_videos']} videos")

# Scrape single channel
result = scraper.scrape_channel('@3blue1brown')
for video in result.videos:
    print(f"- {video.title}")
```

## Notes

- YouTube RSS feeds typically contain the 15 most recent videos per channel
- Rate limiting is important to avoid being blocked
- The scraper uses public RSS feeds and doesn't require API keys
