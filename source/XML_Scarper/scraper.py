"""
YouTube RSS/XML Scraper for K-RSS Recommendation System
========================================================
This module scrapes YouTube channel RSS feeds and extracts video metadata
useful for building a knowledge-aware recommendation system.

Features:
- Reads channel list from CSV file
- Fetches RSS/XML feeds from YouTube channels
- Extracts rich metadata for recommendation system
- Saves structured JSON output
- Handles errors gracefully with retry logic
"""

import csv
import json
import os
import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field
import xml.etree.ElementTree as ET

import requests  # type: ignore
from requests.adapters import HTTPAdapter  # type: ignore
from urllib3.util.retry import Retry  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# YouTube RSS feed base URL
YOUTUBE_RSS_BASE_URL = "https://www.youtube.com/feeds/videos.xml"

# YouTube channel ID pattern
CHANNEL_ID_PATTERN = r'UC[a-zA-Z0-9_-]{22}'

# XML namespaces used in YouTube RSS feeds
NAMESPACES = {
    'atom': 'http://www.w3.org/2005/Atom',
    'yt': 'http://www.youtube.com/xml/schemas/2015',
    'media': 'http://search.yahoo.com/mrss/'
}


@dataclass
class VideoMetadata:
    """
    Structured video metadata for recommendation system.
    Contains all relevant fields for content-based and knowledge-aware recommendations.
    """
    # Core identifiers
    video_id: str
    channel_id: str
    channel_name: str
    
    # Content information
    title: str
    description: str
    published_date: str
    updated_date: str
    
    # Media information
    thumbnail_url: str
    video_url: str
    duration_hint: Optional[str] = None
    
    # Engagement metrics (from RSS if available)
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    
    # Video details (from YouTube API)
    duration: Optional[str] = None  # ISO 8601 duration (e.g., "PT15M33S")
    duration_seconds: Optional[int] = None  # Duration in seconds for easier filtering
    definition: Optional[str] = None  # "hd" or "sd"
    caption: Optional[bool] = None  # Has captions
    licensed_content: Optional[bool] = None
    default_language: Optional[str] = None
    default_audio_language: Optional[str] = None
    
    # Category info (from YouTube API)
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    
    # Derived features for recommendation
    title_length: int = 0
    description_length: int = 0
    has_description: bool = False
    
    # Knowledge graph linking fields
    entities: list = field(default_factory=list)
    categories: list = field(default_factory=list)
    tags: list = field(default_factory=list)  # From YouTube API
    topic_categories: list = field(default_factory=list)  # Wikipedia topics from API
    
    # Metadata
    scraped_at: str = ""
    feed_source: str = "youtube_rss"
    enriched_via_api: bool = False
    
    def __post_init__(self):
        """Calculate derived features after initialization."""
        self.title_length = len(self.title) if self.title else 0
        self.description_length = len(self.description) if self.description else 0
        self.has_description = bool(self.description and self.description.strip())
        self.scraped_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ChannelMetadata:
    """
    Channel-level metadata for recommendation system.
    """
    channel_id: str
    channel_name: str
    channel_url: str
    feed_url: str
    last_updated: str
    video_count: int
    scraped_at: str = ""
    
    def __post_init__(self):
        self.scraped_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ScrapingResult:
    """
    Complete scraping result containing channel and video data.
    """
    channel: ChannelMetadata
    videos: list
    success: bool
    error_message: Optional[str] = None


# YouTube category ID to name mapping
YOUTUBE_CATEGORIES = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "18": "Short Movies",
    "19": "Travel & Events",
    "20": "Gaming",
    "21": "Videoblogging",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
    "30": "Movies",
    "31": "Anime/Animation",
    "32": "Action/Adventure",
    "33": "Classics",
    "34": "Comedy",
    "35": "Documentary",
    "36": "Drama",
    "37": "Family",
    "38": "Foreign",
    "39": "Horror",
    "40": "Sci-Fi/Fantasy",
    "41": "Thriller",
    "42": "Shorts",
    "43": "Shows",
    "44": "Trailers",
}


class YouTubeAPIEnricher:
    """
    Enriches video metadata using YouTube Data API v3.
    
    Adds additional fields not available in RSS feeds:
    - duration, tags, category, comment_count
    - content details (definition, caption availability)
    - topic categories (Wikipedia topics)
    """
    
    YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
    MAX_VIDEOS_PER_REQUEST = 50  # YouTube API limit
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API enricher.
        
        Args:
            api_key: YouTube Data API key. If not provided, reads from 
                     YOUTUBE_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get('YOUTUBE_API_KEY')
        if not self.api_key:
            logger.warning("No YouTube API key provided. API enrichment disabled.")
        
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json'
        })
        
        # Track quota usage (each videos.list costs 1 unit per request)
        self.quota_used = 0
    
    def is_available(self) -> bool:
        """Check if API enrichment is available."""
        return bool(self.api_key)
    
    def _parse_duration(self, iso_duration: str) -> int:
        """
        Parse ISO 8601 duration to seconds.
        
        Examples:
            PT15M33S -> 933
            PT1H2M10S -> 3730
            PT45S -> 45
        """
        if not iso_duration:
            return 0
            
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, iso_duration)
        
        if not match:
            return 0
            
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    def enrich_videos(self, videos: list[dict], delay: float = 0.1) -> list[dict]:
        """
        Enrich a list of videos with YouTube API data.
        
        Args:
            videos: List of video dictionaries to enrich
            delay: Delay between API requests (to avoid rate limits)
            
        Returns:
            Enriched list of video dictionaries
        """
        if not self.is_available():
            logger.warning("API key not available. Returning videos unchanged.")
            return videos
        
        video_ids = [v['video_id'] for v in videos]
        enrichment_data = {}
        
        # Process in batches of 50 (API limit)
        for i in range(0, len(video_ids), self.MAX_VIDEOS_PER_REQUEST):
            batch = video_ids[i:i + self.MAX_VIDEOS_PER_REQUEST]
            batch_data = self._fetch_video_details(batch)
            enrichment_data.update(batch_data)
            
            if i + self.MAX_VIDEOS_PER_REQUEST < len(video_ids):
                time.sleep(delay)
        
        # Apply enrichment to videos
        enriched_videos = []
        for video in videos:
            vid = video['video_id']
            if vid in enrichment_data:
                video.update(enrichment_data[vid])
                video['enriched_via_api'] = True
            enriched_videos.append(video)
        
        logger.info(f"Enriched {len(enrichment_data)}/{len(videos)} videos via API")
        return enriched_videos
    
    def _fetch_video_details(self, video_ids: list[str]) -> dict:
        """
        Fetch detailed video information from YouTube API.
        
        Args:
            video_ids: List of video IDs (max 50)
            
        Returns:
            Dictionary mapping video_id to enrichment data
        """
        if not video_ids:
            return {}
        
        params = {
            'key': self.api_key,
            'id': ','.join(video_ids),
            'part': 'contentDetails,statistics,snippet,topicDetails'
        }
        
        try:
            response = self.session.get(
                f"{self.YOUTUBE_API_BASE}/videos",
                params=params,
                timeout=30
            )
            self.quota_used += 1
            
            if response.status_code == 403:
                logger.error("YouTube API quota exceeded or invalid key")
                return {}
            
            response.raise_for_status()
            data = response.json()
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
        
        enrichment = {}
        for item in data.get('items', []):
            vid = item['id']
            
            # Content details
            content = item.get('contentDetails', {})
            duration = content.get('duration', '')
            
            # Statistics
            stats = item.get('statistics', {})
            
            # Snippet
            snippet = item.get('snippet', {})
            category_id = snippet.get('categoryId', '')
            
            # Topic details
            topics = item.get('topicDetails', {})
            topic_categories = topics.get('topicCategories', [])
            # Clean Wikipedia URLs to just topic names
            topic_names = [url.split('/')[-1].replace('_', ' ') for url in topic_categories]
            
            enrichment[vid] = {
                # Duration
                'duration': duration,
                'duration_seconds': self._parse_duration(duration),
                
                # Content details
                'definition': content.get('definition'),  # 'hd' or 'sd'
                'caption': content.get('caption') == 'true',
                'licensed_content': content.get('licensedContent'),
                
                # Statistics (update with fresh data)
                'view_count': int(stats.get('viewCount', 0)) if stats.get('viewCount') else None,
                'like_count': int(stats.get('likeCount', 0)) if stats.get('likeCount') else None,
                'comment_count': int(stats.get('commentCount', 0)) if stats.get('commentCount') else None,
                
                # Category
                'category_id': category_id,
                'category_name': YOUTUBE_CATEGORIES.get(category_id, 'Unknown'),
                
                # Tags and topics
                'tags': snippet.get('tags', []),
                'topic_categories': topic_names,
                
                # Language
                'default_language': snippet.get('defaultLanguage'),
                'default_audio_language': snippet.get('defaultAudioLanguage'),
            }
        
        return enrichment
    
    def get_quota_usage(self) -> int:
        """Return the number of API quota units used."""
        return self.quota_used


class YouTubeRSSScraper:
    """
    YouTube RSS/XML Scraper for extracting video metadata from channel feeds.
    
    This scraper fetches YouTube's public RSS feeds and extracts structured
    metadata suitable for building recommendation systems.
    """
    
    def __init__(
        self,
        request_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize the scraper.
        
        Args:
            request_delay: Delay between requests in seconds (be respectful to servers)
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set a reasonable User-Agent
        session.headers.update({
            'User-Agent': 'K-RSS-Scraper/1.0 (Academic Research Project)',
            'Accept': 'application/xml, application/rss+xml, text/xml'
        })
        
        return session
    
    def _resolve_channel_id(self, channel_input: str) -> Optional[str]:
        """
        Resolve various channel input formats to a channel ID.
        
        Supports:
        - Direct channel ID (UC...)
        - Channel URL (youtube.com/channel/UC...)
        - Custom URL (youtube.com/@handle)
        - Channel name (requires additional lookup)
        
        Args:
            channel_input: Channel identifier in various formats
            
        Returns:
            Channel ID if resolved, None otherwise
        """
        channel_input = channel_input.strip()
        
        # Already a channel ID
        if channel_input.startswith('UC') and len(channel_input) == 24:
            return channel_input
        
        # Full channel URL
        if 'youtube.com/channel/' in channel_input:
            match = re.search(rf'youtube\.com/channel/({CHANNEL_ID_PATTERN})', channel_input)
            if match:
                return match.group(1)
        
        # Handle format (@username)
        if channel_input.startswith('@') or 'youtube.com/@' in channel_input:
            return self._resolve_handle_to_channel_id(channel_input)
        
        # Try to resolve as channel name/handle
        return self._resolve_handle_to_channel_id(f"@{channel_input}")
    
    def _resolve_handle_to_channel_id(self, handle: str) -> Optional[str]:
        """
        Resolve a YouTube handle (@username) to channel ID.
        
        This fetches the channel page and extracts the channel ID from meta tags.
        """
        # Clean the handle
        if 'youtube.com/@' in handle:
            handle = '@' + handle.split('@')[-1].split('/')[0].split('?')[0]
        elif not handle.startswith('@'):
            handle = f"@{handle}"
        
        channel_url = f"https://www.youtube.com/{handle}"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = self.session.get(channel_url, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            
            # YouTube embeds channel ID in various places
            patterns = [
                rf'"channelId"\s*:\s*"({CHANNEL_ID_PATTERN})"',
                rf'"externalId"\s*:\s*"({CHANNEL_ID_PATTERN})"',
                rf'"browseId"\s*:\s*"({CHANNEL_ID_PATTERN})"',
                rf'channel_id=({CHANNEL_ID_PATTERN})',
                rf'<meta itemprop="channelId" content="({CHANNEL_ID_PATTERN})"',
                rf'/channel/({CHANNEL_ID_PATTERN})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response.text)
                if match:
                    channel_id = match.group(1)
                    logger.info(f"Resolved {handle} to channel ID: {channel_id}")
                    return channel_id
                    
            logger.warning(f"Could not find channel ID for handle: {handle}")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Error resolving handle {handle}: {e}")
            return None
    
    def _build_feed_url(self, channel_id: str) -> str:
        """Build the RSS feed URL for a channel."""
        return f"{YOUTUBE_RSS_BASE_URL}?channel_id={channel_id}"
    
    def _fetch_feed(self, feed_url: str) -> Optional[str]:
        """
        Fetch the RSS feed content.
        
        Args:
            feed_url: URL of the RSS feed
            
        Returns:
            Feed content as string, None if failed
        """
        try:
            response = self.session.get(feed_url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
            return None
    
    def _parse_feed(self, feed_content: str, channel_id: str) -> ScrapingResult:
        """
        Parse the RSS feed XML and extract video metadata.
        
        Args:
            feed_content: Raw XML content of the feed
            channel_id: Channel ID for reference
            
        Returns:
            ScrapingResult containing parsed data
        """
        try:
            root = ET.fromstring(feed_content)
            
            # Extract channel information
            channel_name = self._get_text(root, 'atom:title')
            channel_url = self._get_link(root, 'alternate')
            feed_url = self._build_feed_url(channel_id)
            
            # Find all video entries
            entries = root.findall('atom:entry', NAMESPACES)
            
            videos = []
            for entry in entries:
                video = self._parse_entry(entry, channel_id, channel_name or "Unknown")
                if video:
                    videos.append(video)
            
            # Get last updated from feed
            last_updated = self._get_text(root, 'atom:updated') or datetime.now(timezone.utc).isoformat()
            
            channel_metadata = ChannelMetadata(
                channel_id=channel_id,
                channel_name=channel_name or "Unknown",
                channel_url=channel_url or f"https://www.youtube.com/channel/{channel_id}",
                feed_url=feed_url,
                last_updated=last_updated,
                video_count=len(videos)
            )
            
            return ScrapingResult(
                channel=channel_metadata,
                videos=videos,
                success=True
            )
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error for channel {channel_id}: {e}")
            return ScrapingResult(
                channel=ChannelMetadata(
                    channel_id=channel_id,
                    channel_name="Unknown",
                    channel_url="",
                    feed_url="",
                    last_updated="",
                    video_count=0
                ),
                videos=[],
                success=False,
                error_message=str(e)
            )
    
    def _parse_entry(
        self,
        entry: ET.Element,
        channel_id: str,
        channel_name: str
    ) -> Optional[VideoMetadata]:
        """
        Parse a single video entry from the RSS feed.
        
        Args:
            entry: XML element for the video entry
            channel_id: Parent channel ID
            channel_name: Parent channel name
            
        Returns:
            VideoMetadata object or None if parsing fails
        """
        try:
            # Extract video ID
            video_id_element = entry.find('yt:videoId', NAMESPACES)
            video_id = video_id_element.text if video_id_element is not None else None
            
            if not video_id:
                return None
            
            # Extract basic info
            title = self._get_text(entry, 'atom:title') or ""
            published = self._get_text(entry, 'atom:published') or ""
            updated = self._get_text(entry, 'atom:updated') or ""
            
            # Extract media group information
            media_group = entry.find('media:group', NAMESPACES)
            
            description = ""
            thumbnail_url = ""
            
            if media_group is not None:
                description = self._get_text(media_group, 'media:description') or ""
                
                # Get thumbnail
                thumbnail = media_group.find('media:thumbnail', NAMESPACES)
                if thumbnail is not None:
                    thumbnail_url = thumbnail.get('url', '')
            
            # Extract view count and like count from media:community if available
            view_count = None
            like_count = None
            media_community = entry.find('.//media:community', NAMESPACES)
            if media_community is not None:
                statistics = media_community.find('media:statistics', NAMESPACES)
                if statistics is not None:
                    views_str = statistics.get('views', '')
                    if views_str:
                        try:
                            view_count = int(views_str)
                        except ValueError:
                            pass
                
                # Extract like count from media:starRating
                star_rating = media_community.find('media:starRating', NAMESPACES)
                if star_rating is not None:
                    likes_str = star_rating.get('count', '')
                    if likes_str:
                        try:
                            like_count = int(likes_str)
                        except ValueError:
                            pass
                
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            return VideoMetadata(
                video_id=video_id,
                channel_id=channel_id,
                channel_name=channel_name,
                title=title,
                description=description,
                published_date=published,
                updated_date=updated,
                thumbnail_url=thumbnail_url,
                video_url=video_url,
                view_count=view_count,
                like_count=like_count
            )
            
        except Exception as e:
            logger.warning(f"Error parsing entry: {e}")
            return None
    
    def _get_text(self, element: ET.Element, xpath: str) -> Optional[str]:
        """Get text content from an XML element."""
        child = element.find(xpath, NAMESPACES)
        return child.text if child is not None else None
    
    def _get_link(self, element: ET.Element, rel: str) -> Optional[str]:
        """Get link href from an XML element by rel attribute."""
        for link in element.findall('atom:link', NAMESPACES):
            if link.get('rel') == rel:
                return link.get('href')
        return None
    
    def scrape_channel(self, channel_input: str) -> ScrapingResult:
        """
        Scrape a single YouTube channel.
        
        Args:
            channel_input: Channel ID, URL, or handle
            
        Returns:
            ScrapingResult with channel and video data
        """
        logger.info(f"Processing channel: {channel_input}")
        
        # Resolve to channel ID
        channel_id = self._resolve_channel_id(channel_input)
        
        if not channel_id:
            logger.error(f"Could not resolve channel ID for: {channel_input}")
            return ScrapingResult(
                channel=ChannelMetadata(
                    channel_id="",
                    channel_name=channel_input,
                    channel_url="",
                    feed_url="",
                    last_updated="",
                    video_count=0
                ),
                videos=[],
                success=False,
                error_message=f"Could not resolve channel ID for: {channel_input}"
            )
        
        # Build and fetch feed
        feed_url = self._build_feed_url(channel_id)
        logger.info(f"Fetching feed: {feed_url}")
        
        feed_content = self._fetch_feed(feed_url)
        
        if not feed_content:
            return ScrapingResult(
                channel=ChannelMetadata(
                    channel_id=channel_id,
                    channel_name="Unknown",
                    channel_url="",
                    feed_url=feed_url,
                    last_updated="",
                    video_count=0
                ),
                videos=[],
                success=False,
                error_message="Failed to fetch RSS feed"
            )
        
        # Parse feed
        result = self._parse_feed(feed_content, channel_id)
        
        logger.info(f"Scraped {len(result.videos)} videos from channel {result.channel.channel_name}")
        
        return result
    
    def scrape_channels_from_csv(
        self,
        csv_path: str,
        channel_column: str = "channel",
        output_path: Optional[str] = None
    ) -> dict:
        """
        Scrape multiple channels from a CSV file.
        
        Args:
            csv_path: Path to CSV file containing channel list
            channel_column: Name of column containing channel IDs/names
            output_path: Optional path to save JSON output
            
        Returns:
            Dictionary containing all scraped data
        """
        csv_file = Path(csv_path)
        
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        channels_data = []
        all_videos = []
        failed_channels = []
        
        # Read CSV file
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            
            # Auto-detect delimiter
            if '\t' in sample:
                delimiter = '\t'
            elif ';' in sample:
                delimiter = ';'
            else:
                delimiter = ','
            
            reader = csv.DictReader(f, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            
            # Check if column exists
            if channel_column not in fieldnames:
                # Try common alternatives
                alternatives = ['channel', 'channel_id', 'channelId', 'name', 'id', 'Channel', 'Channel ID']
                for alt in alternatives:
                    if alt in fieldnames:
                        channel_column = alt
                        break
                else:
                    raise ValueError(
                        f"Column '{channel_column}' not found. "
                        f"Available columns: {reader.fieldnames}"
                    )
            
            channels_list = [row[channel_column] for row in reader if row[channel_column].strip()]
        
        total_channels = len(channels_list)
        logger.info(f"Found {total_channels} channels to scrape")
        
        # Scrape each channel
        for idx, channel_input in enumerate(channels_list, 1):
            logger.info(f"Processing {idx}/{total_channels}: {channel_input}")
            
            try:
                result = self.scrape_channel(channel_input)
                
                if result.success:
                    channels_data.append(asdict(result.channel))
                    all_videos.extend([asdict(v) for v in result.videos])
                else:
                    failed_channels.append({
                        'input': channel_input,
                        'error': result.error_message
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {channel_input}: {e}")
                failed_channels.append({
                    'input': channel_input,
                    'error': str(e)
                })
            
            # Respect rate limits
            if idx < total_channels:
                time.sleep(self.request_delay)
        
        # Build output structure optimized for recommendation system
        output = {
            "metadata": {
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "total_channels": len(channels_data),
                "total_videos": len(all_videos),
                "failed_channels": len(failed_channels),
                "source_file": csv_file.name
            },
            "channels": channels_data,
            "videos": all_videos,
            "failed": failed_channels,
            # Index structures for recommendation system
            "indices": {
                "video_by_id": {v['video_id']: idx for idx, v in enumerate(all_videos)},
                "videos_by_channel": self._group_videos_by_channel(all_videos)
            }
        }
        
        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Output saved to: {output_file}")
        
        return output
    
    def _group_videos_by_channel(self, videos: list) -> dict:
        """Group video indices by channel ID."""
        grouped = {}
        for idx, video in enumerate(videos):
            channel_id = video['channel_id']
            if channel_id not in grouped:
                grouped[channel_id] = []
            grouped[channel_id].append(idx)
        return grouped


def enrich_scraped_data(
    data: dict,
    api_key: Optional[str] = None
) -> dict:
    """
    Enrich scraped video data with YouTube API information.
    
    Args:
        data: Scraped data dictionary with 'videos' key
        api_key: Optional YouTube API key (reads from env if not provided)
        
    Returns:
        Enriched data dictionary
    """
    enricher = YouTubeAPIEnricher(api_key)
    
    if not enricher.is_available():
        logger.warning("No API key available. Skipping enrichment.")
        return data
    
    logger.info(f"Enriching {len(data.get('videos', []))} videos via YouTube API...")
    
    data['videos'] = enricher.enrich_videos(data['videos'])
    
    # Update metadata
    data['metadata']['enriched_via_api'] = True
    data['metadata']['api_quota_used'] = enricher.get_quota_usage()
    data['metadata']['enriched_at'] = datetime.now(timezone.utc).isoformat()
    
    return data


def create_sample_csv(output_path: str):
    """
    Create a sample CSV file with example channels.
    
    Args:
        output_path: Path to save the sample CSV
    """
    sample_channels = [
        {"channel": "@3blue1brown", "category": "education"},
        {"channel": "@FavijTV", "category": "entrateiment"},
        {"channel": "@BenFelixCSI", "category": "finance"},
    ]
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['channel', 'category'])
        writer.writeheader()
        writer.writerows(sample_channels)
    
    logger.info(f"Sample CSV created: {output_file}")


def main():
    """Main entry point for the scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='YouTube RSS Scraper for K-RSS Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape channels from CSV file
  python scraper.py --csv channels.csv --output data/videos.json
  
  # Scrape with YouTube API enrichment
  python scraper.py --csv channels.csv --enrich --api-key YOUR_KEY
  
  # Scrape a single channel
  python scraper.py --channel @3blue1brown --output data/videos.json
  
  # Create sample CSV file
  python scraper.py --create-sample data/sample_channels.csv

Environment Variables:
  YOUTUBE_API_KEY: YouTube Data API v3 key for enrichment
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file containing channel list'
    )
    parser.add_argument(
        '--channel',
        type=str,
        help='Single channel to scrape (ID, URL, or handle)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/scraped_videos.json',
        help='Output JSON file path (default: data/scraped_videos.json)'
    )
    parser.add_argument(
        '--column',
        type=str,
        default='channel',
        help='CSV column name containing channel identifiers (default: channel)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--create-sample',
        type=str,
        metavar='PATH',
        help='Create a sample CSV file at the specified path'
    )
    parser.add_argument(
        '--enrich',
        action='store_true',
        help='Enrich videos with YouTube API data (requires API key)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='YouTube Data API v3 key (or set YOUTUBE_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Create sample CSV if requested
    if args.create_sample:
        create_sample_csv(args.create_sample)
        return
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get('YOUTUBE_API_KEY')
    
    # Initialize scraper
    scraper = YouTubeRSSScraper(request_delay=args.delay)
    
    if args.csv:
        # Scrape from CSV
        result = scraper.scrape_channels_from_csv(
            csv_path=args.csv,
            channel_column=args.column,
            output_path=None  # Don't save yet if enriching
        )
        
        # Enrich with API if requested
        if args.enrich:
            if api_key:
                result = enrich_scraped_data(result, api_key)
            else:
                logger.warning("--enrich specified but no API key provided. Skipping.")
        
        # Save output
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write('\n')
        
        print(f"\n{'='*50}")
        print("Scraping Complete!")
        print(f"{'='*50}")
        print(f"Channels scraped: {result['metadata']['total_channels']}")
        print(f"Videos collected: {result['metadata']['total_videos']}")
        print(f"Failed channels: {result['metadata']['failed_channels']}")
        if result['metadata'].get('enriched_via_api'):
            print(f"API Enrichment: ✓ (quota used: {result['metadata'].get('api_quota_used', 0)})")
        print(f"Output saved to: {args.output}")
        
    elif args.channel:
        # Scrape single channel
        result = scraper.scrape_channel(args.channel)
        
        if result.success:
            output = {
                "metadata": {
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "total_channels": 1,
                    "total_videos": len(result.videos)
                },
                "channels": [asdict(result.channel)],
                "videos": [asdict(v) for v in result.videos]
            }
            
            # Enrich with API if requested
            if args.enrich and api_key:
                output = enrich_scraped_data(output, api_key)
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
                f.write('\n')
            
            print(f"\n{'='*50}")
            print("Scraping Complete!")
            print(f"{'='*50}")
            print(f"Channel: {result.channel.channel_name}")
            print(f"Videos collected: {len(result.videos)}")
            if output['metadata'].get('enriched_via_api'):
                print(f"API Enrichment: ✓ (quota used: {output['metadata'].get('api_quota_used', 0)})")
            print(f"Output saved to: {args.output}")
        else:
            print(f"Error: {result.error_message}")
            
    else:
        parser.print_help()
        print("\nError: Please specify either --csv or --channel option")


if __name__ == "__main__":
    main()