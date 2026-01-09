"""
K-RSS Web Application
=====================
Streamlit-based interface for the Knowledge-aware YouTube RSS Recommendation System.

This is a placeholder implementation that will be integrated with the AI_RM module.
"""

import streamlit as st
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="K-RSS - YouTube Recommendations",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_PATH = Path("/app/data") if Path("/app/data").exists() else Path("../../data")
RAW_DATA = DATA_PATH / "raw" / "scraped_videos.json"


def load_videos() -> dict:
    """Load scraped videos from JSON file."""
    if RAW_DATA.exists():
        with open(RAW_DATA, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"videos": [], "channels": [], "metadata": {}}


def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.title("🎬 K-RSS")
        st.markdown("*Knowledge-aware YouTube RSS Recommendations*")
        st.divider()
        
        # XAI Parameters Section (placeholder)
        st.subheader("XAI Parameters")
        
        exploration = st.slider(
            "Exploration vs Exploitation",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher values = more diverse recommendations"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum similarity score for recommendations"
        )
        
        st.divider()
        st.caption("Academic Project")
        st.caption("Francesco Romeo & Matteo Picozzi")
    
    # Main content
    st.title("Video Recommendations")
    
    # Load data
    data = load_videos()
    videos = data.get("videos", [])
    metadata = data.get("metadata", {})
    
    # Show data status
    if videos:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Videos", len(videos))
        with col2:
            st.metric("Channels", metadata.get("total_channels", 0))
        with col3:
            enriched = "✅" if metadata.get("enriched_via_api") else "❌"
            st.metric("API Enriched", enriched)
    else:
        st.warning("⚠️ No videos loaded. Run the scraper first:")
        st.code("docker-compose run --rm scrape-csv", language="bash")
        return
    
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["For You", "Explore", "Profile"])
    
    with tab1:
        st.subheader("Recommended Videos")
        st.info("Recommendation engine not yet integrated. Showing recent videos.")
        
        # Display videos in grid
        cols = st.columns(3)
        for idx, video in enumerate(videos[:9]):
            with cols[idx % 3]:
                render_video_card(video, key_prefix="foryou")
    
    with tab2:
        st.subheader("Explore All Videos")
        
        # Category filter (if available)
        categories = list(set(v.get("category_name", "Unknown") for v in videos if v.get("category_name")))
        if categories:
            selected_category = st.selectbox("Filter by Category", ["All"] + sorted(categories))
        else:
            selected_category = "All"
        
        # Filter videos
        filtered = videos if selected_category == "All" else [
            v for v in videos if v.get("category_name") == selected_category
        ]
        
        # Display filtered videos
        cols = st.columns(3)
        for idx, video in enumerate(filtered[:12]):
            with cols[idx % 3]:
                render_video_card(video, key_prefix="explore")
    
    with tab3:
        st.subheader("User Profile")
        st.info("User profile management coming soon.")
        
        st.markdown("""
        ### Planned Features:
        - **Positive/Negative Voting** - Train your preferences
        - **Profile Vector Visualization** - See your taste evolve
        - **Profile Reset** - Start fresh
        - **Feedback History** - Review your interactions
        """)


def render_video_card(video: dict, key_prefix: str = "default"):
    print(video)
    """Render a single video card."""
    with st.container():
        # Thumbnail
        if video.get("thumbnail_url"):
            st.image(video["thumbnail_url"], use_container_width=True)
        
        # Title (truncated)
        title = video.get("title", "Unknown")[:60]
        if len(video.get("title", "")) > 60:
            title += "..."
        st.markdown(f"**{title}**")

        # Description (trunched)
        desc = video.get("description", "Unknown")[:100]
        if len(video.get("description", "")) > 100:
            desc += "..."
        st.caption(desc)
        
        # Channel
        st.caption(f"{video.get('channel_name', 'Unknown')}")
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            views = video.get("view_count")
            if views:
                st.caption(f"{format_number(views)} views")
        with col3:
            duration = video.get("duration_seconds")
            if duration:
                st.caption(f"{format_duration(duration)}")
        with col2:
            like = video.get("like_count")
            if like:
                st.caption(f'{format_number(like)} like')
        
        # Action buttons (placeholders)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("👍", key=f"{key_prefix}_like_{video['video_id']}", help="Like")
        with col2:
            st.button("👎", key=f"{key_prefix}_dislike_{video['video_id']}", help="Dislike")
        with col3:
            st.link_button("▶️", video.get("video_url", "#"), help="Watch")
        
        st.divider()


def format_number(n: int) -> str:
    """Format large numbers for display."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human readable."""
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m}m"
    elif seconds >= 60:
        m = seconds // 60
        s = seconds % 60
        return f"{m}:{s:02d}"
    return f"{seconds}s"


if __name__ == "__main__":
    main()
