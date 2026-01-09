"""
Analytics Page - K-RSS Dashboard
=================================
Visualizations and statistics for the recommendation system.
"""

import streamlit as st
from pathlib import Path
import json

st.set_page_config(page_title="Analytics - K-RSS", page_icon="📊", layout="wide")

# Load data
DATA_PATH = Path("/app/data") if Path("/app/data").exists() else Path("../../../data")
RAW_DATA = DATA_PATH / "raw" / "scraped_videos.json"


def load_videos() -> dict:
    if RAW_DATA.exists():
        with open(RAW_DATA, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"videos": [], "channels": [], "metadata": {}}


def main():
    st.title("Analytics Dashboard")
    st.markdown("*Insights into the video collection and recommendation performance*")
    
    data = load_videos()
    videos = data.get("videos", [])
    channels = data.get("channels", [])
    
    if not videos:
        st.warning("No data available. Run the scraper first.")
        return
    
    # Overview metrics
    st.subheader("Collection Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Videos", len(videos))
    with col2:
        st.metric("Channels", len(channels))
    with col3:
        enriched = sum(1 for v in videos if v.get("enriched_via_api"))
        st.metric("Enriched Videos", enriched)
    with col4:
        with_tags = sum(1 for v in videos if v.get("tags"))
        st.metric("Videos with Tags", with_tags)
    
    st.divider()
    
    # Category distribution
    st.subheader("Category Distribution")
    categories = {}
    for v in videos:
        cat = v.get("category_name", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    if categories:
        import pandas as pd
        df = pd.DataFrame(list(categories.items()), columns=["Category", "Count"])
        df = df.sort_values("Count", ascending=False)
        st.bar_chart(df.set_index("Category"))
    
    st.divider()
    
    # Channel statistics
    st.subheader("📺 Videos per Channel")
    channel_counts = {}
    for v in videos:
        ch = v.get("channel_name", "Unknown")
        channel_counts[ch] = channel_counts.get(ch, 0) + 1
    
    if channel_counts:
        import pandas as pd
        df = pd.DataFrame(list(channel_counts.items()), columns=["Channel", "Videos"])
        df = df.sort_values("Videos", ascending=False)
        st.dataframe(df, use_container_width=True)
    
    st.divider()
    
    # Placeholder for recommendation metrics
    st.subheader("Recommendation Metrics")
    st.info("Recommendation analytics will be available after AI_RM integration.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("User Interactions", "—", help="Coming soon")
    with col2:
        st.metric("Avg. Feedback Score", "—", help="Coming soon")
    with col3:
        st.metric("Profile Updates", "—", help="Coming soon")


if __name__ == "__main__":
    main()
