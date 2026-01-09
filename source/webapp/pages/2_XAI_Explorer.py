"""
XAI Explorer Page - K-RSS
==========================
Explainable AI interface for understanding recommendation decisions.
"""

import streamlit as st

st.set_page_config(page_title="XAI Explorer - K-RSS", page_icon="XAI", layout="wide")


def main():
    st.title("XAI Explorer")
    st.markdown("*Understand how recommendations are generated*")
    
    st.info("""
    **Coming Soon** - This page will provide explainability tools for the recommendation system.
    """)
    
    st.divider()
    
    # Planned features preview
    st.subheader("Planned Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Recommendation Explanation
        - Why was this video recommended?
        - Which entities/topics drove the recommendation?
        - Similarity breakdown visualization
        
        ### Embedding Visualization
        - 2D/3D projection of video embeddings
        - User profile position in latent space
        - Cluster visualization
        """)
    
    with col2:
        st.markdown("""
        ### Knowledge Graph View
        - Entity relationships from DBpedia/Wikidata
        - Topic connections between videos
        - Interactive graph exploration
        
        ### Parameter Impact
        - How exploration/exploitation affects results
        - A/B comparison of recommendations
        - Parameter sensitivity analysis
        """)
    
    st.divider()
    
    # Interactive parameter demo
    st.subheader("Parameter Playground (Demo)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exploration = st.slider(
            "Exploration Weight",
            0.0, 1.0, 0.5, 0.1,
            help="Higher = more diverse, unexpected recommendations"
        )
        
        knowledge_weight = st.slider(
            "Knowledge Graph Weight",
            0.0, 1.0, 0.3, 0.1,
            help="How much entity linking influences recommendations"
        )
    
    with col2:
        recency_bias = st.slider(
            "Recency Bias",
            0.0, 1.0, 0.2, 0.1,
            help="Preference for newer videos"
        )
        
        diversity_penalty = st.slider(
            "Same-Channel Diversity Penalty",
            0.0, 1.0, 0.5, 0.1,
            help="Reduce same-channel recommendations"
        )
    
    st.info(f"""
    **Current Configuration:**
    - Exploration: {exploration:.0%} | Exploitation: {1-exploration:.0%}
    - Knowledge Graph influence: {knowledge_weight:.0%}
    - Recency preference: {recency_bias:.0%}
    - Channel diversity: {diversity_penalty:.0%}
    """)


if __name__ == "__main__":
    main()
