*Members*: Francesco Romeo 885880 - Matteo Picozzi 890228

*Abstract*: The project aims to develop a recommendation system for YouTube RSS feeds that addresses the cold start problem and dynamic personalization through a human-in-the-loop approach. Unlike systems based on traditional collaborative filters, K-RSS uses a knowledge-aware architecture:
1. Representation: Overcoming the limitations of pre-transformer models, we will use LLM-based embeddings for deep semantic capture of titles and entity linking (via DBpedia/Wikidata) to enrich the information context
2. Dynamics: The system implements a relevance feedback mechanism. Through positive/negative votes, the user profile is dynamically updated in the latent space (vector shifting), allowing the system to quickly converge on the user's tastes from scratch.
3. Explainability: An interactive interface (Streamlit) will allow for the manipulation of critical parameters (e.g., exploration vs. exploitation), making the recommendation process transparent and analyzable (XAI).The project will critically discuss the choice of these techniques compared to alternative solutions such as Retrieval-Augmented Generation (RAG) architectures, focusing on the system's ability to evolve its user knowledge base over time.

*References*:
* Wu, L., et al. (2023). Recommender Systems in the Era of Large Language Models (LLMs): A Survey. (embedding post-BERT).
* Wang, X., et al. (2023). Large Language Models for Interactive Recommendation. (Dynamic feedback).
* Wang, H., et al. (2018). DKN: Deep Knowledge-Aware Network for News Recommendation. (Knowledge Graph implementation).