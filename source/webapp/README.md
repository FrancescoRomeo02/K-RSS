# K-RSS Web Application

Streamlit-based interface for the Knowledge-aware YouTube RSS Recommendation System.

## Structure

```
webapp/
├── app.py                    # Main application entry point
├── pages/
│   ├── 1_📊_Analytics.py    # Dashboard and statistics
│   └── 2_🔬_XAI_Explorer.py # Explainability interface
└── README.md
```

## Running Locally

```bash
# From project root
cd source/webapp
streamlit run app.py

# Or via Docker
docker-compose up webapp
```

## Features

### Current (Placeholder)
- Video display grid
- Basic filtering by category
- XAI parameter sliders (UI only)

### Planned Integration
- [ ] Connect to AI_RM for recommendations
- [ ] Implement user feedback (like/dislike)
- [ ] Add embedding visualizations
- [ ] Knowledge graph explorer
- [ ] User profile management

## Development

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for team guidelines.
