from pathlib import Path

# Dataset configuration
DATASET_CONFIG = {
    "mind": {
        "behaviors_path": Path("data/mind/val/behaviors.tsv"),
        "news_path": Path("data/mind/val/news.tsv"),
        "entity_embedding_path": Path("data/mind/val/entity_embedding.vec"),
        "relation_embedding_path": Path("data/mind/val/relation_embedding.vec"),
        "articles_path": Path("data/mind/articles_mind.pickle"),
        "processed_articles_cache": Path("data/mind/articles_mind_processed.pkl"),
        "model_name": "all-MiniLM-L6-v2",
        "recommenders": ["lstur", "naml", "npa", "nrms", "pop", "random"],
    },
    "ebnerd": {
        "behaviors_path": Path("data/ebnerd/val/behaviors.parquet"),
        "history_path": Path("data/ebnerd/val/history.parquet"),
        "articles_path": Path("data/ebnerd/articles_ebnerd.pickle"),
        "processed_articles_cache": Path("data/ebnerd/articles_ebnerd_processed.pkl"),
        "model_name": "all-MiniLM-L6-v2",
        "recommenders": ["lstur", "naml", "npa", "nrms", "pop", "random"],
    },
    "adressa": {
        "behaviors_path": Path("data/adressa/val/behaviors.tsv"),
        "news_path": Path("data/adressa/val/news.tsv"),
        "articles_path": Path("data/adressa/articles_adressa.pickle"),
        "processed_articles_cache": Path("data/adressa/articles_adressa_processed.pkl"),
        "model_name": "all-MiniLM-L6-v2",
        "recommenders": ["lstur", "naml", "npa", "nrms", "pop", "random"],
    },
}

# Cutoff for top-k evaluation
RADIO_CUTOFF = 10
NDCG_CUTOFF = 10

# Output folder for results
OUTPUT_FOLDER = Path("results")

# Central configuration for tradeoff-enabled metrics
TRADEOFF_METRICS_CONFIG = [
    {
        "name": "topic",
        "mode": "category",
        "baseline_keys": ["topic_calibration"],
    },
    {
        "name": "subtopic",
        "mode": "subcategory",
        "baseline_keys": ["subtopic_calibration"],
    },
    {
        "name": "activation",
        "mode": "activation",
        "baseline_keys": ["activation"],
    },
    {
        "name": "complexity",
        "mode": "complexity",
        "baseline_keys": ["complexity_calibration"],
    },
    {
        "name": "ild_tfidf",
        "mode": "ild_tfidf",
        "baseline_keys": ["tf_idf_ild"],
    },
    {
        "name": "ild_sentbert",
        "mode": "ild_sentbert",
        "baseline_keys": ["sentbert_ild"],
    },
    {
        "name": "gini",
        "mode": "gini",
        "baseline_keys": ["gini"],
    },
    {
        "name": "fragmentation",
        "mode": "fragmentation",
        "baseline_keys": ["fragmentation"],
    },
]
