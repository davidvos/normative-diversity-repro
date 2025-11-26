
import sys
from pathlib import Path
sys.path.append("src")
import pandas as pd
from dart.metrics.calibration import Calibration
from utils import load_dataset, ensure_article_cache
from config import DATASET_CONFIG

def test_calibration():
    dataset = "ebnerd"
    print(f"Loading {dataset}...")
    behaviors, articles_df = load_dataset(dataset, DATASET_CONFIG)
    articles = ensure_article_cache(
        articles_df,
        DATASET_CONFIG[dataset]["model_name"],
        DATASET_CONFIG[dataset].get("processed_articles_cache"),
        build_if_missing=False,
    )
    
    # Mock history and recommendation
    # Pick some article IDs from the dataframe
    article_ids = list(articles.keys())[:10]
    history = [articles[aid] for aid in article_ids[:5]]
    recommendation = [articles[aid] for aid in article_ids[5:]]
    
    print("History categories:", [a.get('category') for a in history])
    print("Recommendation categories:", [a.get('category') for a in recommendation])
    
    config = {"language": "english"} # Force english to match what happens in code
    cal = Calibration(config)
    
    print("Computing calibration...")
    res = cal.calculate(history, recommendation)
    print("Result:", res)

if __name__ == "__main__":
    test_calibration()
