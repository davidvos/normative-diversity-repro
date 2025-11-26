"""Preprocess EBNeRD news articles to match the MIND pipeline outputs."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import re
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from tqdm import tqdm

from numpy import ndarray
from sentiment_models import danish_swn_sentiment, TransformerSentimentScorer
from story_clustering import compute_story_clusters
from text_complexity import lix_score, mtld_score
from utils import (
    STORY_SIMILARITY_THRESHOLDS,
    ensure_article_cache,
    story_column_name,
)

tqdm.pandas()


def _build_transformer_scorer(dataset_name: str) -> TransformerSentimentScorer:
    try:
        return TransformerSentimentScorer()
    except RuntimeError as exc:  # pragma: no cover - dependency issues
        raise RuntimeError(
            "Transformer sentiment scoring failed for EBNeRD. "
            "Ensure torch, transformers, sentencepiece, and tiktoken are installed, "
            "then rerun preprocess_ebnerd_news.py."
        ) from exc


def _normalise_story_id(value: Any) -> Optional[Any]:
    """Return a normalised scalar identifier for a story-like field."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() == "nan":
            return None
        return cleaned
    if isinstance(value, ndarray):
        iterable = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        iterable = list(value)
    else:
        iterable = None

    if iterable is not None:
        for item in iterable:
            normalised = _normalise_story_id(item)
            if normalised is not None:
                return normalised
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def preprocess_ebnerd_news(
    input_path: Path,
    output_path: Path,
    processed_cache: Optional[Path],
    model_name: str,
    *,
    danish_swn_path: Path,
) -> None:
    """Load, enrich, and persist the EBNeRD article catalogue."""
    print("Loading EBNeRD articles...")
    news = pd.read_parquet(input_path)

    # Standardise core text fields in line with the MIND pipeline.
    news["article_id"] = news["article_id"]
    news["title"] = news.get("title", "").fillna("")
    news["subtitle"] = news.get("subtitle", "").fillna("")
    # Maintain backward compatibility with downstream code that expects an 'abstract' column.
    news["abstract"] = news["subtitle"]
    news["category"] = news.get("category_str", news.get("category", "other")).fillna("other")

    if "category_str" not in news.columns:
        news["category_str"] = news["category"]

    if "subcategory" not in news.columns:
        news["subcategory"] = "other"
    news["subcategory"] = news["subcategory"].fillna("other")

    def _normalize_subcategory(value: Any) -> str:
        if value is None:
            return "other"
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or "other"
        if isinstance(value, ndarray):
            iterable = value.tolist()
        elif isinstance(value, (list, tuple, set)):
            iterable = list(value)
        else:
            iterable = None

        if iterable is not None:
            for item in iterable:
                token = str(item).strip()
                if token:
                    return token
            return "other"
        return str(value)

    news["subcategory"] = news["subcategory"].apply(_normalize_subcategory)

    news["text_title"] = news["title"].astype(str).str.strip()
    news["text_subtitle"] = news["subtitle"].astype(str).str.strip()
    news["text_abstract"] = news["text_subtitle"]
    news["text_title_abstract"] = (
        news["text_title"].fillna("") + " " + news["text_subtitle"].fillna("")
    ).str.strip()

    body_candidates = []
    for column in ("body", "content", "text_body", "article_body"):
        if column in news.columns:
            body_candidates.append(news[column].fillna("").astype(str).str.strip())
    if body_candidates:
        merged_body = body_candidates[0]
        for series in body_candidates[1:]:
            merged_body = (merged_body + " " + series).str.strip()
        news["text_body"] = merged_body
    else:
        news["text_body"] = ""

    news["text"] = (
        news["text_title_abstract"].fillna("") + " " + news["text_body"].fillna("")
    ).str.strip()

    if "publication_datetime" not in news.columns:
        if "published_time" in news.columns:
            news["publication_datetime"] = pd.to_datetime(news["published_time"], errors="coerce")
        elif "last_modified_time" in news.columns:
            news["publication_datetime"] = pd.to_datetime(news["last_modified_time"], errors="coerce")

    if "sentiment_score" not in news.columns:
        raise RuntimeError("Expected 'sentiment_score' column in EBNeRD articles parquet.")

    print("Using dataset-provided sentiment from 'sentiment_score'...")
    sentiment_metadata = pd.to_numeric(news["sentiment_score"], errors="coerce").fillna(0.0)
    news["sentiment_metadata"] = sentiment_metadata
    news["sentiment_dataset"] = sentiment_metadata

    lexicon_path = danish_swn_path.expanduser()
    if not lexicon_path.exists():
        raise FileNotFoundError(
            f"Danish SentiWordNet file not found at {lexicon_path}. "
            "Download or place the lexicon before running preprocessing."
        )

    print("Computing lexicon sentiment scores (Danish SentiWordNet)...")
    news["sentiment_lexicon"] = news["text"].progress_apply(
        lambda text: danish_swn_sentiment(text, lexicon_path=lexicon_path)
    )

    print("Computing transformer sentiment scores (cardiffnlp/twitter-xlm-roberta-base-sentiment)...")
    transformer_model = _build_transformer_scorer("ebnerd")
    news["sentiment_transformer"] = transformer_model.score_many(news["text"].tolist())

    print("Computing complexity scores (LIX & MTLD)...")
    news["complexity_lix"] = news["text"].progress_apply(lix_score)
    news["complexity_mtld"] = news["text"].progress_apply(mtld_score)
    news["complexity_readability"] = news["complexity_lix"]
    news["complexity"] = news["complexity_readability"]

    if "topics" not in news.columns:
        raise RuntimeError("Expected 'topics' column in EBNeRD articles parquet to derive story identifiers.")

    print("Deriving story identifiers (topics + TF-IDF clusters)...")
    story_series = news["topics"].progress_apply(_normalise_story_id)
    if story_series.isna().any():
        story_series = story_series.fillna(news["article_id"])
    story_series = story_series.astype(str)
    news["story_topics"] = story_series
    news["story_dataset"] = story_series

    cluster_map = compute_story_clusters(
        news,
        STORY_SIMILARITY_THRESHOLDS,
        time_window_days=3,
        vectorizer_kwargs={"stop_words": "danish"},
    )
    default_column = story_column_name(0.3)
    if cluster_map:
        for threshold, labels in cluster_map.items():
            column_name = story_column_name(threshold)
            news[column_name] = labels.astype(str)
        if default_column in news.columns:
            news["story"] = news[default_column].astype(str)
        else:
            first_series = next(iter(cluster_map.values()))
            news["story"] = first_series.astype(str)
    else:
        for threshold in STORY_SIMILARITY_THRESHOLDS:
            column_name = story_column_name(threshold)
            news[column_name] = story_series
        news["story"] = story_series

    # Persist the processed catalogue.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    news.to_pickle(output_path)
    print(f"Saved preprocessed articles to {output_path}")

    if processed_cache:
        print(f"Building article representation cache at {processed_cache}...")
        cache_source = news.set_index("article_id")
        ensure_article_cache(
            cache_source,
            model_name,
            processed_cache,
            build_if_missing=True,
            overwrite=True,
        )
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess EBNeRD news articles.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ebnerd/train/articles.parquet"),
        help="Path to the raw EBNeRD article parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ebnerd/articles_ebnerd.pickle"),
        help="Destination pickle path for preprocessed EBNeRD articles.",
    )
    parser.add_argument(
        "--processed-cache",
        type=Path,
        default=Path("data/ebnerd/articles_ebnerd_processed.pkl"),
        help="Destination cache for EBNeRD embeddings and TF-IDF features.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model used for article embeddings.",
    )
    parser.add_argument(
        "--danish-swn-path",
        type=Path,
        default=Path("data/ebnerd/Danish_sentiwordnet.txt"),
        help="Path to the Danish SentiWordNet lexicon for lexical sentiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_ebnerd_news(
        input_path=args.input,
        output_path=args.output,
        processed_cache=args.processed_cache,
        model_name=args.model_name,
        danish_swn_path=args.danish_swn_path,
    )


if __name__ == "__main__":
    main()
