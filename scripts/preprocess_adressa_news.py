"""Preprocess Adressa news articles to align with the RADio pipeline outputs."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from sentiment_models import norwegian_swn_sentiment, TransformerSentimentScorer
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
            "Transformer sentiment scoring failed for Adressa. "
            "Ensure torch, transformers, sentencepiece, and tiktoken are installed, "
            "then rerun preprocess_adressa_news.py."
        ) from exc

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


def load_news_tables(root: Path, splits: Iterable[str]) -> pd.DataFrame:
    """Load Adressa news TSV files for the requested splits."""
    frames: List[pd.DataFrame] = []
    for split in splits:
        news_path = root / split / "news.tsv"
        if not news_path.exists():
            warnings.warn(f"Skipping missing news file: {news_path}")
            continue
        df = pd.read_table(
            news_path,
            header=None,
            names=NEWS_COLUMNS,
            quoting=3,
            on_bad_lines="skip",
        )
        df["split_name"] = split
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No news.tsv files found for the provided splits.")

    news = pd.concat(frames, ignore_index=True)
    news = news.drop_duplicates(subset=["news_id"], keep="first")
    return news


def _extract_news_ids(raw: str) -> List[str]:
    """Split whitespace-separated news identifiers and strip labels."""
    if not raw or raw == "nan":
        return []
    tokens = raw.strip().split()
    cleaned: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token or token == "nan":
            continue
        if "-" in token:
            token = token.split("-", 1)[0]
        cleaned.append(token)
    return cleaned


def infer_publication_datetimes(root: Path, splits: Iterable[str]) -> Dict[str, pd.Timestamp]:
    """Infer article publication timestamps from behaviors logs when available."""
    earliest: Dict[str, pd.Timestamp] = {}

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue

        behaviour_paths = sorted(path for path in split_dir.glob("behaviors*.tsv") if path.is_file())
        if not behaviour_paths:
            continue

        for behaviour_path in behaviour_paths:
            with behaviour_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 4:
                        continue

                    timestamp: Optional[pd.Timestamp] = None
                    raw_timestamp = parts[2].strip()
                    if raw_timestamp:
                        if raw_timestamp.isdigit():
                            try:
                                timestamp = pd.to_datetime(int(raw_timestamp), unit="s", errors="coerce")
                            except (OverflowError, ValueError):
                                timestamp = None
                        if timestamp is None:
                            timestamp = pd.to_datetime(raw_timestamp, errors="coerce")
                    if timestamp is None or pd.isna(timestamp):
                        continue

                    candidate_fields: List[str] = []
                    if len(parts) >= 4:
                        candidate_fields.append(parts[3])
                    if len(parts) >= 5:
                        candidate_fields.append(parts[4])
                    if len(parts) >= 6:
                        candidate_fields.append(parts[5])

                    for field in candidate_fields:
                        for news_id in _extract_news_ids(field):
                            previous = earliest.get(news_id)
                            if previous is None or timestamp < previous:
                                earliest[news_id] = timestamp

    return earliest


def preprocess_adressa_news(
    adressa_root: Path,
    output_path: Path,
    splits: Iterable[str],
    *,
    processed_cache: Optional[Path],
    model_name: str,
) -> None:
    """Load, enrich, and persist the Adressa article catalogue."""
    news = load_news_tables(adressa_root, splits)
    news = news.drop(columns=["title_entities", "abstract_entities"], errors="ignore")

    publication_times = infer_publication_datetimes(adressa_root, splits)
    news["publication_datetime"] = pd.to_datetime(
        news["news_id"].map(publication_times), errors="coerce"
    )
    news["publication_date"] = news["publication_datetime"].dt.normalize()

    news["article_id"] = news["news_id"]

    for field in ("category", "subcategory"):
        news[field] = news[field].fillna("").astype(str).str.strip()
        news.loc[news[field] == "", field] = "other"

    news["title"] = news["title"].fillna("").astype(str).str.strip()
    news["abstract"] = news["abstract"].fillna("").astype(str).str.strip()
    news["url"] = news["url"].fillna("").astype(str).str.strip()

    news["text_title"] = news["title"]
    news["text_abstract"] = news["abstract"]
    news["text_title_abstract"] = (news["text_title"] + " " + news["text_abstract"]).str.strip()
    news["text"] = news["text_title_abstract"]

    empty_entities = [[] for _ in range(len(news))]
    news["entities"] = empty_entities
    news["entities_base"] = [list(item) for item in empty_entities]
    news["enriched_entities"] = [[] for _ in range(len(news))]

    lexicon_path = adressa_root / "Norwegian_sentiwordnet.txt"
    if not lexicon_path.exists():
        raise FileNotFoundError(
            f"Norwegian SentiWordNet file not found at {lexicon_path}. "
            "Download or place the lexicon before running preprocessing."
        )

    print("Computing lexicon sentiment scores (Norwegian SentiWordNet)...")
    news["sentiment_lexicon"] = news["text"].progress_apply(
        lambda text: norwegian_swn_sentiment(text, lexicon_path=lexicon_path)
    )

    print("Computing transformer sentiment scores (cardiffnlp/twitter-xlm-roberta-base-sentiment)...")
    transformer_model = _build_transformer_scorer("adressa")
    news["sentiment_transformer"] = transformer_model.score_many(news["text"].tolist())

    print("Computing complexity scores (LIX & MTLD)...")
    news["complexity_lix"] = news["text"].progress_apply(lix_score)
    news["complexity_mtld"] = news["text"].progress_apply(mtld_score)
    news["complexity_readability"] = news["complexity_lix"]
    news["complexity"] = news["complexity_readability"]

    print("Assigning story clusters...")
    cluster_map = compute_story_clusters(
        news,
        STORY_SIMILARITY_THRESHOLDS,
        time_window_days=3,
    )
    if cluster_map:
        for threshold, labels in cluster_map.items():
            column_name = story_column_name(threshold)
            news[column_name] = labels.astype(str)
        default_column = story_column_name(0.3)
        if default_column in news.columns:
            news["story"] = news[default_column]
        else:
            first_series = next(iter(cluster_map.values()))
            news["story"] = first_series.astype(str)
    else:
        news["story"] = news.index.astype(str)

    news = news.set_index("article_id", drop=True)
    news = news.sort_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    news.to_pickle(output_path)
    print(f"Saved preprocessed articles to {output_path}")

    if processed_cache:
        print(f"Building article representation cache at {processed_cache}...")
        ensure_article_cache(
            news,
            model_name,
            processed_cache,
            build_if_missing=True,
            overwrite=True,
        )
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Adressa news articles.")
    parser.add_argument(
        "--adressa-root",
        type=Path,
        default=Path("data/adressa"),
        help="Root directory containing Adressa splits (default: data/adressa).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Adressa splits to include (default: train val test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/adressa/articles_adressa.pickle"),
        help="Destination pickle path for preprocessed Adressa articles.",
    )
    parser.add_argument(
        "--processed-cache",
        type=Path,
        default=Path("data/adressa/articles_adressa_processed.pkl"),
        help="Destination cache for Adressa embeddings and TF-IDF features.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model used for article embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_adressa_news(
        adressa_root=args.adressa_root,
        output_path=args.output,
        splits=args.splits,
        processed_cache=args.processed_cache,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
