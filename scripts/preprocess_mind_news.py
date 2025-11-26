"""Preprocess MIND news articles into the format expected by RADio metrics."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from sentiment_models import textblob_sentiment, TransformerSentimentScorer
from story_clustering import compute_story_clusters
from text_complexity import flesch_kincaid_score, mtld_score
from utils import (
    STORY_SIMILARITY_THRESHOLDS,
    ensure_article_cache,
    story_column_name,
)

try:
    from dart.handler.other.textstat import TextStatHandler
except ImportError:  # pragma: no cover
    TextStatHandler = None  # type: ignore[misc,assignment]

tqdm.pandas()

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


def parse_entities(raw: str, source: str) -> List[dict]:
    """Parse the JSON entity payload that ships with MIND."""
    if not raw or raw == "[]":
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict):
        payload = [payload]

    parsed: List[dict] = []
    for entity in payload:
        if not isinstance(entity, dict):
            continue

        label = str(entity.get("Type", "")).upper()
        wikidata_id = entity.get("WikidataId") or None
        confidence = entity.get("Confidence")
        surfaces = entity.get("SurfaceForms", []) or []
        if isinstance(surfaces, str):
            try:
                surfaces = json.loads(surfaces)
            except json.JSONDecodeError:
                surfaces = []

        if not isinstance(surfaces, list):
            surfaces = [surfaces]

        for surface in surfaces:
            text = ""
            offset = 0
            if isinstance(surface, dict):
                text = surface.get("SurfaceForm", "") or ""
                offset = surface.get("Offset", 0) or 0
            elif isinstance(surface, str):
                text = surface

            if not text:
                continue

            parsed.append(
                {
                    "label": label,
                    "text": text,
                    "spans": [offset],
                    "source": source,
                    "confidence": confidence,
                    "wikidata_id": wikidata_id,
                }
            )
    return parsed


def merge_entities(row: pd.Series) -> List[dict]:
    """Combine title and abstract entities into a single list."""
    title_entities = parse_entities(row.get("title_entities", ""), "title")
    abstract_entities = parse_entities(row.get("abstract_entities", ""), "abstract")
    return title_entities + abstract_entities




def load_news_tables(root: Path, splits: Iterable[str]) -> pd.DataFrame:
    """Load MIND news TSV files for the requested splits."""
    frames: List[pd.DataFrame] = []
    for split in splits:
        news_path = root / split / "news.tsv"
        if not news_path.exists():
            raise FileNotFoundError(f"Missing news file: {news_path}")
        df = pd.read_table(news_path, header=None, names=NEWS_COLUMNS, quoting=3, on_bad_lines="skip")
        df["split_name"] = split
        frames.append(df)

    news = pd.concat(frames, ignore_index=True)
    news = news.drop_duplicates(subset=["news_id"], keep="first")
    return news


def infer_publication_datetimes(root: Path, splits: Iterable[str]) -> Dict[str, pd.Timestamp]:
    """Infer article publication timestamps from behaviors logs."""
    earliest: Dict[str, datetime] = {}
    time_format = "%m/%d/%Y %I:%M:%S %p"

    for split in splits:
        behaviors_path = root / split / "behaviors.tsv"
        if not behaviors_path.exists():
            warnings.warn(f"Skipping missing behaviors file: {behaviors_path}")
            continue

        with behaviors_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue

                raw_timestamp = parts[2].strip()
                try:
                    timestamp = datetime.strptime(raw_timestamp, time_format)
                except ValueError:
                    continue

                history_raw = parts[3].strip()
                impressions_raw = parts[4].strip()

                if history_raw and history_raw != "nan":
                    for news_id in history_raw.split(" "):
                        if news_id and news_id != "nan":
                            previous = earliest.get(news_id)
                            if previous is None or timestamp < previous:
                                earliest[news_id] = timestamp

                if impressions_raw and impressions_raw != "nan":
                    for impression in impressions_raw.split(" "):
                        if not impression:
                            continue
                        news_id = impression.split("-", 1)[0]
                        if not news_id or news_id == "nan":
                            continue
                        previous = earliest.get(news_id)
                        if previous is None or timestamp < previous:
                            earliest[news_id] = timestamp

    return {news_id: pd.Timestamp(ts) for news_id, ts in earliest.items()}




def preprocess_mind_news(
    mind_root: Path,
    output_path: Path,
    splits: Iterable[str],
    *,
    processed_cache: Optional[Path],
    model_name: str,
) -> None:
    """Load, enrich, and persist the MIND article catalogue."""
    complexity_handler: Optional["TextStatHandler"] = None

    if TextStatHandler is not None:
        try:
            complexity_handler = TextStatHandler("english")
        except Exception:
            warnings.warn("Failed to initialise TextStatHandler; complexity scores will be 0.0.")

    news = load_news_tables(mind_root, splits)

    publication_times = infer_publication_datetimes(mind_root, splits)
    news["publication_datetime"] = pd.to_datetime(
        news["news_id"].map(publication_times), errors="coerce"
    )
    news["publication_date"] = news["publication_datetime"].dt.normalize()

    news["title"] = news["title"].fillna("")
    news["abstract"] = news["abstract"].fillna("")
    news["category"] = news["category"].fillna("other")
    news["subcategory"] = news["subcategory"].fillna("other")

    news["article_id"] = news["news_id"]
    news["text_title"] = news["title"].str.strip()
    news["text_abstract"] = news["abstract"].str.strip()
    news["text_title_abstract"] = (
        news["text_title"].fillna("") + " " + news["text_abstract"].fillna("")
    ).str.strip()
    news["text"] = news["text_title_abstract"]

    print("Parsing entity annotations...")
    news["entities"] = news.progress_apply(merge_entities, axis=1)
    news["entities_base"] = news["entities"]
    news["enriched_entities"] = [[] for _ in range(len(news))]

    print("Computing lexicon sentiment scores (TextBlob)...")
    news["sentiment_lexicon"] = news["text"].progress_apply(textblob_sentiment)

    print("Computing transformer sentiment scores (cardiffnlp/twitter-xlm-roberta-base-sentiment)...")
    transformer_model = TransformerSentimentScorer()
    news["sentiment_transformer"] = transformer_model.score_many(news["text"].tolist())

    print("Computing complexity scores (Flesch–Kincaid & MTLD)...")
    news["complexity_flesch_kincaid"] = news["text"].progress_apply(
        lambda text: flesch_kincaid_score(text, scorer=complexity_handler)
    )
    news["complexity_mtld"] = news["text"].progress_apply(mtld_score)
    news["complexity_readability"] = news["complexity_flesch_kincaid"]
    news["complexity"] = news["complexity_readability"]

    print("Assigning story clusters...")
    cluster_map = compute_story_clusters(
        news,
        STORY_SIMILARITY_THRESHOLDS,
        time_window_days=3,
        vectorizer_kwargs={"stop_words": "english"},
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MIND news articles.")
    parser.add_argument(
        "--mind-root",
        type=Path,
        default=Path("data/mind"),
        help="Root directory containing MIND splits (default: data/mind).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        help="MIND splits to include (default: val).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mind/articles_mind.pickle"),
        help="Destination pickle path.",
    )
    parser.add_argument(
        "--processed-cache",
        type=Path,
        default=Path("data/mind/articles_mind_processed.pkl"),
        help="Destination cache for embeddings and TF-IDF features.",
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
    preprocess_mind_news(
        args.mind_root,
        args.output,
        args.splits,
        processed_cache=args.processed_cache,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
