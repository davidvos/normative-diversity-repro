"""Utility functions for the recommendation system."""

import glob
import math
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import dart.metrics.activation
import dart.metrics.calibration
import dart.metrics.fragmentation
import dart.metrics.representation
import dart.metrics.alternative_voices

from intra_list_diversity import ILD
from gini_coefficient import GiniCoefficient
from config import NDCG_CUTOFF

EPSILON = 1e-12
STORY_SIMILARITY_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def story_column_name(threshold: float) -> str:
    """Return the column name used for a specific similarity threshold."""
    scaled = int(round(threshold * 100))
    return f"story_sim_{scaled:02d}"

def process_candidates(candidates):
    """Process candidate strings to extract article IDs."""
    return [candidate.split('-')[0] for candidate in candidates]

def process_labels(candidates):
    """Process candidate strings to extract relevance labels."""
    return [int(candidate.split('-')[1]) for candidate in candidates]

def get_articles(article_ids, articles):
    """Get article objects for given article IDs."""
    filtered_articles = [articles[article_id] for article_id in article_ids if article_id in articles]
    return filtered_articles


def attach_article_metadata_field(
    articles: Dict,
    data: Optional[pd.Series],
    target_key: str,
) -> None:
    """Attach a column from the article DataFrame to the cached article dictionary."""
    if not isinstance(articles, dict) or data is None:
        return
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    for article_id, value in data.items():
        entry = articles.get(article_id)
        key = article_id
        if entry is None:
            str_id = str(article_id)
            entry = articles.get(str_id)
            key = str_id if entry is not None else None
        if entry is None or key is None:
            continue
        entry[target_key] = value
        articles[key] = entry


def select_complexity_series(
    articles_df: pd.DataFrame,
    source: str,
    dataset: str,
) -> Tuple[str, pd.Series]:
    """Select the appropriate complexity column from the article catalogue."""
    if not isinstance(articles_df, pd.DataFrame):
        raise ValueError("articles_df must be a pandas DataFrame.")

    available = set(articles_df.columns)
    preferred_column: Optional[str]

    if source in {"auto", "readability"}:
        preferred_column = "complexity_readability"
    elif source == "mtld":
        preferred_column = "complexity_mtld"
    else:
        preferred_column = source

    if preferred_column not in available:
        fallback = "complexity" if "complexity" in available else None
        if fallback is None:
            raise KeyError(
                f"Column '{preferred_column}' not found for dataset '{dataset}'. "
                "Re-run preprocessing to generate the requested complexity scores."
            )
        preferred_column = fallback

    return preferred_column, articles_df[preferred_column]


def select_sentiment_series(
    articles_df: pd.DataFrame,
    source: str,
    dataset: str,
) -> Tuple[str, pd.Series]:
    """Select the sentiment column corresponding to the requested source."""
    if not isinstance(articles_df, pd.DataFrame):
        raise ValueError("articles_df must be a pandas DataFrame.")

    source_map = {
        "lexicon": "sentiment_lexicon",
        "transformer": "sentiment_transformer",
        "dataset": "sentiment_dataset",
    }
    column = source_map.get(source)
    if column is None:
        raise ValueError(f"Unsupported sentiment source '{source}'.")

    if column not in articles_df.columns:
        raise KeyError(
            f"Column '{column}' not found for dataset '{dataset}'. "
            "Re-run preprocessing to generate it."
        )

    return column, articles_df[column]


def _article_text(row: Dict) -> str:
    """Combine title and abstract text for embedding generation."""
    title = row.get("title", "") or ""
    abstract = row.get("abstract", "") or ""
    return f"{title} {abstract}".strip()


def build_article_representations(articles: pd.DataFrame, model_name: str) -> Dict:
    """
    Compute sentence-transformer and TF-IDF representations for each article.

    Returns a dictionary keyed by article_id with the new representations attached.
    """
    articles_dict: Dict = articles.to_dict(orient="index")
    if not articles_dict:
        return {}

    article_ids = list(articles_dict.keys())
    texts = [_article_text(articles_dict[article_id]) for article_id in article_ids]

    st_model = SentenceTransformer(model_name)
    st_embeddings = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    for idx, article_id in enumerate(article_ids):
        payload = articles_dict[article_id]
        payload["st_vector"] = st_embeddings[idx]
        payload["tfidf_vector"] = tfidf_matrix[idx]

    return articles_dict


def load_article_cache(cache_path: Path) -> Dict:
    """Load a serialized article representation cache."""
    with open(cache_path, "rb") as cache_file:
        return pickle.load(cache_file)


def ensure_article_cache(
    articles: pd.DataFrame,
    model_name: str,
    cache_path: Optional[Path],
    *,
    build_if_missing: bool = True,
    overwrite: bool = False,
) -> Dict:
    """
    Ensure an article representation cache exists and return it.

    If `cache_path` is provided and exists, it is loaded. When `build_if_missing`
    is True the cache will be generated (and optionally overwritten) using the
    supplied articles; otherwise a missing cache raises FileNotFoundError.
    """
    if cache_path is None:
        if not build_if_missing:
            raise ValueError("A cache path is required when build_if_missing=False.")
        return build_article_representations(articles, model_name)

    cache_path = Path(cache_path)
    cache_exists = cache_path.exists()

    if cache_exists and not overwrite:
        return load_article_cache(cache_path)

    if not build_if_missing and not cache_exists:
        raise FileNotFoundError(
            f"Article cache not found at {cache_path}. Run the preprocessing "
            "pipeline to generate it."
        )

    representations = build_article_representations(articles, model_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as cache_file:
        pickle.dump(representations, cache_file)

    return representations


def process_articles(articles, model_name, cache_path=None):
    """Process articles to add embeddings and TF-IDF vectors with optional caching."""
    return ensure_article_cache(
        articles,
        model_name,
        cache_path,
        build_if_missing=True,
        overwrite=False,
    )

def ranking_to_scores(ranking):
    """Convert a ranking list (1-indexed) into a score vector."""
    n = len(ranking)
    scores = [0] * n
    for rank, candidate in enumerate(ranking):
        scores[candidate - 1] = n - rank
    return scores


def _dcg_from_relevance(relevance: Sequence[int], cutoff: int) -> float:
    """Compute DCG for a relevance list."""
    dcg = 0.0
    for position, rel in enumerate(relevance[:cutoff]):
        gain = (2.0 ** rel) - 1.0
        discount = 1.0 / math.log2(position + 2.0)
        dcg += gain * discount
    return dcg


def indices_to_pred_rank(order: Sequence[int], total_candidates: int) -> List[int]:
    """Convert an ordered list of candidate indices into a 1-indexed rank vector."""
    pred_rank = [0] * total_candidates
    for position, candidate_idx in enumerate(order, start=1):
        pred_rank[candidate_idx] = position
    return pred_rank


def compute_ndcg_from_rank(
    pred_rank: Sequence[int],
    dataset: str,
    gt_relevance_scores: Sequence[int],
) -> float:
    """Compute NDCG for a given ranking without relying on sklearn."""
    if not pred_rank or not gt_relevance_scores:
        return 0.0

    num_candidates = len(gt_relevance_scores)
    usable = min(len(pred_rank), num_candidates)
    if usable == 0:
        return 0.0

    cutoff = min(NDCG_CUTOFF, usable)
    sentinel = float(num_candidates + usable + 1)
    try:
        sanitized = []
        for idx in range(usable):
            raw_rank = pred_rank[idx]
            try:
                value = float(raw_rank)
                if not math.isfinite(value) or value <= 0:
                    value = sentinel + idx
            except (TypeError, ValueError):
                value = sentinel + idx
            sanitized.append(value)
    except Exception:
        sanitized = [sentinel + idx for idx in range(usable)]

    ordered_indices = sorted(range(usable), key=lambda idx: (sanitized[idx], idx))[:cutoff]
    if not ordered_indices:
        return 0.0

    ranked_relevance = [gt_relevance_scores[idx] for idx in ordered_indices]
    dcg = _dcg_from_relevance(ranked_relevance, cutoff)

    ideal_relevance = sorted(gt_relevance_scores[:usable], reverse=True)
    idcg = _dcg_from_relevance(ideal_relevance, cutoff)
    if idcg <= EPSILON:
        return 0.0

    return float(dcg / idcg)


def normalize_metric_gain(
    metric_value: Optional[float],
    baseline_metric: Optional[float],
    target_metric: Optional[float],
) -> Optional[float]:
    """Normalize metric improvement into [0, 1]."""
    if metric_value is None or baseline_metric is None:
        return None

    effective_target = target_metric if target_metric is not None else baseline_metric
    effective_target = max(effective_target, baseline_metric)
    denom = effective_target - baseline_metric

    if denom <= EPSILON:
        return 1.0 if metric_value >= baseline_metric else 0.0

    gain = (metric_value - baseline_metric) / denom
    return float(np.clip(gain, 0.0, 1.0))


def normalize_ndcg(ndcg_value: Optional[float], baseline_ndcg: Optional[float]) -> Optional[float]:
    """Normalize NDCG relative to the baseline into [0, 1]."""
    if ndcg_value is None or baseline_ndcg is None:
        return None

    if baseline_ndcg <= EPSILON:
        return 1.0 if ndcg_value >= baseline_ndcg else 0.0

    ratio = ndcg_value / baseline_ndcg
    return float(np.clip(ratio, 0.0, 1.0))


def greedy_optimize_tradeoff(
    metric_evaluator: Callable[[Sequence[int]], Optional[float]],
    num_candidates: int,
    baseline_order: Sequence[int],
    gt_relevance_scores: Sequence[int],
    dataset: str,
    cutoff: int,
    baseline_metric: Optional[float],
    baseline_ndcg: Optional[float],
    target_metric: Optional[float],
    lambda_weight: float,
    allowed_indices: Optional[Sequence[int]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Greedily optimise a trade-off between a RADio metric and NDCG."""
    if (
        num_candidates == 0
        or baseline_metric is None
        or baseline_ndcg is None
        or gt_relevance_scores is None
    ):
        return None, None

    cutoff = min(cutoff, num_candidates)
    if cutoff == 0:
        return None, None

    selected_indices: List[int] = []
    if allowed_indices is not None:
        remaining_indices = set(idx for idx in allowed_indices if 0 <= idx < num_candidates)
    else:
        remaining_indices = set(range(num_candidates))
    if not remaining_indices:
        return None, None
    baseline_order = list(baseline_order)

    for _ in range(cutoff):
        best_candidate_idx = None
        best_objective = None

        for candidate_idx in remaining_indices:
            tentative_selected = selected_indices + [candidate_idx]
            tentative_selected_set = set(tentative_selected)
            remaining_after_select = [
                idx for idx in baseline_order if idx not in tentative_selected_set
            ]
            full_order = tentative_selected + remaining_after_select

            metric_value = metric_evaluator(full_order[:cutoff])
            if metric_value is None:
                continue

            pred_rank = indices_to_pred_rank(full_order, num_candidates)
            ndcg_value = compute_ndcg_from_rank(pred_rank, dataset, gt_relevance_scores)

            metric_norm = normalize_metric_gain(metric_value, baseline_metric, target_metric)
            ndcg_norm = normalize_ndcg(ndcg_value, baseline_ndcg)
            if metric_norm is None or ndcg_norm is None:
                continue

            objective = lambda_weight * metric_norm + (1.0 - lambda_weight) * ndcg_norm

            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_candidate_idx = candidate_idx

        if best_candidate_idx is None:
            break

        selected_indices.append(best_candidate_idx)
        remaining_indices.remove(best_candidate_idx)

    if not selected_indices:
        return None, None

    selected_set = set(selected_indices)
    remaining_after_select = [idx for idx in baseline_order if idx not in selected_set]
    final_order = selected_indices + remaining_after_select

    final_metric = metric_evaluator(final_order[:cutoff])
    pred_rank = indices_to_pred_rank(final_order, num_candidates)
    final_ndcg = compute_ndcg_from_rank(pred_rank, dataset, gt_relevance_scores)

    return final_metric, final_ndcg

def load_dataset(dataset, config):
    """Load dataset-specific behaviors and articles."""
    dataset_config = config[dataset]
    
    if dataset == 'mind':
        behavior_path = Path(dataset_config['behaviors_path'])
        if not behavior_path.exists():
            fallback = Path('data/mind/val/behaviors.tsv')
            if fallback.exists():
                behavior_path = fallback
            else:
                raise FileNotFoundError(f"No behaviors file found for MIND. Tried {behavior_path} and {fallback}.")
        behaviors = pd.read_csv(behavior_path, delimiter='\t', header=None)
        behaviors = behaviors.replace({np.nan: None})
        articles = pd.read_pickle(dataset_config['articles_path'])
        articles['title'] = articles['title'].fillna('')
        articles['abstract'] = articles['abstract'].fillna('')

    elif dataset == 'ebnerd':
        behaviors = pd.read_parquet(dataset_config['behaviors_path'])
        behaviors = behaviors.rename(columns={
            'user_id': 'user',
            'impression_time': 'time',
            'article_ids_clicked': 'clicked_news',
            'article_ids_inview': 'impressions'
        })
        behaviors['clicked_news'] = behaviors['clicked_news'].apply(lambda x: " ".join(map(str, x)) if isinstance(x, (list, np.ndarray)) else "")
        behaviors['impressions'] = behaviors['impressions'].apply(lambda x: " ".join(map(str, x)) if isinstance(x, (list, np.ndarray)) else "")
        behaviors = behaviors[['impression_id', 'user', 'time', 'clicked_news', 'impressions']]
        behaviors = behaviors.replace({np.nan: None})

        articles = pd.read_pickle(dataset_config['articles_path'])
        if 'article_id' in articles.columns:
            articles = articles.set_index('article_id')
        articles['title'] = articles['title'].fillna('').astype(str)

        if 'abstract' in articles.columns:
            abstract_series = articles['abstract']
        elif 'subtitle' in articles.columns:
            abstract_series = articles['subtitle']
        else:
            abstract_series = pd.Series('', index=articles.index, dtype=str)
        if not isinstance(abstract_series, pd.Series):
            abstract_series = pd.Series(abstract_series, index=articles.index, dtype=str)
        articles['abstract'] = abstract_series.fillna('').astype(str)

        if 'category' in articles.columns:
            category_series = articles['category']
        elif 'category_str' in articles.columns:
            category_series = articles['category_str']
        else:
            category_series = pd.Series('other', index=articles.index, dtype=str)
        if not isinstance(category_series, pd.Series):
            category_series = pd.Series(category_series, index=articles.index, dtype=str)
        articles['category'] = category_series.fillna('other').astype(str)

        if 'subcategory' not in articles.columns:
            articles['subcategory'] = 'other'
        articles['subcategory'] = articles['subcategory'].fillna('other').astype(str)

    elif dataset == 'adressa':
        behavior_pattern = dataset_config['behaviors_path']
        behavior_files = sorted(glob.glob(behavior_pattern))
        if not behavior_files:
            raise FileNotFoundError(f"No behavior files found for pattern: {behavior_pattern}")
        frames = [
            pd.read_table(path, header=None, usecols=[0, 1, 2, 3, 4])
            for path in behavior_files
        ]
        behaviors = pd.concat(frames, ignore_index=True)
        # Adressa splits reuse impression ids per shard; reassign to a global sequence to align with predictions.
        behaviors[0] = np.arange(1, len(behaviors) + 1)
        behaviors = behaviors.replace({np.nan: None})

        articles = pd.read_pickle(dataset_config['articles_path'])
        if 'article_id' in articles.columns:
            articles = articles.set_index('article_id')
        articles['title'] = articles['title'].fillna('').astype(str)

        if 'abstract' in articles.columns:
            abstract_series = articles['abstract']
        else:
            abstract_series = pd.Series('', index=articles.index, dtype=str)
        if not isinstance(abstract_series, pd.Series):
            abstract_series = pd.Series(abstract_series, index=articles.index, dtype=str)
        articles['abstract'] = abstract_series.fillna('').astype(str)

        if 'category' not in articles.columns:
            articles['category'] = 'other'
        articles['category'] = articles['category'].fillna('other').astype(str)

        if 'subcategory' not in articles.columns:
            articles['subcategory'] = 'other'
        articles['subcategory'] = articles['subcategory'].fillna('other').astype(str)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return behaviors, articles

def greedy_optimize_metric(
    num_candidates: int,
    cutoff: int,
    evaluate_fn: Callable[[Sequence[int]], Optional[float]],
    maximize: bool = True
) -> Tuple[Optional[float], List[int]]:
    """
    Greedily select candidate indices to optimise an arbitrary metric.

    Args:
        num_candidates: Total number of candidate items.
        cutoff: Maximum number of items to select.
        evaluate_fn: Function returning a metric value for the provided order
                     of candidate indices.
        maximize: Whether to maximize (True) or minimize (False) the metric.

    Returns:
        Tuple of (metric score for the selected set, list of selected indices).
    """
    if num_candidates == 0 or cutoff <= 0:
        return None, []

    cutoff = min(cutoff, num_candidates)
    selected: List[int] = []
    remaining = set(range(num_candidates))

    for _ in range(cutoff):
        best_idx = None
        best_score = None

        for candidate_idx in remaining:
            tentative = selected + [candidate_idx]
            score = evaluate_fn(tentative)
            if score is None:
                continue

            if best_idx is None:
                best_idx = candidate_idx
                best_score = score
                continue

            if maximize and score > best_score:
                best_idx = candidate_idx
                best_score = score
            elif not maximize and score < best_score:
                best_idx = candidate_idx
                best_score = score

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    if not selected:
        return None, []

    final_score = evaluate_fn(selected)
    return final_score, selected


def greedy_optimize_topic_calibration(user_history_articles, candidate_articles, calibration_metric, maximize=True, cutoff=10):
    """
    Greedily re-rank candidate articles to optimize Topic Calibration.
    
    Args:
        user_history_articles: List of articles in user history
        candidate_articles: List of candidate articles to rank
        calibration_metric: Calibration metric object to use for calculations
        maximize: If True, maximize divergence; if False, minimize divergence
        cutoff: Number of articles to include in the recommendation
        
    Returns:
        A tuple containing (optimal calibration score, list of selected candidate indices)
    """
    if len(candidate_articles) == 0 or len(user_history_articles) == 0:
        return None, []

    def evaluate(indices: Sequence[int]) -> Optional[float]:
        articles = [candidate_articles[idx] for idx in indices]
        if not articles:
            return None
        topic_divergence, _ = calibration_metric.calculate(user_history_articles, articles)
        if not topic_divergence:
            return None
        return topic_divergence[0][1]

    return greedy_optimize_metric(len(candidate_articles), cutoff, evaluate, maximize=maximize)

def initialize_metrics(dataset: str = "mind"):
    """Initialize all metric calculators with dataset-aware configuration."""
    dataset_locales = {
        "mind": {"language": "english", "country": "us"},
        "ebnerd": {"language": "danish", "country": "dk"},
        "adressa": {"language": "norwegian", "country": "no"},
    }
    locale = dataset_locales.get(dataset, dataset_locales["mind"])
    # TextStat currently supports only english, dutch, and german. Fallback to english when unavailable.
    calibration_language = locale["language"] if locale["language"] in {"english", "dutch", "german"} else "english"
    calibration_locale = {"language": calibration_language, "country": locale["country"]}

    return {
        'calibration': dart.metrics.calibration.Calibration(calibration_locale),
        'fragmentation': dart.metrics.fragmentation.Fragmentation(),
        'activation': dart.metrics.activation.Activation(locale),
        'representation': dart.metrics.representation.Representation({'language': 'english', 'country': 'us'}),
        'alternative_voices': dart.metrics.alternative_voices.AlternativeVoices(),
        'ild': ILD(),
        'gini': GiniCoefficient()
    }
