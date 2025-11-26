"""Shared helpers for building TF-IDF based story clusters."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
try:
    from scipy.sparse import csr_matrix  # type: ignore
    from scipy.sparse.csgraph import connected_components  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except ImportError:  # pragma: no cover
    csr_matrix = None  # type: ignore[misc,assignment]
    connected_components = None  # type: ignore[misc,assignment]
    TfidfVectorizer = None  # type: ignore[misc,assignment]
    NearestNeighbors = None  # type: ignore[misc,assignment]


def compute_story_clusters(
    news: pd.DataFrame,
    similarity_thresholds: Iterable[float],
    *,
    time_window_days: Optional[int] = 3,
    text_column: str = "text",
    date_columns: Sequence[str] = ("publication_datetime", "publication_date"),
    vectorizer_kwargs: Optional[dict] = None,
) -> Dict[float, pd.Series]:
    """
    Cluster articles into stories via TF-IDF cosine similarity and connected components.

    Args:
        news: DataFrame with article metadata and text.
        similarity_thresholds: Cosine similarity cutoffs (0-1).
        time_window_days: Maximum gap between linked articles; None disables filtering.
        text_column: Column containing text used for clustering (default: combined title/subtitle/body).
        date_columns: Ordered list of columns used for publication timestamps.
        vectorizer_kwargs: Optional overrides passed to TfidfVectorizer.
    """
    text_series = news[text_column].fillna("").astype(str)
    thresholds = sorted({round(thr, 5) for thr in similarity_thresholds if 0.0 < thr < 1.0})
    if not thresholds:
        return {}
    if text_series.empty:
        basis = text_series.str.lower().str.replace(r"\s+", " ", regex=True)
        labels, _ = pd.factorize(basis)
        fallback = pd.Series(labels, index=news.index, dtype="int64")
        return {thr: fallback.copy(deep=True) for thr in thresholds}

    date_series: Optional[pd.Series] = None
    time_window: Optional[pd.Timedelta] = None
    if time_window_days is not None:
        time_window = pd.Timedelta(days=time_window_days)
        for column in date_columns:
            if column in news.columns:
                date_series = pd.to_datetime(news[column], errors="coerce")
                break

    if (
        TfidfVectorizer is None
        or NearestNeighbors is None
        or connected_components is None
        or csr_matrix is None
    ):
        basis = text_series.str.lower().str.replace(r"\s+", " ", regex=True)
        labels, _ = pd.factorize(basis)
        fallback = pd.Series(labels, index=news.index, dtype="int64")
        return {thr: fallback.copy(deep=True) for thr in thresholds}

    base_vectorizer_kwargs = {
        "lowercase": True,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_features": 20000,
    }
    if vectorizer_kwargs:
        base_vectorizer_kwargs.update(vectorizer_kwargs)
    vectorizer = TfidfVectorizer(**base_vectorizer_kwargs)
    try:
        tfidf_matrix = vectorizer.fit_transform(text_series)
    except ValueError:
        basis = text_series.str.lower().str.replace(r"\s+", " ", regex=True)
        labels, _ = pd.factorize(basis)
        fallback = pd.Series(labels, index=news.index, dtype="int64")
        return {thr: fallback.copy(deep=True) for thr in thresholds}

    distance_thresholds = {thr: max(1.0 - thr, 0.0) for thr in thresholds}
    max_radius = max(distance_thresholds.values())
    nn = NearestNeighbors(metric="cosine", radius=max_radius)
    nn.fit(tfidf_matrix)
    neighbor_distances, neighbor_indices = nn.radius_neighbors(tfidf_matrix, return_distance=True)

    rows_dict = {thr: [] for thr in thresholds}
    cols_dict = {thr: [] for thr in thresholds}
    data_dict = {thr: [] for thr in thresholds}
    for row_idx, (distances, indices) in enumerate(zip(neighbor_distances, neighbor_indices)):
        for dist, col_idx in zip(distances, indices):
            if row_idx == col_idx:
                continue
            if date_series is not None and time_window is not None:
                date_i = date_series.iat[row_idx]
                date_j = date_series.iat[col_idx]
                if pd.notna(date_i) and pd.notna(date_j):
                    if abs(date_i - date_j) > time_window:
                        continue
            for thr, radius in distance_thresholds.items():
                if dist <= radius + 1e-9:
                    rows_dict[thr].append(row_idx)
                    cols_dict[thr].append(col_idx)
                    data_dict[thr].append(1)

    basis = text_series.str.lower().str.replace(r"\s+", " ", regex=True)
    hash_labels, _ = pd.factorize(basis)
    fallback = pd.Series(hash_labels, index=news.index, dtype="int64")

    n_docs = len(text_series)
    clusters: Dict[float, pd.Series] = {}
    for thr in thresholds:
        rows = rows_dict[thr]
        if not rows:
            clusters[thr] = fallback.copy(deep=True)
            continue
        adjacency = csr_matrix((data_dict[thr], (rows, cols_dict[thr])), shape=(n_docs, n_docs))
        adjacency = adjacency.maximum(adjacency.transpose())
        _, labels = connected_components(csgraph=adjacency, directed=False, return_labels=True)
        clusters[thr] = pd.Series(labels, index=news.index, dtype="int64")

    return clusters


__all__ = ["compute_story_clusters"]
