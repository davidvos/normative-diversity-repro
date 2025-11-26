"""
Reranking functions for balancing relevance and diversity in recommendations.

This module provides two main approaches:
1. rerank_production_topk: Production-ready reranker that infers relevance from baseline rankings
2. rerank_tradeoff_topk: Research-focused reranker that requires ground truth relevance labels

The production reranker is designed for real-world deployments where ground truth labels
are not available, while the tradeoff reranker is used for offline evaluation and research.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from dart.external.discount import harmonic_number
from utils import EPSILON

DEFAULT_SENTIMENT_FIELD = "__activation_sentiment"

def _get_topic_value(article: dict, attr_key: str) -> str:
    if not isinstance(article, dict):
        return "other"
    if attr_key == "category":
        value = article.get("category") or article.get("category_str") or "other"
    elif attr_key == "subcategory":
        value = article.get("subcategory") or article.get("sub_category") or "other"
    else:
        value = article.get(attr_key, "other")
    if value is None or value == "":
        value = "other"
    return str(value)


def _build_category_mapping(
    history_articles: Sequence[dict],
    candidate_articles: Sequence[dict],
    attr_key: str,
) -> Dict[str, int]:
    categories = {_get_topic_value(article, attr_key) for article in history_articles}
    categories.update(_get_topic_value(article, attr_key) for article in candidate_articles)
    categories.discard(None)
    categories_list = sorted(categories)
    return {category: idx for idx, category in enumerate(categories_list)}


def _discounted_topic_vector(
    articles: Sequence[dict],
    category_to_idx: Dict[str, int],
    attr_key: str,
) -> np.ndarray:
    n_articles = len(articles)
    num_categories = len(category_to_idx)
    if n_articles == 0 or num_categories == 0:
        return np.zeros(num_categories, dtype=np.float64)

    harmonic_norm = harmonic_number(n_articles)
    vector = np.zeros(num_categories, dtype=np.float64)
    for position, article in enumerate(articles, start=1):
        category_idx = category_to_idx.get(_get_topic_value(article, attr_key))
        if category_idx is None:
            continue
        weight = 1.0 / position / harmonic_norm
        vector[category_idx] += weight

    total = vector.sum()
    if total > 0:
        vector /= total
    return vector


def _compute_jsd_divergence(history_vec: np.ndarray, recommendation_vec: np.ndarray, alpha: float = 0.001) -> Optional[float]:
    if history_vec.size == 0 or recommendation_vec.size == 0:
        return None

    p = history_vec.astype(np.float64, copy=False)
    q = recommendation_vec.astype(np.float64, copy=False)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= EPSILON or q_sum <= EPSILON:
        return None

    p = p / p_sum
    q = q / q_sum

    p_smooth = (1.0 - alpha) * p + alpha * q
    q_smooth = (1.0 - alpha) * q + alpha * p

    p_smooth = np.clip(p_smooth, EPSILON, 1.0)
    q_smooth = np.clip(q_smooth, EPSILON, 1.0)
    m = 0.5 * (p_smooth + q_smooth)

    jsd = 0.5 * (
        np.sum(p_smooth * (np.log2(p_smooth) - np.log2(m)))
        + np.sum(q_smooth * (np.log2(q_smooth) - np.log2(m)))
    )
    jsd = max(jsd, 0.0)
    return float(np.sqrt(jsd))


def _make_divergence_evaluator(
    mode: str,
    candidate_articles: Sequence[dict],
    history_articles: Sequence[dict],
    allowed_set: Sequence[int],
    weight_lookup: Dict[int, np.ndarray],
    metrics,
    *,
    fragmentation_reference: Optional[Sequence[Sequence[dict]]] = None,
    sentiment_field: str = DEFAULT_SENTIMENT_FIELD,
):
    if mode == "activation":
        def _sent_val(article: dict) -> float:
            if not isinstance(article, dict):
                return 0.0
            value = article.get(sentiment_field)
            if value is None:
                value = article.get("sentiment")
            try:
                return abs(float(value))
            except (TypeError, ValueError):
                return 0.0

        hist_vals = np.array([_sent_val(a) for a in history_articles], dtype=np.float64).reshape(-1, 1)
        binner = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
        try:
            binner.fit(hist_vals)
        except ValueError:
            return None
        hist_bins = binner.transform(hist_vals).astype(int).flatten()
        cand_vals = np.array([_sent_val(a) for a in candidate_articles], dtype=np.float64)
        cand_bins_full = binner.transform(cand_vals.reshape(-1, 1)).astype(int).flatten()

        n_bins_raw = getattr(binner, "n_bins_", 5)
        num_bins = int(n_bins_raw[0] if isinstance(n_bins_raw, (list, tuple, np.ndarray)) else n_bins_raw)
        if num_bins <= 0:
            return None

        harmonic_norm = harmonic_number(len(hist_bins))
        if harmonic_norm <= EPSILON:
            return None
        hist_vec = np.zeros(num_bins, dtype=np.float64)
        for pos, b in enumerate(hist_bins, start=1):
            b = int(np.clip(b, 0, num_bins - 1))
            hist_vec[b] += 1.0 / pos / harmonic_norm
        total = hist_vec.sum()
        if total <= EPSILON:
            return None
        hist_vec /= total

        def _eval(order: Sequence[int]) -> Optional[float]:
            L = len(order)
            if L == 0:
                return None
            weights = weight_lookup[L][:, None]
            topic_vec = np.zeros(num_bins, dtype=np.float64)
            for i, idx in enumerate(order, start=1):
                bin_idx = int(np.clip(cand_bins_full[idx], 0, num_bins - 1))
                topic_vec[bin_idx] += float(weights[i - 1, 0])
            s = topic_vec.sum()
            if s > EPSILON:
                topic_vec /= s
            return _compute_jsd_divergence(hist_vec, topic_vec)

        return _eval

    if mode == "complexity":
        def _cpx(article: dict) -> float:
            if not isinstance(article, dict):
                return 0.0
            try:
                return float(article.get("complexity", 0.0))
            except (TypeError, ValueError):
                return 0.0

        hist_vals = np.array([_cpx(a) for a in history_articles], dtype=np.float64).reshape(-1, 1)
        binner = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
        try:
            binner.fit(hist_vals)
        except ValueError:
            return None
        hist_bins = binner.transform(hist_vals).astype(int).flatten()
        cand_vals = np.array([_cpx(a) for a in candidate_articles], dtype=np.float64)
        cand_bins_full = binner.transform(cand_vals.reshape(-1, 1)).astype(int).flatten()

        n_bins_raw = getattr(binner, "n_bins_", 5)
        num_bins = int(n_bins_raw[0] if isinstance(n_bins_raw, (list, tuple, np.ndarray)) else n_bins_raw)
        if num_bins <= 0:
            return None

        harmonic_norm = harmonic_number(len(hist_bins))
        if harmonic_norm <= EPSILON:
            return None
        hist_vec = np.zeros(num_bins, dtype=np.float64)
        for pos, b in enumerate(hist_bins, start=1):
            b = int(np.clip(b, 0, num_bins - 1))
            hist_vec[b] += 1.0 / pos / harmonic_norm
        total = hist_vec.sum()
        if total <= EPSILON:
            return None
        hist_vec /= total

        def _eval(order: Sequence[int]) -> Optional[float]:
            L = len(order)
            if L == 0:
                return None
            weights = weight_lookup[L][:, None]
            topic_vec = np.zeros(num_bins, dtype=np.float64)
            for i, idx in enumerate(order, start=1):
                bin_idx = int(np.clip(cand_bins_full[idx], 0, num_bins - 1))
                topic_vec[bin_idx] += float(weights[i - 1, 0])
            s = topic_vec.sum()
            if s > EPSILON:
                topic_vec /= s
            return _compute_jsd_divergence(hist_vec, topic_vec)

        return _eval

    if mode in {"category", "subcategory"}:
        attr_key = "category" if mode == "category" else "subcategory"
        candidate_subset = [candidate_articles[idx] for idx in allowed_set]
        category_to_idx = _build_category_mapping(history_articles, candidate_subset, attr_key)
        num_categories = len(category_to_idx)
        if num_categories == 0:
            return None
        history_vec = _discounted_topic_vector(history_articles, category_to_idx, attr_key)
        if history_vec.sum() <= EPSILON:
            return None

        def _eval(order: Sequence[int]) -> Optional[float]:
            L = len(order)
            if L == 0:
                return None
            weights = weight_lookup[L][:, None]
            topic_vec = np.zeros(num_categories, dtype=np.float64)
            for i, idx in enumerate(order, start=1):
                cat_idx = category_to_idx.get(_get_topic_value(candidate_articles[idx], attr_key))
                if cat_idx is not None:
                    topic_vec[int(cat_idx)] += float(weights[i - 1, 0])
            s = topic_vec.sum()
            if s > EPSILON:
                topic_vec /= s
            return _compute_jsd_divergence(history_vec, topic_vec)

        return _eval

    if mode == "fragmentation":
        if not fragmentation_reference:
            return None

        def _eval(order: Sequence[int]) -> Optional[float]:
            if not order:
                return None
            rec_articles = [candidate_articles[idx] for idx in order if 0 <= idx < len(candidate_articles)]
            if not rec_articles:
                return None
            try:
                value = metrics["fragmentation"].calculate(fragmentation_reference, rec_articles)
            except Exception:
                return None
            if not value:
                return None
            try:
                return float(value[0][1])
            except (TypeError, ValueError, IndexError):
                return None

        return _eval

    if mode in {"ild_tfidf", "ild_sentbert", "gini"}:
        def _eval(order: Sequence[int]) -> Optional[float]:
            if not order:
                return None
            rec_articles = [candidate_articles[idx] for idx in order if 0 <= idx < len(candidate_articles)]
            try:
                if mode == "ild_tfidf":
                    v = metrics["ild"].calculate_ild(rec_articles, representation="tfidf")
                    return float(v) if v is not None else None
                if mode == "ild_sentbert":
                    v = metrics["ild"].calculate_ild(rec_articles, representation="st")
                    return float(v) if v is not None else None
                if mode == "gini":
                    v = metrics["gini"].calculate_list_gini(rec_articles, key="category")
                    return float(v) if v is not None else None
            except Exception:
                return None
            return None

        return _eval

    return None


def rerank_production_topk(
    recommendations: Sequence[int],
    candidate_articles: Sequence[dict],
    history_articles: Sequence[dict],
    cutoff: int,
    lambda_weight: float = 0.5,
    allowed_indices: Optional[Sequence[int]] = None,
    attr_key: str = "category",
    metrics=None,
    sentiment_field: str = DEFAULT_SENTIMENT_FIELD,
) -> List[int]:
    """
    Re-rank candidate articles to balance relevance (from baseline ranking) and diversity.
    
    This is a production-ready function that doesn't require ground truth relevance labels.
    It infers relevance from the baseline recommendation ranking and optimizes a tradeoff
    between relevance preservation and diversity enhancement.
    
    Args:
        recommendations: 1-indexed ranking of candidates from the baseline recommender
        candidate_articles: List of candidate article dictionaries
        history_articles: List of user's historical articles for diversity calibration
        cutoff: Number of articles to include in the final ranking
        lambda_weight: Tradeoff weight (0-1), where 1.0 = maximize diversity, 0.0 = preserve relevance
        allowed_indices: Optional subset of candidate indices to consider
        attr_key: Diversity metric to optimize ("category", "subcategory", "activation", etc.)
        metrics: Metrics object for ILD and Gini calculations
        sentiment_field: Article field containing sentiment scores for Activation mode
        
    Returns:
        List of candidate indices in the re-ranked order
    """
    if not recommendations or not candidate_articles or cutoff <= 0:
        return []
    
    num_candidates = len(recommendations)
    baseline_order = sorted(range(num_candidates), key=lambda idx: recommendations[idx])
    
    if allowed_indices:
        allowed = [int(idx) for idx in allowed_indices if 0 <= int(idx) < num_candidates]
    else:
        allowed = baseline_order[:cutoff]
    allowed = [idx for idx in allowed if 0 <= idx < num_candidates]
    if not allowed:
        return []
    
    allowed_set = set(allowed)
    max_positions = min(cutoff, len(allowed_set))
    if max_positions == 0:
        return []
    
    # Precompute helpers
    discounts_lookup = {
        length: 1.0 / np.log2(np.arange(length, dtype=np.float64) + 2.0)
        for length in range(1, max_positions + 1)
    }
    harmonic_lookup = {length: harmonic_number(length) for length in range(1, max_positions + 1)}
    weight_lookup = {
        length: (1.0 / (np.arange(1, length + 1, dtype=np.float64) * harmonic_lookup[length]))
        for length in range(1, max_positions + 1)
    }
    
    # Infer relevance from baseline ranking position
    # Items ranked higher in baseline are more "relevant"
    relevance_scores = np.zeros(num_candidates, dtype=np.float64)
    for idx, rank in enumerate(recommendations):
        if isinstance(rank, (int, float)) and rank > 0:
            # Lower rank number = higher relevance, so invert
            relevance_scores[idx] = 1.0 / float(rank)
        else:
            relevance_scores[idx] = 0.0
    
    # Compute ideal DCG for normalization
    sorted_relevances = sorted(relevance_scores, reverse=True)
    ideal_dcg_lookup: Dict[int, float] = {}
    for length in range(1, max_positions + 1):
        top_rels = sorted_relevances[:length]
        if not top_rels:
            ideal_dcg_lookup[length] = 1.0  # Avoid division by zero
            continue
        discounts = discounts_lookup[length]
        gains = np.array(top_rels, dtype=np.float64)
        ideal_dcg_lookup[length] = float(np.sum(gains * discounts))
    
    evaluation_cache: Dict[Tuple[int, ...], Tuple[Optional[float], Optional[float]]] = {}
    
    def build_full_order(selected_indices: Sequence[int]) -> List[int]:
        selected_set_local = set(selected_indices)
        fallback = [idx for idx in baseline_order if idx in allowed_set and idx not in selected_set_local]
        order = list(selected_indices) + fallback
        return order[:max_positions]
    
    # Build divergence evaluator
    mode = attr_key
    evaluator = _make_divergence_evaluator(
        mode,
        candidate_articles,
        history_articles,
        allowed_set,
        weight_lookup,
        metrics,
        fragmentation_reference=fragmentation_reference,
        sentiment_field=sentiment_field,
    )
    if evaluator is None:
        return baseline_order[:max_positions]
    
    def evaluate_order(order: Sequence[int]) -> Tuple[Optional[float], Optional[float]]:
        key = tuple(order)
        if key in evaluation_cache:
            return evaluation_cache[key]
        
        order_len = len(order)
        if order_len == 0:
            evaluation_cache[key] = (None, None)
            return evaluation_cache[key]
        
        divergence = evaluator(order)
        
        # Compute DCG based on inferred relevance
        discounts = discounts_lookup[order_len]
        gains = relevance_scores[np.array(order, dtype=np.int64)]
        dcg = float(np.sum(gains * discounts))
        
        ideal_dcg = ideal_dcg_lookup.get(order_len, 1.0)
        ndcg = dcg / ideal_dcg if ideal_dcg > EPSILON else 0.0
        
        evaluation_cache[key] = (divergence, ndcg)
        return evaluation_cache[key]
    
    baseline_order_topk = build_full_order([])
    if not baseline_order_topk:
        return []
    
    baseline_eval_divergence, baseline_ndcg_val = evaluate_order(baseline_order_topk)
    if baseline_eval_divergence is None or baseline_ndcg_val is None:
        return baseline_order[:max_positions]
    
    baseline_divergence_value = float(baseline_eval_divergence)
    baseline_ndcg_value = float(baseline_ndcg_val)
    
    # Greedy optimization
    remaining = set(allowed_set)
    selected: List[int] = []
    
    for _ in range(max_positions):
        best_candidate = None
        best_objective = None
        
        for candidate_idx in remaining:
            tentative_selected = selected + [candidate_idx]
            full_order = build_full_order(tentative_selected)
            divergence, ndcg = evaluate_order(full_order)
            if divergence is None or ndcg is None:
                continue
            
            # Normalize metric and NDCG to [0,1]
            # For diversity metrics, higher divergence is better
            if baseline_divergence_value < 1.0:
                metric_norm = (divergence - baseline_divergence_value) / max(EPSILON, (1.0 - baseline_divergence_value))
                metric_norm = float(np.clip(metric_norm, 0.0, 1.0))
            else:
                # If already at maximum, just check if we're maintaining it
                metric_norm = 1.0 if divergence >= baseline_divergence_value else 0.0
            
            # Preserve relative NDCG
            ndcg_norm = ndcg / baseline_ndcg_value if baseline_ndcg_value > EPSILON else 0.0
            ndcg_norm = float(np.clip(ndcg_norm, 0.0, 1.0))
            
            objective = lambda_weight * metric_norm + (1.0 - lambda_weight) * ndcg_norm
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_candidate = candidate_idx
        
        if best_candidate is None:
            break
        
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    
    final_order = build_full_order(selected)
    return final_order[:max_positions]


def rerank_tradeoff_topk(
    lambda_values: Sequence[float],
    recommendations: Sequence[int],
    candidate_articles: Sequence[dict],
    history_articles: Sequence[dict],
    gt_relevance: Sequence[int],
    predicted_relevance: Optional[Sequence[float]],
    baseline_divergence: Optional[float],
    baseline_ndcg: Optional[float],
    cutoff: int,
    allowed_indices: Optional[Sequence[int]] = None,
    attr_key: str = "category",
    metrics=None,
    *,
    fragmentation_reference: Optional[Sequence[Sequence[dict]]] = None,
    sentiment_field: str = DEFAULT_SENTIMENT_FIELD,
    **unused_kwargs: Any,
) -> Dict[float, Tuple[float, float]]:
    """Greedy MMR-style reranker using predicted relevance for selection."""
    if unused_kwargs:
        unused_kwargs.clear()
    if not lambda_values or cutoff <= 0:
        return {}
    if not history_articles or not candidate_articles:
        return {}

    num_candidates = len(recommendations)
    if num_candidates == 0 or not gt_relevance:
        return {}

    baseline_order = sorted(range(num_candidates), key=lambda idx: recommendations[idx])
    if allowed_indices:
        initial_allowed = [int(idx) for idx in allowed_indices if 0 <= int(idx) < num_candidates]
    else:
        initial_allowed = baseline_order[:cutoff]
    if not initial_allowed:
        return {}

    seen = set()
    allowed: List[int] = []
    for idx in initial_allowed:
        if 0 <= idx < num_candidates and idx not in seen:
            seen.add(idx)
            allowed.append(idx)
    if not allowed:
        return {}

    allowed_set = set(allowed)
    max_positions = min(cutoff, len(allowed_set))
    if max_positions == 0:
        return {}

    baseline_positions = {idx: pos for pos, idx in enumerate(baseline_order)}

    discounts_lookup = {
        length: 1.0 / np.log2(np.arange(length, dtype=np.float64) + 2.0)
        for length in range(1, max_positions + 1)
    }
    harmonic_lookup = {length: harmonic_number(length) for length in range(1, max_positions + 1)}
    weight_lookup = {
        length: (1.0 / (np.arange(1, length + 1, dtype=np.float64) * harmonic_lookup[length]))
        for length in range(1, max_positions + 1)
    }
    position_discounts = 1.0 / np.log2(np.arange(max_positions, dtype=np.float64) + 2.0)

    evaluator = _make_divergence_evaluator(
        attr_key,
        candidate_articles,
        history_articles,
        allowed_set,
        weight_lookup,
        metrics,
        fragmentation_reference=fragmentation_reference,
        sentiment_field=sentiment_field,
    )
    if evaluator is None:
        return {}

    rel_count = min(num_candidates, len(gt_relevance))
    eval_gain_values = np.zeros(num_candidates, dtype=np.float64)
    if rel_count:
        rel_array = np.array(gt_relevance[:rel_count], dtype=np.float64)
        eval_gain_values[:rel_count] = (2.0 ** rel_array) - 1.0

    selection_gains = np.array(eval_gain_values, copy=True)
    if predicted_relevance is not None:
        pred_array = np.asarray(predicted_relevance, dtype=np.float64)
        usable_pred = min(len(pred_array), num_candidates)
        if usable_pred:
            selection_gains[:usable_pred] = pred_array[:usable_pred]

    sorted_relevances = sorted((float(val) for val in gt_relevance[:rel_count]), reverse=True)
    ideal_dcg_lookup: Dict[int, float] = {}
    for length in range(1, max_positions + 1):
        top_rels = sorted_relevances[:length]
        if not top_rels:
            ideal_dcg_lookup[length] = 0.0
            continue
        discounts = discounts_lookup[length]
        gains = (2.0 ** np.array(top_rels, dtype=np.float64)) - 1.0
        ideal_dcg_lookup[length] = float(np.sum(gains * discounts))

    diversity_cache: Dict[Tuple[int, ...], Optional[float]] = {}

    def compute_diversity(order: Sequence[int]) -> Optional[float]:
        key = tuple(order)
        if key in diversity_cache:
            return diversity_cache[key]
        value = evaluator(order)
        diversity_cache[key] = value
        return value

    def build_full_order(selected_indices: Sequence[int]) -> List[int]:
        if len(selected_indices) >= max_positions:
            return list(selected_indices[:max_positions])
        selected_set_local = set(selected_indices)
        fallback = [idx for idx in baseline_order if idx in allowed_set and idx not in selected_set_local]
        order = list(selected_indices) + fallback
        return order[:max_positions]

    def compute_ndcg(order: Sequence[int]) -> Optional[float]:
        length = len(order)
        if length == 0:
            return None
        gains = np.array(
            [eval_gain_values[idx] if 0 <= idx < len(eval_gain_values) else 0.0 for idx in order],
            dtype=np.float64,
        )
        discounts = discounts_lookup[length]
        dcg = float(np.sum(gains * discounts))
        ideal_dcg = ideal_dcg_lookup.get(length, 0.0)
        if ideal_dcg <= EPSILON:
            return 0.0
        return dcg / ideal_dcg

    def _normalize(value: float, maximum: float) -> float:
        return float(value / maximum) if maximum > EPSILON else 0.0

    results: Dict[float, Tuple[float, float]] = {}
    for lambda_weight in lambda_values:
        lambda_val = float(lambda_weight)
        remaining = set(allowed_set)
        selected: List[int] = []

        for position in range(max_positions):
            if not remaining:
                break

            candidate_stats: List[Tuple[int, float, float]] = []
            discount = float(position_discounts[position])
            for candidate_idx in remaining:
                raw_gain = selection_gains[candidate_idx] if 0 <= candidate_idx < len(selection_gains) else 0.0
                delta_dcg = float(max(raw_gain, 0.0) * discount)
                diversity_value = compute_diversity(selected + [candidate_idx])
                diversity_float = float(diversity_value) if diversity_value is not None else 0.0
                candidate_stats.append((candidate_idx, delta_dcg, diversity_float))

            if not candidate_stats:
                break

            max_delta = max(stat[1] for stat in candidate_stats)
            max_diversity = max(stat[2] for stat in candidate_stats)

            best_candidate = None
            best_score = None
            for candidate_idx, delta_val, diversity_val in candidate_stats:
                relevance_term = _normalize(delta_val, max_delta)
                diversity_term = _normalize(diversity_val, max_diversity)
                score = (1.0 - lambda_val) * relevance_term + lambda_val * diversity_term

                if (
                    best_score is None
                    or score > best_score
                    or (
                        best_candidate is not None
                        and abs(score - best_score) <= 1e-12
                        and baseline_positions.get(candidate_idx, num_candidates)
                        < baseline_positions.get(best_candidate, num_candidates)
                    )
                ):
                    best_score = score
                    best_candidate = candidate_idx

            if best_candidate is None:
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        final_order = build_full_order(selected)
        diversity_score = compute_diversity(final_order)
        if diversity_score is None:
            continue
        ndcg_score = compute_ndcg(final_order)
        if ndcg_score is None:
            continue
        results[lambda_val] = (float(ndcg_score), float(diversity_score))

    return results
