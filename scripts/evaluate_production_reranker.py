"""
Evaluate production reranker using ground truth relevance labels.

This script demonstrates how to use rerank_production_topk for offline evaluation
by computing ground truth NDCG on the reranked results.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rerank import rerank_production_topk
from utils import (
    STORY_SIMILARITY_THRESHOLDS,
    attach_article_metadata_field,
    ensure_article_cache,
    get_articles,
    initialize_metrics,
    load_dataset,
    process_candidates,
    process_labels,
    ranking_to_scores,
    indices_to_pred_rank,
    compute_ndcg_from_rank,
    select_complexity_series,
    select_sentiment_series,
    story_column_name,
)
from config import DATASET_CONFIG, RADIO_CUTOFF, TRADEOFF_METRICS_CONFIG

# Configuration for production reranker evaluation


def _parse_lambda_grid(lambda_spec: Optional[str]) -> List[float]:
    """Parse a comma-separated lambda grid specification into a sorted float list."""
    if not lambda_spec:
        return []
    
    values: List[float] = []
    for token in lambda_spec.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        try:
            value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"Invalid lambda weight '{stripped}'.") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Lambda weight '{value}' must be within [0, 1].")
        values.append(value)
    
    if not values:
        return []
    
    # Deduplicate while preserving ordering semantics.
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate production reranker with ground truth labels.")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), default="mind")
    parser.add_argument("--recommender", type=str, default="nrms")
    parser.add_argument("--max-behaviors", type=int, default=100)
    parser.add_argument(
        "--complexity-source",
        choices=["auto", "readability", "mtld"],
        default="auto",
        help="Select which article complexity column to use during Complexity Calibration.",
    )
    parser.add_argument(
        "--story-threshold",
        type=float,
        default=0.3,
        choices=STORY_SIMILARITY_THRESHOLDS,
        help="Cosine similarity threshold used for story clustering in the Fragmentation metric.",
    )
    parser.add_argument(
        "--sentiment-source",
        choices=["lexicon", "transformer", "dataset"],
        default="lexicon",
        help="Select which article sentiment column to use for Activation.",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated list of lambda weights (0-1) for tradeoff reranking.",
    )
    return parser.parse_args()


def evaluate_single_lambda(
    lambda_val: float,
    dataset: str,
    recommender: str,
    metrics,
    lambda_values: Sequence[float],
    max_behaviors: int,
    complexity_source: str,
    story_threshold: float,
    sentiment_source: str,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate production reranker for a single recommender across all metrics.
    
    Returns a dictionary mapping metric name to DataFrame with columns:
    lambda, recommender, ndcg, divergence
    """
    behaviors, articles_df = load_dataset(dataset, DATASET_CONFIG)
    articles = ensure_article_cache(
        articles_df,
        DATASET_CONFIG[dataset]["model_name"],
        DATASET_CONFIG[dataset].get("processed_articles_cache"),
        build_if_missing=False,
    )
    activation_field = "__activation_sentiment"
    if isinstance(articles, dict):
        try:
            _, complexity_series = select_complexity_series(articles_df, complexity_source, dataset)
            attach_article_metadata_field(articles, complexity_series, "complexity")
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
        if isinstance(articles_df, pd.DataFrame):
            requested_story_column = story_column_name(story_threshold)
            story_column_used = requested_story_column if requested_story_column in articles_df.columns else None
            if story_column_used is None and "story" in articles_df.columns:
                story_column_used = "story"
            if story_column_used:
                attach_article_metadata_field(articles, articles_df[story_column_used], "story")
        try:
            _, sentiment_series = select_sentiment_series(articles_df, sentiment_source, dataset)
            attach_article_metadata_field(articles, sentiment_series, activation_field)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
    
    # Load baseline recommendations
    df = pd.read_json(f"data/recommendations/{dataset}/{recommender}_prediction.json", lines=True)
    if "impr_index" in df.columns:
        df = df.set_index("impr_index", drop=False)
        recommender_rankings = df["pred_rank"].to_dict()
    else:
        recommender_rankings = {}
    
    # Helper to attach sentiment field for activation calculations
    def with_activation_sentiment(articles_seq: Sequence[dict]) -> List[dict]:
        return [
            dict(article, sentiment=float(article.get(activation_field, 0.0)))
            for article in articles_seq
        ]

    # Initialize results for each metric
    results_by_metric = {cfg["name"]: {"lambda": [], "ndcg": [], "divergence": []} 
                        for cfg in TRADEOFF_METRICS_CONFIG}
    
    processed = 0
    for row_idx, behavior in behaviors.iterrows():
        if processed >= max_behaviors:
            break
            
        behavior_id, _, _, history_raw, candidates_raw = behavior
        if dataset == "ebnerd":
            behavior_id = row_idx + 1
        
        if not history_raw:
            continue
        
        history_tokens = history_raw.split(" ")[::-1]
        if dataset == "ebnerd":
            history_tokens = [int(i) for i in history_tokens]
        history_articles = get_articles(history_tokens, articles)
        
        candidates = candidates_raw.split(" ")
        candidate_ids = process_candidates(candidates)
        if dataset == "ebnerd":
            candidate_ids = [int(i) for i in candidate_ids]
        candidate_articles = get_articles(candidate_ids, articles)
        
        gt_relevance = process_labels(candidates)
        
        # Get baseline ranking
        baseline_ranking = recommender_rankings.get(behavior_id)
        if not baseline_ranking:
            continue
        
        # Rerank for each metric and lambda combination
        for cfg in TRADEOFF_METRICS_CONFIG:
            metric_name = cfg["name"]
            metric_mode = cfg["mode"]
            
            for lam_val in lambda_values:
                reranked_indices = rerank_production_topk(
                    recommendations=baseline_ranking,
                    candidate_articles=candidate_articles,
                    history_articles=history_articles,
                    cutoff=RADIO_CUTOFF,
                    lambda_weight=lam_val,
                    attr_key=metric_mode,
                    metrics=metrics,
                    sentiment_field=activation_field,
                )
                
                # Convert to ranking format for NDCG computation
                pred_rank = indices_to_pred_rank(reranked_indices, len(baseline_ranking))
                gt_ndcg = compute_ndcg_from_rank(pred_rank, dataset, gt_relevance)
                
                # Compute diversity metric on the reranked list
                reranked_articles = [candidate_articles[idx] for idx in reranked_indices]
                diversity_value = None
                
                # Compute diversity based on metric mode
                if metric_mode == "category":
                    topic_div, _ = metrics["calibration"].calculate(history_articles, reranked_articles)
                    diversity_value = topic_div[0][1] if topic_div else None
                elif metric_mode == "subcategory":
                    subtopic_div, _ = metrics["calibration"].calculate(
                        history_articles, reranked_articles, complexity=False, subcategory=True
                    )
                    diversity_value = subtopic_div[0][1] if subtopic_div else None
                elif metric_mode == "complexity":
                    _, complexity_div = metrics["calibration"].calculate(history_articles, reranked_articles)
                    diversity_value = complexity_div[0][1] if complexity_div else None
                elif metric_mode == "activation":
                    activation_candidates = with_activation_sentiment(candidate_articles)
                    activation_recommendations = with_activation_sentiment(reranked_articles)
                    activation = metrics["activation"].calculate(activation_candidates, activation_recommendations)
                    diversity_value = activation[0][1] if activation else None
                elif metric_mode == "ild_tfidf":
                    diversity_value = metrics["ild"].calculate_ild(reranked_articles, representation="tfidf")
                elif metric_mode == "ild_sentbert":
                    diversity_value = metrics["ild"].calculate_ild(reranked_articles, representation="st")
                elif metric_mode == "gini":
                    diversity_value = metrics["gini"].calculate_list_gini(reranked_articles, key="category")
                
                if gt_ndcg is not None and diversity_value is not None:
                    results_by_metric[metric_name]["lambda"].append(lam_val)
                    results_by_metric[metric_name]["ndcg"].append(gt_ndcg)
                    results_by_metric[metric_name]["divergence"].append(diversity_value)
        
        processed += 1
    
    # Convert to DataFrames and aggregate
    output_dict = {}
    for metric_name, results in results_by_metric.items():
        if not results["lambda"]:
            continue
        
        df_results = pd.DataFrame(results)
        
        # Aggregate by lambda (compute means) and add recommender column
        aggregated = df_results.groupby("lambda").agg({
            "ndcg": "mean",
            "divergence": "mean",
        }).reset_index()
        aggregated["recommender"] = recommender
        
        # Reorder columns to match expected format
        aggregated = aggregated[["lambda", "recommender", "ndcg", "divergence"]]
        output_dict[metric_name] = aggregated
    
    return output_dict


def evaluate_production_reranker_all_metrics(
    dataset: str = "mind",
    recommender: str = "nrms",
    lambda_values: Sequence[float] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
    max_behaviors: int = 100,
    output_dir: str = "results",
    complexity_source: str = "auto",
    story_threshold: float = 0.3,
    sentiment_source: str = "lexicon",
) -> None:
    """
    Evaluate production reranker across all diversity metrics.
    
    Saves CSV files matching the format of compute_radio.py tradeoff outputs:
    - {dataset}_{metric}_production_tradeoff_k@{cutoff}.csv
    
    Args:
        dataset: Dataset name ('mind', 'ebnerd', etc.)
        recommender: Baseline recommender to rerank
        lambda_values: Lambda weights to test
        max_behaviors: Maximum number of behaviors to evaluate
        output_dir: Directory to save results
    """
    metrics = initialize_metrics(dataset)
    
    print(f"Evaluating production reranker: {recommender} on {dataset}")
    print(f"Lambda values: {lambda_values}")
    print(f"Processing up to {max_behaviors} behaviors...")
    print(f"Story similarity threshold: {story_threshold}")
    print(f"Sentiment source: {sentiment_source}")
    
    # Evaluate for all metrics
    results_dict = evaluate_single_lambda(
        lambda_val=0.0,  # Not used, keeping all lambda values
        dataset=dataset,
        recommender=recommender,
        metrics=metrics,
        lambda_values=lambda_values,
        max_behaviors=max_behaviors,
        complexity_source=complexity_source,
        story_threshold=story_threshold,
        sentiment_source=sentiment_source,
    )
    
    # Save each metric's results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for metric_name, df_results in results_dict.items():
        filename = f"{dataset}_{metric_name}_production_tradeoff_k@{RADIO_CUTOFF}.csv"
        filepath = output_path / filename
        df_results.to_csv(filepath, index=False)
        print(f"  Saved {metric_name}: {filepath}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    args = parse_args()
    try:
        lambda_values = _parse_lambda_grid(args.lambdas)
    except ValueError as exc:
        raise SystemExit(str(exc))
    
    evaluate_production_reranker_all_metrics(
        dataset=args.dataset,
        recommender=args.recommender,
        lambda_values=lambda_values,
        max_behaviors=args.max_behaviors,
        complexity_source=args.complexity_source,
        story_threshold=args.story_threshold,
        sentiment_source=args.sentiment_source,
    )
