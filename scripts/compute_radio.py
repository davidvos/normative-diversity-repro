"""Main script for computing recommendation metrics."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATASET_CONFIG, RADIO_CUTOFF, OUTPUT_FOLDER, TRADEOFF_METRICS_CONFIG
from rerank import rerank_tradeoff_topk
from utils import (
    STORY_SIMILARITY_THRESHOLDS,
    attach_article_metadata_field,
    ensure_article_cache,
    get_articles,
    initialize_metrics,
    load_dataset,
    process_candidates,
    process_labels,
    select_complexity_series,
    select_sentiment_series,
    story_column_name,
    compute_ndcg_from_rank,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ACTIVATION_SENTIMENT_FIELD = "__activation_sentiment"

def parse_args():
    parser = argparse.ArgumentParser(description="Compute recommendation metrics.")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()) + ["all"], default="mind")
    parser.add_argument("--metric", choices=["all", "complexity", "activation", "fragmentation"], default="all")
    parser.add_argument("--max-behaviors", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--complexity-source", choices=["auto", "readability", "mtld", "both"], default="auto")
    parser.add_argument("--story-threshold", type=float, default=0.3, choices=STORY_SIMILARITY_THRESHOLDS)
    parser.add_argument("--sentiment-source", choices=["lexicon", "transformer", "dataset"], default="lexicon")
    parser.add_argument("--rerank-all", action="store_true")
    parser.add_argument("--tradeoff-lambdas", type=str, default=None)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--use-predicted-relevance", action="store_true")
    return parser.parse_args()

def compute_base_metrics(candidate_articles, recommendation_articles, history_articles, dataset, metrics, fragmentation_ref, activation_field, enabled_metrics=None):
    results = {}
    def wants(key): return enabled_metrics is None or key in enabled_metrics

    if recommendation_articles and history_articles:
        if any(wants(k) for k in ("topic_calibration", "complexity_calibration", "subtopic_calibration")):
            topic_div, complexity_div = metrics["calibration"].calculate(history_articles, recommendation_articles)
            if topic_div and wants("topic_calibration"): results["topic_calibration"] = topic_div[0][1]
            if complexity_div and wants("complexity_calibration"): results["complexity_calibration"] = complexity_div[0][1]
            if wants("subtopic_calibration"):
                subtopic_div, _ = metrics["calibration"].calculate(history_articles, recommendation_articles, complexity=False, subcategory=True)
                if subtopic_div: results["subtopic_calibration"] = subtopic_div[0][1]

    if recommendation_articles:
        if wants("tf_idf_ild"):
            val = metrics["ild"].calculate_ild(recommendation_articles, representation="tfidf")
            if val: results["tf_idf_ild"] = val
        if wants("sentbert_ild"):
            val = metrics["ild"].calculate_ild(recommendation_articles, representation="st")
            if val: results["sentbert_ild"] = float(val)
        if wants("gini"):
            val = metrics["gini"].calculate_list_gini(recommendation_articles, key="category")
            if val: results["gini"] = float(val)
        if wants("activation"):
            cands = [dict(a, sentiment=float(a.get(activation_field, 0.0))) for a in candidate_articles]
            recs = [dict(a, sentiment=float(a.get(activation_field, 0.0))) for a in recommendation_articles]
            val = metrics["activation"].calculate(cands, recs)
            results["activation"] = val[0][1] if val else None
        if dataset == "mind":
            if wants("representation"):
                val = metrics["representation"].calculate(candidate_articles, recommendation_articles)
                if val: results["representation"] = val[0][1]
            if wants("alternative_voices"):
                val = metrics["alternative_voices"].calculate(candidate_articles, recommendation_articles)
                if val: results["alternative_voices"] = val[0][1]
        if wants("fragmentation"):
            if fragmentation_ref and all("story" in a for a in recommendation_articles):
                try:
                    val = metrics["fragmentation"].calculate(fragmentation_ref, recommendation_articles)
                    results["fragmentation"] = val[0][1] if val else None
                except KeyError: results["fragmentation"] = None
            else: results["fragmentation"] = None
    return results

def build_fragmentation_references(recommender_to_df, behaviors, articles, dataset):
    if not isinstance(articles, dict) or not any("story" in a for a in articles.values()): return {}
    references = {}
    behaviors_indexed = behaviors.set_index("impression_id" if "impression_id" in behaviors.columns else 0, drop=False) if "impression_id" in behaviors.columns or 0 in behaviors.columns else behaviors

    for rec, df in recommender_to_df.items():
        if df.empty:
            references[rec] = None
            continue
        frag_articles = []
        for _, row in df.sample(min(5, len(df)), random_state=42).iterrows():
            impr = row.get("impr_index")
            behavior_row = behaviors_indexed.loc[impr] if pd.notna(impr) and impr in behaviors_indexed.index else None
            if behavior_row is None and dataset in {"mind", "ebnerd"} and "__row_pos" in row:
                try: behavior_row = behaviors.iloc[int(row["__row_pos"]) - 1]
                except: pass
            if behavior_row is None: continue
            
            if isinstance(behavior_row, pd.DataFrame): behavior_row = behavior_row.iloc[0]
            cands = behavior_row["impressions"] if isinstance(behavior_row, pd.Series) and "impressions" in behavior_row.index else (behavior_row[4] if len(behavior_row) > 4 else None)
            if not isinstance(cands, str): continue
            
            cids = process_candidates(cands.split(" "))
            if dataset == "ebnerd": cids = [int(c) for c in cids if c]
            
            ranking = row["pred_rank"]
            if not isinstance(ranking, list): continue
            
            a_ids = [cids[i-1] for i in ranking[:RADIO_CUTOFF] if 0 <= i-1 < len(cids)]
            objs = [a for a in get_articles(a_ids, articles) if "story" in a]
            if objs: frag_articles.append(objs)
        references[rec] = [a for a in frag_articles if a]
    return references

def process_recommender(dataset, rec, recommendations, candidate_articles, history_articles, metrics, gt_relevance, predicted_relevance, fragmentation_ref, precomputed_indices, lambda_values, rerank_all, activation_field, enabled_metrics, tradeoff_metric_names):
    baseline_ndcg = compute_ndcg_from_rank(recommendations, dataset, gt_relevance)
    
    rec_indices = list(precomputed_indices) if precomputed_indices is not None else sorted(range(len(recommendations)), key=lambda idx: recommendations[idx])[:RADIO_CUTOFF]
    rec_articles = [candidate_articles[idx] for idx in rec_indices if 0 <= idx < len(candidate_articles)]
    
    metric_values = compute_base_metrics(candidate_articles, rec_articles, history_articles, dataset, metrics, fragmentation_ref, activation_field, enabled_metrics)
    
    tradeoff_bundle = {}
    if lambda_values:
        allowed = list(range(len(recommendations))) if rerank_all else precomputed_indices
        for cfg in TRADEOFF_METRICS_CONFIG:
            name = cfg["name"]
            if tradeoff_metric_names is not None and name not in tradeoff_metric_names: continue
            
            baseline_val = next((metric_values.get(k) for k in cfg["baseline_keys"] if metric_values.get(k) is not None), None)
            if baseline_val is None: continue

            tradeoff_bundle[name] = rerank_tradeoff_topk(
                lambda_values, recommendations, candidate_articles, history_articles, gt_relevance, predicted_relevance,
                baseline_val, baseline_ndcg, RADIO_CUTOFF, allowed, attr_key=cfg["mode"], metrics=metrics,
                fragmentation_reference=fragmentation_ref if name == "fragmentation" else None,
                sentiment_field=activation_field
            )
    return rec, metric_values, baseline_ndcg, tradeoff_bundle

def run_single_configuration(dataset, **kwargs):
    metrics = initialize_metrics(dataset)
    behaviors, articles_df = load_dataset(dataset, DATASET_CONFIG)
    articles = ensure_article_cache(articles_df, DATASET_CONFIG[dataset]["model_name"], DATASET_CONFIG[dataset].get("processed_articles_cache"), build_if_missing=False)
    
    enabled_metrics = kwargs.get("enabled_metrics")
    def wants(key): return enabled_metrics is None or key in enabled_metrics

    if isinstance(articles, dict):
        if any(wants(k) for k in ("topic_calibration", "complexity_calibration", "subtopic_calibration")):
            _, series = select_complexity_series(articles_df, kwargs["complexity_source"], dataset)
            attach_article_metadata_field(articles, series, "complexity")
        if wants("activation"):
            _, series = select_sentiment_series(articles_df, kwargs["sentiment_source"], dataset)
            attach_article_metadata_field(articles, series, ACTIVATION_SENTIMENT_FIELD)
        if wants("fragmentation") and isinstance(articles_df, pd.DataFrame):
            col = kwargs.get("story_column") or story_column_name(kwargs["story_threshold"])
            if col not in articles_df.columns and "story" in articles_df.columns: col = "story"
            if col in articles_df.columns: attach_article_metadata_field(articles, articles_df[col], "story")

    recommenders = DATASET_CONFIG[dataset]["recommenders"]
    rec_data = {}
    
    def get_top_k(ranks, k):
        arr = np.asarray(ranks)
        k = min(k, len(arr))
        if k == len(arr): return sorted(range(len(arr)), key=lambda i: ranks[i])
        try:
            parts = np.argpartition(arr, k-1)[:k]
            return parts[np.argsort(arr[parts])].tolist()
        except TypeError: return sorted(range(len(arr)), key=lambda i: ranks[i])[:k]

    for rec in recommenders:
        path = Path(f"data/recommendations/{dataset}/{rec}_prediction.json")
        if not path.exists(): continue
        try: df = pd.read_json(path, lines=True)
        except ValueError: df = pd.read_json(path, lines=False)
        df = df.reset_index(drop=True)
        df["__row_pos"] = df.index + 1
        
        data = {"df": df, "rankings": {}, "top_indices": {}, "scores": {}, "seq_rankings": {}, "seq_top_indices": {}, "seq_scores": {}}
        
        if "impr_index" in df.columns:
            df_idx = df.set_index("impr_index", drop=False)
            data["rankings"] = df_idx["pred_rank"].to_dict()
            data["top_indices"] = {k: get_top_k(v, RADIO_CUTOFF) if isinstance(v, (list, np.ndarray)) else [] for k, v in data["rankings"].items()}
            if "pred_rel" in df.columns: data["scores"] = df_idx["pred_rel"].to_dict()
            
        seq_df = df.set_index("__row_pos")
        data["seq_rankings"] = seq_df["pred_rank"].to_dict()
        data["seq_top_indices"] = {k: get_top_k(v, RADIO_CUTOFF) if isinstance(v, (list, np.ndarray)) else [] for k, v in data["seq_rankings"].items()}
        if "pred_rel" in seq_df.columns: data["seq_scores"] = seq_df["pred_rel"].to_dict()
        rec_data[rec] = data

    frag_refs = build_fragmentation_references({r: d["df"] for r, d in rec_data.items()}, behaviors, articles, dataset) if wants("fragmentation") else {}
    
    results = {k: {r: [] for r in recommenders} for k in ["topic_calibration", "subtopic_calibration", "complexity_calibration", "fragmentation", "activation", "representation", "alternative_voices", "tf_idf_ild", "sentbert_ild", "gini", "ndcg_values"] if wants(k) or k == "ndcg_values"}
    tradeoff_results = {}
    lambda_values = kwargs.get("lambda_values") or []
    if lambda_values:
        tradeoff_results = {cfg["name"]: {l: {r: {"divergence": [], "ndcg": []} for r in recommenders} for l in lambda_values} for cfg in TRADEOFF_METRICS_CONFIG if kwargs.get("tradeoff_metric_names") is None or cfg["name"] in kwargs.get("tradeoff_metric_names")}

    executor = ThreadPoolExecutor(max_workers=kwargs["workers"]) if kwargs["workers"] > 1 else None
    futures = []
    
    for row_idx, behavior in enumerate(tqdm(behaviors.itertuples(index=False, name=None), total=len(behaviors))):
        if 0 <= kwargs["max_behaviors"] <= row_idx: break
        
        hist_tokens = (behavior[3] if len(behavior) > 3 else "").split(" ")[::-1]
        cand_tokens = (behavior[4] if len(behavior) > 4 else "").split(" ")
        if not hist_tokens or not cand_tokens or not behavior[3] or not behavior[4]: continue
        
        if dataset == "ebnerd":
            hist_tokens = [int(i) for i in hist_tokens if i]
            cand_tokens = [int(i) for i in cand_tokens if i]
            
        hist_arts = get_articles(hist_tokens, articles)
        cand_ids = process_candidates(cand_tokens if dataset != "ebnerd" else [str(c) for c in cand_tokens])
        if dataset == "ebnerd": cand_ids = [int(c) for c in cand_ids]
        cand_arts = get_articles(cand_ids, articles)
        if dataset == "ebnerd":
            clicked_set = set(hist_tokens)
            gt_rel = [1 if c in clicked_set else 0 for c in cand_tokens]
        else:
            gt_rel = process_labels(cand_tokens)

        for rec in recommenders:
            data = rec_data.get(rec)
            if not data: continue
            
            recs = data["rankings"].get(behavior[0])
            precomputed = None
            scores = data["scores"].get(behavior[0]) if kwargs["use_predicted_relevance"] else None
            
            if recs is None and dataset in {"ebnerd", "mind"}:
                recs = data["seq_rankings"].get(row_idx + 1)
                precomputed = data["seq_top_indices"].get(row_idx + 1)
                if kwargs["use_predicted_relevance"] and scores is None:
                    scores = data["seq_scores"].get(row_idx + 1)
            
            if recs is None: continue
            if precomputed is None:
                precomputed = data["top_indices"].get(behavior[0])
                
            args = (dataset, rec, recs, cand_arts, hist_arts, metrics, gt_rel, scores, frag_refs.get(rec), precomputed, lambda_values, kwargs["rerank_all"], ACTIVATION_SENTIMENT_FIELD, enabled_metrics, kwargs.get("tradeoff_metric_names"))
            
            if executor: futures.append(executor.submit(process_recommender, *args))
            else:
                _, vals, ndcg, tradeoffs = process_recommender(*args)
                if "ndcg_values" in results: results["ndcg_values"][rec].append(ndcg)
                for k, v in vals.items(): results[k][rec].append(v)
                for m, map_ in tradeoffs.items():
                    for l, (n, d) in map_.items():
                        tradeoff_results[m][l][rec]["ndcg"].append(n)
                        tradeoff_results[m][l][rec]["divergence"].append(d)

    if executor:
        for f in futures:
            rec, vals, ndcg, tradeoffs = f.result()
            if "ndcg_values" in results: results["ndcg_values"][rec].append(ndcg)
            for k, v in vals.items(): results[k][rec].append(v)
            for m, map_ in tradeoffs.items():
                for l, (n, d) in map_.items():
                    tradeoff_results[m][l][rec]["ndcg"].append(n)
                    tradeoff_results[m][l][rec]["divergence"].append(d)
        executor.shutdown()

    return pd.DataFrame({k: {r: np.mean([v for v in vs if v is not None]) if vs else None for r, vs in d.items()} for k, d in results.items()}).T, tradeoff_results

if __name__ == "__main__":
    args = parse_args()
    datasets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]
    lambdas = sorted({float(x) for x in args.tradeoff_lambdas.split(",") if x.strip()}) if args.tradeoff_lambdas else []
    
    for ds in datasets:
        print(f"Processing {ds}...")
        df, tradeoffs = run_single_configuration(
            ds, max_behaviors=args.max_behaviors, workers=args.workers, complexity_source=args.complexity_source,
            story_threshold=args.story_threshold, sentiment_source=args.sentiment_source, rerank_all=args.rerank_all,
            lambda_values=lambdas, output_suffix=args.output_suffix,
            enabled_metrics=None if args.metric == "all" else {args.metric},
            use_predicted_relevance=args.use_predicted_relevance
        )
        print(df)
        suffix = f"_{args.output_suffix.strip().replace(' ', '_')}" if args.output_suffix.strip() else ""
        df.to_csv(OUTPUT_FOLDER / f"{ds}_results{suffix}.csv")
        for metric, data in tradeoffs.items():
            rows = []
            for lam, rec_data in data.items():
                for rec, vals in rec_data.items():
                    rows.append({"lambda": lam, "recommender": rec, "ndcg": np.mean(vals["ndcg"]), "divergence": np.mean(vals["divergence"])})
            pd.DataFrame(rows).to_csv(OUTPUT_FOLDER / f"{ds}_{metric}_tradeoff{suffix}.csv", index=False)
