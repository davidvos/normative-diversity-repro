"""Main script for computing recommendation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import warnings
import argparse

from utils import (
    process_candidates, process_labels, get_articles,
    process_articles, ranking_to_scores, load_dataset,
    greedy_optimize_topic_calibration, initialize_metrics
)
from config import DATASET_CONFIG, NDCG_CUTOFF, RADIO_CUTOFF, OUTPUT_FOLDER

# Suppress specific warning
warnings.filterwarnings("ignore", message="In version 1.5 onwards, subsample=200_000 will be used by default.", category=FutureWarning)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compute recommendation metrics for a dataset.')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mind', 'ebnerd'],
        default='mind',
        help='Dataset to process (mind or ebnerd)'
    )
    parser.add_argument(
        '--greedy',
        action='store_true',
        help='Enable greedy re-ranking optimization'
    )
    parser.add_argument(
        '--max-behaviors',
        type=int,
        default=-1,
        help='Maximum number of behaviors to process (default: process all behaviors)'
    )
    return parser.parse_args()

def process_recommendation(recommender, recommendations, candidate_articles, user_history_articles, dataset, metrics, enable_greedy=False, recommender_to_df=None, behaviors=None, articles=None):
    """Process a single recommendation and compute metrics."""
    
    # Get recommendation articles
    recommendation_article_indices = [i-1 for i in recommendations][:RADIO_CUTOFF]
    recommendation_articles = [candidate_articles[i] for i in recommendation_article_indices if i < len(candidate_articles)]
    
    results = {}
    
    # Calculate original topic calibration
    if len(recommendation_articles) > 0 and len(user_history_articles) > 0:
        topic_divergence, complexity_divergence = metrics['calibration'].calculate(user_history_articles, recommendation_articles)
        if topic_divergence:
            results['topic_calibration'] = topic_divergence[0][1]
            results['complexity_calibration'] = complexity_divergence[0][1]
            
            # Calculate subtopic calibration
            subtopic_divergence, _ = metrics['calibration'].calculate(
                user_history_articles, recommendation_articles, 
                complexity=False, subcategory=True
            )
            if subtopic_divergence:
                results['subtopic_calibration'] = subtopic_divergence[0][1]

    # Calculate max/min topic calibration if greedy re-ranking is enabled
    if enable_greedy:
        max_calibration_score = greedy_optimize_topic_calibration(
            user_history_articles, candidate_articles, metrics['calibration'],
            maximize=True, cutoff=RADIO_CUTOFF
        )
        min_calibration_score = greedy_optimize_topic_calibration(
            user_history_articles, candidate_articles, metrics['calibration'],
            maximize=False, cutoff=RADIO_CUTOFF
        )
        
        results.update({
            'original_topic_calibration': results['topic_calibration'],
            'max_topic_calibration': max_calibration_score,
            'min_topic_calibration': min_calibration_score
        })
    
    # Calculate diversity metrics
    if len(recommendation_articles) > 0:
        # TF-IDF ILD
        # tfidf_ild_value = metrics['ild'].calculate_ild(recommendation_articles, representation='tfidf')
        # if tfidf_ild_value:
        #     results['tf_idf_ild'] = tfidf_ild_value
            
        # # Sentence Transformer ILD
        # sentbert_ild_value = metrics['ild'].calculate_ild(recommendation_articles, representation='st')
        # if sentbert_ild_value:
        #     results['sentbert_ild'] = float(sentbert_ild_value)
            
        # Gini coefficient
        gini_coefficient = metrics['gini'].calculate_list_gini(recommendation_articles, key="category")
        if gini_coefficient:
            results['gini'] = float(gini_coefficient)
            
        # Activation
        activation = metrics['activation'].calculate(candidate_articles, recommendation_articles)
        if activation:
            results['activation'] = activation[0][1]
            
        # For MIND dataset, calculate additional metrics
        if dataset == 'mind':
            # Representation
            representation = metrics['representation'].calculate(candidate_articles, recommendation_articles)
            if representation:
                results['representation'] = representation[0][1]
                
            # Alternative voices
            alternative_voices = metrics['alternative_voices'].calculate(candidate_articles, recommendation_articles)
            if alternative_voices:
                results['alternative_voices'] = alternative_voices[0][1]
            
            # Fragmentation
            if recommender_to_df is not None and behaviors is not None and articles is not None:
                rec_sample = recommender_to_df[recommender].sample(5)
                frag_articles = []
                for rec_sample_index, rec_sample_row in rec_sample.iterrows():
                    sample_impr = rec_sample_row['impr_index']
                    sample_recs = rec_sample_row['pred_rank']
                    
                    samples_behavior = behaviors[behaviors[0] == sample_impr].iloc[0]
                    samples_behavior_candidates = samples_behavior[4].split(' ')
                    samples_candidates_nids = process_candidates(samples_behavior_candidates)
                    
                    sample_ranking_nids = [samples_candidates_nids[i-1] for i in sample_recs][:RADIO_CUTOFF]
                    frag_articles.append(get_articles(sample_ranking_nids, articles))
                
                fragmentation = metrics['fragmentation'].calculate(frag_articles, recommendation_articles)
                if fragmentation:
                    results['fragmentation'] = fragmentation[0][1]
    
    return results

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()
    dataset = args.dataset
    enable_greedy = args.greedy
    max_behaviors = args.max_behaviors
    
    # Initialize metrics
    metrics = initialize_metrics()
    
    # Load dataset
    behaviors, articles = load_dataset(dataset, DATASET_CONFIG)
    articles = process_articles(articles, DATASET_CONFIG[dataset]['model_name'])
    
    # Load recommender predictions
    recommenders = DATASET_CONFIG[dataset]['recommenders']
    recommender_to_df = {
        recommender: pd.read_json(f"data/recommendations/{dataset}/{recommender}_prediction.json", lines=True)
        for recommender in recommenders
    }
    
    # Initialize results dictionary
    results = {
        "topic_calibration": {r: [] for r in recommenders},
        "subtopic_calibration": {r: [] for r in recommenders},
        "complexity_calibration": {r: [] for r in recommenders},
        "fragmentation": {r: [] for r in recommenders},
        "activation": {r: [] for r in recommenders},
        "representation": {r: [] for r in recommenders},
        "alternative_voices": {r: [] for r in recommenders},
        "tf_idf_ild": {r: [] for r in recommenders},
        "sentbert_ild": {r: [] for r in recommenders},
        "gini": {r: [] for r in recommenders},
        "ndcg_values": {r: [] for r in recommenders},
        "original_topic_calibration": {r: [] for r in recommenders}
    }
    
    # Add greedy re-ranking metrics if enabled
    if enable_greedy:
        results.update({
            "max_topic_calibration": {r: [] for r in recommenders},
            "min_topic_calibration": {r: [] for r in recommenders}
        })
    
    # Process each behavior
    for progress_index, behavior in tqdm(behaviors.iterrows()):
        if max_behaviors >= 0 and progress_index > max_behaviors:
            break
            
        behavior_id, _, _, behavior_history, behavior_candidates = behavior
        
        if dataset == 'ebnerd':
            behavior_id = progress_index + 1
            
        if not behavior_history:
            continue
        
        # Process user history
        user_history_nids = behavior_history.split(' ')
        user_history_nids.reverse()
        if dataset == 'ebnerd':
            user_history_nids = [int(i) for i in user_history_nids]
        user_history_articles = get_articles(user_history_nids, articles)
        
        # Process candidates
        candidates = behavior_candidates.split(' ')
        candidates_nids = process_candidates(candidates)
        if dataset == 'ebnerd':
            candidates_nids = [int(i) for i in candidates_nids]
        candidate_articles = get_articles(candidates_nids, articles)
        
        # Get ground truth relevance scores
        gt_relevance_scores = process_labels(candidates)
        
        # Process each recommender
        for recommender in recommenders:
            recommendations = recommender_to_df[recommender][
                recommender_to_df[recommender]['impr_index'] == behavior_id
            ]['pred_rank'].values.tolist()[0]
            
            # Process recommendation and update results
            recommendation_results = process_recommendation(
                recommender, recommendations, candidate_articles,
                user_history_articles, dataset, metrics, enable_greedy,
                recommender_to_df, behaviors, articles
            )
            
            # Update results dictionary
            for key, value in recommendation_results.items():
                if key in results:
                    results[key][recommender].append(value)
            
            # Calculate NDCG
            if dataset == 'mind':
                n = len(recommendations)
                pred_scores = (n + 1) - np.array(recommendations)
            else:
                pred_scores = ranking_to_scores(recommendations)
            
            ndcg_value = ndcg_score([gt_relevance_scores], [pred_scores], k=NDCG_CUTOFF)
            results['ndcg_values'][recommender].append(ndcg_value)
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(f'{OUTPUT_FOLDER}{dataset}_results_k@{RADIO_CUTOFF}.csv')
    
    # Print summary statistics
    print("\nNDCG Values:")
    print(results_df['ndcg_values'].apply(lambda x: np.mean(x)))

    print("\nTopic Calibration Values:")
    print(results_df['topic_calibration'].apply(lambda x: np.mean(x)))

    print("\nSubtopic Calibration Values:")
    print(results_df['subtopic_calibration'].apply(lambda x: np.mean(x)))

    if enable_greedy:

        print("\nOriginal Topic Calibration Values:")
        print(results_df['original_topic_calibration_scores'].apply(lambda x: np.mean(x) if x else None))

        print("\nMaximum Topic Calibration Values:")
        print(results_df['max_topic_calibration_scores'].apply(lambda x: np.mean(x) if x else None))
        
        print("\nMinimum Topic Calibration Values:")
        print(results_df['min_topic_calibration_scores'].apply(lambda x: np.mean(x) if x else None))
        
        # Calculate and print improvement percentages
        print("\nPotential Improvement Percentage in Topic Calibration:")
        for recommender in recommenders:
            orig_values = results_df['original_topic_calibration_scores'][recommender]
            max_values = results_df['max_topic_calibration_scores'][recommender]
            
            valid_pairs = [(orig, max_) for orig, max_ in zip(orig_values, max_values) if orig is not None and max_ is not None]
            
            if valid_pairs:
                avg_orig = np.mean([pair[0] for pair in valid_pairs])
                avg_max = np.mean([pair[1] for pair in valid_pairs])
                improvement_pct = ((avg_max - avg_orig) / avg_orig) * 100
                print(f"{recommender}: {improvement_pct:.2f}%")
            else:
                print(f"{recommender}: N/A")
        
        # Calculate and print degradation percentages
        print("\nPotential Degradation Percentage in Topic Calibration:")
        for recommender in recommenders:
            orig_values = results_df['original_topic_calibration_scores'][recommender]
            min_values = results_df['min_topic_calibration_scores'][recommender]
            
            valid_pairs = [(orig, min_) for orig, min_ in zip(orig_values, min_values) if orig is not None and min_ is not None]
            
            if valid_pairs:
                avg_orig = np.mean([pair[0] for pair in valid_pairs])
                avg_min = np.mean([pair[1] for pair in valid_pairs])
                degradation_pct = ((avg_min - avg_orig) / avg_orig) * 100
                print(f"{recommender}: {degradation_pct:.2f}%")
            else:
                print(f"{recommender}: N/A")

if __name__ == "__main__":
    main()