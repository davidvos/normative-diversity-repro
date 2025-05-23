"""Utility functions for the recommendation system."""

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

def process_articles(articles, model_name):
    """Process articles to add embeddings and TF-IDF vectors."""
    articles = articles.to_dict(orient='index')
    news_ids = list(articles.keys())
    news_texts = [articles[news_id]["title"] + " " + articles[news_id]["abstract"] for news_id in news_ids]

    # Get sentence transformer embeddings
    st_model = SentenceTransformer(model_name)
    st_embeddings = st_model.encode(news_texts, convert_to_numpy=True, show_progress_bar=True)
    for idx, news_id in enumerate(news_ids):
        articles[news_id]['st_vector'] = st_embeddings[idx]

    # Get TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(news_texts)
    for idx, news_id in enumerate(news_ids):
        articles[news_id]['tfidf_vector'] = tfidf_matrix[idx]

    return articles

def ranking_to_scores(ranking):
    """Convert a ranking list (1-indexed) into a score vector."""
    n = len(ranking)
    scores = [0] * n
    for rank, candidate in enumerate(ranking):
        scores[candidate - 1] = n - rank
    return scores

def load_dataset(dataset, config):
    """Load dataset-specific behaviors and articles."""
    dataset_config = config[dataset]
    
    if dataset == 'mind':
        behaviors = pd.read_csv(dataset_config['behaviors_path'], delimiter='\t', header=None)
        behaviors = behaviors.replace({np.nan: None})
        articles = pd.read_pickle(dataset_config['articles_path'])
        articles['title'] = articles['title'].fillna('')
        articles['abstract'] = articles['abstract'].fillna('')

    elif dataset == 'ebnerd':
        behaviors = pd.read_table(dataset_config['behaviors_path'], delimiter='\t', header=None, 
                                names=['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidate_news', 'clicked'])
        behaviors = behaviors[['impression_id', 'user', 'time', 'clicked_news', 'impressions']]
        behaviors = behaviors.replace({np.nan: None})
        
        articles = pd.read_pickle(dataset_config['articles_path'])
        articles = articles.set_index('article_id')
        articles['title'] = articles['title'].fillna('')
        articles['abstract'] = articles['subtitle'].fillna('')
        articles['category'] = articles['category_str'].fillna('other')
        articles['subcategory'] = articles['subcategory'].apply(lambda x: x[0] if len(x) > 0 else 0)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return behaviors, articles

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
        The optimal calibration score
    """
    if len(candidate_articles) == 0 or len(user_history_articles) == 0:
        return None
    
    recommendation_indices = []
    remaining_candidate_indices = list(range(len(candidate_articles)))
    best_score = float('-inf') if maximize else float('inf')
    
    for _ in range(min(cutoff, len(candidate_articles))):
        best_candidate_idx = -1
        
        for candidate_idx in remaining_candidate_indices:
            tentative_indices = recommendation_indices + [candidate_idx]
            tentative_articles = [candidate_articles[idx] for idx in tentative_indices]
            
            if len(tentative_articles) > 0:
                topic_divergence, _ = calibration_metric.calculate(user_history_articles, tentative_articles)
                if topic_divergence:
                    jsd_score = topic_divergence[0][1]
                    
                    if (maximize and jsd_score > best_score) or (not maximize and jsd_score < best_score):
                        best_score = jsd_score
                        best_candidate_idx = candidate_idx
        
        if best_candidate_idx != -1:
            recommendation_indices.append(best_candidate_idx)
            remaining_candidate_indices.remove(best_candidate_idx)
    
    if recommendation_indices:
        optimal_articles = [candidate_articles[idx] for idx in recommendation_indices]
        topic_divergence, _ = calibration_metric.calculate(user_history_articles, optimal_articles)
        final_jsd_score = topic_divergence[0][1] if topic_divergence else None
    else:
        final_jsd_score = None
    
    return final_jsd_score

def initialize_metrics():
    """Initialize all metric calculators."""
    return {
        'calibration': dart.metrics.calibration.Calibration({'language': 'english', 'country': 'us'}),
        'fragmentation': dart.metrics.fragmentation.Fragmentation(),
        'activation': dart.metrics.activation.Activation({'language': 'english', 'country': 'us'}),
        'representation': dart.metrics.representation.Representation({'language': 'english', 'country': 'us'}),
        'alternative_voices': dart.metrics.alternative_voices.AlternativeVoices(),
        'ild': ILD(),
        'gini': GiniCoefficient()
    }
