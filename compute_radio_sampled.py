from utils import process_candidates, process_labels

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import warnings

import dart.metrics.activation
import dart.metrics.calibration
import dart.metrics.fragmentation
import dart.metrics.representation
import dart.metrics.alternative_voices

# Suppressing the specific warning
warnings.filterwarnings("ignore", message="In version 1.5 onwards, subsample=200_000 will be used by default.", category=FutureWarning)

config = {
  "language": "english",
  "political_file": "data/term-116.csv",
  "output_folder": "output/",
    "metadata_folder": "metadata/",
  "country" : "us"
}

def get_articles(article_ids, articles):
    filtered_articles = [articles[article_id] for article_id in article_ids if article_id in articles]
    return filtered_articles

Calibration = dart.metrics.calibration.Calibration(config)
Fragmentation = dart.metrics.fragmentation.Fragmentation()
Activation = dart.metrics.activation.Activation(config)
Representation = dart.metrics.representation.Representation(config)
AlternativeVoices = dart.metrics.alternative_voices.AlternativeVoices()

ndcg_cutoff = 10
radio_cutoff = 10

behaviors = pd.read_csv('data/sampled_user_behaviors.csv', delimiter='\t', header=None)
behaviors = behaviors.replace({np.nan: None})

articles = pd.read_pickle('data/articles.pickle')
articles = articles.to_dict(orient='index')

recommenders = ['random', 'incorrect_random', 'pop']

recommender_to_df = {recommender: pd.read_json(f"data/recommendations_sampled/{recommender}_predictions.json", lines=True) for recommender in recommenders}
print(recommender_to_df['random'].head())

results = {
    "topic_calibrations": {recommender: [] for recommender in recommender_to_df},
    "complexity_calibrations": {recommender: [] for recommender in recommender_to_df},
    "fragmentations": {recommender: [] for recommender in recommender_to_df},
    "activations": {recommender: [] for recommender in recommender_to_df},
    "representations": {recommender: [] for recommender in recommender_to_df},
    "alternative_voices": {recommender: [] for recommender in recommender_to_df},
    "ndcg_values": {recommender: [] for recommender in recommender_to_df}
}

for progress_index, behavior in tqdm(behaviors.iterrows()):
    behavior_id, behavior_history, behavior_candidates = behavior

    if not behavior_history:
        continue
    
    user_history_nids = behavior_history.split(' ')
    user_history_nids.reverse()
    user_history_articles = get_articles(user_history_nids, articles)

    candidates = behavior_candidates.split(' ')
    candidates_nids = process_candidates(candidates)
    gt_relevance_scores = process_labels(candidates)

    recommendations_collection = {recommender: recommender_to_df[recommender][recommender_to_df[recommender]['impr_index'] + 1 == behavior_id]['pred_rank'].values.tolist()[0] for recommender in recommenders}

    candidate_articles = get_articles(candidates_nids, articles)

    for recommender in recommendations_collection:

        recommendations = recommendations_collection[recommender]

        if len(recommendations) != len(candidate_articles):
            print(f"Skipping {recommender}, {behavior_id} due to length mismatch")
            continue

        predicted_relevance_scores = [-recommendation for recommendation in recommendations]
        
        ndcg_value = ndcg_score([gt_relevance_scores], [predicted_relevance_scores], k=ndcg_cutoff)
        results['ndcg_values'][recommender].append(ndcg_value)

        recommendation_articles = [candidate_articles[i-1] for i in recommendations][:radio_cutoff]
        topic_divergence, complexity_divergence = Calibration.calculate(user_history_articles, recommendation_articles)

        topic_jsd_with_discount = topic_divergence[0][1]
        complexity_jsd_with_discount = complexity_divergence[0][1]
        results['topic_calibrations'][recommender].append(topic_jsd_with_discount)
        results['complexity_calibrations'][recommender].append(complexity_jsd_with_discount)

        rec_sample = recommender_to_df[recommender].sample(5)
        frag_articles = []
        for rec_sample_index, rec_sample_row in rec_sample.iterrows():
            sample_impr = rec_sample_row['impr_index'] + 1
            sample_recs = rec_sample_row['pred_rank']

            samples_behavior = behaviors[behaviors[0] == sample_impr].iloc[0]
            samples_behavior_candidates = samples_behavior[2].split(' ')
            samples_candidates_nids = process_candidates(samples_behavior_candidates)

            sample_ranking_nids = [samples_candidates_nids[i-1] for i in sample_recs][:radio_cutoff]
            frag_articles.append(get_articles(sample_ranking_nids, articles))

        fragmentation = Fragmentation.calculate(frag_articles, recommendation_articles)
        fragmentation_jsd_with_discount = fragmentation[0][1]
        results['fragmentations'][recommender].append(fragmentation_jsd_with_discount)

        activation = Activation.calculate(candidate_articles, recommendation_articles)
        activation_jsd_with_discount = activation[0][1]
        results['activations'][recommender].append(activation_jsd_with_discount)

        representation = Representation.calculate(candidate_articles, recommendation_articles)
        if representation:
            representation_jsd_with_discount = representation[0][1]
            results['representations'][recommender].append(representation_jsd_with_discount)

        alternative_voices = AlternativeVoices.calculate(candidate_articles, recommendation_articles)
        if alternative_voices:
            alternative_voices_jsd_with_discount = alternative_voices[0][1]
            results['alternative_voices'][recommender].append(alternative_voices_jsd_with_discount)

results_df = pd.DataFrame.from_dict(results)
results_df.to_csv(f'results/mind_results_sampled.csv')