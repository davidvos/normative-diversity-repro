from utils import read_mind_files, read_recommendation_files, get_articles, process_candidates, process_labels, get_recommendations
# from metrecs.metrecs import calculate_topic_calibration, calculate_complexity_calibration, calculate_activation

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from scipy.stats import ttest_ind
from tqdm import tqdm

import dart.metrics.activation
import dart.metrics.calibration
import dart.metrics.fragmentation
import dart.metrics.representation
import dart.metrics.alternative_voices

# Importing the warning module to suppress the warning
import warnings

# Suppressing the specific warning
warnings.filterwarnings("ignore", message="In version 1.5 onwards, subsample=200_000 will be used by default.", category=FutureWarning)

config = {
  "distance": ["jsd"],
  "metrics": [
    "calibration",
    "fragmentation", 
    "activation",
    "representation",
    "alternative_voices"
    ],
  "cutoff": [10],
  "append": "N",
  "language": "english",
  "political_file": "data/term-116.csv",
  "output_folder": "output/",
    "metadata_folder": "metadata/",
  "mind_type": "small",
  "country" : "us"
}

Calibration = dart.metrics.calibration.Calibration(config)
Fragmentation = dart.metrics.fragmentation.Fragmentation()
Activation = dart.metrics.activation.Activation(config)
Representation = dart.metrics.representation.Representation(config)
AlternativeVoices = dart.metrics.alternative_voices.AlternativeVoices()

extended = False
save_results = True
candidate_size = 1000

ndcg_cutoff = 10
radio_cutoff = 10

behaviors = pd.read_csv('data/MIND/MINDlarge_dev/behaviors.tsv', delimiter='\t', header=None)
behaviors = behaviors.replace({np.nan: None})

articles = pd.read_csv('data/MIND/MINDlarge_dev/news.tsv', delimiter='\t', header=None, names=['article_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']) 

# for seed in range(1, 6):
#     for candidate_size in [200, 300, 400, 500, 600, 700, 800, 900, 980]:

new_behaviors = pd.read_csv(f'data/MIND/MINDlarge_dev/behaviors.tsv', delimiter='\t', header=None, names=['uid', 'date', 'history', 'candidates'])

lstur_df = pd.read_json(f"data/predictions_final/lstur_prediction.json", lines=True)
nrms_df = pd.read_json(f"data/predictions_final/nrms_prediction.json", lines=True)
random_df = pd.read_json(f"data/predictions_final/random_prediction.json", lines=True)
pop_df = pd.read_json(f"data/predictions_final/pop_prediction.json", lines=True)

recommender_to_df = {
    'lstur': lstur_df,
    'nrms': nrms_df,
    # 'incorrect_random': incorrect_random_df,
    'random': random_df,
    'pop': pop_df
} 

results = {}
results['topic_calibrations'] = {'lstur': [], 'nrms': [], 'pop': [], 'random': [], 'incorrect_random': []}
results['ndcg_values'] = {'lstur': [], 'nrms': [], 'pop': [], 'random': [], 'incorrect_random': []}

for progress_index, behavior in tqdm(behaviors[:100000].iterrows()):

    behavior_id = behavior[0]
    behavior_user = behavior[1]
    behavior_datetime = behavior[2]
    behavior_history = behavior[3]
    behavior_candidates = behavior[4]

    if not behavior_history:
        continue
    
    user_history_nids = behavior_history.split(' ')
    user_history_nids.reverse()
    user_history_articles = get_articles(user_history_nids, articles)

    candidates = new_behaviors[new_behaviors.index == behavior_id].iloc[0]['candidates'].split(' ')
    gt_relevance_scores = process_labels(candidates)

    candidates_nids = process_candidates(candidates)
    candidate_articles = get_articles(candidates_nids, articles)
    
    recommendations_collection = {}
    recommendations_collection['random'] = random_df[random_df['impr_index'] == behavior_id]['pred_rank'].values.tolist()[0]
    recommendations_collection['pop'] = pop_df[pop_df['impr_index'] == behavior_id + 1]['pred_rank'].values.tolist()[0]
    recommendations_collection['lstur'] = lstur_df[lstur_df['impr_index'] == behavior_id]['pred_rank'].values.tolist()[0]
    recommendations_collection['nrms'] = nrms_df[nrms_df['impr_index'] == behavior_id]['pred_rank'].values.tolist()[0]

    for recommender in recommendations_collection:
        
        recommendations = recommendations_collection[recommender]
        predicted_relevance_scores = [-recommendation for recommendation in recommendations]

        ndcg_value = ndcg_score([gt_relevance_scores], [predicted_relevance_scores], k=ndcg_cutoff)
        results['ndcg_values'][recommender].append(ndcg_value)
        recommendation_articles = [candidate_articles[i-1] for i in recommendations][:radio_cutoff]

        topic_divergence, complexity_divergence = Calibration.calculate(user_history_articles, recommendation_articles, candidate_articles, complexity=False)

        topic_jsd_with_discount = topic_divergence[0][1]
        results['topic_calibrations'][recommender].append(topic_jsd_with_discount)

if save_results:
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(f'pop_ndcg_results.csv')

for recommender in results['ndcg_values']:
    print()
    print(f"{recommender} topic calibration: {np.mean(results['topic_calibrations'][recommender])}")
    print(f"{recommender} ndcg: {np.mean(results['ndcg_values'][recommender])}")
