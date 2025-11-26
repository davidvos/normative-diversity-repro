import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import json
import pandas as pd
import random
from tqdm import tqdm
import numpy as np
from typing import Sequence


def _scores_from_ranks(ranks: Sequence[int]) -> list[float]:
    length = len(ranks)
    if length <= 0:
        return []
    denom = float(length)
    scores: list[float] = []
    for rank in ranks:
        try:
            value = int(rank)
        except (TypeError, ValueError):
            value = length
        value = max(1, min(value, length))
        scores.append(float(length - value + 1) / denom)
    return scores


def _write_prediction(plain_handle, scored_handle, payload: dict, ranks: Sequence[int]) -> None:
    json.dump(payload, plain_handle)
    plain_handle.write("\n")
    if scored_handle is not None:
        scored_payload = dict(payload)
        scored_payload["pred_rel"] = _scores_from_ranks(ranks)
        json.dump(scored_payload, scored_handle)
        scored_handle.write("\n")

def _sequence_to_list(field):
    """Return a list representation for optional sequence-like fields."""
    if field is None:
        return []
    if isinstance(field, str):
        tokens = field.split()
        return [token for token in tokens if token]
    if isinstance(field, np.ndarray):
        return [item for item in field.tolist() if item not in ("", None)]
    if isinstance(field, (list, tuple, set)):
        return [item for item in field if item not in ("", None)]
    return [field]

def generate_incorrect_random():

    prediction_file = f'incorrect_random_prediction.json'
    scored_file = f'incorrect_random_prediction_with_rel.json'
    example_file = f'random_prediction.json'

    with open('data/recommendations/ebnerd/' + prediction_file, 'w') as write_file, \
        open('data/recommendations/ebnerd/' + scored_file, 'w') as scored_file_handle:
        with open('data/recommendations/ebnerd/' + example_file , 'r') as read_file:
            for index, line in tqdm(enumerate(read_file)):
                parsed_line = json.loads(line)
                pred_rank = parsed_line['pred_rank']

                incorrect_random_recommendations = []
                for _ in range(len(pred_rank)):
                    random_item = random.randint(1, len(pred_rank))
                    incorrect_random_recommendations.append(random_item)

                payload = {
                    'impr_index': parsed_line['impr_index'],
                    'pred_rank': incorrect_random_recommendations,
                }

                _write_prediction(write_file, scored_file_handle, payload, incorrect_random_recommendations)

def generate_random():

    prediction_file = f'random_prediction.json'
    scored_file = f'random_prediction_with_rel.json'
    behavior_file = f'behaviors_parsed.tsv'

    with open('data/recommendations/ebnerd/' + prediction_file, 'w') as write_file, \
        open('data/recommendations/ebnerd/' + scored_file, 'w') as scored_file_handle:

        valid_behaviors = pd.read_table(
            f'data/ebnerd/val/{behavior_file}',
            delimiter='\t',
            header=None,
            names=['impression_id', 'user', 'time', 'history', 'candidates'],
        )
        valid_behaviors = valid_behaviors.replace({np.nan: None})

        for index, line in tqdm(valid_behaviors.iterrows()):
            candidate_tokens = _sequence_to_list(line['candidates'])
            if not candidate_tokens:
                continue

            candidates = [str(candidate).split('-')[0] for candidate in candidate_tokens]

            random_candidates = candidates.copy()
            random.shuffle(random_candidates)
            
            # create a list of indices for the candidates where the index is the index of the sorted candidate in the list
            random_recommendations = [random_candidates.index(candidate)+1 for candidate in candidates]

            impr_index = line['impression_id']
            try:
                impr_index = int(impr_index)
            except (TypeError, ValueError):
                impr_index = index + 1

            payload = {
                'impr_index': impr_index,
                'pred_rank': random_recommendations,
            }

            _write_prediction(write_file, scored_file_handle, payload, random_recommendations)

def generate_pop():

    train_behaviors = pd.read_parquet('data/ebnerd/train/behaviors.parquet')
    train_histories = pd.read_parquet('data/ebnerd/train/history.parquet')
    train_behaviors = train_behaviors.merge(train_histories, on='user_id', how='left')

    train_behaviors = train_behaviors[['impression_id', 'user_id', 'impression_time', 'article_id_fixed', 
                           'article_ids_inview', 'article_ids_clicked']]
    train_behaviors.columns = ['impression_id', 'uid', 'date', 'history', 'candidates', 'labels']
    train_behaviors = train_behaviors.replace({np.nan: None})

    pop_count_dict = {}

    # Iterate through each row in the specified column
    for index, row in train_behaviors.iterrows():
        history_items = _sequence_to_list(row['history'])
        for article in history_items:
            pop_count_dict[article] = pop_count_dict.get(article, 0) + 1
        
        label_items = _sequence_to_list(row['labels'])
        for label in label_items:
            pop_count_dict[label] = pop_count_dict.get(label, 0) + 1
    
    prediction_file = f'pop_prediction.json'
    scored_file = f'pop_prediction_with_rel.json'

    with open('data/recommendations/ebnerd/' + prediction_file, 'w') as write_file, \
        open('data/recommendations/ebnerd/' + scored_file, 'w') as scored_file_handle:

        valid_behaviors = pd.read_parquet('data/ebnerd/val/behaviors.parquet')
        valid_histories = pd.read_parquet('data/ebnerd/val/history.parquet')
        valid_behaviors = valid_behaviors.merge(valid_histories, on='user_id', how='left')
        valid_behaviors = valid_behaviors[['impression_id', 'user_id', 'impression_time', 'article_id_fixed',
                            'article_ids_inview', 'article_ids_clicked']]
        valid_behaviors.columns = ['impression_id', 'uid', 'date', 'history', 'candidates', 'labels']
        valid_behaviors = valid_behaviors.replace({np.nan: None})
        
        for index, line in tqdm(valid_behaviors.iterrows()):
            candidates = _sequence_to_list(line['candidates'])
            if not candidates:
                continue
            impr_index = line['impression_id']
            try:
                impr_index = int(impr_index)
            except (TypeError, ValueError):
                impr_index = index + 1

            sorted_candidates = sorted(candidates, key=lambda x: pop_count_dict.get(x, 0), reverse=True)
            
            # create a list of indices for the candidates where the index is the index of the sorted candidate in the list
            pop_recommendations = [sorted_candidates.index(candidate) + 1 for candidate in candidates]

            payload = {
                'impr_index': impr_index,
                'pred_rank': pop_recommendations,
            }

            _write_prediction(write_file, scored_file_handle, payload, pop_recommendations)

if __name__ == '__main__':   
    generate_random()
    generate_incorrect_random()
    generate_pop()
