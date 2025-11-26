import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import json
import random
from pathlib import Path
from typing import Sequence

import pandas as pd
from tqdm import tqdm


VAL_BEHAVIOR_CANDIDATES = [
    Path("data/mind/MINDlarge_dev/behaviors.tsv"),
    Path("data/mind/val/behaviors.tsv"),
]
TRAIN_BEHAVIOR_CANDIDATES = [
    Path("data/mind/MINDlarge_train/behaviors.tsv"),
    Path("data/mind/train/behaviors.tsv"),
]


def _resolve_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the paths exist: {', '.join(str(p) for p in candidates)}")


def _scores_from_ranks(ranks: Sequence[int]) -> list[float]:
    """Return a monotonic score vector aligned with 1-indexed ranks."""
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

def generate_incorrect_random():

    prediction_file = f'incorrect_random_prediction.json'
    scored_file = f'incorrect_random_prediction_with_rel.json'
    example_file = f'random_prediction.json'

    with open('data/recommendations/mind/' + prediction_file, 'w') as write_file, \
        open('data/recommendations/mind/' + scored_file, 'w') as scored_file_handle:
        with open('data/recommendations/mind/' + example_file , 'r') as read_file:
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
    behavior_path = _resolve_path(VAL_BEHAVIOR_CANDIDATES)

    with open('data/recommendations/mind/' + prediction_file, 'w') as write_file, \
        open('data/recommendations/mind/' + scored_file, 'w') as scored_file_handle:

        valid_behaviors = pd.read_csv(
            behavior_path,
            delimiter='\t',
            header=None,
        )
        valid_behaviors.columns = list(range(valid_behaviors.shape[1]))

        for _, row in tqdm(valid_behaviors.iterrows()):
            impr_index = row.get(0)
            if pd.isna(impr_index):
                continue
            impr_index = int(impr_index)
            candidates_raw = row.get(4)
            if not isinstance(candidates_raw, str):
                continue
            candidates = [candidate.split('-')[0] for candidate in candidates_raw.split()]

            random_candidates = candidates.copy()
            random.shuffle(random_candidates)
            
            # create a list of indices for the candidates where the index is the index of the sorted candidate in the list
            random_recommendations = [random_candidates.index(candidate) + 1 for candidate in candidates]

            payload = {
                'impr_index': impr_index,
                'pred_rank': random_recommendations,
            }

            _write_prediction(write_file, scored_file_handle, payload, random_recommendations)

def generate_pop():

    train_behaviors_original = pd.read_csv(
        _resolve_path(TRAIN_BEHAVIOR_CANDIDATES),
        delimiter='\t',
        header=None,
    )
    train_behaviors_original.columns = list(range(train_behaviors_original.shape[1]))
    pop_count_dict = {}

    # Iterate through each row in the specified column
    for index, row in train_behaviors_original.iterrows():
        history_raw = row.get(3)
        if isinstance(history_raw, str) and history_raw.strip():
            # Split the text by whitespace
            history = history_raw.split()
            # Iterate through each word
            for article in history:
                # Update the count in the dictionary
                pop_count_dict[article] = pop_count_dict.get(article, 0) + 1  
        
        candidates_raw = row.get(4)
        if isinstance(candidates_raw, str) and candidates_raw.strip():
            candidates = candidates_raw.split()
            for article in candidates:
                article = article.split('-')
                article, label = article[0], article[1]
                if label == '1':
                    pop_count_dict[article] = pop_count_dict.get(article, 0) + 1

    
    prediction_file = f'pop_prediction.json'
    scored_file = f'pop_prediction_with_rel.json'
    behavior_path = _resolve_path(VAL_BEHAVIOR_CANDIDATES)

    with open('data/recommendations/mind/' + prediction_file, 'w') as write_file, \
        open('data/recommendations/mind/' + scored_file, 'w') as scored_file_handle:

        valid_behaviors = pd.read_csv(
            behavior_path,
            delimiter='\t',
            header=None,
        )
        valid_behaviors.columns = list(range(valid_behaviors.shape[1]))
        
        for _, row in tqdm(valid_behaviors.iterrows()):
            impr_index = int(row.get(0))
            candidates_raw = row.get(4)
            if not isinstance(candidates_raw, str) or not candidates_raw.strip():
                continue

            candidates = [candidate.split('-')[0] for candidate in candidates_raw.split()]

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
