import json
import pandas as pd
import random
from tqdm import tqdm
import numpy as np

def generate_incorrect_random():

    # for sample in range(1, 6):
    #     for num_candidates in [10, 20, 40, 60, 80, 100]:

    prediction_file = f'incorrect_random_prediction.json'
    example_file = f'random_prediction.json'

    with open('data/recommendations/ebnerd/' + prediction_file, 'w') as write_file:
        with open('data/recommendations/ebnerd/' + example_file , 'r') as read_file:
            for index, line in enumerate(read_file):
                parsed_line = json.loads(line)
                pred_rank = parsed_line['pred_rank']

                incorrect_random_recommendations = []
                for _ in range(len(pred_rank)):
                    random_item = random.randint(1, len(pred_rank))
                    incorrect_random_recommendations.append(random_item)

                json_dict = {
                    'impr_index': index + 1,
                    'pred_rank': incorrect_random_recommendations
                }

                json.dump(json_dict, write_file)
                write_file.write("\n")

def generate_random():

    # for sample in range(1, 6):
    #     for num_candidates in [10, 20, 40, 60, 80, 100]:  

    prediction_file = f'random_prediction.json'
    behavior_file = f'behaviors_parsed_0.tsv'

    with open('data/recommendations/ebnerd/' + prediction_file, 'w') as write_file:

        valid_behaviors = pd.read_table(f'data/ebnerd/val/{behavior_file}', delimiter='\t', header=None, names=['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidates', 'clicked'])
        valid_behaviors = valid_behaviors[['impression_id', 'user', 'time', 'clicked_news', 'impressions', 'candidates']]
        valid_behaviors = valid_behaviors.replace({np.nan: None})

        for index, line in tqdm(valid_behaviors.iterrows()):
            candidates = [candidate.split('-')[0] for candidate in line['candidates'].split()]

            random_candidates = candidates.copy()
            random.shuffle(random_candidates)
            
            # create a list of indices for the candidates where the index is the index of the sorted candidate in the list
            random_recommendations = [random_candidates.index(candidate)+1 for candidate in candidates]

            json_dict = {
                'impr_index': index + 1,
                'pred_rank': random_recommendations
            }

            json.dump(json_dict, write_file)
            write_file.write("\n")

def generate_pop():

    train_behaviors = pd.read_parquet('data/ebnerd/train/behaviors.parquet')
    train_histories = pd.read_parquet('data/ebnerd/train/history.parquet')
    train_behaviors = train_behaviors.merge(train_histories, on='user_id', how='left')

    train_behaviors = train_behaviors[['impression_id', 'user_id', 'impression_time', 'article_id_fixed', 
                           'article_ids_inview', 'article_ids_clicked']]
    train_behaviors.columns = ['impression_id', 'uid', 'date', 'history', 'candidates', 'labels']
    train_behaviors = train_behaviors.replace({np.nan: None})

    valid_behaviors = pd.read_parquet('data/ebnerd/val/behaviors.parquet')
    valid_histories = pd.read_parquet('data/ebnerd/val/history.parquet')
    valid_behaviors = valid_behaviors.merge(valid_histories, on='user_id', how='left')
    valid_behaviors = valid_behaviors[['impression_id', 'user_id', 'impression_time', 'article_id_fixed',
                           'article_ids_inview', 'article_ids_clicked']]
    valid_behaviors.columns = ['impression_id', 'uid', 'date', 'history', 'candidates', 'labels']
    valid_behaviors = valid_behaviors.replace({np.nan: None})

    pop_count_dict = {}

    # Iterate through each row in the specified column
    for index, row in valid_behaviors.iterrows():
        if len(row['history']) > 0:
            # Split the text by whitespace
            history = row['history']
            # Iterate through each word
            for article in history:
                # Update the count in the dictionary
                pop_count_dict[article] = pop_count_dict.get(article, 0) + 1

    # Iterate through each row in the specified column
    for index, row in train_behaviors.iterrows():
        if len(row['history']) > 0:
            # Split the text by whitespace
            history = row['history']
            # Iterate through each word
            for article in history:
                # Update the count in the dictionary
                pop_count_dict[article] = pop_count_dict.get(article, 0) + 1
        
        if len(row['labels']) > 0:
            labels = row['labels']
            for label in labels:
                pop_count_dict[label] = pop_count_dict.get(label, 0) + 1

    # for sample in range(1, 6):
    #     for num_candidates in [10, 20, 40, 60, 80, 100]:
    
    prediction_file = f'pop_prediction.json'

    with open('data/recommendations/ebnerd/' + prediction_file, 'w') as write_file:

        valid_behaviors = pd.read_parquet('data/ebnerd/val/behaviors.parquet')
        valid_histories = pd.read_parquet('data/ebnerd/val/history.parquet')
        valid_behaviors = valid_behaviors.merge(valid_histories, on='user_id', how='left')
        valid_behaviors = valid_behaviors[['impression_id', 'user_id', 'impression_time', 'article_id_fixed',
                            'article_ids_inview', 'article_ids_clicked']]
        valid_behaviors.columns = ['impression_id', 'uid', 'date', 'history', 'candidates', 'labels']
        valid_behaviors = valid_behaviors.replace({np.nan: None})
        
        for index, line in tqdm(valid_behaviors.iterrows()):
            
            candidates = [candidate for candidate in line['candidates']]

            sorted_candidates = sorted(candidates, key=lambda x: pop_count_dict.get(x, 0), reverse=True)
            
            # create a list of indices for the candidates where the index is the index of the sorted candidate in the list
            pop_recommendations = [sorted_candidates.index(candidate) + 1 for candidate in candidates]

            json_dict = {
                'impr_index': index + 1,
                'pred_rank': pop_recommendations
            }

            json.dump(json_dict, write_file)
            write_file.write("\n")

if __name__ == '__main__':   
    # generate_random()
    # generate_incorrect_random()
    generate_pop()