import json
import pandas as pd
import random
from tqdm import tqdm

def generate_incorrect_random():

    # for sample in range(1, 6):
    #     for num_candidates in [10, 20, 40, 60, 80, 100]:

    prediction_file = f'random_prediction.json'
    example_file = f'behaviors.tsv'

    with open('data/recommendations/' + prediction_file, 'w') as write_file:
        with open('data/recommendations/' + example_file , 'r') as read_file:
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
    behavior_file = f'behaviors.tsv'

    with open('data/recommendations/' + prediction_file, 'w') as write_file:

        valid_behaviors = pd.read_csv('data/MIND/MINDlarge_dev/' + behavior_file, delimiter='\t', header=None, names=['uid', 'date', 'history', 'candidates'])

        for index, line in tqdm(valid_behaviors.iterrows()):
            candidates = [candidate.split('-')[0] for candidate in line['candidates'].split()]

            random_candidates = candidates.copy()
            random.shuffle(random_candidates)
            
            # create a list of indices for the candidates where the index is the index of the sorted candidate in the list
            random_recommendations = [random_candidates.index(candidate) + 1 for candidate in candidates]

            json_dict = {
                'impr_index': index,
                'pred_rank': random_recommendations
            }

            json.dump(json_dict, write_file)
            write_file.write("\n")

def generate_pop():

    valid_behaviors_original = pd.read_csv('data/MIND/MINDlarge_dev/behaviors.tsv', delimiter='\t', header=None, names=['uid', 'date', 'history', 'candidates'])
    train_behaviors_original = pd.read_csv('data/MIND/MINDlarge_train/behaviors.tsv', delimiter='\t', header=None, names=['uid', 'date', 'history', 'candidates'])

    pop_count_dict = {}

    # Iterate through each row in the specified column
    for index, row in valid_behaviors_original.iterrows():
        print(row['history'])
        if pd.notna(row['history']):
            # Split the text by whitespace
            history = row['history'].split()
            # Iterate through each word
            for article in history:
                # Update the count in the dictionary
                pop_count_dict[article] = pop_count_dict.get(article, 0) + 1

    # Iterate through each row in the specified column
    for index, row in train_behaviors_original.iterrows():
        if pd.notna(row['history']):
            # Split the text by whitespace
            history = row['history'].split()
            # Iterate through each word
            for article in history:
                # Update the count in the dictionary
                pop_count_dict[article] = pop_count_dict.get(article, 0) + 1  
        
        if pd.notna(row['candidates']):
            candidates = row['candidates'].split()
            for article in candidates:
                article = article.split('-')
                article, label = article[0], article[1]
                if label == '1':
                    pop_count_dict[article] = pop_count_dict.get(article, 0) + 1

    # for sample in range(1, 6):
    #     for num_candidates in [10, 20, 40, 60, 80, 100]:
    
    prediction_file = f'pop_prediction.json'
    behavior_file = f'behaviors.tsv'

    with open('data/recommendations/' + prediction_file, 'w') as write_file:

        valid_behaviors = pd.read_csv('data/MIND/MINDsmall_dev/' + behavior_file, delimiter='\t', header=None, names=['uid', 'date', 'history', 'candidates'])
        
        for index, line in tqdm(valid_behaviors.iterrows()):
            
            candidates = [candidate.split('-')[0] for candidate in line['candidates'].split()]

            sorted_candidates = sorted(candidates, key=lambda x: pop_count_dict.get(x, 0), reverse=True)
            
            # create a list of indices for the candidates where the index is the index of the sorted candidate in the list
            pop_recommendations = [sorted_candidates.index(candidate) + 1 for candidate in candidates]

            json_dict = {
                'impr_index': index,
                'pred_rank': pop_recommendations
            }

            json.dump(json_dict, write_file)
            write_file.write("\n")

if __name__ == '__main__':   
    # generate_random()
    # generate_incorrect_random()
    generate_pop()