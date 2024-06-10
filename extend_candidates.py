import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def extend_candidates(article_nids):

    for seed in range(1, 6):
        random.seed(seed)
        for candidate_size in [10, 20, 40, 60, 80, 100]:

            behaviors = pd.read_csv('data/MIND/MINDlarge_dev/behaviors.tsv', delimiter='\t', header=None, names=['uid', 'datetime', 'history', 'candidates'])
            behaviors['date'] = pd.to_datetime(behaviors['datetime'])

            for index, behavior in tqdm(behaviors.iterrows()):
                
                candidates = behavior['candidates'].split()
                candidate_nids = [candidate.split('-')[0] for candidate in candidates]

                correct_candidate = None
                for candidate in candidates:
                    if candidate.split('-')[1] == '1':
                        correct_candidate = candidate.split('-')[0]
                    
                articles_of_that_day = random.sample(article_nids, candidate_size)
                nids_of_that_day = [article['article_id'] for article in articles_of_that_day]

                new_candidates = []
                for nid_of_that_day in nids_of_that_day:
                    new_candidates.append(nid_of_that_day)
                new_candidates += candidate_nids
                new_candidates = set(new_candidates)

                new_candidates_final = []
                for new_candidate in new_candidates:
                    if new_candidate == correct_candidate:
                        new_candidates_final.append(new_candidate + '-1')
                    else:
                        new_candidates_final.append(new_candidate + '-0')

                behaviors.loc[index, 'candidates'] = ' '.join(list(new_candidates_final))
                
            behaviors = behaviors.drop(columns=['date'])
            behaviors.to_csv(f'data/extended_candidates/new_behaviors_{candidate_size}_sample_{seed}.tsv', sep='\t', index=False, header=False)

if __name__ == '__main__':

    articles_df = pd.read_csv('data/MIND/MINDlarge_dev/news.tsv', delimiter='\t', header=None, names=['article_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    articles_df = articles_df.replace({np.nan: None})
    article_nids = articles_df['article_id'].values

    extend_candidates(article_nids)