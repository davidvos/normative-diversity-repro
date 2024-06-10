import pandas as pd
import numpy as np

def read_mind_files(articles_path, behavior_path):
    articles_df = pd.read_pickle(articles_path)  

    behavior_df = pd.read_csv(behavior_path, delimiter='\t', header=None)
    behavior_df = behavior_df.replace({np.nan: None})
    return articles_df, behavior_df

def read_recommendation_files(lstur_path, naml_path, nrms_path, npa_path, incorrect_random_path, random_path, pop_path):
    lstur_df = pd.read_json(lstur_path, lines=True)
    naml_df = pd.read_json(naml_path, lines=True)
    nrms_df = pd.read_json(nrms_path, lines=True)
    npa_df = pd.read_json(npa_path, lines=True)
    incorrect_random_df = pd.read_json(incorrect_random_path, lines=True)
    random_df = pd.read_json(random_path, lines=True)
    pop_df = pd.read_json(pop_path, lines=True)
    return lstur_df, naml_df, nrms_df, npa_df, incorrect_random_df, random_df, pop_df

def get_articles(article_ids, articles_df):
    articles_df = articles_df[articles_df.article_id.isin(article_ids)].set_index('article_id', drop=False).reindex(article_ids)
    articles = articles_df[['article_id', 'category', 'subcategory', 'sentiment', 'complexity', 'text', 'publication_date', 'entities_base', 'enriched_entities', 'story']].to_dict('records')
    # articles = articles_df[['article_id', 'category', 'subcategory']].to_dict('records')

    return articles

def process_candidates(candidates):
    return [candidate.split('-')[0] for candidate in candidates]

def process_labels(candidates):
    return [int(candidate.split('-')[1]) for candidate in candidates]
