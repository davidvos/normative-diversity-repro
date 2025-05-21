"""Configuration parameters for the recommendation system."""

DATASET_CONFIG = {
    'mind': {
        'behaviors_path': 'data/mind/MINDlarge_train/behaviors.tsv',
        'articles_path': 'data/mind/articles_mind.pickle',
        'model_name': 'all-MiniLM-L6-v2',
        'recommenders': ['pop', 'random', 'bert4rec', 'sasrec', 'gru4rec', 'npa', 'nrms', 'lstur', 'naml']
    },
    'ebnerd': {
        'behaviors_path': 'data/ebnerd/train/behaviors.tsv',
        'articles_path': 'data/ebnerd/articles_ebnerd.pickle',
        'model_name': 'all-MiniLM-L6-v2',
        'recommenders': ['pop', 'random', 'bert4rec', 'sasrec', 'gru4rec', 'npa', 'nrms', 'lstur', 'naml']
    }
}

# General configuration
NDCG_CUTOFF = 10
RADIO_CUTOFF = 10

# Output configuration
OUTPUT_FOLDER = 'results/' 