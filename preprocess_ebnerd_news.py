import pandas as pd
import dacy
import re
from tqdm import tqdm

# Initialize tqdm for pandas
tqdm.pandas()

def lix_score(text: str) -> float:
    sentences = re.split(r'[\.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sent = max(len(sentences), 1)
    words = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    num_words = max(len(words), 1)
    num_long = sum(1 for w in words if len(w) > 6)
    
    complexity = num_words / num_sent + (num_long * 100.0) / num_words
    return complexity

def sentiment_score(nlp, text):
    """
    Compute a continuous sentiment score in [-1, 1] for Danish text
    using DaCy's `dacy/polarity` model.
    """
    doc = nlp(text)
    # polarity_prob is a dict: {'prob': array([pos, neu, neg]), 'labels': [...]}
    probs = doc._.polarity_prob["prob"]
    pos, neu, neg = probs  # arrays of shape (3,)
    # Map to a single score: +1 * P(pos) + 0 * P(neu) + (–1) * P(neg)
    score = float(pos - neg)
    return score

def main():
    nlp = dacy.load("small")
    # Add the polarity component to the pipeline
    nlp.add_pipe("dacy/polarity") 

    print('Preprocessing Ebnerd News dataset...')
    news_df = pd.read_parquet('data/ebnerd/train/articles.parquet')
    news_df['text'] = news_df['title'] + ' ' + news_df['subtitle']

    # Sentiment
    print('Calculating sentiment...')
    news_df['sentiment'] = news_df['text'].progress_apply(lambda x: sentiment_score(nlp, x))

    # Complexity (LIX)
    print('Calculating complexity...')
    news_df['complexity'] = news_df['text'].progress_apply(lix_score)

    print(news_df[['sentiment', 'complexity']].head())

    # Save the preprocessed data to 'articles_ebnerd.pickle'
    print('Saving preprocessed data...')
    news_df.to_pickle('data/ebnerd/articles_ebnerd.pickle')
    print('Done.')


if __name__ == '__main__':
    main()