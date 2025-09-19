import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path='data/reviews.csv'):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Multi-class sentiment labeling
    positive_words = ['amazing', 'great', 'excellent', 'tasty', 'fresh', 'friendly', 'fast']
    negative_words = ['cold', 'bad', 'soggy', 'overcooked', 'dry', 'bland', 'slow']

    def label_sentiment(text):
        text_lower = text.lower()
        if any(word in text_lower for word in positive_words):
            return 2  # Positive
        elif any(word in text_lower for word in negative_words):
            return 0  # Negative
        else:
            return 1  # Neutral

    df['sentiment'] = df['review'].apply(label_sentiment)
    return df

def split_data(df, test_size=0.2):
    X = df['review']
    y = df['sentiment']
    return train_test_split(X, y, test_size=test_size, random_state=42)
