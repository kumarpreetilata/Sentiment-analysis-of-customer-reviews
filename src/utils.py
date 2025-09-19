def print_sample_predictions(df, n=5):
    print(df[['review', 'predicted_sentiment']].head(n))
