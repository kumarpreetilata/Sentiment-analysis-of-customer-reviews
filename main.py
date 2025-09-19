from src.data_loader import load_data, preprocess_data, split_data
from src.sentiment_model import train_model
from src.predict import predict_sentiment, save_predictions, evaluate_model
from src.utils import print_sample_predictions
import pandas as pd

# 1. Load and preprocess
df = load_data()
df = preprocess_data(df)

# 2. Split data
X_train, X_test, y_train, y_test = split_data(df)

# 3. Train model
model, vectorizer = train_model(X_train, y_train)

# 4. Predict on test data
predictions = predict_sentiment(X_test)

# 5. Save predictions
save_predictions(pd.DataFrame({'review': X_test}), predictions)

# 6. Evaluate
evaluate_model(y_test, predictions)

# 7. Sample predictions
print_sample_predictions(pd.DataFrame({'review': X_test, 'predicted_sentiment': predictions}))
