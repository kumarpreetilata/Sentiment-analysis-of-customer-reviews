
Restaurant Review Sentiment Analyzer (Multi-class)

Analyze customer reviews for restaurants to detect Positive, Neutral, and Negative sentiment using TF-IDF + Logistic Regression, fully local and interactive via Streamlit.

################################################################

Project Structure:

restaurant_sentiment_local/
│
├── data/
│   └── reviews.csv             # Sample review dataset
│
├── src/
│   ├── data_loader.py          # Load & preprocess data
│   ├── sentiment_model.py      # Train & save TF-IDF + Logistic Regression model
│   ├── predict.py              # Predict sentiment and evaluate model
│   └── utils.py                # Helper functions
│
├── outputs/
│   ├── sentiment_model.pkl     # Saved model & vectorizer
│   └── predictions.csv         # Optional saved predictions
│
├── app.py                      # Streamlit web app for interactive use
├── main.py                     # Train & test pipeline (CLI)
└── requirements.txt



################################################
Setup Instructions:

python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
streamlit run app.py
