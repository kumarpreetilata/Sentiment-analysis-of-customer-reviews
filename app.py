import streamlit as st
import pandas as pd
from src.data_loader import load_data, preprocess_data
from src.predict import predict_sentiment, evaluate_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load trained multi-class model
with open('outputs/sentiment_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

label_map = {0:'Negative', 1:'Neutral', 2:'Positive'}

# -----------------------
# Streamlit App Layout
# -----------------------

st.title("Restaurant Review Sentiment Analyzer (Multi-class)")

# Sidebar: Analyze a Single Review
st.sidebar.header("Analyze a Single Review")
user_review = st.sidebar.text_area("Enter your review here:")

if st.sidebar.button("Predict Sentiment"):
    if user_review.strip():
        vect = vectorizer.transform([user_review])
        pred = model.predict(vect)[0]
        sentiment = label_map[pred]
        st.sidebar.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.sidebar.warning("Please enter a review.")

# Main: Dataset display and predictions
st.header("Sample Reviews Dataset")
df = load_data()
df = preprocess_data(df)
predictions = predict_sentiment(df['review'])
df['predicted_sentiment'] = [label_map[p] for p in predictions]
st.dataframe(df)

# Evaluation metrics and charts
st.header("Evaluation & Sentiment Distribution")

# Create two columns
col1, col2 = st.columns([1, 2])  # Adjust width ratios if needed

# Metrics on the left
# Metrics on the left
with col1:
    st.subheader("Metrics")
    acc = accuracy_score(df['sentiment'], predictions)
    prec = precision_score(df['sentiment'], predictions, average='weighted')
    rec = recall_score(df['sentiment'], predictions, average='weighted')
    f1 = f1_score(df['sentiment'], predictions, average='weighted')

    st.write(f"Accuracy: {acc:.2f}")
    st.write(f"Precision: {prec:.2f}")
    st.write(f"Recall: {rec:.2f}")
    st.write(f"F1-score: {f1:.2f}")

# Plots on the right column
with col2:
    fig = evaluate_model(df['sentiment'], predictions, return_fig=True)
    st.pyplot(fig)
