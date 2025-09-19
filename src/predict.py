import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

label_map = {0:'Negative', 1:'Neutral', 2:'Positive'}

def predict_sentiment(reviews):
    with open('outputs/sentiment_model.pkl', 'rb') as f:
        model, vectorizer = pickle.load(f)
    X_vect = vectorizer.transform(reviews)
    predictions = model.predict(X_vect)
    return predictions

def save_predictions(df, predictions, file_path='outputs/predictions.csv'):
    df['predicted_sentiment'] = [label_map[p] for p in predictions]
    df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")

def evaluate_model(y_true, y_pred, return_fig=False):
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    # Display metrics if not returning figure
    if not return_fig:
        st.write("**Multi-class Model Evaluation Metrics:**")
        st.write(f"Accuracy: {acc:.2f}")
        st.write(f"Precision: {prec:.2f}")
        st.write(f"Recall: {rec:.2f}")
        st.write(f"F1-score: {f1:.2f}")

    # ------------------------
    # Side-by-side plots
    # ------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative','Neutral','Positive'],
                yticklabels=['Negative','Neutral','Positive'],
                ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title('Confusion Matrix')

    # Sentiment Distribution
    sns.countplot(x=y_pred, ax=axes[1])
    axes[1].set_xticks([0,1,2])
    axes[1].set_xticklabels(['Negative','Neutral','Positive'])
    axes[1].set_title('Predicted Sentiment Distribution')

    # Return or display figure
    if return_fig:
        return fig
    else:
        st.pyplot(fig)
