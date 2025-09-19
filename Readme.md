# 🍽️ Restaurant Review Sentiment Analyzer (Multi-Class)

A complete **end-to-end NLP project** to classify restaurant reviews into **Negative**, **Neutral**, or **Positive** sentiments.  
This project demonstrates data preprocessing, model training, evaluation, and deployment using **Streamlit**.

---

## Features

✅ **Multi-Class Sentiment Classification** – Negative, Neutral, Positive  
✅ **End-to-End Pipeline** – From raw dataset to predictions  
✅ **Interactive Streamlit App** – Analyze reviews in real time  
✅ **Evaluation Dashboard** – Accuracy, Classification Report, and Confusion Matrix  
✅ **Modular Codebase** – Easy to extend or retrain with new data  

---

## 📂 Project Structure

```bash
.
├── data/
│   └── restaurant_reviews.csv      # Raw dataset
├── outputs/
│   └── sentiment_model.pkl         # Trained model + vectorizer
├── src/
│   ├── data_loader.py              # Data loading & preprocessing
│   └── predict.py                  # Prediction & evaluation utilities
├── app.py                          # Streamlit app entry point
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
