import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X_train_vect = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train_vect, y_train)
    
    # Save model and vectorizer
    with open('outputs/sentiment_model.pkl', 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print("Multi-class model saved to outputs/sentiment_model.pkl")
    return model, vectorizer
