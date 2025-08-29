import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from preprocess import clean_text, preprocess

def train_model(data_path="data/Tweets.csv", model_path="sentiment_model.pkl"):
    # Load dataset
    df = pd.read_csv(data_path)
    df = df[['text','airline_sentiment']]
    df = df[df['airline_sentiment'] != 'neutral']  # binary only
    df['target'] = df['airline_sentiment'].map({'negative':0, 'positive':1})

    # Clean + preprocess
    df['clean_text'] = df['text'].apply(clean_text)
    df['processed_text'] = df['clean_text'].apply(preprocess)

    # TF-IDF
    X = df['processed_text']
    y = df['target']
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model + vectorizer
    joblib.dump(model, model_path)
    joblib.dump(tfidf, "tfidf.pkl")

    print("✅ Model trained and saved!")

if __name__ == "__main__":
    train_model()
