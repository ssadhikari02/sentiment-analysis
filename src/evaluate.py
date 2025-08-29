import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from preprocess import clean_text, preprocess

def evaluate_model(data_path="data/Tweets.csv", model_path="sentiment_model.pkl"):
    # Load dataset
    df = pd.read_csv(data_path)
    df = df[['text','airline_sentiment']]
    df = df[df['airline_sentiment'] != 'neutral']
    df['target'] = df['airline_sentiment'].map({'negative':0, 'positive':1})

    # Preprocess
    df['clean_text'] = df['text'].apply(clean_text)
    df['processed_text'] = df['clean_text'].apply(preprocess)

    # Features
    tfidf = joblib.load("tfidf.pkl")
    X = tfidf.transform(df['processed_text'])
    y = df['target']

    # Load model
    model = joblib.load(model_path)
    y_pred = model.predict(X)

    # Metrics
    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred))

    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
