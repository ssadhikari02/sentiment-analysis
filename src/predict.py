import joblib
from preprocess import clean_text, preprocess

def predict_sentiment(text, model_path="sentiment_model.pkl"):
    model = joblib.load(model_path)
    tfidf = joblib.load("tfidf.pkl")
    
    clean = preprocess(clean_text(text))
    vector = tfidf.transform([clean])
    prediction = model.predict(vector)[0]

    return "Positive" if prediction == 1 else "Negative"

if __name__ == "__main__":
    sample = "This flight was terrible, I will never use this airline again!"
    print("Input:", sample)
    print("Prediction:", predict_sentiment(sample))
