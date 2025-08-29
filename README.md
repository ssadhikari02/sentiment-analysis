![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)  ![License](https://img.shields.io/badge/license-MIT-green.svg)  ![Status](https://img.shields.io/badge/status-active-success.svg)  ![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

# ✈️ Sentiment Analysis on Airline Tweets

## 📌 Overview
This project analyzes airline-related tweets and predicts whether they are **positive** or **negative** using **NLP + Machine Learning**.

- Dataset: [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- Techniques: Data Cleaning, Lemmatization, TF-IDF, Logistic Regression
- Accuracy: ~80%
- Outputs: Top words analysis, Confusion Matrix, Custom predictions

## 📂 Project Structure
```
Sentiment Analysis/
│── data/                  # dataset(not uploaded, see below)
│── src/
│   ├── preprocess.py       # text cleaning & preprocessing
│   ├── train.py            # train model and save .pkl files
│   ├── evaluate.py         # evaluate model performance
│   └── predict.py          # predict custom text sentiment
│── requirements.txt        # dependencies
│── sentiment_notebook.ipynb # optional Jupyter/Colab notebook
│── README.md               # project documentation
```


## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Train the model:
   ```bash
   python src/train.py

3. Evaluate the model:
   ```bash
   python src/evaluate.py

4. Test predictions:
   ```bash
   python src/predict.py

## 🏆 Results
   - Positive tweets often mention words like: thank, great, love.
   - Negative tweets often include: delay, cancelled, bad.
   - Achieved ~80% accuracy on test data.

## 🏆 Key Learnings
   - Hands-on experience with text preprocessing.
   - Used TF-IDF for feature extraction.
   - Applied Logistic Regression for binary classification.
   - Understood evaluation metrics: accuracy, precision, recall, F1-score.
   - Built a modular project structure for professional deployment.

## 👨‍💻 Author
Sachin Singh Adhikari