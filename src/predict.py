import joblib
from src.preprocess import clean_text

model = joblib.load("models/best_model.joblib")
bow = joblib.load("models/bow_vectorizer.joblib")

def predict_sentiment(text: str) -> str:
    vec = bow.transform([clean_text(text)])
    return "Positive Sentiment" if model.predict(vec)[0] == 1 else "Negative Sentiment"

if __name__ == "__main__":
    print(predict_sentiment("The product quality is excellent and delivery was very fast."))
    print(predict_sentiment("The product stopped working after a week. Very disappointed."))