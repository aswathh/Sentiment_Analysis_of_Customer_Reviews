import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

df = pd.read_csv("data/cleaned.csv").dropna()
x_train, x_test, y_train, y_test = train_test_split(
    df["clean_review"], df["label"],
    test_size=0.2, stratify=df["label"], random_state=42,
)

model = joblib.load("models/best_model.joblib")
bow = joblib.load("models/bow_vectorizer.joblib")

pred = model.predict(bow.transform(x_test))

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["negative", "positive"]))

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(cmap="Blues")
plt.title("Confusion matrix: Logistic Regression + BoW")
plt.savefig("models/confusion_matrix.png", bbox_inches="tight")
plt.show()