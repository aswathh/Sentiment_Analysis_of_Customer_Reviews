import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocess import load_data
from src.feature import review_vector

# ---------- Data ----------
df = load_data()
df[["clean_review", "label"]].to_csv("data/cleaned.csv", index=False)

x_train, x_test, y_train, y_test = train_test_split(
    df["clean_review"], df["label"],
    test_size=0.2, stratify=df["label"], random_state=42,
)

# Feature engineering
bow = CountVectorizer(max_features=20000, ngram_range=(1, 2))
x_train_bow = bow.fit_transform(x_train)
x_test_bow = bow.transform(x_test)

train_tokens = [t.split() for t in x_train]
test_tokens = [t.split() for t in x_test]
w2v = Word2Vec(sentences=train_tokens, vector_size=100, window=5,
               min_count=2, workers=4, epochs=10, seed=42)
x_train_w2v = np.array([review_vector(t, w2v) for t in train_tokens])
x_test_w2v = np.array([review_vector(t, w2v) for t in test_tokens])

fitted = {}

#Naive Bayes
nb = MultinomialNB()
nb.fit(x_train_bow, y_train)
pred = nb.predict(x_test_bow)
nb_bow = [accuracy_score(y_test, pred), precision_score(y_test, pred),
          recall_score(y_test, pred), f1_score(y_test, pred)]
fitted[("Naive Bayes", "BoW")] = nb


#Logistic Regression
lr_b = LogisticRegression(max_iter=1000)
lr_b.fit(x_train_bow, y_train)
pred = lr_b.predict(x_test_bow)
lr_bow = [accuracy_score(y_test, pred), precision_score(y_test, pred),
          recall_score(y_test, pred), f1_score(y_test, pred)]
fitted[("Logistic Regression", "BoW")] = lr_b

lr_w = LogisticRegression(max_iter=1000)
lr_w.fit(x_train_w2v, y_train)
pred = lr_w.predict(x_test_w2v)
lr_w2v = [accuracy_score(y_test, pred), precision_score(y_test, pred),
          recall_score(y_test, pred), f1_score(y_test, pred)]
fitted[("Logistic Regression", "Word2Vec")] = lr_w


#SVM 
svm_b = LinearSVC()
svm_b.fit(x_train_bow, y_train)
pred = svm_b.predict(x_test_bow)
svm_bow = [accuracy_score(y_test, pred), precision_score(y_test, pred),
           recall_score(y_test, pred), f1_score(y_test, pred)]
fitted[("SVM", "BoW")] = svm_b

svm_w = LinearSVC()
svm_w.fit(x_train_w2v, y_train)
pred = svm_w.predict(x_test_w2v)
svm_w2v = [accuracy_score(y_test, pred), precision_score(y_test, pred),
           recall_score(y_test, pred), f1_score(y_test, pred)]
fitted[("SVM", "Word2Vec")] = svm_w


#Random Forest
rf_b = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_b.fit(x_train_bow, y_train)
pred = rf_b.predict(x_test_bow)
rf_bow = [accuracy_score(y_test, pred), precision_score(y_test, pred),
          recall_score(y_test, pred), f1_score(y_test, pred)]
fitted[("Random Forest", "BoW")] = rf_b

rf_w = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf_w.fit(x_train_w2v, y_train)
pred = rf_w.predict(x_test_w2v)
rf_w2v = [accuracy_score(y_test, pred), precision_score(y_test, pred),
          recall_score(y_test, pred), f1_score(y_test, pred)]
fitted[("Random Forest", "Word2Vec")] = rf_w


# Comparison 
comparison = pd.DataFrame(
    [
        ["Naive Bayes",         "BoW"]      + nb_bow,
        ["Logistic Regression", "BoW"]      + lr_bow,
        ["Logistic Regression", "Word2Vec"] + lr_w2v,
        ["SVM",                 "BoW"]      + svm_bow,
        ["SVM",                 "Word2Vec"] + svm_w2v,
        ["Random Forest",       "BoW"]      + rf_bow,
        ["Random Forest",       "Word2Vec"] + rf_w2v,
    ],
    columns=["Model", "Features", "Accuracy", "Precision", "Recall", "F1"],
).sort_values("F1", ascending=False).reset_index(drop=True)

print("\n", comparison.round(4))
comparison.to_csv("models/results.csv", index=False)

# Save 
best = comparison.iloc[0]
artifact = {
    "model_name": best["Model"],
    "features": best["Features"],
    "clf": fitted[(best["Model"], best["Features"])],
    "bow": bow if best["Features"] == "BoW" else None,
    "w2v": w2v if best["Features"] == "Word2Vec" else None,
}
joblib.dump(artifact, "models/best_model.joblib")
print(f"\nBest: {best['Model']} + {best['Features']} (F1={best['F1']:.4f}) saved")