import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

Stop = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
Lemm = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(Lemm.lemmatize(w) for w in text.split() if w not in Stop)

def load_data(path: str = "data/IMDB Dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df