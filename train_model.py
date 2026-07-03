import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


df = pd.read_csv("data.csv")


df['text'] = df['tweet_text']


df['label'] = df['cyberbullying_type'].apply(
    lambda x: 0 if x == "not_cyberbullying" else 1
)


def clean(text):
    return re.sub(r'[^a-zA-Z ]', '', str(text).lower())

df['text'] = df['text'].apply(clean)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']


model = LinearSVC()
model.fit(X, y)

joblib.dump(model, "model/svm_model.pkl")
joblib.dump(vectorizer, "model/tfidf.pkl")

print("Model trained!")    model train.py
